"""Live, read-only Vercel adapter for the BTC Alpha research terminal."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import secrets
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse

from forecast_core import HORIZONS, MODEL_VERSION, build_forecasts, classify_regime
from forecast_store import record_settle_and_score

KRAKEN_BASE = "https://api.kraken.com/0/public"
COINBASE_TICKER = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
POLYMARKET_SEARCH = "https://gamma-api.polymarket.com/public-search"
CACHE_SECONDS = 60
MAX_CROSS_VENUE_DEVIATION_BPS = 75.0

app = FastAPI(
    title="BTC Alpha Research Engine",
    version="0.2.0",
    description="Live BTC market intelligence and probabilistic research forecasts. Paper only.",
)

_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = asyncio.Lock()


async def _json(
    client: httpx.AsyncClient, url: str, *, params: dict[str, Any] | None = None
) -> Any:
    response = await client.get(url, params=params)
    response.raise_for_status()
    return response.json()


def _cached(key: str) -> Any | None:
    item = _cache.get(key)
    if item and time.monotonic() - item[0] <= CACHE_SECONDS:
        return item[1]
    return None


def _set_cache(key: str, value: Any) -> Any:
    _cache[key] = (time.monotonic(), value)
    return value


def _kraken_result(payload: dict[str, Any]) -> Any:
    errors = payload.get("error") or []
    if errors:
        raise ValueError(f"Kraken API error: {', '.join(str(error) for error in errors)}")
    result = payload.get("result")
    if not isinstance(result, dict) or not result:
        raise ValueError("Kraken returned an empty result")
    return next(value for key, value in result.items() if key != "last")


def _normalize_candles(rows: list[list[Any]]) -> list[dict[str, Any]]:
    # Kraken documents that the final row is always the current, uncommitted candle.
    completed = rows[:-1]
    if not completed:
        raise ValueError("No completed Kraken candles returned")
    output: list[dict[str, Any]] = []
    for row in completed:
        values = [float(row[index]) for index in range(1, 7)]
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("Kraken returned a non-finite or negative OHLCV value")
        if values[1] < values[2] or values[0] <= 0 or values[3] <= 0:
            raise ValueError("Kraken returned an invalid OHLC candle")
        output.append(
            {
                "timestamp": datetime.fromtimestamp(int(row[0]), tz=UTC).isoformat(),
                "open": values[0],
                "high": values[1],
                "low": values[2],
                "close": values[3],
                "vwap": values[4],
                "volume": values[5],
                "trades": int(row[7]),
            }
        )
    return output


async def _market_data() -> dict[str, Any]:
    cached = _cached("market")
    if cached is not None:
        return cached
    async with _cache_lock:
        cached = _cached("market")
        if cached is not None:
            return cached
        intervals = sorted({horizon.interval_minutes for horizon in HORIZONS})
        observed_at = datetime.now(UTC)
        headers = {"User-Agent": "btc-alpha-research/0.2 (read-only)"}
        async with httpx.AsyncClient(timeout=20, headers=headers) as client:
            requests = [
                _json(
                    client,
                    f"{KRAKEN_BASE}/OHLC",
                    params={"pair": "XBTUSD", "interval": interval, "assetVersion": 1},
                )
                for interval in intervals
            ]
            ticker_request = _json(
                client,
                f"{KRAKEN_BASE}/Ticker",
                params={"pair": "XBTUSD", "assetVersion": 1},
            )
            coinbase_request = _json(client, COINBASE_TICKER)
            results = await asyncio.gather(
                *requests,
                ticker_request,
                coinbase_request,
                return_exceptions=True,
            )
        candle_results = results[: len(intervals)]
        ticker_result = results[len(intervals)]
        coinbase_result = results[len(intervals) + 1]
        if isinstance(ticker_result, Exception):
            raise HTTPException(
                status_code=503, detail=f"Primary BTC feed unavailable: {ticker_result}"
            )
        candles_by_interval: dict[int, list[dict[str, Any]]] = {}
        for interval, result in zip(intervals, candle_results, strict=True):
            if isinstance(result, Exception):
                raise HTTPException(
                    status_code=503,
                    detail=f"Kraken {interval}m candle feed unavailable: {result}",
                )
            candles_by_interval[interval] = _normalize_candles(_kraken_result(result))
        ticker = _kraken_result(ticker_result)
        kraken_price = float(ticker["c"][0])
        bid = float(ticker["b"][0])
        ask = float(ticker["a"][0])
        if not (0 < bid <= kraken_price <= ask * 1.01):
            raise HTTPException(status_code=503, detail="Primary BTC quote failed validation")
        coinbase_price: float | None = None
        coinbase_timestamp: str | None = None
        cross_venue_bps: float | None = None
        if not isinstance(coinbase_result, Exception):
            try:
                coinbase_price = float(coinbase_result["price"])
                coinbase_timestamp = str(coinbase_result.get("time") or "")
                cross_venue_bps = abs(kraken_price / coinbase_price - 1) * 10_000
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                coinbase_price = None
        quality_status = "VERIFIED"
        quality_reason = "Kraken primary quote agrees with Coinbase cross-check."
        if coinbase_price is None:
            quality_status = "DEGRADED"
            quality_reason = "Kraken is live; Coinbase cross-check is unavailable."
        elif cross_venue_bps is not None and cross_venue_bps > MAX_CROSS_VENUE_DEVIATION_BPS:
            quality_status = "SUSPECT"
            quality_reason = "Cross-venue price deviation exceeds the quality threshold."
        bundle = {
            "observed_at": observed_at.isoformat(),
            "instrument": "BTC/USD",
            "price": round(kraken_price, 2),
            "bid": round(bid, 2),
            "ask": round(ask, 2),
            "spread_bps": (ask / bid - 1) * 10_000,
            "day_open": float(ticker["o"]),
            "day_high": float(ticker["h"][0]),
            "day_low": float(ticker["l"][0]),
            "day_volume_btc": float(ticker["v"][0]),
            "sources": {
                "primary": {
                    "venue": "Kraken",
                    "pair": "BTC/USD",
                    "price": round(kraken_price, 2),
                    "observed_at": observed_at.isoformat(),
                },
                "cross_check": {
                    "venue": "Coinbase Exchange",
                    "pair": "BTC-USD",
                    "price": round(coinbase_price, 2) if coinbase_price is not None else None,
                    "source_timestamp": coinbase_timestamp,
                },
            },
            "quality": {
                "status": quality_status,
                "reason": quality_reason,
                "cross_venue_deviation_bps": cross_venue_bps,
                "threshold_bps": MAX_CROSS_VENUE_DEVIATION_BPS,
                "synthetic_fallback": False,
            },
            "candles_by_interval": candles_by_interval,
        }
        if quality_status == "SUSPECT":
            raise HTTPException(status_code=503, detail=quality_reason)
        return _set_cache("market", bundle)


def _decode_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    return []


def _threshold_from_question(question: str) -> float | None:
    match = re.search(r"\$\s*([\d,.]+)\s*([kKmM])?", question)
    if not match:
        return None
    value = float(match.group(1).replace(",", ""))
    suffix = (match.group(2) or "").lower()
    if suffix == "k":
        value *= 1_000
    elif suffix == "m":
        value *= 1_000_000
    return value


async def _prediction_markets(current_price: float) -> dict[str, Any]:
    cached = _cached("prediction_markets")
    if cached is not None:
        return cached
    try:
        async with httpx.AsyncClient(
            timeout=15,
            headers={"User-Agent": "btc-alpha-research/0.2 (read-only)"},
        ) as client:
            payload = await _json(
                client,
                POLYMARKET_SEARCH,
                params={"q": "bitcoin", "limit_per_type": 20, "events_status": "active"},
            )
    except (httpx.HTTPError, ValueError) as error:
        return {
            "status": "UNAVAILABLE",
            "source": "Polymarket",
            "observed_at": datetime.now(UTC).isoformat(),
            "error": str(error),
            "markets": [],
            "directional_score": None,
            "model_weight": 0.0,
        }
    markets: list[dict[str, Any]] = []
    now = datetime.now(UTC)
    for event in payload.get("events", []):
        for market in event.get("markets", []):
            question = str(market.get("question") or "")
            if not re.search(r"\b(bitcoin|btc)\b", question, flags=re.IGNORECASE):
                continue
            outcomes = _decode_json_list(market.get("outcomes"))
            prices = _decode_json_list(market.get("outcomePrices"))
            try:
                yes_index = [str(outcome).lower() for outcome in outcomes].index("yes")
                probability = float(prices[yes_index])
                expiry = datetime.fromisoformat(str(market["endDate"]).replace("Z", "+00:00"))
            except (KeyError, ValueError, TypeError, IndexError):
                continue
            if expiry < now or not (0 <= probability <= 1):
                continue
            threshold = _threshold_from_question(question)
            directional = None
            if threshold and current_price > 0:
                directional = (1 if threshold > current_price else -1) * (2 * probability - 1)
            markets.append(
                {
                    "question": question,
                    "yes_probability": probability,
                    "liquidity_usd": float(
                        market.get("liquidityNum") or market.get("liquidity") or 0
                    ),
                    "volume_usd": float(market.get("volumeNum") or market.get("volume") or 0),
                    "expiry": expiry.astimezone(UTC).isoformat(),
                    "updated_at": market.get("updatedAt"),
                    "threshold_price": threshold,
                    "directional_signal": directional,
                    "url": f"https://polymarket.com/event/{event.get('slug')}",
                }
            )
    markets.sort(key=lambda item: item["liquidity_usd"], reverse=True)
    markets = markets[:8]
    scored = [market for market in markets if market["directional_signal"] is not None]
    directional_score = None
    if scored:
        weights = [math.sqrt(max(float(market["liquidity_usd"]), 1)) for market in scored]
        directional_score = sum(
            float(market["directional_signal"]) * weight
            for market, weight in zip(scored, weights, strict=True)
        ) / sum(weights)
    result = {
        "status": "LIVE" if markets else "NO_MATCHING_MARKETS",
        "source": "Polymarket Gamma API",
        "observed_at": now.isoformat(),
        "markets": markets,
        "directional_score": directional_score,
        "model_weight": 0.0,
        "model_note": (
            "Displayed as public context. Weight remains zero until timestamped market history "
            "has enough settled samples for leakage-safe validation."
        ),
    }
    return _set_cache("prediction_markets", result)


def _public_market(bundle: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in bundle.items() if key != "candles_by_interval"}


async def _terminal(periods: int) -> dict[str, Any]:
    market = await _market_data()
    latest_completed = market["candles_by_interval"][15][-1]
    decision_time = datetime.fromisoformat(
        str(latest_completed["timestamp"]).replace("Z", "+00:00")
    ).astimezone(UTC) + timedelta(minutes=15)
    forecasts_task = asyncio.to_thread(
        build_forecasts,
        market["candles_by_interval"],
        now=decision_time,
        anchor_price=float(latest_completed["close"]),
    )
    prediction_task = _prediction_markets(float(market["price"]))
    forecasts, prediction_markets = await asyncio.gather(forecasts_task, prediction_task)
    try:
        tracking = await asyncio.to_thread(
            record_settle_and_score,
            forecasts,
            market["candles_by_interval"][15],
            prediction_market_score=prediction_markets["directional_score"],
            prediction_market_weight=prediction_markets["model_weight"],
        )
    except Exception as error:  # The market terminal remains usable if persistence is degraded.
        tracking = {
            "status": "DATABASE_ERROR",
            "error_type": type(error).__name__,
            "open_predictions": 0,
            "settled_predictions": 0,
            "settled_now": 0,
            "scorecards": [],
            "recent": [],
        }
    series = market["candles_by_interval"][15][-periods:]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": "PAPER_RESEARCH_ONLY",
        "paper_trading_only": True,
        "synthetic_data": False,
        "market": _public_market(market),
        "regime": classify_regime(market["candles_by_interval"][15]),
        "forecasts": forecasts,
        "prediction_markets": prediction_markets,
        "tracking": tracking,
        "model": {
            "version": MODEL_VERSION,
            "input_status": "LIVE_KRAKEN_OHLCV",
            "prediction_market_weight": prediction_markets["model_weight"],
            "promotion_policy": "manual review only",
        },
        "series": series,
    }


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "mode": "paper",
        "paper_trading_only": True,
        "synthetic_fallback": False,
    }


@app.get("/api/market/live")
async def market_live() -> dict[str, Any]:
    return _public_market(await _market_data())


@app.get("/api/forecast/live")
async def forecast_live() -> dict[str, Any]:
    snapshot = await _terminal(60)
    return {
        "generated_at": snapshot["generated_at"],
        "instrument": "BTC/USD",
        "model_version": MODEL_VERSION,
        "forecasts": snapshot["forecasts"],
        "tracking": snapshot["tracking"],
        "paper_trading_only": True,
    }


@app.get("/api/prediction-markets")
async def prediction_markets() -> dict[str, Any]:
    market = await _market_data()
    return await _prediction_markets(float(market["price"]))


@app.get("/api/research/terminal")
async def terminal(periods: int = Query(240, ge=60, le=719)) -> dict[str, Any]:
    return await _terminal(periods)


@app.get("/api/research/latest")
async def latest() -> dict[str, Any]:
    return await _terminal(240)


@app.post("/api/research/run")
async def run() -> dict[str, Any]:
    _cache.clear()
    return await _terminal(240)


@app.get("/api/tracking")
async def tracking() -> dict[str, Any]:
    snapshot = await _terminal(60)
    return snapshot["tracking"]


@app.get("/api/cron/forecasts")
async def cron_forecasts(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    cron_secret = os.getenv("CRON_SECRET")
    expected = f"Bearer {cron_secret}" if cron_secret else None
    if expected is None:
        raise HTTPException(status_code=503, detail="Cron is not configured")
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")
    _cache.clear()
    snapshot = await _terminal(60)
    return {
        "status": "ok",
        "generated_at": snapshot["generated_at"],
        "tracking": snapshot["tracking"],
    }


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    try:
        from dashboard_content import DASHBOARD_HTML

        return str(DASHBOARD_HTML)
    except ImportError:
        return Path("src/alpha_engine/web/dashboard.html").read_text(encoding="utf-8")


__all__ = ["app"]
