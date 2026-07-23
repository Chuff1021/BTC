"""Read-only public intelligence connectors for the BTC research terminal.

Every connector fails independently. Missing licensed feeds are represented as
provider requirements; they are never replaced with synthetic observations.
"""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime, timedelta
from statistics import median
from typing import Any

import httpx

USER_AGENT = "btc-alpha-research/0.3 (read-only; paper research)"
KRAKEN_FUTURES = "https://futures.kraken.com"
DERIBIT_OPTIONS = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
STABLECOINS = "https://stablecoins.llama.fi/stablecoincharts/all"
FEAR_GREED = "https://api.alternative.me/fng/"
MEMPOOL_FEES = "https://mempool.space/api/v1/fees/recommended"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart"


def _utc_from_epoch(value: int | float) -> str:
    if value > 10_000_000_000:
        value /= 1000
    return datetime.fromtimestamp(value, tz=UTC).isoformat()


async def _json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
) -> Any:
    response = await client.get(url, params=params)
    response.raise_for_status()
    return response.json()


async def _derivatives(client: httpx.AsyncClient) -> dict[str, Any]:
    now = datetime.now(UTC)
    since = int((now - timedelta(hours=24)).timestamp())
    tickers_url = f"{KRAKEN_FUTURES}/derivatives/api/v3/tickers"
    analytics = (
        "open-interest",
        "funding",
        "future-basis",
        "liquidation-volume",
        "long-short-ratio",
    )
    tasks = [
        _json(client, tickers_url),
        *[
            _json(
                client,
                f"{KRAKEN_FUTURES}/api/charts/v1/analytics/PF_XBTUSD/{kind}",
                params={"since": since, "interval": 3600},
            )
            for kind in analytics
        ],
    ]
    payloads = await asyncio.gather(*tasks)
    ticker = next(item for item in payloads[0]["tickers"] if item.get("symbol") == "PF_XBTUSD")
    parsed = {
        name: payload["result"] for name, payload in zip(analytics, payloads[1:], strict=True)
    }
    open_interest_rows = parsed["open-interest"]["data"]
    oi_first = float(open_interest_rows[0][0])
    oi_last = float(open_interest_rows[-1][3])
    funding_rows = parsed["funding"]["data"]["relativeRate"]
    funding_hourly = float(funding_rows[-1][3])
    basis_values = [float(value) for value in parsed["future-basis"]["data"]["basis"]]
    liquidations = [float(value) for value in parsed["liquidation-volume"]["data"]]
    ratios = [float(value) for value in parsed["long-short-ratio"]["data"]]
    mark = float(ticker["markPrice"])
    index = float(ticker["indexPrice"])
    return {
        "status": "LIVE",
        "source": "Kraken Futures public API",
        "source_url": "https://docs.kraken.com/api/docs/futures-api/charts/market-analytics",
        "observed_at": now.isoformat(),
        "contract": "PF_XBTUSD",
        "mark_price": mark,
        "index_price": index,
        "perpetual_basis_bps": (mark / index - 1) * 10_000,
        "open_interest_btc": oi_last,
        "open_interest_24h_change": oi_last / oi_first - 1 if oi_first else None,
        "funding_rate_hourly": funding_hourly,
        "funding_rate_8h_equivalent": funding_hourly * 8,
        "basis_24h_latest": basis_values[-1] if basis_values else None,
        "liquidation_volume_btc_24h": sum(liquidations),
        "long_short_ratio": ratios[-1] if ratios else None,
        "note": "Public exchange analytics. Funding is the latest hourly relative rate.",
    }


async def _options(client: httpx.AsyncClient) -> dict[str, Any]:
    payload = await _json(
        client,
        DERIBIT_OPTIONS,
        params={"currency": "BTC", "kind": "option"},
    )
    rows = payload.get("result") or []
    if not rows:
        raise ValueError("Deribit returned no BTC option summaries")
    underlying = median(
        float(row["underlying_price"])
        for row in rows
        if row.get("underlying_price") not in (None, 0)
    )
    call_oi = 0.0
    put_oi = 0.0
    total_volume_usd = 0.0
    near_atm_iv: list[float] = []
    for row in rows:
        name = str(row.get("instrument_name") or "")
        parts = name.split("-")
        option_type = parts[-1] if parts else ""
        oi = float(row.get("open_interest") or 0)
        if option_type == "C":
            call_oi += oi
        elif option_type == "P":
            put_oi += oi
        total_volume_usd += float(row.get("volume_usd") or 0)
        try:
            strike = float(parts[-2])
            mark_iv = float(row["mark_iv"])
        except (IndexError, TypeError, ValueError):
            continue
        if underlying and abs(strike / underlying - 1) <= 0.10 and math.isfinite(mark_iv):
            near_atm_iv.append(mark_iv)
    return {
        "status": "LIVE",
        "source": "Deribit public API",
        "source_url": (
            "https://docs.deribit.com/api-reference/market-data/public-get_book_summary_by_currency"
        ),
        "observed_at": datetime.now(UTC).isoformat(),
        "contracts": len(rows),
        "underlying_price": underlying,
        "near_atm_mark_iv_pct": median(near_atm_iv) if near_atm_iv else None,
        "call_open_interest_btc": call_oi,
        "put_open_interest_btc": put_oi,
        "put_call_open_interest_ratio": put_oi / call_oi if call_oi else None,
        "volume_usd_24h": total_volume_usd,
        "note": "Snapshot aggregation; this is not a volatility-surface model.",
    }


async def _stablecoins(client: httpx.AsyncClient) -> dict[str, Any]:
    rows = await _json(client, STABLECOINS)
    if not isinstance(rows, list) or len(rows) < 8:
        raise ValueError("Stablecoin history is incomplete")
    current = rows[-1]
    week_ago = rows[-8]
    current_value = float(current["totalCirculatingUSD"]["peggedUSD"])
    previous_value = float(week_ago["totalCirculatingUSD"]["peggedUSD"])
    return {
        "status": "LIVE",
        "source": "DefiLlama stablecoins API",
        "source_url": "https://defillama.com/docs/api",
        "observed_at": _utc_from_epoch(float(current["date"])),
        "total_supply_usd": current_value,
        "change_7d_usd": current_value - previous_value,
        "change_7d_pct": current_value / previous_value - 1 if previous_value else None,
        "note": "Aggregate USD-pegged circulating supply; not direct BTC exchange inflow.",
    }


async def _sentiment(client: httpx.AsyncClient) -> dict[str, Any]:
    payload = await _json(client, FEAR_GREED, params={"limit": 2})
    rows = payload.get("data") or []
    if len(rows) < 2:
        raise ValueError("Sentiment history is incomplete")
    latest, previous = rows[0], rows[1]
    return {
        "status": "LIVE",
        "source": "Alternative.me Crypto Fear & Greed Index",
        "source_url": "https://alternative.me/crypto/fear-and-greed-index/",
        "observed_at": _utc_from_epoch(float(latest["timestamp"])),
        "value": int(latest["value"]),
        "classification": str(latest["value_classification"]),
        "daily_change": int(latest["value"]) - int(previous["value"]),
        "note": "Daily BTC sentiment composite. Displayed with source attribution.",
    }


async def _network(client: httpx.AsyncClient) -> dict[str, Any]:
    fees = await _json(client, MEMPOOL_FEES)
    return {
        "status": "LIVE",
        "source": "mempool.space public API",
        "source_url": "https://mempool.space/docs/api/rest",
        "observed_at": datetime.now(UTC).isoformat(),
        "fastest_fee_sat_vb": int(fees["fastestFee"]),
        "half_hour_fee_sat_vb": int(fees["halfHourFee"]),
        "hour_fee_sat_vb": int(fees["hourFee"]),
        "economy_fee_sat_vb": int(fees["economyFee"]),
        "note": (
            "Network fee pressure only; exchange reserves and miner selling "
            "require licensed feeds."
        ),
    }


MACRO_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("^TNX", "US 10Y yield"),
    ("DX-Y.NYB", "US Dollar Index"),
    ("^IXIC", "Nasdaq Composite"),
    ("GC=F", "Gold futures"),
    ("^VIX", "VIX"),
)


async def _macro_series(
    client: httpx.AsyncClient,
    symbol: str,
    label: str,
) -> dict[str, Any]:
    payload = await _json(
        client,
        f"{YAHOO_CHART}/{symbol}",
        params={"range": "5d", "interval": "1d"},
    )
    result = payload["chart"]["result"][0]
    timestamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    values = [
        (timestamp, float(value))
        for timestamp, value in zip(timestamps, closes, strict=True)
        if value is not None and math.isfinite(float(value))
    ]
    if not values:
        raise ValueError(f"Macro chart returned no observations for {symbol}")
    latest = values[-1]
    previous = values[-2] if len(values) > 1 else latest
    return {
        "series_id": symbol,
        "label": label,
        "value": latest[1],
        "previous_value": previous[1],
        "change": latest[1] - previous[1],
        "observation_date": datetime.fromtimestamp(latest[0], tz=UTC).date().isoformat(),
    }


async def _macro(client: httpx.AsyncClient) -> dict[str, Any]:
    results = await asyncio.gather(
        *[_macro_series(client, symbol, label) for symbol, label in MACRO_SYMBOLS],
        return_exceptions=True,
    )
    series = [row for row in results if isinstance(row, dict)]
    if not series:
        raise ValueError("All macro market series are unavailable")
    return {
        "status": "LIVE",
        "source": "Yahoo Finance public chart feed",
        "source_url": "https://finance.yahoo.com/markets/",
        "observed_at": datetime.now(UTC).isoformat(),
        "series": series,
        "available_series": len(series),
        "requested_series": len(MACRO_SYMBOLS),
        "note": "Daily public market snapshots; each row carries its observation date.",
    }


def _unavailable(name: str, source: str, error: Exception) -> dict[str, Any]:
    return {
        "status": "UNAVAILABLE",
        "source": source,
        "observed_at": datetime.now(UTC).isoformat(),
        "error_type": type(error).__name__,
        "error": str(error)[:180],
        "name": name,
    }


async def build_intelligence(
    *,
    spot_price: float,
    coinbase_price: float | None,
) -> dict[str, Any]:
    """Fetch independent public intelligence domains concurrently."""

    async with httpx.AsyncClient(
        timeout=30,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        names = ("derivatives", "options", "stablecoins", "sentiment", "network", "macro")
        sources = (
            "Kraken Futures",
            "Deribit",
            "DefiLlama",
            "Alternative.me",
            "mempool.space",
            "Yahoo Finance",
        )
        tasks = (
            _derivatives(client),
            _options(client),
            _stablecoins(client),
            _sentiment(client),
            _network(client),
            _macro(client),
        )
        results = await asyncio.gather(*tasks, return_exceptions=True)
    domains: dict[str, Any] = {}
    for name, source, result in zip(names, sources, results, strict=True):
        domains[name] = (
            _unavailable(name, source, result) if isinstance(result, Exception) else result
        )
    premium = (
        (coinbase_price / spot_price - 1) * 10_000
        if coinbase_price is not None and spot_price > 0
        else None
    )
    domains["coinbase_premium"] = {
        "status": "LIVE" if premium is not None else "UNAVAILABLE",
        "source": "Coinbase Exchange vs Kraken",
        "observed_at": datetime.now(UTC).isoformat(),
        "premium_bps": premium,
        "note": "Cross-venue spot premium, not the institutional Coinbase Premium Index.",
    }
    domains["etf_flows"] = {
        "status": "PROVIDER_REQUIRED",
        "source": "Licensed timestamped ETF flow provider",
        "observed_at": None,
        "note": "No verified free machine-readable intraday feed is configured.",
    }
    domains["onchain_flows"] = {
        "status": "PROVIDER_REQUIRED",
        "source": "Licensed on-chain provider",
        "observed_at": None,
        "note": "Exchange reserves, miner selling and wallet clusters are not fabricated.",
    }
    domains["information_velocity"] = {
        "status": "PROVIDER_REQUIRED",
        "source": "Licensed social/news firehose",
        "observed_at": None,
        "note": "Cross-platform propagation history is required before modeling.",
    }
    return domains


def data_catalog(
    domains: dict[str, Any],
    *,
    prediction_market_status: str,
) -> list[dict[str, Any]]:
    rows = [
        ("Spot OHLCV", "LIVE", "Kraken + Coinbase", "price, candles, volume, venue gap"),
        (
            "Derivatives",
            domains["derivatives"]["status"],
            domains["derivatives"]["source"],
            "funding, OI, basis, liquidations, positioning",
        ),
        (
            "Options",
            domains["options"]["status"],
            domains["options"]["source"],
            "IV, put/call OI, option volume",
        ),
        (
            "ETF flows",
            domains["etf_flows"]["status"],
            domains["etf_flows"]["source"],
            "daily creations/redemptions",
        ),
        (
            "Stablecoin supply",
            domains["stablecoins"]["status"],
            domains["stablecoins"]["source"],
            "aggregate supply and 7d change",
        ),
        (
            "On-chain flows",
            domains["onchain_flows"]["status"],
            domains["onchain_flows"]["source"],
            "reserves, miners, whale clusters",
        ),
        (
            "Macro assets",
            domains["macro"]["status"],
            domains["macro"]["source"],
            "USD, yields, Nasdaq, gold, VIX",
        ),
        ("Network", domains["network"]["status"], domains["network"]["source"], "fee pressure"),
        (
            "Sentiment",
            domains["sentiment"]["status"],
            domains["sentiment"]["source"],
            "Fear & Greed",
        ),
        (
            "Information velocity",
            domains["information_velocity"]["status"],
            domains["information_velocity"]["source"],
            "social/news diffusion",
        ),
        (
            "Coinbase premium",
            domains["coinbase_premium"]["status"],
            domains["coinbase_premium"]["source"],
            "cross-venue premium",
        ),
        (
            "Prediction markets",
            prediction_market_status,
            "Polymarket Gamma API",
            "BTC event probabilities",
        ),
    ]
    return [
        {"domain": domain, "status": status, "source": source, "coverage": coverage}
        for domain, status, source, coverage in rows
    ]
