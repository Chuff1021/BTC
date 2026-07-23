from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from alpha_engine.backtesting.engine import Backtester, BuyAndHold, CostModel, RegimeTrend
from alpha_engine.config import Settings
from alpha_engine.data.connectors import BinanceKlines, SyntheticMarketSource
from alpha_engine.data.storage import ResearchStore
from alpha_engine.research.features import FEATURE_COLUMNS, FeatureStore
from alpha_engine.research.hypotheses import HypothesisEngine
from alpha_engine.research.regimes import RegimeDetector
from alpha_engine.research.similarity import MarketStateIndex
from alpha_engine.tracking import ExperimentTracker
from alpha_engine.types import FetchRequest


def run_research(
    settings: Settings,
    days: int = 1200,
    source: str = "synthetic",
) -> dict[str, Any]:
    settings.ensure_directories()
    end = pd.Timestamp(datetime.now(UTC).date(), tz=UTC)
    start = end - timedelta(days=days)
    connector = SyntheticMarketSource() if source == "synthetic" else BinanceKlines("spot")
    request = FetchRequest("BTCUSDT", start, end)
    raw = connector.fetch(request)
    features = FeatureStore().build(raw)
    features["regime"] = RegimeDetector().predict(features)

    store = ResearchStore(settings.database_path, settings.parquet_root)
    store.write("market_daily", raw)
    store.write("features_daily", features)

    cost_model = CostModel(settings.fee_bps, settings.slippage_bps)
    backtester = Backtester(cost_model)
    backtests: list[dict[str, str | float | int]] = []
    for strategy in (BuyAndHold(), RegimeTrend()):
        result, curve = backtester.run(features, strategy)
        backtests.append(result.to_dict())
        store.write(f"equity_{strategy.name}", curve)

    hypotheses = [result.to_dict() for result in HypothesisEngine().discover(features)[:10]]
    similarities = [
        {
            "timestamp": state.timestamp.isoformat(),
            "distance": state.distance,
            "forward_return": state.forward_return,
        }
        for state in MarketStateIndex(FEATURE_COLUMNS).query(features)
    ]
    fingerprint = _fingerprint(raw)
    best = max(backtests, key=lambda row: float(row["sharpe"]))
    run = ExperimentTracker(settings.artifact_root).log(
        "research_pipeline",
        {"source": source, "days": days, "fee_bps": settings.fee_bps},
        {
            "best_sharpe": float(best["sharpe"]),
            "best_total_return": float(best["total_return"]),
        },
        fingerprint,
    )
    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": source,
        "synthetic_data": source == "synthetic",
        "paper_trading_only": settings.paper_trading_only,
        "observations": len(raw),
        "feature_rows": len(features),
        "latest_regime": str(features["regime"].iloc[-1]),
        "data_fingerprint": fingerprint,
        "run_id": run.run_id,
        "backtests": backtests,
        "hypotheses": hypotheses,
        "similar_states": similarities,
    }
    (settings.artifact_root / "latest_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def latest_summary(settings: Settings) -> dict[str, Any] | None:
    path = settings.artifact_root / "latest_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def terminal_snapshot(settings: Settings, periods: int = 240) -> dict[str, Any] | None:
    """Return a compact chart-ready view without exposing arbitrary database queries."""
    summary = latest_summary(settings)
    store = ResearchStore(settings.database_path, settings.parquet_root)
    if summary is None or "features_daily" not in store.tables():
        return None
    features = store.read("features_daily").tail(periods)
    points = [
        {
            "timestamp": pd.Timestamp(str(timestamp)).isoformat(),
            "close": round(float(row["close"]), 2),
            "regime": str(row["regime"]),
            "return_30d": _finite_or_none(row.get("return_30d")),
            "volatility_30d": _finite_or_none(row.get("volatility_30d")),
            "funding_z_30d": _finite_or_none(row.get("funding_z_30d")),
        }
        for timestamp, row in features.iterrows()
    ]
    return {**summary, "series": points}


def _fingerprint(frame: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="uint64").tobytes()
    return hashlib.sha256(payload).hexdigest()


def _finite_or_none(value: Any) -> float | None:
    return round(float(value), 6) if pd.notna(value) else None
