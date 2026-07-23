from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_engine.forecast_tracking import ForecastLedger, evaluate_challenger
from forecast_core import HORIZONS, build_forecasts


def _candles(interval: int, rows: int = 500) -> list[dict[str, object]]:
    rng = np.random.default_rng(interval)
    prices = 60_000 * np.exp(np.cumsum(rng.normal(0.0001, 0.006, rows)))
    end = datetime(2025, 12, 31, tzinfo=UTC)
    start = end - timedelta(minutes=interval * rows)
    return [
        {
            "timestamp": (start + timedelta(minutes=interval * index)).isoformat(),
            "open": float(price * 0.999),
            "high": float(price * 1.004),
            "low": float(price * 0.996),
            "close": float(price),
            "volume": float(100 + index),
        }
        for index, price in enumerate(prices)
    ]


def test_all_requested_horizons_are_forecast_without_future_rows() -> None:
    intervals = {horizon.interval_minutes for horizon in HORIZONS}
    candles = {interval: _candles(interval) for interval in intervals}
    now = datetime(2026, 1, 1, tzinfo=UTC)
    forecasts = build_forecasts(candles, now=now, anchor_price=65_000)
    assert [forecast["horizon"] for forecast in forecasts] == [
        "15m",
        "1h",
        "4h",
        "8h",
        "12h",
        "1d",
        "1w",
        "1mo",
    ]
    assert all(forecast["current_price"] == 65_000 for forecast in forecasts)
    assert all(
        forecast["latest_completed_candle"] < forecast["issued_at"] for forecast in forecasts
    )
    assert all(forecast["validation"]["samples"] >= 100 for forecast in forecasts)
    assert all(
        len(forecast["explanation"]["feature_contributions"]) == 13 for forecast in forecasts
    )
    assert all(forecast["analogs"] for forecast in forecasts)
    assert all(forecast["hypotheses"] for forecast in forecasts)
    assert all(forecast["validation"]["walk_forward"]["folds"] == 4 for forecast in forecasts)
    assert all(
        forecast["validation"]["baseline_backtest"][
            "assumed_fee_plus_slippage_bps_per_position_change"
        ]
        == 10.0
        for forecast in forecasts
    )


def test_forecast_ledger_is_idempotent_and_scores_due_predictions(tmp_path: Path) -> None:
    ledger = ForecastLedger(tmp_path / "forecasts.duckdb")
    issued = datetime(2026, 1, 1, tzinfo=UTC)
    forecast = {
        "issued_at": issued.isoformat(),
        "target_at": (issued + timedelta(minutes=15)).isoformat(),
        "horizon": "15m",
        "current_price": 100.0,
        "predicted_price": 105.0,
        "low_price": 98.0,
        "high_price": 108.0,
        "model_version": "test-v1",
        "latest_completed_candle": issued.isoformat(),
    }
    first = ledger.record([forecast])
    second = ledger.record([forecast])
    assert first == second
    observations = pd.DataFrame(
        {"price": [104.0]},
        index=[issued + timedelta(minutes=16)],
    )
    assert ledger.settle(observations, settled_at=issued + timedelta(minutes=20)) == 1
    assert ledger.settle(observations, settled_at=issued + timedelta(minutes=20)) == 0
    score = ledger.accuracy()[0]
    assert score["samples"] == 1
    assert score["direction_accuracy"] == 1.0
    assert score["interval_coverage"] == 1.0


def test_challenger_cannot_auto_promote_or_win_with_too_few_samples() -> None:
    champion = {
        "median_absolute_percentage_error": 0.04,
        "direction_accuracy": 0.54,
        "interval_coverage": 0.78,
    }
    challenger = {
        "median_absolute_percentage_error": 0.03,
        "direction_accuracy": 0.56,
        "interval_coverage": 0.79,
    }
    premature = evaluate_challenger(champion, challenger, settled_predictions=40)
    assert premature.eligible is False
    eligible = evaluate_challenger(champion, challenger, settled_predictions=120)
    assert eligible.eligible is True
    assert eligible.requires_manual_approval is True
