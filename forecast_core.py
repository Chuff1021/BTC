"""Small, auditable BTC forecasting primitives used by the Vercel adapter.

The model is intentionally conservative: a fixed ridge regression, chronological
holdout validation, and empirical residual intervals. It is a research baseline,
not a claim that future BTC prices are knowable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Horizon:
    key: str
    label: str
    interval_minutes: int
    steps: int
    target_minutes: int


HORIZONS: tuple[Horizon, ...] = (
    Horizon("15m", "15 MIN", 15, 1, 15),
    Horizon("1h", "1 HOUR", 60, 1, 60),
    Horizon("4h", "4 HOURS", 240, 1, 240),
    Horizon("8h", "8 HOURS", 240, 2, 480),
    Horizon("12h", "12 HOURS", 240, 3, 720),
    Horizon("1d", "DAILY", 1440, 1, 1440),
    Horizon("1w", "WEEKLY", 10080, 1, 10080),
    Horizon("1mo", "MONTHLY", 21600, 2, 43200),
)

MODEL_VERSION = "btc-ridge-ohlcv-v1"
MIN_ROWS = 120


def _features(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    index: int,
) -> np.ndarray:
    log_price = np.log(close)
    returns = np.diff(log_price, prepend=log_price[0])
    windows = (1, 2, 3, 6, 12, 24)
    momentum = [float(log_price[index] - log_price[index - window]) for window in windows]
    volatility = [
        float(np.std(returns[index - window + 1 : index + 1], ddof=1)) for window in (6, 12, 24)
    ]
    trend = [
        float(close[index] / np.mean(close[index - window + 1 : index + 1]) - 1)
        for window in (6, 24)
    ]
    ranges = np.log(np.maximum(high, 1e-12) / np.maximum(low, 1e-12))
    range_mean = float(np.mean(ranges[index - 11 : index + 1]))
    log_volume = np.log1p(np.maximum(volume, 0))
    volume_window = log_volume[index - 23 : index + 1]
    volume_z = float((log_volume[index] - np.mean(volume_window)) / (np.std(volume_window) + 1e-12))
    return np.asarray(momentum + volatility + trend + [range_mean, volume_z], dtype=float)


def _matrix(
    candles: list[dict[str, Any]], steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close = np.asarray([float(row["close"]) for row in candles], dtype=float)
    high = np.asarray([float(row["high"]) for row in candles], dtype=float)
    low = np.asarray([float(row["low"]) for row in candles], dtype=float)
    volume = np.asarray([float(row["volume"]) for row in candles], dtype=float)
    if len(close) < MIN_ROWS + steps:
        raise ValueError(f"At least {MIN_ROWS + steps} completed candles are required")
    start = 24
    stop = len(close) - steps
    x = np.vstack([_features(close, high, low, volume, index) for index in range(start, stop)])
    y = np.asarray(
        [np.log(close[index + steps] / close[index]) for index in range(start, stop)],
        dtype=float,
    )
    observed = close[start:stop]
    actual_future = close[start + steps : stop + steps]
    latest = _features(close, high, low, volume, len(close) - 1)
    return x, y, observed, actual_future, latest


def _fit_ridge(
    x: np.ndarray, y: np.ndarray, latest: np.ndarray, alpha: float = 10.0
) -> tuple[float, np.ndarray, int]:
    split = max(60, int(len(x) * 0.72))
    if len(x) - split < 30:
        raise ValueError("Insufficient chronological validation rows")
    x_train = x[:split]
    y_train = y[:split]
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    x_train_scaled = (x_train - mean) / scale
    x_valid_scaled = (x[split:] - mean) / scale
    latest_scaled = (latest - mean) / scale
    y_mean = float(y_train.mean())
    centered = y_train - y_mean
    penalty = np.eye(x_train_scaled.shape[1], dtype=float) * alpha
    coefficients = np.linalg.solve(
        x_train_scaled.T @ x_train_scaled + penalty, x_train_scaled.T @ centered
    )
    validation = y_mean + x_valid_scaled @ coefficients
    prediction = float(y_mean + latest_scaled @ coefficients)
    return prediction, validation, split


def forecast_horizon(
    candles: list[dict[str, Any]],
    horizon: Horizon,
    *,
    now: datetime | None = None,
    anchor_price: float | None = None,
) -> dict[str, Any]:
    """Fit and validate one horizon using completed candles only."""

    x, y, observed, actual_future, latest = _matrix(candles, horizon.steps)
    prediction, validation, split = _fit_ridge(x, y, latest)
    actual_returns = y[split:]
    residuals = actual_returns - validation
    q10, q90 = np.quantile(residuals, [0.10, 0.90])
    low_return = float(min(prediction, prediction + q10))
    high_return = float(max(prediction, prediction + q90))
    current_price = float(anchor_price if anchor_price is not None else candles[-1]["close"])
    validation_prices = observed[split:] * np.exp(validation)
    actual_prices = actual_future[split:]
    absolute_pct_errors = np.abs(validation_prices / actual_prices - 1)
    coverage = np.mean((actual_returns >= validation + q10) & (actual_returns <= validation + q90))
    direction_accuracy = np.mean(np.sign(validation) == np.sign(actual_returns))
    issued_at = now or datetime.now(UTC)
    latest_candle = datetime.fromisoformat(str(candles[-1]["timestamp"]).replace("Z", "+00:00"))
    return {
        "horizon": horizon.key,
        "label": horizon.label,
        "issued_at": issued_at.astimezone(UTC).isoformat(),
        "target_at": (
            issued_at.astimezone(UTC) + timedelta(minutes=horizon.target_minutes)
        ).isoformat(),
        "training_interval_minutes": horizon.interval_minutes,
        "forecast_steps": horizon.steps,
        "latest_completed_candle": latest_candle.astimezone(UTC).isoformat(),
        "current_price": round(current_price, 2),
        "predicted_price": round(current_price * float(np.exp(prediction)), 2),
        "low_price": round(current_price * float(np.exp(low_return)), 2),
        "high_price": round(current_price * float(np.exp(high_return)), 2),
        "predicted_return": float(np.exp(prediction) - 1),
        "interval": "empirical 80% residual interval",
        "validation": {
            "method": "chronological holdout",
            "samples": int(len(actual_returns)),
            "direction_accuracy": float(direction_accuracy),
            "median_absolute_pct_error": float(np.median(absolute_pct_errors)),
            "mean_absolute_return_error": float(np.mean(np.abs(residuals))),
            "interval_coverage": float(coverage),
        },
        "model_version": MODEL_VERSION,
        "status": "RESEARCH_ONLY_NOT_LIVE_VALIDATED",
    }


def build_forecasts(
    candles_by_interval: dict[int, list[dict[str, Any]]],
    *,
    now: datetime | None = None,
    anchor_price: float | None = None,
) -> list[dict[str, Any]]:
    return [
        forecast_horizon(
            candles_by_interval[horizon.interval_minutes],
            horizon,
            now=now,
            anchor_price=anchor_price,
        )
        for horizon in HORIZONS
    ]


def classify_regime(candles: list[dict[str, Any]]) -> dict[str, Any]:
    closes = np.asarray([float(row["close"]) for row in candles], dtype=float)
    returns = np.diff(np.log(closes))
    fast = float(np.mean(closes[-20:]))
    slow = float(np.mean(closes[-80:]))
    momentum = float(closes[-1] / closes[-20] - 1)
    annualized_volatility = float(np.std(returns[-96:], ddof=1) * np.sqrt(365 * 96))
    if fast > slow and momentum > 0:
        name = "BULL EXPANSION"
    elif fast > slow:
        name = "BULL PULLBACK"
    elif fast < slow and momentum < 0:
        name = "BEAR CONTRACTION"
    else:
        name = "NEUTRAL / TRANSITION"
    return {
        "name": name,
        "momentum_20_bars": momentum,
        "fast_slow_spread": fast / slow - 1,
        "annualized_volatility": annualized_volatility,
        "method": "15m price-only descriptive regime",
    }
