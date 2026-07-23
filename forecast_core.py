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
FEATURE_NAMES: tuple[str, ...] = (
    "momentum_1",
    "momentum_2",
    "momentum_3",
    "momentum_6",
    "momentum_12",
    "momentum_24",
    "volatility_6",
    "volatility_12",
    "volatility_24",
    "trend_vs_ma_6",
    "trend_vs_ma_24",
    "range_mean_12",
    "volume_z_24",
)
FEATURE_LABELS = {
    "momentum_1": "1-bar momentum",
    "momentum_2": "2-bar momentum",
    "momentum_3": "3-bar momentum",
    "momentum_6": "6-bar momentum",
    "momentum_12": "12-bar momentum",
    "momentum_24": "24-bar momentum",
    "volatility_6": "6-bar volatility",
    "volatility_12": "12-bar volatility",
    "volatility_24": "24-bar volatility",
    "trend_vs_ma_6": "Price vs 6-bar mean",
    "trend_vs_ma_24": "Price vs 24-bar mean",
    "range_mean_12": "12-bar intrabar range",
    "volume_z_24": "24-bar volume impulse",
}


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
) -> dict[str, Any]:
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
    return {
        "prediction": prediction,
        "validation": validation,
        "split": split,
        "mean": mean,
        "scale": scale,
        "latest_scaled": latest_scaled,
        "coefficients": coefficients,
        "intercept": y_mean,
    }


def _explain_model(latest: np.ndarray, fit: dict[str, Any]) -> dict[str, Any]:
    contributions = fit["latest_scaled"] * fit["coefficients"]
    ranked = sorted(
        (
            {
                "feature": name,
                "label": FEATURE_LABELS[name],
                "raw_value": float(latest[index]),
                "standardized_value": float(fit["latest_scaled"][index]),
                "coefficient": float(fit["coefficients"][index]),
                "contribution_log_return": float(contributions[index]),
                "direction": (
                    "UPSIDE"
                    if contributions[index] > 0
                    else "DOWNSIDE"
                    if contributions[index] < 0
                    else "NEUTRAL"
                ),
            }
            for index, name in enumerate(FEATURE_NAMES)
        ),
        key=lambda item: abs(item["contribution_log_return"]),
        reverse=True,
    )
    return {
        "intercept_log_return": float(fit["intercept"]),
        "feature_contributions": ranked,
        "positive_driver_count": sum(item["contribution_log_return"] > 0 for item in ranked),
        "negative_driver_count": sum(item["contribution_log_return"] < 0 for item in ranked),
    }


def _market_analogs(
    candles: list[dict[str, Any]],
    x: np.ndarray,
    y: np.ndarray,
    latest: np.ndarray,
    steps: int,
    fit: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return comparable completed historical states with known forward outcomes."""

    embargo = max(steps, 4)
    usable = max(0, len(x) - embargo)
    if usable < 20:
        return []
    standardized = (x[:usable] - fit["mean"]) / fit["scale"]
    distances = np.sqrt(np.mean((standardized - fit["latest_scaled"]) ** 2, axis=1))
    nearest = np.argsort(distances)[:5]
    start = 24
    return [
        {
            "state_at": str(candles[start + int(index)]["timestamp"]),
            "similarity": float(1 / (1 + distances[index])),
            "distance": float(distances[index]),
            "forward_return": float(np.exp(y[index]) - 1),
            "outcome": "UP" if y[index] > 0 else "DOWN" if y[index] < 0 else "FLAT",
        }
        for index in nearest
    ]


def _hypotheses(
    x: np.ndarray,
    y: np.ndarray,
    latest: np.ndarray,
    split: int,
) -> list[dict[str, Any]]:
    """Evaluate a small, fixed hypothesis library without optimizing thresholds."""

    definitions = (
        ("momentum_continuation", "6-bar momentum is elevated", 3, "high"),
        ("momentum_reversal", "6-bar momentum is depressed", 3, "low"),
        ("volatility_compression", "24-bar volatility is compressed", 8, "low"),
        ("range_expansion", "12-bar intrabar range is elevated", 11, "high"),
        ("volume_impulse", "24-bar volume is elevated", 12, "high"),
    )
    output: list[dict[str, Any]] = []
    train_y = y[:split]
    valid_y = y[split:]
    benchmark = float(np.mean(valid_y))
    for key, label, feature_index, side in definitions:
        quantile = 0.75 if side == "high" else 0.25
        threshold = float(np.quantile(x[:split, feature_index], quantile))
        train_mask = (
            x[:split, feature_index] >= threshold
            if side == "high"
            else x[:split, feature_index] <= threshold
        )
        valid_mask = (
            x[split:, feature_index] >= threshold
            if side == "high"
            else x[split:, feature_index] <= threshold
        )
        active = bool(
            latest[feature_index] >= threshold
            if side == "high"
            else latest[feature_index] <= threshold
        )
        valid_values = valid_y[valid_mask]
        train_values = train_y[train_mask]
        samples = int(valid_mask.sum())
        mean_return = float(np.mean(np.exp(valid_values) - 1)) if samples else None
        hit_rate = float(np.mean(valid_values > 0)) if samples else None
        train_hit = float(np.mean(train_values > 0)) if len(train_values) else None
        stable = bool(
            samples >= 15
            and train_hit is not None
            and hit_rate is not None
            and (train_hit - 0.5) * (hit_rate - 0.5) > 0
        )
        output.append(
            {
                "key": key,
                "hypothesis": label,
                "feature": FEATURE_NAMES[feature_index],
                "active_now": active,
                "threshold": threshold,
                "current_value": float(latest[feature_index]),
                "validation_samples": samples,
                "validation_mean_forward_return": mean_return,
                "validation_hit_rate": hit_rate,
                "benchmark_mean_log_return": benchmark,
                "stable_direction": stable,
                "status": "SUPPORTED" if stable else "INSUFFICIENT",
            }
        )
    return output


def _walk_forward(x: np.ndarray, y: np.ndarray, *, alpha: float = 10.0) -> dict[str, Any]:
    """Expanding-window validation with four chronological test folds."""

    start = max(60, int(len(x) * 0.55))
    fold_edges = np.linspace(start, len(x), 5, dtype=int)
    predictions: list[float] = []
    actuals: list[float] = []
    folds = 0
    for edge_index in range(4):
        train_end = int(fold_edges[edge_index])
        test_end = int(fold_edges[edge_index + 1])
        if test_end <= train_end:
            continue
        x_train = x[:train_end]
        y_train = y[:train_end]
        mean = x_train.mean(axis=0)
        scale = x_train.std(axis=0)
        scale[scale < 1e-12] = 1.0
        train_scaled = (x_train - mean) / scale
        test_scaled = (x[train_end:test_end] - mean) / scale
        intercept = float(y_train.mean())
        coefficients = np.linalg.solve(
            train_scaled.T @ train_scaled + np.eye(x.shape[1]) * alpha,
            train_scaled.T @ (y_train - intercept),
        )
        predictions.extend((intercept + test_scaled @ coefficients).tolist())
        actuals.extend(y[train_end:test_end].tolist())
        folds += 1
    predicted = np.asarray(predictions)
    actual = np.asarray(actuals)
    return {
        "method": "expanding window",
        "folds": folds,
        "samples": int(len(actual)),
        "direction_accuracy": float(np.mean(np.sign(predicted) == np.sign(actual))),
        "median_absolute_return_error": float(np.median(np.abs(actual - predicted))),
        "shuffle": False,
        "fixed_hyperparameters": True,
    }


def _baseline_backtest(validation: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    """Compare a long/cash model rule with buy-and-hold after conservative costs."""

    cost_per_position_change = 0.0010  # 10 bps per entry/exit.
    position = (validation > 0).astype(float)
    prior = np.concatenate(([0.0], position[:-1]))
    turnover = np.abs(position - prior)
    strategy_returns = position * actual - turnover * cost_per_position_change
    buy_hold_returns = actual

    def stats(values: np.ndarray) -> dict[str, float]:
        equity = np.exp(np.cumsum(values))
        peak = np.maximum.accumulate(equity)
        return {
            "total_return": float(equity[-1] - 1),
            "max_drawdown": float(np.min(equity / peak - 1)),
        }

    return {
        "model_long_cash": stats(strategy_returns),
        "buy_hold": stats(buy_hold_returns),
        "assumed_fee_plus_slippage_bps_per_position_change": 10.0,
        "note": (
            "Illustrative holdout backtest; overlapping long horizons are not "
            "executable trades."
        ),
    }


def forecast_horizon(
    candles: list[dict[str, Any]],
    horizon: Horizon,
    *,
    now: datetime | None = None,
    anchor_price: float | None = None,
) -> dict[str, Any]:
    """Fit and validate one horizon using completed candles only."""

    x, y, observed, actual_future, latest = _matrix(candles, horizon.steps)
    fit = _fit_ridge(x, y, latest)
    prediction = float(fit["prediction"])
    validation = fit["validation"]
    split = int(fit["split"])
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
    walk_forward = _walk_forward(x, y)
    baseline_backtest = _baseline_backtest(validation, actual_returns)
    issued_at = now or datetime.now(UTC)
    latest_candle = datetime.fromisoformat(str(candles[-1]["timestamp"]).replace("Z", "+00:00"))
    explanation = _explain_model(latest, fit)
    analogs = _market_analogs(candles, x, y, latest, horizon.steps, fit)
    hypotheses = _hypotheses(x, y, latest, split)
    top_up = next(
        (
            item
            for item in explanation["feature_contributions"]
            if item["contribution_log_return"] > 0
        ),
        None,
    )
    top_down = next(
        (
            item
            for item in explanation["feature_contributions"]
            if item["contribution_log_return"] < 0
        ),
        None,
    )
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
            "training_samples": int(split),
            "samples": int(len(actual_returns)),
            "direction_accuracy": float(direction_accuracy),
            "median_absolute_pct_error": float(np.median(absolute_pct_errors)),
            "mean_absolute_return_error": float(np.mean(np.abs(residuals))),
            "interval_coverage": float(coverage),
            "always_up_direction_accuracy": float(np.mean(actual_returns > 0)),
            "walk_forward": walk_forward,
            "baseline_backtest": baseline_backtest,
        },
        "explanation": {
            **explanation,
            "summary": (
                f"{top_up['label'] if top_up else 'No feature'} is the strongest upside "
                f"driver; {top_down['label'] if top_down else 'no feature'} is the strongest "
                "downside driver."
            ),
            "decision_trace": [
                {
                    "stage": "SOURCE",
                    "detail": "Completed Kraken OHLCV candles only; live partial candle excluded.",
                    "status": "PASS",
                },
                {
                    "stage": "FEATURES",
                    "detail": (
                        f"{len(FEATURE_NAMES)} fixed price, volatility, range "
                        "and volume features."
                    ),
                    "status": "PASS",
                },
                {
                    "stage": "VALIDATION",
                    "detail": (
                        "Chronological 72/28 holdout; no random shuffle or "
                        "future-derived features."
                    ),
                    "status": "PASS",
                },
                {
                    "stage": "MODEL",
                    "detail": (
                        "Fixed ridge regression with alpha=10; no per-run "
                        "hyperparameter search."
                    ),
                    "status": "PASS",
                },
                {
                    "stage": "OUTPUT",
                    "detail": "Point estimate plus empirical 10th–90th percentile residual range.",
                    "status": "RESEARCH",
                },
            ],
        },
        "analogs": analogs,
        "hypotheses": hypotheses,
        "training": {
            "rows_total": int(len(x)),
            "rows_train": int(split),
            "rows_validation": int(len(x) - split),
            "feature_count": len(FEATURE_NAMES),
            "leakage_embargo_rows_for_analogs": max(horizon.steps, 4),
            "hyperparameter_search": False,
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
