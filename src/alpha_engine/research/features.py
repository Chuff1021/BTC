from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_engine.research.leakage import assert_features_are_lagged, assert_point_in_time

FEATURE_COLUMNS = [
    "return_1d",
    "return_7d",
    "return_30d",
    "volatility_30d",
    "trend_50_200",
    "drawdown_90d",
    "volume_z_30d",
    "funding_z_30d",
    "oi_change_7d",
    "basis_z_30d",
    "stablecoin_growth_30d",
    "sentiment_z_30d",
]


class FeatureStore:
    """Point-in-time daily feature materialization with release-time lineage."""

    def build(self, raw: pd.DataFrame) -> pd.DataFrame:
        assert_point_in_time(raw)
        data = raw.copy()
        release = pd.to_datetime(data.pop("available_at"), utc=True)
        # A close is actionable at the following timestamp; shift all raw values one row.
        lagged = data.select_dtypes(include=[np.number]).shift(1)
        close = lagged["close"]
        features = pd.DataFrame(index=data.index)
        features["return_1d"] = close.pct_change()
        features["return_7d"] = close.pct_change(7)
        features["return_30d"] = close.pct_change(30)
        features["volatility_30d"] = features["return_1d"].rolling(30).std() * np.sqrt(365)
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        features["trend_50_200"] = ma50 / ma200 - 1
        features["drawdown_90d"] = close / close.rolling(90).max() - 1
        volume = lagged.get("volume", pd.Series(index=data.index, dtype=float))
        features["volume_z_30d"] = _rolling_z(volume, 30)
        features["funding_z_30d"] = _rolling_z(
            lagged.get("funding_rate", pd.Series(0.0, index=data.index)), 30
        )
        oi = lagged.get("open_interest", pd.Series(index=data.index, dtype=float))
        features["oi_change_7d"] = oi.pct_change(7)
        features["basis_z_30d"] = _rolling_z(
            lagged.get("basis", pd.Series(0.0, index=data.index)), 30
        )
        stable = lagged.get("stablecoin_supply", pd.Series(index=data.index, dtype=float))
        features["stablecoin_growth_30d"] = stable.pct_change(30)
        features["sentiment_z_30d"] = _rolling_z(
            lagged.get("sentiment", pd.Series(0.0, index=data.index)), 30
        )
        features["close"] = close
        features["max_available_at"] = release.shift(1)
        features = features.dropna(subset=["close", "trend_50_200", "max_available_at"])
        assert_features_are_lagged(features, raw)
        return features


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std

