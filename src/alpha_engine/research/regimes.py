from enum import StrEnum

import pandas as pd


class Regime(StrEnum):
    BULL_EXPANSION = "bull_expansion"
    BULL_EXHAUSTION = "bull_exhaustion"
    BEAR = "bear"
    CAPITULATION = "capitulation"
    NEUTRAL = "neutral"


class RegimeDetector:
    """Explainable baseline classifier; replaceable with HMM/clustering implementations."""

    def predict(self, features: pd.DataFrame) -> pd.Series:
        labels = pd.Series(Regime.NEUTRAL.value, index=features.index, name="regime")
        trend = features["trend_50_200"]
        momentum = features["return_30d"]
        drawdown = features["drawdown_90d"]
        volatility = features["volatility_30d"]
        labels.loc[(trend > 0) & (momentum > 0)] = Regime.BULL_EXPANSION.value
        labels.loc[(trend > 0) & (momentum <= 0)] = Regime.BULL_EXHAUSTION.value
        labels.loc[(trend <= 0) & (drawdown > -0.25)] = Regime.BEAR.value
        high_vol = volatility.rolling(180, min_periods=30).quantile(0.8)
        labels.loc[(drawdown <= -0.25) & (volatility >= high_vol)] = Regime.CAPITULATION.value
        return labels

