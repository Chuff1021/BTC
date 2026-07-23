from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SimilarState:
    timestamp: pd.Timestamp
    distance: float
    forward_return: float | None


class MarketStateIndex:
    """Causal nearest-neighbor search fitted only on states before the query."""

    def __init__(self, feature_columns: list[str]) -> None:
        self.feature_columns = feature_columns

    def query(
        self,
        features: pd.DataFrame,
        at: pd.Timestamp | None = None,
        neighbors: int = 5,
        horizon: int = 10,
        embargo: int = 30,
    ) -> list[SimilarState]:
        clean = features[self.feature_columns + ["close"]].dropna()
        if clean.empty:
            return []
        query_time = clean.index[-1] if at is None else at
        eligible = clean.loc[clean.index < query_time - timedelta(days=embargo)]
        if eligible.empty or query_time not in clean.index:
            return []
        scaler = StandardScaler().fit(eligible[self.feature_columns])
        history = scaler.transform(eligible[self.feature_columns])
        target = scaler.transform(clean.loc[[query_time], self.feature_columns])[0]
        distances = np.sqrt(np.square(history - target).mean(axis=1))
        positions = np.argsort(distances)[:neighbors]
        results: list[SimilarState] = []
        for position in positions:
            timestamp = eligible.index[position]
            loc_result = clean.index.get_loc(timestamp)
            if not isinstance(loc_result, int):
                continue
            loc = loc_result
            future_loc = loc + horizon
            forward = None
            if future_loc < len(clean) and clean.index[future_loc] < query_time:
                forward = float(clean["close"].iloc[future_loc] / clean["close"].iloc[loc] - 1)
            results.append(SimilarState(timestamp, float(distances[position]), forward))
        return results
