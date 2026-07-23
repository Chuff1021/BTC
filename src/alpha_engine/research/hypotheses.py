from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HypothesisResult:
    name: str
    samples: int
    hit_rate: float
    mean_forward_return: float
    benchmark_return: float
    uplift: float
    train_hit_rate: float
    test_hit_rate: float
    stable: bool

    def to_dict(self) -> dict[str, str | int | float | bool]:
        return asdict(self)


class HypothesisEngine:
    """Generates transparent univariate hypotheses and validates on a held-out tail."""

    def discover(
        self, features: pd.DataFrame, horizon: int = 10, minimum_samples: int = 30
    ) -> list[HypothesisResult]:
        data = features.copy()
        data["forward_return"] = data["close"].shift(-horizon) / data["close"] - 1
        numeric = data.select_dtypes(include=[np.number]).columns
        results: list[HypothesisResult] = []
        split = int(len(data) * 0.7)
        for column in numeric:
            if column in {"close", "forward_return"}:
                continue
            train_threshold = data[column].iloc[:split].quantile(0.75)
            mask = data[column] >= train_threshold
            selected = data.loc[mask, "forward_return"].dropna()
            train = data.loc[mask.iloc[:split].index[mask.iloc[:split]], "forward_return"].dropna()
            test_mask = mask.iloc[split:]
            test = data.loc[test_mask.index[test_mask], "forward_return"].dropna()
            if len(selected) < minimum_samples or len(train) < 10 or len(test) < 5:
                continue
            hit = float((selected > 0).mean())
            train_hit = float((train > 0).mean())
            test_hit = float((test > 0).mean())
            benchmark = float(data["forward_return"].mean())
            mean_return = float(selected.mean())
            results.append(
                HypothesisResult(
                    name=f"high_{column}",
                    samples=len(selected),
                    hit_rate=hit,
                    mean_forward_return=mean_return,
                    benchmark_return=benchmark,
                    uplift=mean_return - benchmark,
                    train_hit_rate=train_hit,
                    test_hit_rate=test_hit,
                    stable=abs(train_hit - test_hit) <= 0.15 and test_hit > 0.5,
                )
            )
        return sorted(results, key=lambda result: result.uplift, reverse=True)
