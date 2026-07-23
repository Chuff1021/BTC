from __future__ import annotations

import numpy as np
import pandas as pd


class LeakageError(ValueError):
    pass


def assert_point_in_time(frame: pd.DataFrame) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise LeakageError("Dataset index must be a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise LeakageError("Timestamps must be sorted and unique")
    if "available_at" not in frame:
        raise LeakageError("Every raw dataset must include available_at")
    available = pd.to_datetime(frame["available_at"], utc=True)
    observed = pd.Series(frame.index, index=frame.index)
    if bool((available < observed).any()):
        raise LeakageError("available_at cannot predate its observation timestamp")


def assert_features_are_lagged(features: pd.DataFrame, source: pd.DataFrame) -> None:
    """Ensure a feature row cannot use a source value released after decision time."""
    if "max_available_at" not in features:
        raise LeakageError("Feature frame must retain max_available_at lineage")
    available = pd.to_datetime(features["max_available_at"], utc=True)
    decision = pd.Series(features.index, index=features.index)
    if bool((available > decision).any()):
        raise LeakageError("Feature rows contain information unavailable at decision time")
    if len(features) > len(source):
        raise LeakageError("Feature construction unexpectedly increased row count")


def purged_time_series_splits(
    length: int, folds: int = 5, embargo: int = 5, min_train: int = 120
) -> list[tuple[np.ndarray, np.ndarray]]:
    if folds < 2 or length <= min_train + embargo + folds:
        raise ValueError("Not enough observations for requested purged folds")
    test_size = max(1, (length - min_train) // folds)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(folds):
        test_start = min_train + fold * test_size
        test_end = length if fold == folds - 1 else min(length, test_start + test_size)
        train_end = max(0, test_start - embargo)
        train = np.arange(train_end)
        test = np.arange(test_start, test_end)
        if len(train) >= min_train and len(test):
            splits.append((train, test))
    return splits

