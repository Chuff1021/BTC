from datetime import timedelta

import pandas as pd
import pytest

from alpha_engine.data.connectors import SyntheticMarketSource
from alpha_engine.research.features import FeatureStore
from alpha_engine.research.leakage import LeakageError, assert_point_in_time
from alpha_engine.types import FetchRequest


def synthetic(days: int = 400) -> pd.DataFrame:
    return SyntheticMarketSource().fetch(
        FetchRequest(
            "BTCUSDT",
            pd.Timestamp("2020-01-01", tz="UTC"),
            pd.Timestamp("2020-01-01", tz="UTC") + timedelta(days=days),
        )
    )


def test_feature_store_is_point_in_time() -> None:
    raw = synthetic()
    features = FeatureStore().build(raw)
    assert not features.empty
    assert (features["max_available_at"] <= features.index).all()
    assert features["return_1d"].iloc[-1] != raw["close"].pct_change().iloc[-1]


def test_rejects_impossible_release_time() -> None:
    raw = synthetic(20)
    raw.loc[raw.index[0], "available_at"] = raw.index[0] - timedelta(days=1)
    with pytest.raises(LeakageError):
        assert_point_in_time(raw)
