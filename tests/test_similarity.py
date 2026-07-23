from datetime import timedelta

import pandas as pd

from alpha_engine.research.similarity import MarketStateIndex


def test_similarity_never_returns_future_state() -> None:
    index = pd.date_range("2020-01-01", periods=120, tz="UTC")
    frame = pd.DataFrame(
        {"a": range(120), "b": [value % 7 for value in range(120)], "close": range(100, 220)},
        index=index,
    )
    query = index[-1]
    results = MarketStateIndex(["a", "b"]).query(frame, at=query, embargo=10)
    assert results
    assert all(result.timestamp < query - timedelta(days=10) for result in results)
