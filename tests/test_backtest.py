import pandas as pd

from alpha_engine.backtesting.engine import Backtester, BuyAndHold, CostModel


def test_costs_are_charged_and_signal_is_delayed() -> None:
    index = pd.date_range("2024-01-01", periods=40, tz="UTC")
    frame = pd.DataFrame({"close": [100 * 1.01**i for i in range(40)]}, index=index)
    result, curve = Backtester(CostModel(10, 0)).run(frame, BuyAndHold())
    assert curve["position"].iloc[0] == 0
    assert curve["position"].iloc[1] == 1
    assert result.total_return < frame["close"].iloc[-1] / frame["close"].iloc[0] - 1
    assert result.trades == 1

