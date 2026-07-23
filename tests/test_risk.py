import pytest

from alpha_engine.paper import PaperBroker
from alpha_engine.risk import RiskLimitBreached, RiskLimits, RiskManager


def test_paper_broker_cannot_exceed_exposure() -> None:
    broker = PaperBroker(100_000, RiskManager(RiskLimits(max_position_fraction=0.25)))
    with pytest.raises(RiskLimitBreached):
        broker.market_order("buy", 2, 20_000)


def test_paper_broker_has_no_live_exchange_surface() -> None:
    broker = PaperBroker(100_000, RiskManager(RiskLimits()))
    assert not hasattr(broker, "withdraw")
    assert not hasattr(broker, "exchange")

