from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class Strategy(Protocol):
    name: str

    def positions(self, frame: pd.DataFrame) -> pd.Series: ...


@dataclass(frozen=True)
class CostModel:
    fee_bps: float = 6
    slippage_bps: float = 4

    @property
    def one_way_rate(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 10_000


@dataclass(frozen=True)
class BacktestResult:
    strategy: str
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    turnover: float
    trades: int
    exposure: float

    def to_dict(self) -> dict[str, str | float | int]:
        return asdict(self)


class BuyAndHold:
    name = "buy_and_hold"

    def positions(self, frame: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=frame.index)


class RegimeTrend:
    name = "regime_trend"

    def positions(self, frame: pd.DataFrame) -> pd.Series:
        bullish = frame["regime"].isin(["bull_expansion", "bull_exhaustion"])
        return bullish.astype(float)


class Backtester:
    """Close-to-close vectorized backtest; signals are delayed one bar before execution."""

    def __init__(self, cost_model: CostModel) -> None:
        self.cost_model = cost_model

    def run(self, frame: pd.DataFrame, strategy: Strategy) -> tuple[BacktestResult, pd.DataFrame]:
        if len(frame) < 30:
            raise ValueError("Backtest requires at least 30 observations")
        desired = strategy.positions(frame).clip(0, 1)
        held = desired.shift(1).fillna(0)  # signal at t is filled for return t+1
        asset_return = frame["close"].pct_change().fillna(0)
        turnover = held.diff().abs().fillna(held.abs())
        costs = turnover * self.cost_model.one_way_rate
        net = held * asset_return - costs
        equity = (1 + net).cumprod()
        drawdown = equity / equity.cummax() - 1
        years = max(len(net) / 365, 1 / 365)
        total = float(equity.iloc[-1] - 1)
        annualized = float(equity.iloc[-1] ** (1 / years) - 1)
        volatility = float(net.std() * np.sqrt(365))
        sharpe = float(net.mean() / net.std() * np.sqrt(365)) if net.std() else 0.0
        result = BacktestResult(
            strategy=strategy.name,
            total_return=total,
            annualized_return=annualized,
            annualized_volatility=volatility,
            sharpe=sharpe,
            max_drawdown=float(drawdown.min()),
            turnover=float(turnover.sum()),
            trades=int((turnover > 0).sum()),
            exposure=float(held.mean()),
        )
        curve = pd.DataFrame(
            {
                "close": frame["close"],
                "position": held,
                "return": net,
                "equity": equity,
                "drawdown": drawdown,
            }
        )
        return result, curve

