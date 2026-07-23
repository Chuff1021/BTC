from dataclasses import dataclass, field
from datetime import UTC, datetime

from alpha_engine.risk import RiskManager


@dataclass(frozen=True)
class Fill:
    timestamp: str
    side: str
    quantity: float
    price: float
    fee: float


@dataclass
class PaperBroker:
    """In-memory cash broker. It has no exchange client and cannot place live orders."""

    cash: float
    risk: RiskManager
    fee_bps: float = 6
    slippage_bps: float = 4
    quantity: float = 0
    peak_equity: float = field(init=False)
    day_start_equity: float = field(init=False)
    fills: list[Fill] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.peak_equity = self.cash
        self.day_start_equity = self.cash

    def equity(self, price: float) -> float:
        return self.cash + self.quantity * price

    def market_order(self, side: str, quantity: float, mark_price: float) -> Fill:
        if side not in {"buy", "sell"} or quantity <= 0:
            raise ValueError("Order must be a positive buy or sell")
        signed = quantity if side == "buy" else -quantity
        slip = self.slippage_bps / 10_000 * (1 if side == "buy" else -1)
        fill_price = mark_price * (1 + slip)
        notional = signed * fill_price
        equity = self.equity(mark_price)
        self.risk.approve(
            notional, equity, self.quantity * mark_price, self.peak_equity, self.day_start_equity
        )
        fee = abs(notional) * self.fee_bps / 10_000
        if self.cash - notional - fee < 0:
            raise ValueError("Insufficient paper cash")
        if self.quantity + signed < 0:
            raise ValueError("Shorting is disabled")
        self.cash -= notional + fee
        self.quantity += signed
        self.peak_equity = max(self.peak_equity, self.equity(mark_price))
        fill = Fill(datetime.now(UTC).isoformat(), side, quantity, fill_price, fee)
        self.fills.append(fill)
        return fill
