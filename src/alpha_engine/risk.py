from dataclasses import dataclass


class RiskLimitBreached(RuntimeError):
    pass


@dataclass(frozen=True)
class RiskLimits:
    max_position_fraction: float = 0.25
    max_order_notional: float = 10_000
    max_drawdown: float = 0.15
    daily_loss_limit: float = 0.03


class RiskManager:
    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits

    def approve(
        self,
        order_notional: float,
        equity: float,
        position_notional: float,
        peak_equity: float,
        day_start_equity: float,
    ) -> None:
        if abs(order_notional) > self.limits.max_order_notional:
            raise RiskLimitBreached("Order exceeds max notional")
        if abs(position_notional + order_notional) > equity * self.limits.max_position_fraction:
            raise RiskLimitBreached("Position exceeds exposure limit")
        if equity < peak_equity * (1 - self.limits.max_drawdown):
            raise RiskLimitBreached("Drawdown kill switch is active")
        if equity < day_start_equity * (1 - self.limits.daily_loss_limit):
            raise RiskLimitBreached("Daily loss cutoff is active")

