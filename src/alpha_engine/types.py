from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import pandas as pd


class DataDomain(StrEnum):
    SPOT = "spot"
    DERIVATIVES = "derivatives"
    ETF = "etf"
    OPTIONS = "options"
    STABLECOIN = "stablecoin"
    ONCHAIN = "onchain"
    MACRO = "macro"
    SENTIMENT = "sentiment"
    PREDICTION_MARKET = "prediction_market"


@dataclass(frozen=True)
class FetchRequest:
    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    interval: str = "1d"


class DataSource(Protocol):
    name: str
    domain: DataDomain

    def fetch(self, request: FetchRequest) -> pd.DataFrame:
        """Return UTC-indexed, point-in-time observations with an available_at column."""
        ...
