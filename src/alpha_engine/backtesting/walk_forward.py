from dataclasses import dataclass

import pandas as pd

from alpha_engine.backtesting.engine import Backtester, BacktestResult, Strategy


@dataclass(frozen=True)
class WalkForwardFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    result: BacktestResult


def walk_forward(
    frame: pd.DataFrame,
    strategy: Strategy,
    backtester: Backtester,
    train_size: int = 365,
    test_size: int = 90,
    embargo: int = 5,
) -> list[WalkForwardFold]:
    folds: list[WalkForwardFold] = []
    cursor = train_size + embargo
    while cursor + test_size <= len(frame):
        train = frame.iloc[cursor - embargo - train_size : cursor - embargo]
        test = frame.iloc[cursor : cursor + test_size]
        result, _ = backtester.run(test, strategy)
        folds.append(
            WalkForwardFold(
                train.index[0],
                train.index[-1],
                test.index[0],
                test.index[-1],
                result,
            )
        )
        cursor += test_size
    return folds

