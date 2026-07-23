from pathlib import Path

import pandas as pd

from alpha_engine.backtesting.engine import Backtester, CostModel, RegimeTrend
from alpha_engine.backtesting.walk_forward import walk_forward
from alpha_engine.config import Settings
from alpha_engine.data.connectors import CsvPointInTimeSource, connector_catalog
from alpha_engine.data.storage import ResearchStore
from alpha_engine.modeling import train_direction_model
from alpha_engine.pipeline import latest_summary, run_research, terminal_snapshot
from alpha_engine.research.features import FEATURE_COLUMNS
from alpha_engine.types import DataDomain, FetchRequest


def test_end_to_end_research_pipeline(tmp_path: Path) -> None:
    settings = Settings(
        database_path=tmp_path / "alpha.duckdb",
        parquet_root=tmp_path / "parquet",
        artifact_root=tmp_path / "artifacts",
    )
    summary = run_research(settings, days=800)
    assert summary["synthetic_data"] is True
    assert len(summary["backtests"]) == 2
    assert latest_summary(settings) == summary
    snapshot = terminal_snapshot(settings, periods=100)
    assert snapshot is not None
    assert len(snapshot["series"]) == 100
    assert snapshot["series"][-1]["close"] > 0
    store = ResearchStore(settings.database_path, settings.parquet_root)
    assert {"market_daily", "features_daily", "equity_regime_trend"}.issubset(store.tables())
    features = store.read("features_daily")
    assert not features.empty


def test_model_and_walk_forward(tmp_path: Path) -> None:
    settings = Settings(
        database_path=tmp_path / "alpha.duckdb",
        parquet_root=tmp_path / "parquet",
        artifact_root=tmp_path / "artifacts",
    )
    run_research(settings, days=1200)
    frame = ResearchStore(settings.database_path, settings.parquet_root).read("features_daily")
    model, metrics = train_direction_model(frame, FEATURE_COLUMNS)
    assert len(metrics) >= 2
    assert hasattr(model, "predict_proba")
    folds = walk_forward(frame, RegimeTrend(), Backtester(CostModel()), train_size=365)
    assert folds
    assert all(fold.test_start > fold.train_end for fold in folds)


def test_csv_connector_contract_and_catalog(tmp_path: Path) -> None:
    path = tmp_path / "macro.csv"
    pd.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02"],
            "available_at": ["2024-01-02", "2024-01-03"],
            "dxy": [100.0, 101.0],
        }
    ).to_csv(path, index=False)
    source = CsvPointInTimeSource(path, DataDomain.MACRO, "macro")
    result = source.fetch(
        FetchRequest(
            "DXY",
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-03", tz="UTC"),
        )
    )
    assert len(result) == 2
    assert set(connector_catalog(tmp_path)) == {
        "spot",
        "derivatives_candles",
        "funding",
        "open_interest",
        "basis",
        "etf_flows",
        "options",
        "stablecoins",
        "onchain",
        "macro",
        "sentiment",
        "prediction_markets",
    }
