from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from alpha_engine.research.leakage import purged_time_series_splits


@dataclass(frozen=True)
class FoldMetric:
    fold: int
    accuracy: float
    roc_auc: float
    train_rows: int
    test_rows: int


def train_direction_model(
    frame: pd.DataFrame, feature_columns: list[str], horizon: int = 10
) -> tuple[Pipeline, list[FoldMetric]]:
    clean = frame.copy()
    target = (clean["close"].shift(-horizon) > clean["close"]).astype(int)
    valid = clean[feature_columns].notna().all(axis=1) & target.notna()
    x = clean.loc[valid, feature_columns]
    y = target.loc[valid]
    splits = purged_time_series_splits(len(x), folds=5, embargo=horizon, min_train=120)
    metrics: list[FoldMetric] = []
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=4,
                    min_samples_leaf=20,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    for fold, (train, test) in enumerate(splits):
        model.fit(x.iloc[train], y.iloc[train])
        probability = model.predict_proba(x.iloc[test])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        auc = roc_auc_score(y.iloc[test], probability) if y.iloc[test].nunique() > 1 else np.nan
        metrics.append(
            FoldMetric(
                fold,
                float(accuracy_score(y.iloc[test], prediction)),
                float(auc),
                len(train),
                len(test),
            )
        )
    model.fit(x, y)
    return model, metrics

