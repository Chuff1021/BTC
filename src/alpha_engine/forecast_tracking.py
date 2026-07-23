from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


@dataclass(frozen=True)
class PromotionDecision:
    eligible: bool
    reasons: tuple[str, ...]
    requires_manual_approval: bool = True


class ForecastLedger:
    """Append predictions, settle them once, and report horizon-specific accuracy."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(database_path)) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS forecast_predictions (
                    prediction_id VARCHAR PRIMARY KEY,
                    issued_at TIMESTAMPTZ NOT NULL,
                    target_at TIMESTAMPTZ NOT NULL,
                    horizon VARCHAR NOT NULL,
                    anchor_price DOUBLE NOT NULL,
                    predicted_price DOUBLE NOT NULL,
                    low_price DOUBLE NOT NULL,
                    high_price DOUBLE NOT NULL,
                    model_version VARCHAR NOT NULL,
                    input_fingerprint VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    actual_price DOUBLE,
                    settled_at TIMESTAMPTZ,
                    absolute_percentage_error DOUBLE,
                    direction_correct BOOLEAN,
                    interval_covered BOOLEAN
                )
                """
            )

    @staticmethod
    def _fingerprint(forecast: dict[str, Any]) -> str:
        material = {
            key: forecast.get(key)
            for key in (
                "issued_at",
                "target_at",
                "horizon",
                "current_price",
                "predicted_price",
                "low_price",
                "high_price",
                "model_version",
                "latest_completed_candle",
            )
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()

    def record(self, forecasts: list[dict[str, Any]]) -> list[str]:
        identifiers: list[str] = []
        with duckdb.connect(str(self.database_path)) as connection:
            for forecast in forecasts:
                fingerprint = self._fingerprint(forecast)
                prediction_id = str(uuid.uuid5(uuid.NAMESPACE_URL, fingerprint))
                identifiers.append(prediction_id)
                connection.execute(
                    """
                    INSERT INTO forecast_predictions VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN',
                        NULL, NULL, NULL, NULL, NULL
                    )
                    ON CONFLICT (prediction_id) DO NOTHING
                    """,
                    [
                        prediction_id,
                        forecast["issued_at"],
                        forecast["target_at"],
                        forecast["horizon"],
                        forecast["current_price"],
                        forecast["predicted_price"],
                        forecast["low_price"],
                        forecast["high_price"],
                        forecast["model_version"],
                        fingerprint,
                    ],
                )
        return identifiers

    def settle(
        self,
        observations: pd.DataFrame,
        *,
        settled_at: datetime | None = None,
    ) -> int:
        """Settle due forecasts using the first observed price at or after target_at."""

        if observations.empty or "price" not in observations:
            return 0
        frame = observations.copy()
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame = frame.sort_index()
        now = settled_at or datetime.now(UTC)
        settled = 0
        with duckdb.connect(str(self.database_path)) as connection:
            rows = connection.execute(
                """
                SELECT
                    prediction_id, target_at, anchor_price,
                    predicted_price, low_price, high_price
                FROM forecast_predictions
                WHERE status = 'OPEN' AND target_at <= ?
                ORDER BY target_at
                """,
                [now],
            ).fetchall()
            for prediction_id, target_at, anchor, prediction, low, high in rows:
                candidates = frame.loc[frame.index >= pd.Timestamp(target_at)]
                if candidates.empty:
                    continue
                actual = float(candidates.iloc[0]["price"])
                direction_correct = (prediction - anchor) * (actual - anchor) >= 0
                connection.execute(
                    """
                    UPDATE forecast_predictions
                    SET status = 'SETTLED',
                        actual_price = ?,
                        settled_at = ?,
                        absolute_percentage_error = ?,
                        direction_correct = ?,
                        interval_covered = ?
                    WHERE prediction_id = ? AND status = 'OPEN'
                    """,
                    [
                        actual,
                        now,
                        abs(prediction / actual - 1),
                        direction_correct,
                        low <= actual <= high,
                        prediction_id,
                    ],
                )
                settled += 1
        return settled

    def accuracy(self) -> list[dict[str, Any]]:
        with duckdb.connect(str(self.database_path), read_only=True) as connection:
            rows = connection.execute(
                """
                SELECT
                    horizon,
                    model_version,
                    COUNT(*) AS samples,
                    MEDIAN(absolute_percentage_error) AS median_absolute_percentage_error,
                    AVG(CASE WHEN direction_correct THEN 1.0 ELSE 0.0 END) AS direction_accuracy,
                    AVG(CASE WHEN interval_covered THEN 1.0 ELSE 0.0 END) AS interval_coverage
                FROM forecast_predictions
                WHERE status = 'SETTLED'
                GROUP BY horizon, model_version
                ORDER BY horizon, model_version
                """
            ).fetchall()
        return [
            {
                "horizon": row[0],
                "model_version": row[1],
                "samples": row[2],
                "median_absolute_percentage_error": row[3],
                "direction_accuracy": row[4],
                "interval_coverage": row[5],
            }
            for row in rows
        ]


def evaluate_challenger(
    champion: dict[str, float],
    challenger: dict[str, float],
    *,
    settled_predictions: int,
    minimum_predictions: int = 100,
) -> PromotionDecision:
    """Apply conservative gates; an eligible challenger still needs human approval."""

    reasons: list[str] = []
    if settled_predictions < minimum_predictions:
        reasons.append(f"needs {minimum_predictions} settled predictions")
    if (
        challenger["median_absolute_percentage_error"]
        >= champion["median_absolute_percentage_error"]
    ):
        reasons.append("median absolute percentage error did not improve")
    if challenger["direction_accuracy"] < champion["direction_accuracy"]:
        reasons.append("direction accuracy regressed")
    if abs(challenger["interval_coverage"] - 0.80) > abs(champion["interval_coverage"] - 0.80):
        reasons.append("80% interval calibration regressed")
    return PromotionDecision(eligible=not reasons, reasons=tuple(reasons))
