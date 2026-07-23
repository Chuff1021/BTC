"""Neon-backed live forecast ledger for the Vercel adapter."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from psycopg.rows import dict_row

SCHEMA_VERSION = 1


def database_url() -> str | None:
    return os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")


def _prediction_id(forecast: dict[str, Any]) -> str:
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
    digest = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def _observations(candles: list[dict[str, Any]]) -> list[tuple[datetime, float]]:
    """Return completed 15m candle closes keyed by the time each close became known."""

    return [
        (
            datetime.fromisoformat(str(candle["timestamp"]).replace("Z", "+00:00")).astimezone(UTC)
            + timedelta(minutes=15),
            float(candle["close"]),
        )
        for candle in candles
    ]


def ensure_schema(connection: psycopg.Connection[Any]) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS forecast_predictions (
            prediction_id UUID PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            issued_at TIMESTAMPTZ NOT NULL,
            target_at TIMESTAMPTZ NOT NULL,
            horizon TEXT NOT NULL,
            anchor_price DOUBLE PRECISION NOT NULL,
            predicted_price DOUBLE PRECISION NOT NULL,
            low_price DOUBLE PRECISION NOT NULL,
            high_price DOUBLE PRECISION NOT NULL,
            model_version TEXT NOT NULL,
            latest_completed_candle TIMESTAMPTZ NOT NULL,
            prediction_market_score DOUBLE PRECISION,
            prediction_market_weight DOUBLE PRECISION NOT NULL DEFAULT 0,
            status TEXT NOT NULL CHECK (status IN ('OPEN', 'SETTLED')),
            actual_price DOUBLE PRECISION,
            actual_observed_at TIMESTAMPTZ,
            settled_at TIMESTAMPTZ,
            absolute_percentage_error DOUBLE PRECISION,
            direction_correct BOOLEAN,
            interval_covered BOOLEAN,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS forecast_due_idx
        ON forecast_predictions (status, target_at)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS forecast_score_idx
        ON forecast_predictions (horizon, model_version, status)
        """
    )


def _settle_due(
    connection: psycopg.Connection[Any],
    observations: list[tuple[datetime, float]],
) -> int:
    if not observations:
        return 0
    latest_observed_at = observations[-1][0]
    due = connection.execute(
        """
        SELECT
            prediction_id, target_at, anchor_price,
            predicted_price, low_price, high_price
        FROM forecast_predictions
        WHERE status = 'OPEN' AND target_at <= %s
        ORDER BY target_at
        FOR UPDATE SKIP LOCKED
        """,
        (latest_observed_at,),
    ).fetchall()
    settled = 0
    for row in due:
        prediction_id = row["prediction_id"]
        target_at = row["target_at"]
        anchor = row["anchor_price"]
        prediction = row["predicted_price"]
        low = row["low_price"]
        high = row["high_price"]
        match = next(
            (
                (observed_at, price)
                for observed_at, price in observations
                if observed_at >= target_at
            ),
            None,
        )
        if match is None:
            continue
        actual_observed_at, actual = match
        direction_correct = (prediction - anchor) * (actual - anchor) >= 0
        connection.execute(
            """
            UPDATE forecast_predictions
            SET
                status = 'SETTLED',
                actual_price = %s,
                actual_observed_at = %s,
                settled_at = NOW(),
                absolute_percentage_error = %s,
                direction_correct = %s,
                interval_covered = %s
            WHERE prediction_id = %s AND status = 'OPEN'
            """,
            (
                actual,
                actual_observed_at,
                abs(prediction / actual - 1),
                direction_correct,
                low <= actual <= high,
                prediction_id,
            ),
        )
        settled += 1
    return settled


def _scorecards(connection: psycopg.Connection[Any]) -> list[dict[str, Any]]:
    return list(
        connection.execute(
            """
            SELECT
                horizon,
                model_version,
                COUNT(*)::INTEGER AS samples,
                PERCENTILE_CONT(0.5) WITHIN GROUP (
                    ORDER BY absolute_percentage_error
                ) AS median_absolute_percentage_error,
                AVG(CASE WHEN direction_correct THEN 1.0 ELSE 0.0 END) AS direction_accuracy,
                AVG(CASE WHEN interval_covered THEN 1.0 ELSE 0.0 END) AS interval_coverage
            FROM forecast_predictions
            WHERE status = 'SETTLED'
            GROUP BY horizon, model_version
            ORDER BY
                CASE horizon
                    WHEN '15m' THEN 1 WHEN '1h' THEN 2 WHEN '4h' THEN 3
                    WHEN '8h' THEN 4 WHEN '12h' THEN 5 WHEN '1d' THEN 6
                    WHEN '1w' THEN 7 WHEN '1mo' THEN 8 ELSE 9
                END,
                model_version
            """
        ).fetchall()
    )


def _recent(connection: psycopg.Connection[Any]) -> list[dict[str, Any]]:
    return list(
        connection.execute(
            """
            SELECT
                prediction_id::TEXT,
                issued_at,
                target_at,
                horizon,
                anchor_price,
                predicted_price,
                low_price,
                high_price,
                status,
                actual_price,
                actual_observed_at,
                absolute_percentage_error,
                direction_correct,
                interval_covered,
                model_version
            FROM forecast_predictions
            ORDER BY issued_at DESC, target_at
            LIMIT 40
            """
        ).fetchall()
    )


def record_settle_and_score(
    forecasts: list[dict[str, Any]],
    candles_15m: list[dict[str, Any]],
    *,
    prediction_market_score: float | None,
    prediction_market_weight: float,
) -> dict[str, Any]:
    url = database_url()
    if not url:
        return {
            "status": "DATABASE_NOT_CONFIGURED",
            "open_predictions": 0,
            "settled_predictions": 0,
            "settled_now": 0,
            "scorecards": [],
            "recent": [],
        }
    with psycopg.connect(url, autocommit=False, row_factory=dict_row) as connection:
        ensure_schema(connection)
        for forecast in forecasts:
            connection.execute(
                """
                INSERT INTO forecast_predictions (
                    prediction_id, schema_version, issued_at, target_at, horizon,
                    anchor_price, predicted_price, low_price, high_price,
                    model_version, latest_completed_candle,
                    prediction_market_score, prediction_market_weight, status
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, 'OPEN'
                )
                ON CONFLICT (prediction_id) DO NOTHING
                """,
                (
                    _prediction_id(forecast),
                    SCHEMA_VERSION,
                    forecast["issued_at"],
                    forecast["target_at"],
                    forecast["horizon"],
                    forecast["current_price"],
                    forecast["predicted_price"],
                    forecast["low_price"],
                    forecast["high_price"],
                    forecast["model_version"],
                    forecast["latest_completed_candle"],
                    prediction_market_score,
                    prediction_market_weight,
                ),
            )
        settled_now = _settle_due(connection, _observations(candles_15m))
        counts = connection.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE status = 'OPEN')::INTEGER AS open_predictions,
                COUNT(*) FILTER (WHERE status = 'SETTLED')::INTEGER AS settled_predictions
            FROM forecast_predictions
            """
        ).fetchone()
        scorecards = _scorecards(connection)
        recent = _recent(connection)
        connection.commit()
    return {
        "status": "LIVE",
        "open_predictions": counts["open_predictions"],
        "settled_predictions": counts["settled_predictions"],
        "settled_now": settled_now,
        "scorecards": scorecards,
        "recent": recent,
        "promotion_policy": "challenger gates plus manual approval",
    }


def read_tracking() -> dict[str, Any]:
    url = database_url()
    if not url:
        return {
            "status": "DATABASE_NOT_CONFIGURED",
            "open_predictions": 0,
            "settled_predictions": 0,
            "settled_now": 0,
            "scorecards": [],
            "recent": [],
        }
    with psycopg.connect(url, autocommit=True, row_factory=dict_row) as connection:
        ensure_schema(connection)
        counts = connection.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE status = 'OPEN')::INTEGER AS open_predictions,
                COUNT(*) FILTER (WHERE status = 'SETTLED')::INTEGER AS settled_predictions
            FROM forecast_predictions
            """
        ).fetchone()
        return {
            "status": "LIVE",
            "open_predictions": counts["open_predictions"],
            "settled_predictions": counts["settled_predictions"],
            "settled_now": 0,
            "scorecards": _scorecards(connection),
            "recent": _recent(connection),
            "promotion_policy": "challenger gates plus manual approval",
        }


def migrate() -> None:
    url = database_url()
    if not url:
        raise RuntimeError("POSTGRES_URL or DATABASE_URL is required")
    with psycopg.connect(url, autocommit=True) as connection:
        ensure_schema(connection)


if __name__ == "__main__":
    migrate()
