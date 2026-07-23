from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    created_at: str
    name: str
    params: dict[str, Any]
    metrics: dict[str, float]
    data_fingerprint: str


class ExperimentTracker:
    """Dependency-light JSONL tracker; records map directly to MLflow params/metrics."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        data_fingerprint: str,
    ) -> RunRecord:
        record = RunRecord(
            run_id=str(uuid.uuid4()),
            created_at=datetime.now(UTC).isoformat(),
            name=name,
            params=params,
            metrics=metrics,
            data_fingerprint=data_fingerprint,
        )
        with (self.root / "runs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        return record

