import os
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Secrets are environment-only."""

    model_config = SettingsConfigDict(env_prefix="CAE_", env_file=".env", extra="ignore")

    env: str = "development"
    database_path: Path = Field(default_factory=lambda: _runtime_path("data/alpha.duckdb"))
    parquet_root: Path = Field(default_factory=lambda: _runtime_path("data/parquet"))
    artifact_root: Path = Field(default_factory=lambda: _runtime_path("artifacts"))
    log_level: str = "INFO"
    allow_network: bool = True
    paper_trading_only: bool = True
    starting_cash: float = Field(100_000, gt=0)
    fee_bps: float = Field(6, ge=0)
    slippage_bps: float = Field(4, ge=0)

    @model_validator(mode="after")
    def live_trading_is_forbidden(self) -> "Settings":
        if not self.paper_trading_only:
            raise ValueError("This release is intentionally paper-trading only")
        return self

    def ensure_directories(self) -> None:
        for path in (self.database_path.parent, self.parquet_root, self.artifact_root):
            path.mkdir(parents=True, exist_ok=True)


def _runtime_path(relative: str) -> Path:
    """Use Vercel's writable scratch space while preserving local defaults."""
    return Path("/tmp/crypto-alpha") / relative if os.getenv("VERCEL") else Path(relative)
