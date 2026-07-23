from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from alpha_engine.config import Settings
from alpha_engine.pipeline import latest_summary, run_research, terminal_snapshot

settings = Settings()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings.ensure_directories()
    yield


app = FastAPI(
    title="Crypto Alpha Research Engine",
    version="0.1.0",
    description="Research and paper-trading only. No live order routes exist.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "mode": "paper", "paper_trading_only": True}


@app.post("/api/research/run")
def research_run(
    days: int = Query(1200, ge=365, le=5000),
    source: str = Query("synthetic", pattern="^(synthetic|binance)$"),
) -> dict[str, object]:
    run_research(settings, days=days, source=source)
    snapshot = terminal_snapshot(settings)
    if snapshot is None:
        raise HTTPException(500, "Research pipeline completed without a terminal snapshot")
    return snapshot


@app.get("/api/research/latest")
def research_latest() -> dict[str, object]:
    summary = latest_summary(settings)
    if summary is None:
        raise HTTPException(404, "Run `alpha demo` or POST /api/research/run first")
    return summary


@app.get("/api/research/terminal")
def research_terminal(periods: int = Query(240, ge=60, le=1000)) -> dict[str, object]:
    snapshot = terminal_snapshot(settings, periods)
    if snapshot is None:
        # Vercel functions have ephemeral storage, so each cold instance can seed its
        # own deterministic research demo without credentials or durable writes.
        run_research(settings, days=max(800, periods + 250), source="synthetic")
        snapshot = terminal_snapshot(settings, periods)
    if snapshot is None:
        raise HTTPException(500, "Unable to initialize the research snapshot")
    return snapshot


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return files("alpha_engine.web").joinpath("dashboard.html").read_text(encoding="utf-8")
