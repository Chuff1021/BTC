import json

import typer

from alpha_engine.config import Settings
from alpha_engine.logging import configure_logging
from alpha_engine.pipeline import run_research

app = typer.Typer(no_args_is_help=True, help="Crypto Alpha Research Engine")


@app.callback()
def main() -> None:
    """Research, validate, and paper-test crypto market hypotheses."""


@app.command()
def demo(
    days: int = typer.Option(1200, min=365, help="Synthetic daily observations"),
    source: str = typer.Option("synthetic", help="synthetic or binance"),
) -> None:
    """Run ingestion → features → regimes → hypotheses → backtests."""
    settings = Settings()
    configure_logging(settings.log_level)
    if source not in {"synthetic", "binance"}:
        raise typer.BadParameter("source must be synthetic or binance")
    summary = run_research(settings, days=days, source=source)
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
