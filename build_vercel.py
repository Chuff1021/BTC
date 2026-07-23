"""Embed the dashboard asset in a direct module for Vercel's function tracer."""

from pathlib import Path


def main() -> None:
    dashboard = Path("src/alpha_engine/web/dashboard.html").read_text(encoding="utf-8")
    Path("dashboard_content.py").write_text(
        f"DASHBOARD_HTML = {dashboard!r}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
