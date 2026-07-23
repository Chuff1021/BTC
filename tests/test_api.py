from fastapi.testclient import TestClient

from alpha_engine.api import app


def test_health_is_explicitly_paper_only() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["paper_trading_only"] is True
    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "Intelligence Workstation" in dashboard.text
    assert "PAPER RESEARCH ONLY" in dashboard.text
