from fastapi.testclient import TestClient

import app as vercel_adapter


def test_tracking_reads_database_without_refetching_market(monkeypatch) -> None:
    expected = {
        "status": "LIVE",
        "open_predictions": 8,
        "settled_predictions": 1,
        "settled_now": 0,
        "scorecards": [],
        "recent": [],
    }
    monkeypatch.setattr(vercel_adapter, "read_tracking", lambda: expected)
    monkeypatch.setattr(
        vercel_adapter,
        "_market_data",
        lambda: (_ for _ in ()).throw(AssertionError("market feed must not be called")),
    )
    response = TestClient(vercel_adapter.app).get("/api/tracking")
    assert response.status_code == 200
    assert response.json() == expected


def test_cron_rejects_missing_or_invalid_secret(monkeypatch) -> None:
    client = TestClient(vercel_adapter.app)
    monkeypatch.delenv("CRON_SECRET", raising=False)
    assert client.get("/api/cron/forecasts").status_code == 503
    monkeypatch.setenv("CRON_SECRET", "expected")
    response = client.get(
        "/api/cron/forecasts",
        headers={"Authorization": "Bearer wrong"},
    )
    assert response.status_code == 401
