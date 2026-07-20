from app.api.routes import health as health_route


def test_readiness_fails_when_required_models_are_not_warmed(client, monkeypatch):
    monkeypatch.setattr(health_route.settings, "warm_models_on_startup", True)
    monkeypatch.setattr(health_route, "models_warmed", lambda: False)
    monkeypatch.setattr(health_route, "HEALTH_PROBES", {"dependency": lambda: None})

    response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["models_warmed"] is False
