def test_evaluation_requires_authentication(client):
    response = client.get("/api/v1/test/metrics")
    assert response.status_code == 401
