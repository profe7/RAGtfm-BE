def test_response_carries_request_id(client):
    response = client.get("/api/v1/health/live")

    assert response.status_code == 200
    assert response.headers.get("x-request-id")


def test_metrics_endpoint_exposes_http_counter(client):
    client.get("/api/v1/health/live")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "http_requests_total" in response.text
