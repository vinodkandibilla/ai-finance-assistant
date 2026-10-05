from fastapi.testclient import TestClient

from ai_finance_assistant.main import create_app


def test_health_endpoint_returns_service_status() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}
