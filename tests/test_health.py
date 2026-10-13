from fastapi.testclient import TestClient

from ai_finance_assistant.agents.finance_qa.config import FINANCE_PDF_PATH
from ai_finance_assistant.main import create_app


def test_health_endpoint_returns_service_status() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}


def test_finance_pdf_path_exists() -> None:
    assert FINANCE_PDF_PATH.exists()
    assert FINANCE_PDF_PATH.is_file()
