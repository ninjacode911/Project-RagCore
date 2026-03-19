from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "components" in data


def test_root_redirects():
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (301, 302, 307, 308)


def test_docs_page():
    response = client.get("/docs")
    assert response.status_code == 200
