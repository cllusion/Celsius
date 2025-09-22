from fastapi.testclient import TestClient

from project.app.main import app


def test_root():
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert r.json().get("message") == "Celsius minimal app"
