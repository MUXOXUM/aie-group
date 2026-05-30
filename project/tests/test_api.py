"""
Базовые тесты сервиса (pytest).
Запуск: pytest tests/ -v
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    # Импортируем здесь, чтобы lifespan отработал
    from src.service.app import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    data = r.json()
    assert "model_name" in data
    assert "metrics" in data
    assert len(data["features"]) > 0


def test_predict_basic(client):
    payload = {
        "manufacturer": "toyota",
        "condition": "good",
        "cylinders": "4 cylinders",
        "fuel": "gas",
        "odometer": 60000,
        "title_status": "clean",
        "transmission": "automatic",
        "drive": "fwd",
        "type": "sedan",
        "paint_color": "white",
        "state": "ca",
        "car_age": 5,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["predicted_price"] > 0
    assert data["ci_lower"] <= data["predicted_price"] <= data["ci_upper"]
    assert 0 <= data["confidence"] <= 1


def test_predict_old_car(client):
    """Старый автомобиль должен стоить дешевле нового."""
    base = {
        "manufacturer": "ford",
        "condition": "fair",
        "cylinders": "6 cylinders",
        "fuel": "gas",
        "odometer": 150000,
        "title_status": "clean",
        "transmission": "automatic",
        "drive": "rwd",
        "type": "sedan",
        "paint_color": "blue",
        "state": "tx",
    }
    old  = client.post("/predict", json={**base, "car_age": 20}).json()
    new_ = client.post("/predict", json={**base, "car_age": 2}).json()
    assert old["predicted_price"] < new_["predicted_price"]


def test_predict_validation_error(client):
    r = client.post("/predict", json={"odometer": -100, "car_age": 5})
    assert r.status_code == 422


def test_batch_predict(client):
    cars = [
        {
            "manufacturer": "bmw",
            "condition": "excellent",
            "cylinders": "6 cylinders",
            "fuel": "gas",
            "odometer": 30000,
            "title_status": "clean",
            "transmission": "automatic",
            "drive": "rwd",
            "type": "sedan",
            "paint_color": "black",
            "state": "ny",
            "car_age": 3,
        },
        {
            "manufacturer": "honda",
            "condition": "good",
            "cylinders": "4 cylinders",
            "fuel": "gas",
            "odometer": 80000,
            "title_status": "clean",
            "transmission": "automatic",
            "drive": "fwd",
            "type": "sedan",
            "paint_color": "silver",
            "state": "fl",
            "car_age": 8,
        },
    ]
    r = client.post("/predict/batch", json={"cars": cars})
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2
    assert all(p["predicted_price"] > 0 for p in data["predictions"])
