from fastapi.testclient import TestClient
from serve import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert type(response.json()["prediction"]) == int
