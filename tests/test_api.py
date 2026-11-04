from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_api_root_endpoint_returns_online_status():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_prediction_endpoint_rejects_malformed_payloads():
    response = client.post(
        "/predict",
        json={"tenure": 12, "MonthlyCharges": 70.0},
    )
    assert response.status_code == 422


def test_prediction_endpoint_reports_missing_model_artifacts(monkeypatch):
    import api.main as api_module

    monkeypatch.setattr(api_module, "predictor", None)
    response = client.post(
        "/predict",
        json={
            "tenure": 12,
            "MonthlyCharges": 70.0,
            "TotalCharges": 840.0,
            "Contract_One_year": 0,
            "Contract_Two_year": 0,
            "PaperlessBilling_Yes": 1,
        },
    )
    assert response.status_code == 503


def test_prediction_endpoint_returns_churn_probability(monkeypatch):
    import api.main as api_module

    class StubPredictor:
        def predict(self, input_data):
            return [1] * len(input_data), [0.8] * len(input_data)

    monkeypatch.setattr(api_module, "predictor", StubPredictor())
    response = client.post(
        "/predict",
        json={
            "tenure": 12,
            "MonthlyCharges": 70.0,
            "TotalCharges": 840.0,
            "Contract_One_year": 1,
            "Contract_Two_year": 0,
            "PaperlessBilling_Yes": 1,
        },
    )
    assert response.status_code == 200
    assert response.json()["churn_prediction"] == 1
    assert response.json()["churn_probability"] == 0.8
