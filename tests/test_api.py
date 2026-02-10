import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.api.app import app
from src.prediction_logging.prediction_logger import PredictionLogger


@pytest.fixture
def mock_state():
    """Mock model, preprocessor, and metadata for API tests."""
    model = MagicMock()
    model.classes_ = np.array([0, 1])
    model.predict_proba.return_value = np.array([[0.7, 0.3]])

    preprocessor = MagicMock()
    preprocessor.transform.return_value = np.array([[1.0, 2.0, 3.0]])

    metadata = {
        "best_model": "test_model",
        "model_version": "20240101.abc1234",
        "threshold": 0.5,
        "ref_date": "2023-06-01",
    }

    return model, preprocessor, metadata


@pytest.fixture
def client(mock_state, tmp_path):
    """TestClient with mocked app state."""
    model, preprocessor, metadata = mock_state

    app.state.model = model
    app.state.preprocessor = preprocessor
    app.state.metadata = metadata

    # Use a temp DB for the prediction logger
    app.state.logger = PredictionLogger(db_path=tmp_path / "test.db")

    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_healthy(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_unhealthy_returns_503(self):
        """When model is not loaded, health should return 503."""
        test_app = TestClient(app, raise_server_exceptions=False)
        # Remove model from state to simulate failure
        if hasattr(app.state, "model"):
            delattr(app.state, "model")
        response = test_app.get("/health")
        assert response.status_code == 503
        assert response.json()["status"] == "unhealthy"


class TestPredictEndpoint:
    SAMPLE_PAYLOAD = {
        "esent": 10,
        "eopenrate": 0.3,
        "eclickrate": 0.1,
        "avgorder": 50.0,
        "ordfreq": 5,
        "paperless": 1,
        "refill": 0,
        "doorstep": 1,
        "favday": "Mon",
        "city": "DEL",
        "created": "2023-01-01",
        "firstorder": "2023-01-10",
        "lastorder": "2023-06-01",
    }

    def test_predict_returns_200(self, client):
        response = client.post("/predict", json=self.SAMPLE_PAYLOAD)
        assert response.status_code == 200

    def test_predict_response_fields(self, client):
        response = client.post("/predict", json=self.SAMPLE_PAYLOAD)
        data = response.json()
        assert "request_id" in data
        assert "churn_score" in data
        assert "predicted_label" in data
        assert "threshold" in data
        assert "model_name" in data

    def test_predict_with_true_label(self, client):
        payload = {**self.SAMPLE_PAYLOAD, "true_label": 1}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_missing_field_returns_422(self, client):
        payload = {"esent": 10}  # Missing required fields
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestMetricsEndpoint:
    def test_metrics_empty_db(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data or "service_metrics" in data

    def test_metrics_after_prediction(self, client):
        # Make a prediction first
        client.post("/predict", json=TestPredictEndpoint.SAMPLE_PAYLOAD)
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "service_metrics" in data
        assert data["service_metrics"]["total_predictions"] >= 1
