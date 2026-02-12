import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.prediction_logging.prediction_logger import PredictionLogger


@pytest.fixture
def logger(tmp_path):
    """Create a PredictionLogger with a temporary database."""
    return PredictionLogger(db_path=tmp_path / "test_predictions.db")


class TestPredictionLogger:
    def test_creates_db_file(self, tmp_path):
        db_path = tmp_path / "subdir" / "test.db"
        PredictionLogger(db_path=db_path)
        assert db_path.exists()

    def test_log_returns_request_id(self, logger):
        request_id = logger.log_prediction(
            model_name="test_model",
            model_version="1.0",
            features_hash="abc123",
            score=0.75,
            predicted_label=1,
            latency_ms=5.0,
        )
        assert isinstance(request_id, str)
        assert len(request_id) == 36  # UUID4 format

    def test_count_increments(self, logger):
        assert logger.count() == 0
        logger.log_prediction(
            model_name="m", model_version="1", features_hash="h",
            score=0.5, predicted_label=0, latency_ms=1.0,
        )
        assert logger.count() == 1

    def test_get_predictions_returns_dicts(self, logger):
        logger.log_prediction(
            model_name="m", model_version="1", features_hash="h",
            score=0.5, predicted_label=0, latency_ms=1.0,
        )
        preds = logger.get_predictions(limit=10)
        assert len(preds) == 1
        assert isinstance(preds[0], dict)
        assert "score" in preds[0]
        assert "model_name" in preds[0]

    def test_get_predictions_filter_by_model(self, logger):
        logger.log_prediction(
            model_name="xgboost", model_version="1", features_hash="h",
            score=0.8, predicted_label=1, latency_ms=1.0,
        )
        logger.log_prediction(
            model_name="lr", model_version="1", features_hash="h",
            score=0.3, predicted_label=0, latency_ms=1.0,
        )
        xgb_preds = logger.get_predictions(model_name="xgboost")
        assert len(xgb_preds) == 1
        assert xgb_preds[0]["model_name"] == "xgboost"

    def test_get_scores_returns_tuples(self, logger):
        logger.log_prediction(
            model_name="m", model_version="1", features_hash="h",
            score=0.5, predicted_label=0, latency_ms=1.0,
        )
        scores = logger.get_scores(limit=10)
        assert len(scores) == 1

    def test_log_with_true_label(self, logger):
        logger.log_prediction(
            model_name="m", model_version="1", features_hash="h",
            score=0.9, predicted_label=1, true_label=1, latency_ms=1.0,
        )
        preds = logger.get_predictions()
        assert preds[0]["true_label"] == 1

    def test_log_error_status(self, logger):
        logger.log_prediction(
            model_name="m", model_version="1", features_hash="error",
            score=0.0, predicted_label=-1, latency_ms=1.0,
            status="error: something failed",
        )
        preds = logger.get_predictions()
        assert "error" in preds[0]["status"]
