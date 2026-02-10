import sys
import json
import pickle
import time
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ARTIFACTS_DIR, LOGS_DIR, MONITORING_DIR, CHURN_CLASS
from data.preprocessing import compute_features_hash
from logging.prediction_logger import PredictionLogger

app = FastAPI(
    title="Baubap Challenge Predictor API",
    version="1.0.0",
    description="Predict user churn with logged predictions and monitoring metrics.",
)

# Singleton pattern for model artifacts
_model = None
_preprocessor = None
_metadata = None
_logger = None


def get_artifacts():
    """Lazy-load model artifacts on first request."""
    global _model, _preprocessor, _metadata, _logger
    if _model is None:
        with open(ARTIFACTS_DIR / "best_model.pkl", "rb") as f:
            _model = pickle.load(f)
        with open(ARTIFACTS_DIR / "preprocessor.pkl", "rb") as f:
            _preprocessor = pickle.load(f)
        with open(ARTIFACTS_DIR / "model_metadata.json", "r") as f:
            _metadata = json.load(f)
        _logger = PredictionLogger()
    return _model, _preprocessor, _metadata, _logger


# Pydantic Models
class PredictionRequest(BaseModel):
    esent: int = Field(..., description="Number of emails sent")
    eopenrate: float = Field(..., description="Email open rate (%)")
    eclickrate: float = Field(..., description="Email click rate (%)")
    avgorder: float = Field(..., description="Average order value")
    ordfreq: float = Field(..., description="Order frequency")
    paperless: int = Field(..., description="Paperless billing (0/1)")
    refill: int = Field(..., description="Auto-refill enabled (0/1)")
    doorstep: int = Field(..., description="Doorstep delivery (0/1)")
    favday: str = Field(..., description="Favorite day of the week")
    city: str = Field(..., description="City code (DEL, BOM, MAA, BLR)")
    tenure_days: float = Field(0, description="Days since account creation")
    days_first_to_last_order: float = Field(0)
    days_since_first_order: float = Field(0)
    order_recency_ratio: float = Field(0, ge=0, le=1)
    true_label: Optional[int] = Field(None, description="Ground truth if available")


class PredictionResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    churn_score: float
    predicted_label: int
    threshold: float


# Endpoints
@app.get("/health")
def health_check():
    """Smoke test: can we load the model and respond?"""
    try:
        get_artifacts()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)},
        )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Generate a churn prediction for a single user.

    Flow: validate input → preprocess → predict → log → respond.
    Total latency target: < 50ms (dominated by model inference, not I/O).
    """
    model, preprocessor, metadata, logger = get_artifacts()
    threshold = metadata["threshold"] # type: ignore
    start = time.perf_counter()

    try:
        # Build DataFrame matching the preprocessor's expected columns
        features = request.model_dump(exclude={"true_label"})
        df = pd.DataFrame([features])

        # Preprocess (same pipeline as training)
        X_processed = preprocessor.transform(df) # type: ignore

        # Predict: P(churn) = P(retained=0) = predict_proba[:, 0]
        churn_score = float(model.predict_proba(X_processed)[0, 0])
        predicted_label = int(churn_score >= threshold)

        latency_ms = (time.perf_counter() - start) * 1000

        # Log prediction for monitoring
        features_hash = compute_features_hash(features)
        request_id = logger.log_prediction( # type: ignore
            model_name=metadata["best_model"], # type: ignore
            model_version=metadata["model_version"], # type: ignore
            features_hash=features_hash,
            score=churn_score,
            predicted_label=predicted_label,
            true_label=request.true_label,
            latency_ms=latency_ms,
            status="success",
        )

        return PredictionResponse(
            request_id=request_id,
            model_name=metadata["best_model"], # type: ignore
            model_version=metadata["model_version"], # type: ignore
            churn_score=round(churn_score, 6),
            predicted_label=predicted_label,
            threshold=threshold,
        )

    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.log_prediction( # type: ignore
            model_name=metadata.get("best_model", "unknown"), # type: ignore
            model_version=metadata.get("model_version", "0.0.0"), # type: ignore
            features_hash="error",
            score=0.0,
            predicted_label=-1,
            latency_ms=latency_ms,
            status=f"error: {str(e)}",
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    """
    Return service metrics and latest drift report.

    This endpoint serves both operational monitoring (latency, error rate)
    and ML monitoring (drift report) in a single call.
    """
    _, _, _, logger = get_artifacts()

    predictions = logger.get_predictions(limit=1000) # type: ignore
    if not predictions:
        return {"message": "No predictions logged yet"}

    latencies = [p["latency_ms"] for p in predictions if p["status"] == "success"]
    scores = [p["score"] for p in predictions if p["status"] == "success"]
    error_count = sum(1 for p in predictions if p["status"] != "success")

    service = {
        "total_predictions": len(predictions),
        "error_count": error_count,
        "error_rate": error_count / len(predictions) if predictions else 0,
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else 0,
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0,
        "avg_churn_score": round(float(np.mean(scores)), 4) if scores else 0,
        "predicted_positive_rate": round(
            sum(1 for p in predictions if p["predicted_label"] == 1) / len(predictions), 4
        ) if predictions else 0,
    }

    # Attach latest drift report if available
    drift_report = None
    if MONITORING_DIR.exists():
        json_reports = sorted(MONITORING_DIR.glob("drift_report_*.json"))
        if json_reports:
            with open(json_reports[-1]) as f:
                drift_report = json.load(f)

    return {"service_metrics": service, "latest_drift_report": drift_report}
