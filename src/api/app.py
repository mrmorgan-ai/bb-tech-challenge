import sys
import json
import time
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ARTIFACTS_DIR, MONITORING_DIR, CHURN_CLASS
from data.preprocessing import engineer_features, compute_features_hash
from prediction_logging.prediction_logger import PredictionLogger
from utils.artifacts import load_all_artifacts, get_churn_probability


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup, clean up on shutdown."""
    model, preprocessor, metadata = load_all_artifacts(ARTIFACTS_DIR)
    app.state.model = model
    app.state.preprocessor = preprocessor
    app.state.metadata = metadata
    app.state.logger = PredictionLogger()
    yield


app = FastAPI(
    title="Baubap Challenge Predictor API",
    version="1.0.0",
    description="Predict user churn with logged predictions and monitoring metrics.",
    lifespan=lifespan,
)


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
    created: str = Field(..., description="Account creation date (ISO format)")
    firstorder: str = Field(..., description="First order date (ISO format)")
    lastorder: str = Field(..., description="Last order date (ISO format)")
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
def health_check(request: Request):
    """Smoke test: can we load the model and respond?"""
    try:
        _ = request.app.state.model
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)},
        )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: Request, payload: PredictionRequest):
    """
    Generate a churn prediction for a single user.

    Flow: validate input -> engineer features -> preprocess -> predict -> log -> respond.
    Total latency target: < 50ms (dominated by model inference, not I/O).
    """
    model = request.app.state.model
    preprocessor = request.app.state.preprocessor
    metadata = request.app.state.metadata
    logger = request.app.state.logger

    threshold = metadata["threshold"]
    ref_date = pd.Timestamp(metadata["ref_date"])
    start = time.perf_counter()

    try:
        # Build DataFrame with raw features + date columns
        features = payload.model_dump(exclude={"true_label"})
        df = pd.DataFrame([features])

        # Parse dates and compute engineered features server-side
        for col in ["created", "firstorder", "lastorder"]:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        df, _ = engineer_features(df, ref_date=ref_date)

        # Preprocess (same pipeline as training)
        X_processed = preprocessor.transform(df)

        # Predict via classes_ lookup (not hardcoded index)
        churn_score = float(get_churn_probability(model, X_processed, CHURN_CLASS)[0])
        predicted_label = int(churn_score >= threshold)

        latency_ms = (time.perf_counter() - start) * 1000

        # Log prediction for monitoring
        features_hash = compute_features_hash(features)
        request_id = logger.log_prediction(
            model_name=metadata["best_model"],
            model_version=metadata["model_version"],
            features_hash=features_hash,
            score=churn_score,
            predicted_label=predicted_label,
            true_label=payload.true_label,
            latency_ms=latency_ms,
            status="success",
        )

        return PredictionResponse(
            request_id=request_id,
            model_name=metadata["best_model"],
            model_version=metadata["model_version"],
            churn_score=round(churn_score, 6),
            predicted_label=predicted_label,
            threshold=threshold,
        )

    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.log_prediction(
            model_name=metadata.get("best_model", "unknown"),
            model_version=metadata.get("model_version", "0.0.0"),
            features_hash="error",
            score=0.0,
            predicted_label=-1,
            latency_ms=latency_ms,
            status=f"error: {str(e)}",
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics(request: Request):
    """
    Return service metrics and latest drift report.

    This endpoint serves both operational monitoring (latency, error rate)
    and ML monitoring (drift report) in a single call.
    """
    logger = request.app.state.logger

    predictions = logger.get_predictions(limit=1000)
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
