import json
import pickle
import time
import argparse

import numpy as np

from src.config import ARTIFACTS_DIR, CHURN_CLASS
from src.data.preprocessing import prepare_data, compute_features_hash
from src.prediction_logging.prediction_logger import PredictionLogger


def run_batch_inference(data_path: str, n_samples: int = 200):
    """Run batch inference on test data, logging each prediction."""
    # Load artifacts
    with open(ARTIFACTS_DIR / "best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ARTIFACTS_DIR / "preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open(ARTIFACTS_DIR / "model_metadata.json", "r") as f:
        metadata = json.load(f)

    model_name = metadata["best_model"]
    model_version = metadata["model_version"]
    threshold = metadata["threshold"]

    # Prepare data — use test set to simulate "new" production data
    _, _, X_test, _, _, y_test, _ = prepare_data(data_path)

    # Sample n rows (deterministic for reproducibility)
    if n_samples < len(X_test):
        sample_idx = np.random.RandomState(42).choice(
            len(X_test), n_samples, replace=False
        )
        X_sample = X_test.iloc[sample_idx]
        y_sample = y_test.iloc[sample_idx]
    else:
        X_sample = X_test
        y_sample = y_test

    # Transform features using the saved preprocessor
    X_processed = preprocessor.transform(X_sample)

    logger = PredictionLogger()
    logged_count = 0

    print(f"Running batch inference on {len(X_sample)} samples...")

    for i in range(len(X_sample)):
        start = time.perf_counter()

        try:
            # Score single sample
            x_single = X_processed[i:i + 1]
            churn_score = float(model.predict_proba(x_single)[0, 0])  # P(churn)
            predicted_label = int(churn_score >= threshold)

            latency_ms = (time.perf_counter() - start) * 1000

            # Feature hash for traceability
            features_dict = X_sample.iloc[i].to_dict()
            features_hash = compute_features_hash(features_dict)

            # True label: churned = retained==0
            true_label = int(y_sample.iloc[i] == CHURN_CLASS)

            # Log to SQLite
            logger.log_prediction(
                model_name=model_name,
                model_version=model_version,
                features_hash=features_hash,
                score=churn_score,
                predicted_label=predicted_label,
                true_label=true_label,
                latency_ms=latency_ms,
                status="success",
            )
            logged_count += 1

        except Exception as e:
            # Log the error but don't crash the batch
            latency_ms = (time.perf_counter() - start) * 1000
            logger.log_prediction(
                model_name=model_name,
                model_version=model_version,
                features_hash="error",
                score=0.0,
                predicted_label=-1,
                true_label=None,
                latency_ms=latency_ms,
                status=f"error: {str(e)}",
            )

    total = logger.count()
    print(f"\nLogged {logged_count} predictions (total in DB: {total})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/raw/storedata_total.xlsx")
    parser.add_argument("--n-samples", type=int, default=200)
    args = parser.parse_args()
    run_batch_inference(args.data_path, args.n_samples)
