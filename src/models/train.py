import json
import pickle
import argparse

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)
from xgboost import XGBClassifier

from src.config import (
    MLFLOW_EXPERIMENT, ARTIFACTS_DIR, RANDOM_STATE, CHURN_CLASS,
)
from src.data.preprocessing import prepare_data

def get_models() -> dict:
    """
    Define the three model families with their hyperparameters.

    Return:
    - dict of {name: (model_instance, params_to_log)}.
    """
    lr = LogisticRegression(
        C=1.0,                      # Moderate regularization
        solver="lbfgs",             # Quasi-Newton method, efficient for small-medium datasets. Supports L2 regularization
        class_weight="balanced",    # Automatically adjusts weights for imbalance
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=3.87,      # scale_pos_weight: ratio of negative to positive class ≈ 79.5/20.5 ≈ 3.87. Pay more attention to them during training.        subsample=0.8,
        subsample=0.8,              # subsample=0.8: each tree sees 80% of rows
        colsample_bytree=0.8,       # colsample_bytree=0.8: each tree sees 80% of features. This adds randomness that decorrelates trees and reduces overfitting
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )

    # MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate="adaptive", # learning_rate='adaptive': reduces the learning rate when training loss
        early_stopping=True,
        validation_fraction=0.1,
        max_iter=500,
        random_state=RANDOM_STATE,
    )

    return {
        "logistic_regression": (
            lr,
            {"C": 1.0, "solver": "lbfgs", "class_weight": "balanced"},
        ),
        "xgboost": (
            xgb,
            {
                "n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
                "scale_pos_weight": 3.87, "subsample": 0.8,
            },
        ),
        "mlp": (
            mlp,
            {
                "hidden_layers": "128-64-32", "activation": "relu",
                "alpha": 0.001, "learning_rate": "adaptive",
            },
        ),
    }


def evaluate_model(model, X, y_true) -> dict:
    """
    Compute all evaluation metrics for a fitted model
    """
    # P(churn) = P(retained=0) = first column of predict_proba
    y_churn_prob = model.predict_proba(X)[:, 0]

    # Convert target to churn class
    y_churn_true = (y_true == CHURN_CLASS).astype(int)

    # roc aun metric
    roc_auc = roc_auc_score(y_churn_true, y_churn_prob)

    # pr auc metric PRINCIPAL
    pr_auc = average_precision_score(y_churn_true, y_churn_prob)

    # F1 with 0.5 threshold
    y_pred_churn = (y_churn_prob >= 0.5).astype(int)
    f1 = f1_score(y_churn_true, y_pred_churn)

    # Precision@10% 
    k = int(len(y_churn_true) * 0.10)
    
    # Take last k indices = top k scores
    top_k_idx = np.argsort(y_churn_prob)[-k:]
    precision_at_10 = y_churn_true.values[top_k_idx].mean()

    # Lift@10%
    base_rate = y_churn_true.mean()
    lift_at_10 = precision_at_10 / base_rate if base_rate > 0 else 0

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision_at_10pct": precision_at_10,
        "lift_at_10pct": lift_at_10,
        "base_churn_rate": base_rate,
    }


def find_optimal_threshold(model, X, y_true) -> tuple:
    """
    Find the threshold that maximizes F1 on the churn class in validation set.
    """
    y_churn_prob = model.predict_proba(X)[:, 0]
    y_churn_true = (y_true == CHURN_CLASS).astype(int)

    precisions, recalls, thresholds = precision_recall_curve(
        y_churn_true, y_churn_prob
    )

    # Compute F1 for each threshold
    # precision_recall_curve returns n+1 precisions/recalls but n thresholds
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)

    # Guard: best_idx might equal len(thresholds) due to the n+1 issue
    best_threshold = thresholds[min(best_idx, len(thresholds) - 1)]

    return float(best_threshold), float(f1_scores[best_idx])


def train_all(data_path: str) -> tuple:
    """
    Train all models, log to MLflow, save best model and preprocessor.

    PIPELINE:
    1. Load and prepare data
    2. Fit preprocessor on TRAINING DATA ONLY (prevents data leakage)
    3. Transform all splits using the fitted preprocessor
    4. Train each model family
    5. Evaluate on validation (for selection) and test (for reporting)
    6. Find optimal threshold on validation
    7. Log everything to MLflow
    8. Save best model + preprocessor + metadata to artifacts/
    """
    # Step 1: Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, ref_date = prepare_data(
        data_path
    )

    # Step 2: Fit preprocessor on training data only
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Step 3: Save preprocessor
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    # Step 4: MLflow experiment setup
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    models = get_models()
    results = {}
    best_model_name = None
    best_pr_auc = -1

    for name, (model, params) in models.items():
        print(f"Training: {name}")
        with mlflow.start_run(run_name=name):
            # Log hyperparameters
            mlflow.log_params(params)
            mlflow.log_param("model_family", name)

            # Train
            model.fit(X_train_processed, y_train)

            # Evaluate on validation (for model selection)
            val_metrics = evaluate_model(model, X_val_processed, y_val)
            for metric_name, value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", value)
                print(f"val_{metric_name}: {value:.4f}")

            # Evaluate on test (for unbiased reporting)
            test_metrics = evaluate_model(model, X_test_processed, y_test)
            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)

            # Optimize threshold on validation
            opt_threshold, opt_f1 = find_optimal_threshold(
                model, X_val_processed, y_val
            )
            mlflow.log_metric("optimal_threshold", opt_threshold)
            mlflow.log_metric("optimal_f1", opt_f1)
            print(f"  optimal_threshold: {opt_threshold:.4f} (F1={opt_f1:.4f})")

            # Log model artifact to MLflow
            mlflow.sklearn.log_model(model, name="model") # type: ignore

            # Track results
            results[name] = {
                "model": model,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "threshold": opt_threshold,
            }

            # Track best by PR-AUC (our primary metric)
            if val_metrics["pr_auc"] > best_pr_auc:
                best_pr_auc = val_metrics["pr_auc"]
                best_model_name = name

    # Step 5: Save best model and metadata
    best = results[best_model_name]
    print(f"\n{'=' * 60}")
    print(f"Best model: {best_model_name} (val PR-AUC={best_pr_auc:.4f})")
    print(f"Optimal threshold: {best['threshold']:.4f}")
    print(f"{'=' * 60}")

    with open(ARTIFACTS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best["model"], f)

    metadata = {
        "best_model": best_model_name,
        "model_version": "1.0.0",
        "ref_date": str(ref_date),
        "threshold": best["threshold"],
        "val_metrics": {k: float(v) for k, v in best["val_metrics"].items()},
        "test_metrics": {k: float(v) for k, v in best["test_metrics"].items()},
    }
    with open(ARTIFACTS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return results, best_model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/raw/storedata_total.xlsx")
    args = parser.parse_args()
    train_all(args.data_path)
