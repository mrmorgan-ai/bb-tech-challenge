import json
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report,
    ConfusionMatrixDisplay,
)

from src.config import ARTIFACTS_DIR, CHURN_CLASS
from src.data.preprocessing import prepare_data


def load_artifacts():
    """Load saved model, preprocessor, and metadata."""
    with open(ARTIFACTS_DIR / "best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ARTIFACTS_DIR / "preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open(ARTIFACTS_DIR / "model_metadata.json", "r") as f:
        metadata = json.load(f)
    return model, preprocessor, metadata


def plot_calibration(y_true, y_prob, model_name, save_path):
    """
    Plot calibration curve and score distribution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fraction_pos, mean_predicted = calibration_curve(y_true, y_prob, n_bins=10)
    axes[0].plot(mean_predicted, fraction_pos, "s-", label=model_name)
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Fraction of positives")
    axes[0].set_title("Calibration Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(y_prob[y_true == 1], bins=30, alpha=0.6, label="Churned", density=True)
    axes[1].hist(y_prob[y_true == 0], bins=30, alpha=0.6, label="Retained", density=True)
    axes[1].set_xlabel("Predicted churn probability")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Score Distribution by Class")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_lift_chart(y_true, y_prob, save_path):
    """
    Plot cumulative lift and gains charts.

    LIFT CHART answers: "How many times better than random is the model
    at finding churners when I only look at the top K%?"

    CUMULATIVE GAINS answers: "To capture X% of all churners, what fraction
    of users do I need to score and intervene on?"
    - The gap between model curve and random diagonal = model's value.
    - A perfect model would go straight to 100% capture immediately.
    """
    order = np.argsort(y_prob)[::-1]  # Sort by score descending
    y_sorted = np.array(y_true)[order]

    n = len(y_sorted)
    base_rate = y_sorted.mean()
    percentiles = np.arange(1, n + 1) / n
    cumulative_precision = np.cumsum(y_sorted) / np.arange(1, n + 1)
    lift = cumulative_precision / base_rate

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(percentiles * 100, lift)
    axes[0].axhline(y=1.0, color="r", linestyle="--", label="Random (lift=1)")
    axes[0].set_xlabel("Top K% of predictions")
    axes[0].set_ylabel("Lift")
    axes[0].set_title("Cumulative Lift Chart")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    cumulative_gains = np.cumsum(y_sorted) / y_sorted.sum()
    axes[1].plot(percentiles * 100, cumulative_gains * 100, label="Model")
    axes[1].plot([0, 100], [0, 100], "r--", label="Random")
    axes[1].set_xlabel("Top K% of predictions")
    axes[1].set_ylabel("% of churned users captured")
    axes[1].set_title("Cumulative Gains Chart")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_threshold_analysis(y_true, y_prob, optimal_threshold, save_path):
    """
    Plot precision, recall, and F1 as functions of threshold.

    This visualization shows the trade-off that threshold selection navigates:
    - Lower threshold → higher recall (catch more churners) but lower precision
      (more false alarms)
    - Higher threshold → higher precision (fewer false alarms) but lower recall
      (miss more churners)
    - F1 peaks at the threshold where precision and recall are balanced.
    - The optimal threshold (red line) maximizes F1 on the validation set.
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, label="Precision")
    ax.plot(thresholds, recalls, label="Recall")
    ax.plot(thresholds, f1s, label="F1")
    ax.axvline(
        x=optimal_threshold, color="red", linestyle="--",
        label=f"Optimal={optimal_threshold:.3f}",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Analysis (Churn Class)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate(data_path: str):
    """Run full evaluation pipeline and generate all plots."""
    model, preprocessor, metadata = load_artifacts()
    _, _, X_test, _, _, y_test, _ = prepare_data(data_path)

    X_test_processed = preprocessor.transform(X_test)

    # Get churn probabilities (class 0 = churn)
    y_churn_prob = model.predict_proba(X_test_processed)[:, 0]
    y_churn_true = (y_test == CHURN_CLASS).astype(int)

    threshold = metadata["threshold"]
    y_pred_churn = (y_churn_prob >= threshold).astype(int)

    # Print classification report
    print(f"Evaluation: {metadata['best_model']} v{metadata['model_version']}")
    print(f"Threshold: {threshold:.4f}")
    print(classification_report(
        y_churn_true, y_pred_churn,
        target_names=["Retained", "Churned"],
    ))

    # Generate all plots
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ROC curve
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_churn_true, y_churn_prob, ax=ax, name=metadata["best_model"]
    )
    ax.set_title("ROC Curve (Churn Detection)")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "roc_curve.png", dpi=150)
    plt.close()

    # Precision-Recall curve
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(
        y_churn_true, y_churn_prob, ax=ax, name=metadata["best_model"]
    )
    ax.set_title("Precision-Recall Curve (Churn Detection)")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "pr_curve.png", dpi=150)
    plt.close()

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_churn_true, y_pred_churn,
        display_labels=["Retained", "Churned"], ax=ax, cmap="Blues",
    )
    ax.set_title(f"Confusion Matrix (threshold={threshold:.3f})")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # Calibration + score distribution
    plot_calibration(
        y_churn_true, y_churn_prob, metadata["best_model"],
        ARTIFACTS_DIR / "calibration.png",
    )

    # Lift chart
    plot_lift_chart(
        y_churn_true, y_churn_prob,
        ARTIFACTS_DIR / "lift_chart.png",
    )

    # Threshold analysis
    plot_threshold_analysis(
        y_churn_true, y_churn_prob, threshold,
        ARTIFACTS_DIR / "threshold_analysis.png",
    )

    print(f"\nPlots saved to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/raw/storedata_total.xlsx")
    args = parser.parse_args()
    evaluate(args.data_path)
