import sys
import json
import pickle
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ARTIFACTS_DIR, LOGS_DIR, MONITORING_DIR,
    NUMERIC_COLS, CATEGORICAL_COLS, ENGINEERED_COLS,
    PSI_WARNING, PSI_ALERT, KS_PVALUE_THRESHOLD, CHI2_PVALUE_THRESHOLD,
    PREDICTION_RATE_DRIFT_THRESHOLD, CHURN_CLASS
)
from data.preprocessing import prepare_data


# Calculate PSI
def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index for a single numeric feature.
    """
    # Handle edge cases
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    
    # Create bins
    _, bin_edges = np.histogram(reference, bins=n_bins)
    
    # Extend edges to handle values outside reference range
    bin_edges[0] = -np.inf # All low values
    bin_edges[-1] = np.inf # All high values
    
    # Count values in each bin
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    curr_counts, _ = np.histogram(current, bins=bin_edges)
    
    # Convert percentages
    ref_pct = ref_counts / len(reference)
    curr_pct = curr_counts / len(current)
    
    # Avoid 0 division
    ref_pct = np.maximum(ref_pct,0.0001)
    curr_pct = np.maximum(curr_pct, 0.0001)
    
    # Calculate psi
    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct/ref_pct))   
    
    return psi


def psi_alert_level(psi_value: float) -> str:
    """Map PSI value to alert level."""
    if psi_value >= PSI_ALERT:
        return "X ALERT"
    elif psi_value >= PSI_WARNING:
        return "WARNING!!"
    return "OK"


# Calculate KS Test
def compute_ks_test(reference: np.ndarray, current: np.ndarray) -> dict:
    """
    Two-sample Kolmogorov-Smirnov test.
    """
    stat, pvalue = stats.ks_2samp(reference, current)
    return {
        "ks_statistics": float(stat), # type: ignore
        "p_value": float(pvalue), # type: ignore
        "drift_detected": bool(pvalue < KS_PVALUE_THRESHOLD), # type: ignore
    }


# Calculate chi-square
def compute_chi2_test(reference: pd.Series, current: pd.Series) -> dict:
    """
    Chi-square test for categorical feature drift.
    """
    drift_report = {}
    
    # Get value counts
    ref_counts = reference.value_counts()
    curr_counts = current.value_counts()
    
    # Handle new/missing categories
    all_categories = set(ref_counts.index) | set(curr_counts.index)
    ref_counts = ref_counts.reindex(all_categories, fill_value=0) # type: ignore
    curr_counts = curr_counts.reindex(all_categories, fill_value=0) # type: ignore

    # Chi-Square Test
    try:
        # Expect counts = references proportions x current total
        # References proportions = values / values.sum()
        expected = ref_counts.values * (curr_counts.sum() / ref_counts.sum()) # type: ignore
        
        chi2_stat, p_value = stats.chisquare(
            f_obs = curr_counts.values,
            f_exp = expected
        )
    except Exception as e:
        print(f"Chi2 test failed: {e}")
        chi2_stat, p_value = 0.0, 1.0
    
    drift_report = {
        'chi2_statistics': chi2_stat,
        'p_value': p_value,
        'drift_detected': p_value < CHI2_PVALUE_THRESHOLD
    }
    return drift_report


# Prediction drift
def compute_prediction_drift(
    ref_scores: np.ndarray, 
    cur_scores: np.ndarray,
    ref_rate: float, 
    cur_rate: float,
) -> dict:
    """
    Detect shifts in model output distribution.
    """
    psi = compute_psi(ref_scores, cur_scores)
    rate_drift = abs(cur_rate - ref_rate)

    return {
        "score_psi": psi,
        "score_psi_status": psi_alert_level(psi),
        "ref_positive_rate": float(ref_rate),
        "cur_positive_rate": float(cur_rate),
        "positive_rate_drift": float(rate_drift),
        "rate_drift_alert": rate_drift > PREDICTION_RATE_DRIFT_THRESHOLD
    }


# Service metrics
def compute_service_metrics(db_path: Path) -> dict:
    """
    Query prediction logs for operational health metrics.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql("SELECT * FROM predictions", conn)
    finally:
        conn.close()

    if df.empty:
        return {"total_predictions": 0, "error": "No predictions found"}

    total = len(df)
    errors = (df["status"] != "success").sum()
    latencies = df.loc[df["status"] == "success", "latency_ms"]

    return {
        "total_predictions": int(total),
        "error_count": int(errors),
        "error_rate": float(errors / total) if total > 0 else 0.0,
        "latency_p50_ms": float(latencies.quantile(0.50)) if len(latencies) > 0 else 0.0,
        "latency_p95_ms": float(latencies.quantile(0.95)) if len(latencies) > 0 else 0.0,
        "latency_p99_ms": float(latencies.quantile(0.99)) if len(latencies) > 0 else 0.0,
        "unique_model_versions": df["model_version"].nunique(),
    }


# Report Generation
def generate_report(
    data_drift: dict,
    prediction_drift: dict,
    service_metrics: dict, 
    output_dir: Path,
) -> Path:
    """Generate Markdown + JSON drift report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = output_dir / f"drift_report_{timestamp}.md"

    lines = [
        f"# Monitoring Report - {timestamp}",
        "",
        "## 1. Data Drift",
        "",
        "### Numeric Features (PSI + KS-test)",
        "",
        "| Feature | PSI | Status | KS Stat | KS p-value | KS Drift |",
        "|---------|-----|--------|---------|------------|----------|",
    ]

    for feat, metrics in data_drift.get("numeric", {}).items():
        psi_status = psi_alert_level(metrics["psi"])
        ks_drift = "Yes" if metrics["ks"]["drift_detected"] else "No"
        lines.append(
            f"| {feat} | {metrics['psi']:.4f} | {psi_status} | "
            f"{metrics['ks']['ks_statistics']:.4f} | {metrics['ks']['p_value']:.4f} | {ks_drift} |"
        )

    lines += [
        "",
        "### Categorical Features (Chi-Square)",
        "",
        "| Feature | Chi2 Stat | p-value | Drift Detected |",
        "|---------|-----------|---------|----------------|",
    ]

    for feat, metrics in data_drift.get("categorical", {}).items():
        drift = "Yes" if metrics["drift_detected"] else "No"
        lines.append(
            f"| {feat} | {metrics['chi2_statistics']:.4f} | {metrics['p_value']:.4f} | {drift} |"
        )

    lines += [
        "",
        "## 2. Prediction Drift",
        "",
        f"- **Score PSI**: {prediction_drift['score_psi']:.4f} ({prediction_drift['score_psi_status']})",
        f"- **Reference positive rate**: {prediction_drift['ref_positive_rate']:.4f}",
        f"- **Current positive rate**: {prediction_drift['cur_positive_rate']:.4f}",
        f"- **Rate drift**: {prediction_drift['positive_rate_drift']:.4f} "
        f"({'ALERT' if prediction_drift['rate_drift_alert'] else 'OK'})",
        "",
        "## 3. Service Metrics",
        "",
        f"- **Total predictions**: {service_metrics.get('total_predictions', 'N/A')}",
        f"- **Error rate**: {service_metrics.get('error_rate', 0):.2%}",
        f"- **Latency p50**: {service_metrics.get('latency_p50_ms', 0):.2f} ms",
        f"- **Latency p95**: {service_metrics.get('latency_p95_ms', 0):.2f} ms",
        f"- **Latency p99**: {service_metrics.get('latency_p99_ms', 0):.2f} ms",
        "",
        "## Alert Thresholds Reference",
        "",
        f"| Metric | Threshold | Meaning |",
        f"|--------|-----------|---------|",
        f"| PSI Warning | > {PSI_WARNING} | Moderate distribution shift |",
        f"| PSI Alert | > {PSI_ALERT} | Significant shift - investigate |",
        f"| KS p-value | < {KS_PVALUE_THRESHOLD} | Statistically significant difference |",
        f"| Chi2 p-value | < {CHI2_PVALUE_THRESHOLD} | Category proportions shifted |",
        f"| Prediction rate | > {PREDICTION_RATE_DRIFT_THRESHOLD:.0%} absolute | Model behavior changed |",
        "",
    ]

    report_path.write_text("\n".join(lines))
    print(f"Report saved to {report_path}")

    # JSON for programmatic access
    json_path = output_dir / f"drift_report_{timestamp}.json"
    json_path.write_text(json.dumps({
        "timestamp": timestamp,
        "data_drift": data_drift,
        "prediction_drift": prediction_drift,
        "service_metrics": service_metrics,
    }, indent=2, default=str))

    return report_path


def main(data_path: str):
    """
    Full monitoring pipeline.
    """
    print("Loading reference data (training set)...")
    X_train, X_val, X_test, y_train, y_val, y_test, _ = prepare_data(data_path)

    reference = X_train
    current = X_test

    # Data Drift
    print("Computing data drift...")
    numeric_features = NUMERIC_COLS + ENGINEERED_COLS
    data_drift = {"numeric": {}, "categorical": {}}

    for feat in numeric_features:
        if feat in reference.columns and feat in current.columns:
            ref_vals = reference[feat].dropna().values
            cur_vals = current[feat].dropna().values
            if len(ref_vals) > 0 and len(cur_vals) > 0:
                data_drift["numeric"][feat] = {
                    "psi": compute_psi(ref_vals, cur_vals),
                    "ks": compute_ks_test(ref_vals, cur_vals),
                }

    for feat in CATEGORICAL_COLS:
        if feat in reference.columns and feat in current.columns:
            data_drift["categorical"][feat] = compute_chi2_test(
                reference[feat].dropna(), current[feat].dropna(),
            )

    # Prediction Drift
    print("Computing prediction drift...")
    with open(ARTIFACTS_DIR / "best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ARTIFACTS_DIR / "preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open(ARTIFACTS_DIR / "model_metadata.json", "r") as f:
        metadata = json.load(f)

    threshold = metadata["threshold"]

    # Reference score with validation set
    X_val_proc = preprocessor.transform(X_val)
    ref_scores = model.predict_proba(X_val_proc)[:, 0]
    ref_labels = (ref_scores >= threshold).astype(int)

    # Current scores: test set (simulated production)
    X_test_proc = preprocessor.transform(X_test)
    cur_scores = model.predict_proba(X_test_proc)[:, 0]
    cur_labels = (cur_scores >= threshold).astype(int)

    prediction_drift = compute_prediction_drift(
        ref_scores, cur_scores,
        ref_rate=ref_labels.mean(),
        cur_rate=cur_labels.mean(),
    )

    # Service Metrics
    print("Computing service metrics...")
    db_path = LOGS_DIR / "predictions.db"
    if db_path.exists():
        service_metrics = compute_service_metrics(db_path)
    else:
        service_metrics = {"note": "No prediction logs found. Run 'make infer' first."}

    # Generate Report
    report_path = generate_report(
        data_drift, prediction_drift, service_metrics, MONITORING_DIR,
    )
    print("Monitoring complete.")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/raw/storedata_total.xlsx")
    args = parser.parse_args()
    main(args.data_path)
