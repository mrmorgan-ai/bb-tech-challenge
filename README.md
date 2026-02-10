# bb-tech-challenge
## EDA Conclusions

### Data Quality
- 30,781 valid records after dropping 20 rows with correlated null identifiers
- No duplicate customer IDs detected after that
- All numeric features have valid ranges, no significant outliers or data entry error evidence.

### Target
- Moderate imbalance: 20.5% churn / 79.5% retained (4:1 ratio)
- Handled via class weights, not resampling. Appropriate for this level of imbalance
- Primary metric: PR-AUC. Appropriate for this level of imbalance

### Features
- Strongest raw predictors identify base on Cohen's d analysis over 0.8
- Weak/noise features: identify base on Cohen's d analysis under 0.5
- Zero-inflation detected in: [esent, eopenrate, eclickrate, ordfreq]

### Engineered Features
- `order_recency_ratio` is the strongest engineered feature
- Potential leakage concern: 
  - If retained=0 was labeled based on 'no order in X days', then order_recency_ratio is essentially restating the label definitionthat's leakage. 
  - If retained=0 was labeled by explicit account cancellation, then order_recency_ratio is a legitimate leading indicator. I'll choose last option in this challenge.
- `tenure_days` and `days_since_first_order` are correlated but retained because tree models handle this naturally.

### Leakage Assessment
- `lastorder`derived features partially encode the target — documented and flagged for reviewer discussion
- I will retain these features because they represent information
  available at prediction time in production

### Splitting Strategy
- Stratified random split (70/15/15) — appropriate because features are
  aggregated snapshots, not time-series events
- Temporal split not required: no future-information leakage through features

## Logging and Monitoring
### Known Limitations
- **Prediction store**: SQLite is used for prediction logging for simplicity and portability. In a production deployment with concurrent API instances, replace with PostgreSQL or a managed time-series database to avoid `SQLITE_BUSY` contention under concurrent writes.
- **Drift monitoring**: Currently compares training vs. test splits from the same dataset. In production, the "current" distribution should come from actual production traffic or a new data batch.
- **Drift monitoring data source**: The drift monitor currently compares train vs. test splits from the same dataset — drift will never be detected since both share the same distribution by construction. In production, save training set reference data as `artifacts/reference_data.parquet` during training, and accept a `--current-data-path` argument pointing to genuinely new production data. The statistical tests (PSI, KS, Chi-square) are correct; only the data source needs to change.
- **Calibration**: The evaluation module plots a calibration curve, but no calibration correction is applied. XGBoost scores are useful for ranking but unreliable as true probability estimates. In production, wrap the best model in `CalibratedClassifierCV(cv="prefit", method="isotonic")` fitted on the validation set. The calibrated model replaces the raw one in `best_model.pkl`, so all downstream consumers (API, batch inference) automatically get well-calibrated probabilities.
- **Cross-validation**: Model selection currently relies on a single stratified split. In production, add 5-fold stratified CV on train+val after model selection to validate stability — log `cv_pr_auc_mean` and `cv_pr_auc_std` to MLflow. If std > 0.05, flag the result as potentially unstable. This doesn't replace the held-out test set for final reporting, it's a complementary sanity check.
- **Threshold-metric consistency**: `evaluate_model()` currently reports F1 at a fixed 0.5 threshold, while the actual operating threshold is optimized separately (~0.202). In production, remove F1@0.5 from model comparison metrics (it's misleading for imbalanced problems) and instead compute `f1_at_optimal_threshold` after threshold optimization. Threshold-independent metrics (PR-AUC, ROC-AUC, Precision@10%, Lift@10%) are unaffected and remain the primary basis for model selection.
- **Structured logging**: The codebase uses `print()` for operational output. In production, replace with Python's `logging` module (`logger = logging.getLogger(__name__)`) with configurable log levels, timestamps, and module names. This enables routing logs to files/monitoring systems, silencing verbose output, and filtering by severity (DEBUG/INFO/WARNING/ERROR).
- **Hyperparameter tuning**: Current hyperparameters are manually selected based on domain heuristics (e.g., `scale_pos_weight=3.87` from the class ratio). In production, implement Bayesian optimization via Optuna with a `--tune` flag to search over parameter spaces (XGBoost: n_estimators, max_depth, learning_rate, subsample, colsample_bytree; LR: C; MLP: hidden_layer_sizes, alpha), optimizing PR-AUC on the validation set.
- **CI/CD pipeline**: The current CI (`.github/workflows/ci.yml`) runs lint + tests on push/PR — sufficient for a dev challenge. In production, a full MLOps CI/CD pipeline would include:
  1. **PR gate**: lint (ruff) + unit tests + data validation checks on every PR.
  2. **Training pipeline**: Triggered on data or code changes — runs `train → evaluate → register model` as a DAG (e.g., GitHub Actions, Airflow, or Vertex AI Pipelines). Artifacts versioned with DVC or MLflow Model Registry.
  3. **Model validation gate**: Automated checks before promotion — compare new model PR-AUC vs. production baseline, run A/B shadow scoring, validate calibration, check for data drift in training data.
  4. **Staging deploy**: Auto-deploy to staging on model registry promotion. Run integration tests + load tests against the staging API.
  5. **Production deploy**: Blue/green or canary deployment with automated rollback if error rate or latency SLOs are breached. Container image built from `Dockerfile`, pushed to registry (ECR/GCR), deployed via Kubernetes or ECS.
  6. **Post-deploy monitoring**: Scheduled drift detection job (daily/weekly). Alerting pipeline: drift alert → Slack/PagerDuty → automatic retraining trigger if PSI > threshold for N consecutive days.
