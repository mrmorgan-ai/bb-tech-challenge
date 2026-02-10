from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent # src -> project root
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"
MONITORING_DIR = PROJECT_ROOT / "src" / "monitoring" / "reports"

# Key dataset columns
TARGET_COL = "retained"
ID_COL = "custid"

# retained=0 user churned
CHURN_CLASS = 0

# Column grouped by type
DATE_COLS = ["created", "firstorder", "lastorder"]
NUMERIC_COLS = ["esent", "eopenrate", "eclickrate", "avgorder", "ordfreq"]
BINARY_COLS = ["paperless", "refill", "doorstep"]
CATEGORICAL_COLS = ["favday", "city"]

# Features created during engineering and derived from DATE_COLS
ENGINEERED_COLS = [
    "tenure_days",                # How long the user has been a customer
    "days_first_to_last_order",   # Engagement window duration
    "days_since_first_order",     # Time since first purchase
    "order_recency_ratio",        # How recent is last activity (0=stale, 1=recent)
]


# Splitting values
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# mlflow experiment name 
MLFLOW_EXPERIMENT = "baubap-mlops-challenge"

# Predictions 
PREDICTIONS_DB = LOGS_DIR / "predictions.db"
DATA_SCHEMA_VERSION = "1.0.0"

# Monitoring thesholds
# PSI - Empirical significance
PSI_WARNING = 0.1
PSI_ALERT = 0.2

# Statistical significance for KS and Chi-square tests
KS_PVALUE_THRESHOLD = 0.05
CHI2_PVALUE_THRESHOLD = 0.05

# Drift condig
PREDICTION_RATE_DRIFT_THRESHOLD = 0.05
