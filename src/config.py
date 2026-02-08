from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent # src -> project root
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"
MONITORING_DIR = PROJECT_ROOT / "monitoring" / "reports"

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

# Splitting values
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# mlflow experiment name 
MLFLOW_EXPERIMENT = "user-retention-challenge"

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
