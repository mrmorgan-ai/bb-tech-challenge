.PHONY: setup train evaluate infer monitor api clean test all

# Configuration
PYTHON = python
DATA_PATH = data/raw/storedata_total.xlsx
MLFLOW_URI = mlruns

# Setup
setup:
	uv sync
	$(PYTHON) -c "import os; [os.makedirs(d, exist_ok=True) for d in ['logs', 'artifacts', 'monitoring/reports', 'data']]"
	@echo "Setup complete"

# Data & Training
train:
	$(PYTHON) -m src.models.train --data-path $(DATA_PATH)
	@echo "Training complete — check MLflow UI with: mlflow ui"

# Evaluation
evaluate:
	$(PYTHON) -m src.models.evaluate --data-path $(DATA_PATH)
	@echo "Evaluation complete — reports in artifacts/"

# Inference + Logging
infer:
	$(PYTHON) -m src.prediction_logging.batch_inference --data-path $(DATA_PATH) --n-samples 200
	@echo "Inference complete — predictions logged to logs/predictions.db"

# Monitoring
monitor:
	$(PYTHON) -m src.monitoring.drift_monitor --data-path $(DATA_PATH)
	@echo "Monitoring complete — report in monitoring/reports/"

# API
api:
	$(PYTHON) -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Testing
test:
	$(PYTHON) -m pytest tests/ -v

# Cleanup
clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('logs/predictions.db'), pathlib.Path('mlruns')]]"
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
	@echo "Cleaned"

# Full Pipeline
all: setup train evaluate infer monitor
	@echo "Full pipeline complete"
