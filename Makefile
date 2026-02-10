.PHONY: setup train evaluate infer monitor api clean test all

# Configuration
PYTHON = python3
DATA_PATH = data/raw/storedata_total.xlsx
MLFLOW_URI = mlruns

# Setup
setup:
	pip install -r requirements.txt
	mkdir -p logs artifacts monitoring/reports data
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
	$(PYTHON) -m src.logging.batch_inference --data-path $(DATA_PATH) --n-samples 200
	@echo "Inference complete — predictions logged to logs/predictions.db"

# Monitoring
monitor:
	$(PYTHON) -m src.monitoring.drift_monitor --data-path $(DATA_PATH)
	@echo "Monitoring complete — report in monitoring/reports/"

# API (Bonus)
api:
	$(PYTHON) -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Testing
test:
	$(PYTHON) -m pytest tests/ -v

# Cleanup
clean:
	rm -rf logs/predictions.db monitoring/reports/* mlruns __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "🧹 Cleaned"

# Full Pipeline
all: setup train evaluate infer monitor
	@echo "🚀 Full pipeline complete"
