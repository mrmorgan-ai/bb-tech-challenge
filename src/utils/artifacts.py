import json
import pickle
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer


def load_artifact(path: Path, expected_type: type) -> Any:
    """
    Load a pickle artifact with type verification.

    WARNING: pickle deserialization can execute arbitrary code.
    Only load artifacts from trusted sources. This function adds a
    post-load type check as a minimal safety net.
    """
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")

    with open(path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, expected_type):
        raise TypeError(
            f"Expected {expected_type.__name__}, got {type(obj).__name__} from {path}"
        )
    return obj


def load_model(artifacts_dir: Path) -> BaseEstimator:
    """Load the trained model with type verification."""
    return load_artifact(artifacts_dir / "best_model.pkl", BaseEstimator)


def load_preprocessor(artifacts_dir: Path) -> ColumnTransformer:
    """Load the fitted preprocessor with type verification."""
    return load_artifact(artifacts_dir / "preprocessor.pkl", ColumnTransformer)


def load_metadata(artifacts_dir: Path) -> dict:
    """Load model metadata from JSON."""
    path = artifacts_dir / "model_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")

    with open(path, "r") as f:
        return json.load(f)


def load_all_artifacts(artifacts_dir: Path) -> tuple[BaseEstimator, ColumnTransformer, dict]:
    """Load model, preprocessor, and metadata in one call."""
    model = load_model(artifacts_dir)
    preprocessor = load_preprocessor(artifacts_dir)
    metadata = load_metadata(artifacts_dir)
    return model, preprocessor, metadata


def get_churn_probability(model: BaseEstimator, X: np.ndarray, churn_class: int = 0) -> np.ndarray:
    """
    Get P(churn) from model, using classes_ lookup instead of hardcoded index.

    This avoids the fragile assumption that churn_class is always at index 0
    in predict_proba output. sklearn orders columns by model.classes_, which
    depends on the order classes appear in training data.
    """
    churn_idx = list(model.classes_).index(churn_class) # type: ignore
    return model.predict_proba(X)[:, churn_idx] # type: ignore


def get_model_version() -> str:
    """
    Generate a traceable version string: YYYYMMDD.git_short_hash.

    Every training run produces a unique version tied to the date and
    the exact codebase state (git commit). This replaces the hardcoded
    "1.0.0" and enables tracing any deployed model back to its code.
    """
    date_part = datetime.now().strftime("%Y%m%d")
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_hash = "nogit"
    return f"{date_part}.{git_hash}"
