import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TARGET_COL, ID_COL, DATE_COLS,
    NUMERIC_COLS, BINARY_COLS, CATEGORICAL_COLS,
)

REQUIRED_COLUMNS = [ID_COL, TARGET_COL] + DATE_COLS + NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS

EXPECTED_TYPES = {
    **{col: "numeric" for col in NUMERIC_COLS},
    **{col: "binary" for col in BINARY_COLS},
    **{col: "categorical" for col in CATEGORICAL_COLS},
}

VALID_RANGES = {
    "eopenrate": (0.0, 1.0),
    "eclickrate": (0.0, 1.0),
    "paperless": (0, 1),
    "refill": (0, 1),
    "doorstep": (0, 1),
}


class DataValidationError(Exception):
    """Raised when input data fails validation checks."""
    pass


def validate_schema(df: pd.DataFrame) -> list[str]:
    """
    Check column presence and basic types.
    Returns list of warnings (empty = all good).
    Raises DataValidationError on critical failures.
    """
    warnings: list[str] = []

    # Check required columns exist
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    # Check for completely empty DataFrame
    if len(df) == 0:
        raise DataValidationError("Dataset is empty (0 rows)")

    # Check numeric columns are actually numeric
    for col in NUMERIC_COLS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            warnings.append(f"Column '{col}' expected numeric, got {df[col].dtype}")

    # Check binary columns contain only 0/1 (allow NaN)
    for col in BINARY_COLS:
        unique = set(df[col].dropna().unique())
        if not unique.issubset({0, 1, 0.0, 1.0}):
            warnings.append(f"Column '{col}' has non-binary values: {unique - {0, 1, 0.0, 1.0}}")

    # Check value ranges
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0 and (vals.min() < lo or vals.max() > hi):
                warnings.append(f"Column '{col}' has values outside [{lo}, {hi}]")

    # Check target column values
    if TARGET_COL in df.columns:
        unique_targets = set(df[TARGET_COL].dropna().unique())
        if not unique_targets.issubset({0, 1, 0.0, 1.0}):
            warnings.append(f"Target '{TARGET_COL}' has unexpected values: {unique_targets}")

    return warnings


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all validations. Raises on critical errors, prints warnings otherwise.
    Returns the DataFrame unchanged (pass-through for pipeline chaining).
    """
    warnings = validate_schema(df)
    for w in warnings:
        print(f"[VALIDATION WARNING] {w}")
    return df
