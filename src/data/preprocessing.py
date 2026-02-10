import json
import hashlib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.config import (
    TARGET_COL, ID_COL, DATE_COLS, NUMERIC_COLS, BINARY_COLS,
    CATEGORICAL_COLS, ENGINEERED_COLS,
    TEST_SIZE, VAL_SIZE, RANDOM_STATE,
)


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw dataset from xlsx"""
    if not path:
        raise FileNotFoundError(f"File not found in path: {path}")

    df = pd.read_excel(path, sheet_name='data',header=0,engine='openpyxl')

    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning: drop rows with null identifiers, parse dates.
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)

    # Drop rows where the identifier is null. Corrupted samples
    df_clean = df_clean.dropna(subset=[ID_COL]).reset_index(drop=True)

    # Parse date columns instead of raising an exception
    for col in DATE_COLS:
        df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")

    dropped = initial_rows - len(df_clean)
    print(f"After cleaning: {len(df_clean)} rows (dropped {dropped} with null {ID_COL})")
    return df_clean


def engineer_features(
    df: pd.DataFrame,
    ref_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Create temporal features from raw date columns.

    Feature notes:
    - tenure_days: account age relative to ref_date.
    - days_first_to_last_order: CAUTION — for churned users this is frozen.
      Retained because at prediction time we DO know the last order date.
    - days_since_first_order: historical fact.
    - order_recency_ratio: normalized recency (0=stale, 1=recent).
    """
    df_engineered = df.copy()

    # Reference date: compute from data during training, use stored value at inference
    if ref_date is None:
        ref_date = df_engineered["created"].max()

    # Tenure: how long has this account existed?
    df_engineered["tenure_days"] = (ref_date - df_engineered["created"]).dt.days # type: ignore

    # Engagement window: time between first and last order
    df_engineered["days_first_to_last_order"] = (
        df["lastorder"] - df["firstorder"]
    ).dt.days.fillna(0).clip(lower=0) # type: ignore

    # Time since first order: how long ago did they start buying?
    df_engineered["days_since_first_order"] = (
        ref_date - df["firstorder"]).dt.days.fillna(0).clip(lower=0) # type: ignore

    # Order recency ratio: normalized measure of how recently the user was active
    ## Formula: 1 - (days_since_last_order / tenure_days)
    ## Range: 0 (last order was at account creation) to 1 (ordered today)
    tenure_safe = df_engineered["tenure_days"].replace(0, 1)  # Avoid division by zero
    days_since_last = (ref_date - df_engineered["lastorder"]).dt.days.fillna(tenure_safe) # type: ignore
    df_engineered["order_recency_ratio"] = 1 - (days_since_last / tenure_safe).clip(0, 1)

    # Fill any remaining NaNs in engineered features with 0
    for col in ENGINEERED_COLS:
        df_engineered[col] = df_engineered[col].fillna(0)

    return df_engineered, ref_date # type: ignore


def build_preprocessor() -> ColumnTransformer:
    """
    Build sklearn ColumnTransformer for feature preprocessing.
    """
    numeric_features = NUMERIC_COLS + ENGINEERED_COLS
    binary_features = BINARY_COLS
    categorical_features = CATEGORICAL_COLS

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("bin", "passthrough", binary_features),
            (
                "cat",
                OneHotEncoder(
                    drop="first",           # Avoid dummy variable trap
                    sparse_output=False,    # Dense array for compatibility
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
        ],
        remainder="drop",  # Explicitly drop unlisted columns
    )
    return preprocessor


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Stratified train/validation/test split.
    """
    feature_cols = NUMERIC_COLS + BINARY_COLS + ENGINEERED_COLS + CATEGORICAL_COLS
    X = df[feature_cols]
    y = df[TARGET_COL]

    # Stage 1: tsest set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y, # stratify by target class for moderate imbalance
        random_state=RANDOM_STATE,
    )

    # Stage 2: split remaining into train and validation
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_data(
    path: str, ref_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, ColumnTransformer, pd.Timestamp]:
    """
    Full data pipeline: load -> clean -> engineer -> split -> build preprocessor.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, ref_date)
    """
    df = load_raw_data(path)
    df = clean_data(df)
    df, ref_date = engineer_features(df, ref_date=ref_date)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    preprocessor = build_preprocessor()

    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, ref_date


def compute_features_hash(features: dict) -> str:
    """
    SHA256 hash of a feature dictionary for prediction traceability.
    """
    raw = json.dumps(features, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
