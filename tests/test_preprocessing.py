import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ENGINEERED_COLS, TARGET_COL
from data.preprocessing import (
    clean_data,
    engineer_features,
    build_preprocessor,
    split_data,
    compute_features_hash,
)


@pytest.fixture
def sample_df():
    """Minimal DataFrame that mirrors the raw dataset schema."""
    return pd.DataFrame({
        "custid": [1, 2, 3, 4, 5],
        "created": pd.to_datetime(["2023-01-01", "2023-03-15", "2023-06-01", "2023-02-10", "2023-05-20"]),
        "firstorder": pd.to_datetime(["2023-01-10", "2023-03-20", "2023-06-05", "2023-02-15", "2023-05-25"]),
        "lastorder": pd.to_datetime(["2023-06-01", "2023-06-10", "2023-06-15", "2023-06-05", "2023-06-12"]),
        "esent": [10, 20, 5, 15, 8],
        "eopenrate": [0.3, 0.5, 0.1, 0.4, 0.2],
        "eclickrate": [0.1, 0.2, 0.05, 0.15, 0.08],
        "avgorder": [50.0, 80.0, 30.0, 60.0, 45.0],
        "ordfreq": [5, 10, 2, 7, 3],
        "paperless": [1, 0, 1, 1, 0],
        "refill": [0, 1, 0, 1, 0],
        "doorstep": [1, 1, 0, 0, 1],
        "favday": ["Mon", "Tue", "Wed", "Mon", "Fri"],
        "city": ["DEL", "BOM", "MAA", "BLR", "DEL"],
        "retained": [1, 1, 0, 1, 0],
    })


class TestCleanData:
    def test_drops_null_custid(self):
        df = pd.DataFrame({
            "custid": [1, None, 3],
            "created": ["2023-01-01"] * 3,
            "firstorder": ["2023-02-01"] * 3,
            "lastorder": ["2023-03-01"] * 3,
        })
        cleaned = clean_data(df)
        assert len(cleaned) == 2

    def test_parses_dates(self):
        df = pd.DataFrame({
            "custid": [1],
            "created": ["2023-01-01"],
            "firstorder": ["2023-02-01"],
            "lastorder": ["2023-03-01"],
        })
        cleaned = clean_data(df)
        assert pd.api.types.is_datetime64_any_dtype(cleaned["created"])


class TestEngineerFeatures:
    def test_creates_all_engineered_columns(self, sample_df):
        df_eng, _ = engineer_features(sample_df)
        for col in ENGINEERED_COLS:
            assert col in df_eng.columns

    def test_ref_date_defaults_to_max_created(self, sample_df):
        _, ref_date = engineer_features(sample_df)
        assert ref_date == sample_df["created"].max()

    def test_ref_date_passthrough(self, sample_df):
        fixed_date = pd.Timestamp("2024-01-01")
        _, ref_date = engineer_features(sample_df, ref_date=fixed_date)
        assert ref_date == fixed_date

    def test_tenure_days_positive(self, sample_df):
        df_eng, _ = engineer_features(sample_df)
        assert (df_eng["tenure_days"] >= 0).all()

    def test_order_recency_ratio_bounded(self, sample_df):
        df_eng, _ = engineer_features(sample_df)
        assert (df_eng["order_recency_ratio"] >= 0).all()
        assert (df_eng["order_recency_ratio"] <= 1).all()

    def test_no_nans_in_engineered(self, sample_df):
        df_eng, _ = engineer_features(sample_df)
        for col in ENGINEERED_COLS:
            assert df_eng[col].isna().sum() == 0


class TestBuildPreprocessor:
    def test_returns_column_transformer(self):
        preprocessor = build_preprocessor()
        assert hasattr(preprocessor, "fit_transform")

    def test_fit_transform_produces_array(self, sample_df):
        df_eng, _ = engineer_features(sample_df)
        preprocessor = build_preprocessor()
        X = preprocessor.fit_transform(df_eng)
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == len(sample_df)


class TestSplitData:
    def test_split_preserves_total_rows(self, sample_df):
        """Need enough rows for stratified split (at least 2 per class)."""
        # Expand sample to have enough rows for split
        big_df = pd.concat([sample_df] * 20, ignore_index=True)
        big_df_eng, _ = engineer_features(big_df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(big_df_eng)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(big_df_eng)


class TestFeaturesHash:
    def test_deterministic(self):
        features = {"a": 1, "b": 2.0, "c": "test"}
        assert compute_features_hash(features) == compute_features_hash(features)

    def test_order_independent(self):
        h1 = compute_features_hash({"a": 1, "b": 2})
        h2 = compute_features_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_inputs_different_hashes(self):
        h1 = compute_features_hash({"a": 1})
        h2 = compute_features_hash({"a": 2})
        assert h1 != h2

    def test_hash_length(self):
        h = compute_features_hash({"x": 42})
        assert len(h) == 12
