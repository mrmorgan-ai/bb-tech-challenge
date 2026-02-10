import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import compute_features_hash
from src.monitoring.drift_monitor import compute_psi, compute_ks_test, compute_chi2_test, psi_alert_level


class TestFeaturesHash:
    def test_deterministic(self):
        features = {"a": 1, "b": 2.0, "c": "test"}
        assert compute_features_hash(features) == compute_features_hash(features)

    def test_order_independent(self):
        """sort_keys=True makes hash independent of dict insertion order."""
        h1 = compute_features_hash({"a": 1, "b": 2})
        h2 = compute_features_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_inputs_different_hashes(self):
        h1 = compute_features_hash({"a": 1})
        h2 = compute_features_hash({"a": 2})
        assert h1 != h2


class TestPSI:
    def test_identical_distributions_near_zero(self):
        """Same data → PSI ≈ 0 (only epsilon noise)."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        psi = compute_psi(data, data)
        assert psi < 0.01

    def test_shifted_distribution_detected(self):
        """Mean shift of 2σ → PSI should exceed alert threshold."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(2, 1, 1000)
        psi = compute_psi(ref, cur)
        assert psi > 0.
