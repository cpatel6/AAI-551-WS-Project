"""
Pytest test cases for the VoiceDataset class.

Tests cover data loading, column validation, feature preparation,
and exception handling scenarios.
"""

import pandas as pd
import pytest
from pathlib import Path

from src.dataset import VoiceDataset

DATA_FILE = Path("data/parkinsons.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_df() -> pd.DataFrame:
    """Return a minimal DataFrame that resembles the real dataset."""
    return pd.DataFrame({
        "name": ["phon_R01_S01_1", "phon_R01_S01_2", "phon_R01_S02_1"],
        "MDVP:Fo(Hz)": [119.99, 122.40, 116.68],
        "MDVP:Fhi(Hz)": [157.30, 148.65, 131.11],
        "status": [1, 1, 0],
    })


# ---------------------------------------------------------------------------
# __str__ and __len__ operator overloads
# ---------------------------------------------------------------------------

class TestVoiceDatasetOperators:
    """Tests for __str__ and __len__ operator overloads."""

    def test_str_before_load(self, tmp_path):
        """__str__ should return 'not loaded' before data is read."""
        dummy_csv = tmp_path / "dummy.csv"
        dummy_csv.write_text("name,status\ntest,1\n")
        ds = VoiceDataset(dummy_csv)
        assert "not loaded" in str(ds)

    def test_str_after_load(self, tmp_path):
        """__str__ should include row/column counts after data is loaded."""
        csv_path = tmp_path / "parkinsons.csv"
        _make_minimal_df().to_csv(csv_path, index=False)
        ds = VoiceDataset(csv_path)
        ds.load_data()
        result = str(ds)
        assert "rows=" in result
        assert "columns=" in result

    def test_len_before_load(self, tmp_path):
        """__len__ should return 0 before data is loaded."""
        dummy_csv = tmp_path / "dummy.csv"
        dummy_csv.write_text("name,status\ntest,1\n")
        ds = VoiceDataset(dummy_csv)
        assert len(ds) == 0

    def test_len_after_load(self, tmp_path):
        """__len__ should equal the number of rows in the CSV."""
        csv_path = tmp_path / "parkinsons.csv"
        df = _make_minimal_df()
        df.to_csv(csv_path, index=False)
        ds = VoiceDataset(csv_path)
        ds.load_data()
        assert len(ds) == len(df)


# ---------------------------------------------------------------------------
# load_data – happy path and exception handling
# ---------------------------------------------------------------------------

class TestVoiceDatasetLoad:
    """Tests for load_data method."""

    def test_load_data_returns_dataframe(self, tmp_path):
        """load_data should return a non-empty DataFrame on valid input."""
        csv_path = tmp_path / "parkinsons.csv"
        _make_minimal_df().to_csv(csv_path, index=False)
        ds = VoiceDataset(csv_path)
        result = ds.load_data()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_load_data_raises_file_not_found(self, tmp_path):
        """load_data should raise FileNotFoundError for missing files."""
        missing = tmp_path / "missing.csv"
        ds = VoiceDataset(missing)
        with pytest.raises(FileNotFoundError):
            ds.load_data()

    def test_load_data_raises_value_error_on_empty_csv(self, tmp_path):
        """load_data should raise ValueError when the CSV file is empty."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")
        ds = VoiceDataset(empty_csv)
        with pytest.raises(ValueError):
            ds.load_data()

    def test_load_data_raises_value_error_missing_status_column(self, tmp_path):
        """load_data should raise ValueError when 'status' column is absent."""
        csv_path = tmp_path / "no_status.csv"
        pd.DataFrame({"name": ["a"], "feature": [1.0]}).to_csv(csv_path, index=False)
        ds = VoiceDataset(csv_path)
        with pytest.raises(ValueError):
            ds.load_data()


# ---------------------------------------------------------------------------
# prepare_train_test_data
# ---------------------------------------------------------------------------

class TestVoiceDatasetPrepare:
    """Tests for prepare_train_test_data method."""

    def _make_full_df(self, n: int = 30) -> pd.DataFrame:
        """Create a larger synthetic dataset for train/test splitting."""
        import numpy as np
        rng = np.random.default_rng(42)
        n_positive = n * 3 // 4
        rows = {
            "name": [f"sample_{i}" for i in range(n)],
            "feature_a": rng.uniform(100, 200, n),
            "feature_b": rng.uniform(0.001, 0.01, n),
            "status": ([1] * n_positive) + ([0] * (n - n_positive)),
        }
        return pd.DataFrame(rows)

    def test_prepare_returns_four_arrays(self, tmp_path):
        """prepare_train_test_data should return exactly four splits."""
        csv_path = tmp_path / "parkinsons.csv"
        self._make_full_df().to_csv(csv_path, index=False)
        ds = VoiceDataset(csv_path)
        ds.load_data()
        splits = ds.prepare_train_test_data()
        assert len(splits) == 4

    def test_prepare_raises_for_single_class(self, tmp_path):
        """prepare_train_test_data should raise when only one class exists."""
        csv_path = tmp_path / "one_class.csv"
        pd.DataFrame({
            "name": [f"s{i}" for i in range(20)],
            "feature": [1.0] * 20,
            "status": [1] * 20,
        }).to_csv(csv_path, index=False)
        ds = VoiceDataset(csv_path)
        ds.load_data()
        with pytest.raises(ValueError):
            ds.prepare_train_test_data()

    def test_load_called_automatically_by_prepare(self, tmp_path):
        """prepare_train_test_data should auto-call load_data when needed."""
        csv_path = tmp_path / "parkinsons.csv"
        self._make_full_df().to_csv(csv_path, index=False)
        ds = VoiceDataset(csv_path)
        # Do NOT call load_data manually
        assert ds.data is None
        splits = ds.prepare_train_test_data()
        assert len(splits) == 4

    def test_real_dataset_loads_successfully(self):
        """Integration test: the real parkinsons.csv file loads without errors."""
        if not DATA_FILE.exists():
            pytest.skip("parkinsons.csv not found in data/ directory")
        ds = VoiceDataset(DATA_FILE)
        df = ds.load_data()
        assert len(df) > 0
        assert "status" in df.columns
