"""
Basic PyTest test cases for utility functions.
"""

import pandas as pd
import pytest

from src.utils import select_numeric_features, validate_required_columns


def test_validate_required_columns_passes():
    """Test required column validation when column exists."""
    df = pd.DataFrame({"status": [0, 1], "feature": [1.2, 3.4]})
    assert validate_required_columns(df, ["status"]) is True


def test_validate_required_columns_fails():
    """Test required column validation when column is missing."""
    df = pd.DataFrame({"feature": [1.2, 3.4]})
    with pytest.raises(ValueError):
        validate_required_columns(df, ["status"])


def test_select_numeric_features():
    """Test numeric feature selection."""
    df = pd.DataFrame({
        "name": ["a", "b"],
        "status": [0, 1],
        "MDVP:Fo(Hz)": [120.5, 140.2],
        "text_column": ["x", "y"],
    })
    features = select_numeric_features(df, ["name", "status"])
    assert features == ["MDVP:Fo(Hz)"]
