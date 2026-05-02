"""
Pytest test cases for the ParkinsonPredictor class.

Tests cover model initialization, operator overloads, training,
prediction, evaluation, and exception handling scenarios.
"""

import numpy as np
import pytest

from src.model import ParkinsonPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_training_data(n: int = 60):
    """Return simple linearly separable feature arrays and labels."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 4))
    # Positive class when first feature > 0, negative otherwise
    y = (X[:, 0] > 0).astype(int)
    split = n * 3 // 4
    return X[:split], X[split:], y[:split], y[split:]


# ---------------------------------------------------------------------------
# __str__ and __add__ operator overloads
# ---------------------------------------------------------------------------

class TestParkinsonPredictorOperators:
    """Tests for __str__ and __add__ operator overloads."""

    def test_str_untrained(self):
        """__str__ should show trained=False before training."""
        p = ParkinsonPredictor(n_estimators=10)
        assert "trained=False" in str(p)

    def test_str_trained(self):
        """__str__ should show trained=True after training."""
        X_train, X_test, y_train, y_test = _make_training_data()
        p = ParkinsonPredictor(n_estimators=10)
        p.train(X_train, y_train)
        assert "trained=True" in str(p)

    def test_add_returns_sum_of_estimators(self):
        """__add__ should return the sum of n_estimators from both predictors."""
        p1 = ParkinsonPredictor(n_estimators=100)
        p2 = ParkinsonPredictor(n_estimators=50)
        assert p1 + p2 == 150

    def test_add_symmetric(self):
        """__add__ result should be commutative."""
        p1 = ParkinsonPredictor(n_estimators=30)
        p2 = ParkinsonPredictor(n_estimators=70)
        assert p1 + p2 == p2 + p1


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

class TestParkinsonPredictorTrain:
    """Tests for the train method."""

    def test_train_sets_is_trained(self):
        """train should set is_trained to True."""
        X_train, X_test, y_train, y_test = _make_training_data()
        p = ParkinsonPredictor(n_estimators=10)
        assert not p.is_trained
        p.train(X_train, y_train)
        assert p.is_trained

    def test_train_raises_on_empty_data(self):
        """train should raise ValueError on empty training data."""
        p = ParkinsonPredictor(n_estimators=10)
        with pytest.raises(ValueError, match="empty"):
            p.train(np.array([]).reshape(0, 4), np.array([]))


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class TestParkinsonPredictorPredict:
    """Tests for the predict method."""

    def test_predict_raises_before_training(self):
        """predict should raise RuntimeError if model is not yet trained."""
        p = ParkinsonPredictor(n_estimators=10)
        X = np.random.default_rng(1).standard_normal((5, 4))
        with pytest.raises(RuntimeError, match="trained"):
            p.predict(X)

    def test_predict_returns_array(self):
        """predict should return a numpy array of the same length as input."""
        X_train, X_test, y_train, y_test = _make_training_data()
        p = ParkinsonPredictor(n_estimators=10)
        p.train(X_train, y_train)
        preds = p.predict(X_test)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X_test)

    def test_predict_values_are_binary(self):
        """predict should return only 0 and 1 labels."""
        X_train, X_test, y_train, y_test = _make_training_data()
        p = ParkinsonPredictor(n_estimators=10)
        p.train(X_train, y_train)
        preds = p.predict(X_test)
        assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestParkinsonPredictorEvaluate:
    """Tests for the evaluate method."""

    def test_evaluate_returns_required_keys(self):
        """evaluate should return a dict with accuracy, matrix, and report."""
        X_train, X_test, y_train, y_test = _make_training_data()
        p = ParkinsonPredictor(n_estimators=10)
        p.train(X_train, y_train)
        results = p.evaluate(X_test, y_test)
        assert "accuracy" in results
        assert "confusion_matrix" in results
        assert "classification_report" in results
        assert "sample_predictions" in results

    def test_evaluate_accuracy_between_0_and_1(self):
        """Accuracy returned by evaluate should be between 0 and 1."""
        X_train, X_test, y_train, y_test = _make_training_data()
        p = ParkinsonPredictor(n_estimators=10)
        p.train(X_train, y_train)
        results = p.evaluate(X_test, y_test)
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_evaluate_sample_predictions_are_strings(self):
        """sample_predictions should be a list of human-readable strings."""
        X_train, X_test, y_train, y_test = _make_training_data()
        p = ParkinsonPredictor(n_estimators=10)
        p.train(X_train, y_train)
        results = p.evaluate(X_test, y_test)
        for msg in results["sample_predictions"]:
            assert isinstance(msg, str)
