"""
Model module for training and evaluating the Parkinson's disease predictor.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils import prediction_generator

RANDOM_STATE = 42


class ParkinsonPredictor:
    """
    Trains and evaluates a Random Forest classifier on voice data.

    Depends on VoiceDataset to provide the prepared training and testing splits.
    """

    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=RANDOM_STATE,
        )
        self.is_trained = False

    def __str__(self):
        return f"ParkinsonPredictor(model=RandomForestClassifier, trained={self.is_trained})"

    def __add__(self, other):
        """Add the estimator counts from two predictors."""
        return self.n_estimators + other.n_estimators

    def train(self, X_train, y_train):
        """Train the Random Forest model."""
        if len(X_train) == 0:
            raise ValueError("Training data is empty. Model cannot be trained.")
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X_test):
        """Return predicted class labels for the test set."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict(X_test)

    def get_feature_importances(self, feature_names):
        """Return a dict mapping feature names to importance scores."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before retrieving feature importances.")
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))

    def predict_proba(self, X_test):
        """Return class probabilities for the test set."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict_proba(X_test)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance and return a results dictionary."""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        matrix = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions)

        readable_predictions = []
        for message in prediction_generator(predictions[:5]):
            readable_predictions.append(message)

        return {
            "accuracy": accuracy,
            "confusion_matrix": matrix,
            "classification_report": report,
            "sample_predictions": readable_predictions,
        }
