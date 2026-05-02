"""
Model module for training and evaluating Parkinson's disease prediction model.
"""

from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import RANDOM_STATE
from utils import prediction_generator


class ParkinsonPredictor:
    """
    Class responsible for training and evaluating a machine learning model.

    This class has a dependency relationship with VoiceDataset because it uses
    prepared training and testing data from the dataset class.
    """

    def __init__(self, n_estimators: int = 100):
        """
        Initialize the predictor.

        Parameters:
            n_estimators (int): Number of decision trees in Random Forest.
        """
        self.n_estimators = n_estimators
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=RANDOM_STATE,
        )
        self.is_trained = False

    def __str__(self) -> str:
        """
        Return readable model information.
        """
        return f"ParkinsonPredictor(model=RandomForestClassifier, trained={self.is_trained})"

    def __add__(self, other: "ParkinsonPredictor") -> int:
        """
        Operator overload for adding number of trees from two predictor objects.

        Parameters:
            other (ParkinsonPredictor): Another predictor object.

        Returns:
            int: Total number of estimators.
        """
        return self.n_estimators + other.n_estimators

    def train(self, X_train: np.ndarray, y_train) -> None:
        """
        Train the prediction model.

        Parameters:
            X_train (np.ndarray): Training features.
            y_train: Training labels.
        """
        if len(X_train) == 0:
            raise ValueError("Training data is empty. Model cannot be trained.")

        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels for test samples.

        Parameters:
            X_test (np.ndarray): Test features.

        Returns:
            np.ndarray: Predicted labels.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict(X_test)

    def get_feature_importances(self, feature_names: list) -> dict:
        """
        Return feature importances from the trained Random Forest model.

        Parameters:
            feature_names (list): Names of the model features (same order as training).

        Returns:
            dict: Keys are feature names, values are importance scores.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before retrieving feature importances.")
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Return class probabilities for test samples.

        Parameters:
            X_test (np.ndarray): Test features.

        Returns:
            np.ndarray: Array of shape (n_samples, 2) with class probabilities.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict_proba(X_test)

    def evaluate(self, X_test: np.ndarray, y_test) -> Dict:
        """
        Evaluate model performance.

        Parameters:
            X_test (np.ndarray): Test features.
            y_test: True labels.

        Returns:
            Dict: Evaluation results.
        """
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
