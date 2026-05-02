"""
Dataset module for loading and preparing the Parkinson's voice dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import select_numeric_features, validate_file_path, validate_required_columns

# Column settings
TARGET_COLUMN = "status"
NON_FEATURE_COLUMNS = ["name", "status"]

# Train/test split settings
RANDOM_STATE = 42
TEST_SIZE = 0.20


class VoiceDataset:
    """
    Loads and prepares the biomedical voice dataset for model training.

    VoiceDataset is used by ParkinsonPredictor to provide training and testing splits.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.feature_columns = []
        self.scaler = StandardScaler()

    def __str__(self):
        if self.data is None:
            return f"VoiceDataset(file_path='{self.file_path}', data='not loaded')"
        return f"VoiceDataset(rows={self.data.shape[0]}, columns={self.data.shape[1]})"

    def __len__(self):
        """Returns the number of rows in the loaded dataset."""
        if self.data is None:
            return 0
        return len(self.data)

    def load_data(self):
        """Load the CSV file and return a DataFrame."""
        validate_file_path(self.file_path)

        try:
            self.data = pd.read_csv(self.file_path)
        except pd.errors.EmptyDataError as error:
            raise ValueError("The dataset file is empty.") from error
        except pd.errors.ParserError as error:
            raise ValueError("The dataset file could not be parsed correctly.") from error

        validate_required_columns(self.data, [TARGET_COLUMN])
        return self.data

    def prepare_train_test_data(self):
        """Split the dataset into training and testing sets."""
        if self.data is None:
            self.load_data()

        self.feature_columns = select_numeric_features(self.data, NON_FEATURE_COLUMNS)

        if not self.feature_columns:
            raise ValueError("No numeric feature columns were found for model training.")

        X = self.data[self.feature_columns]
        y = self.data[TARGET_COLUMN]

        # Check whether target has both classes
        if y.nunique() < 2:
            raise ValueError("Target column must contain at least two classes: 0 and 1.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test
