"""
Dataset module for loading and preprocessing biomedical voice data.
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import NON_FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from utils import select_numeric_features, validate_file_path, validate_required_columns


class VoiceDataset:
    """
    Class responsible for reading, validating, and preparing the voice dataset.

    Relationship:
        ParkinsonPredictor depends on the prepared output of VoiceDataset.
    """

    def __init__(self, file_path: Path):
        """
        Initialize VoiceDataset object.

        Parameters:
            file_path (Path): Path to the CSV dataset.
        """
        self.file_path = file_path
        self.data = None
        self.feature_columns: List[str] = []
        self.scaler = StandardScaler()

    def __str__(self) -> str:
        """
        Return readable dataset information.
        """
        if self.data is None:
            return f"VoiceDataset(file_path='{self.file_path}', data='not loaded')"
        return f"VoiceDataset(rows={self.data.shape[0]}, columns={self.data.shape[1]})"

    def __len__(self) -> int:
        """
        Operator overload for len(dataset_object).

        Returns:
            int: Number of rows in the dataset.
        """
        if self.data is None:
            return 0
        return len(self.data)

    def load_data(self) -> pd.DataFrame:
        """
        Load CSV data with exception handling.

        Returns:
            pd.DataFrame: Loaded dataset.
        """
        validate_file_path(self.file_path)

        try:
            self.data = pd.read_csv(self.file_path)
        except pd.errors.EmptyDataError as error:
            raise ValueError("The dataset file is empty.") from error
        except pd.errors.ParserError as error:
            raise ValueError("The dataset file could not be parsed correctly.") from error

        validate_required_columns(self.data, [TARGET_COLUMN])
        return self.data

    def prepare_train_test_data(self) -> Tuple:
        """
        Prepare training and testing data for machine learning.

        Returns:
            Tuple: X_train, X_test, y_train, y_test.
        """
        if self.data is None:
            self.load_data()

        self.feature_columns = select_numeric_features(self.data, NON_FEATURE_COLUMNS)

        if not self.feature_columns:
            raise ValueError("No numeric feature columns were found for model training.")

        X = self.data[self.feature_columns]
        y = self.data[TARGET_COLUMN]

        # If statement checks whether target has both classes.
        if y.nunique() < 2:
            raise ValueError("Target column must contain at least two classes: 0 and 1.")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test
