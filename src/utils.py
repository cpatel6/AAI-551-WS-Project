"""
Utility functions for validation, dataset summary, and result processing.

This module includes regular functions, a generator function, filter(), lambda,
list comprehension, set operations, loops, conditionals, and exception handling.
"""

import time
from pathlib import Path
from typing import Dict, Generator, Iterable, List

import pandas as pd


def validate_file_path(file_path: Path) -> bool:
    """
    Validate whether the dataset file exists.

    Parameters:
        file_path (Path): Path of the dataset file.

    Returns:
        bool: True if file exists.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found: {file_path}\n"
            "Please place parkinsons.csv inside the data folder."
        )
    return True


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> bool:
    """
    Check whether all required columns are present using set operations.

    Parameters:
        df (pd.DataFrame): Input dataset.
        required_columns (Iterable[str]): Required column names.

    Returns:
        bool: True if all required columns are available.

    Raises:
        ValueError: If required columns are missing.
    """
    available_columns = set(df.columns)
    required_set = set(required_columns)
    missing_columns = required_set - available_columns

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True


def select_numeric_features(df: pd.DataFrame, excluded_columns: List[str]) -> List[str]:
    """
    Select numeric feature columns using filter(), lambda, and list comprehension.

    Parameters:
        df (pd.DataFrame): Input dataset.
        excluded_columns (List[str]): Columns to remove from model features.

    Returns:
        List[str]: List of selected numeric feature names.
    """
    numeric_columns = list(filter(lambda col: pd.api.types.is_numeric_dtype(df[col]), df.columns))

    # List comprehension removes target and identifier columns.
    feature_columns = [col for col in numeric_columns if col not in excluded_columns]
    return feature_columns


def prediction_generator(predictions: Iterable[int]) -> Generator[str, None, None]:
    """
    Generator function that converts numeric predictions into readable messages.

    Parameters:
        predictions (Iterable[int]): Model prediction values.

    Yields:
        str: Human-readable prediction result.
    """
    for index, prediction in enumerate(predictions, start=1):
        if prediction == 1:
            yield f"Sample {index}: Parkinson's disease detected"
        else:
            yield f"Sample {index}: Healthy voice sample"


def save_dataset_summary(df: pd.DataFrame, output_file: Path) -> Dict[str, int]:
    """
    Summarize the dataset and save the summary to a text file.

    Parameters:
        df (pd.DataFrame): Input dataset.
        output_file (Path): Output file path.

    Returns:
        Dict[str, int]: Basic dataset summary information.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
    }

    current_time = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(output_file, "w", encoding="utf-8") as file:
        file.write("Parkinson's Voice Dataset Summary\n")
        file.write("=" * 40 + "\n")
        file.write(f"Generated on: {current_time}\n")

        for key, value in summary.items():
            file.write(f"{key}: {value}\n")

        file.write("\nColumn Names:\n")
        for column in df.columns:
            file.write(f"- {column}\n")

    return summary
