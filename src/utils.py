"""
Utility functions for validation, dataset summary, and prediction formatting.

Uses: time (timestamps), math (numeric calculations), filter(), lambda,
list comprehension, set operations, and a generator function.
"""

import math
import time
from pathlib import Path

import pandas as pd


def validate_file_path(file_path):
    """Raise FileNotFoundError if the dataset file does not exist."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found: {file_path}\n"
            "Please place parkinsons.csv inside the data folder."
        )
    return True


def validate_required_columns(df, required_columns):
    """Raise ValueError if any required columns are missing from the DataFrame."""
    available_columns = set(df.columns)
    required_set = set(required_columns)
    missing_columns = required_set - available_columns

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True


def select_numeric_features(df, excluded_columns):
    """Return a list of numeric columns, excluding the specified ones."""
    numeric_columns = list(filter(lambda col: pd.api.types.is_numeric_dtype(df[col]), df.columns))

    # List comprehension removes target and identifier columns
    feature_columns = [col for col in numeric_columns if col not in excluded_columns]
    return feature_columns


def prediction_generator(predictions):
    """Generator that converts numeric predictions to readable strings."""
    for index, prediction in enumerate(predictions, start=1):
        if prediction == 1:
            yield f"Sample {index}: Parkinson's disease detected"
        else:
            yield f"Sample {index}: Healthy voice sample"


def save_dataset_summary(df, output_file):
    """Save a text summary of the dataset and return a summary dict."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = df.shape[0]
    columns = df.shape[1]
    missing_values = int(df.isnull().sum().sum())

    # math.log2 and math.ceil: compute the number of bits needed to index every row
    index_bits = math.ceil(math.log2(rows)) if rows > 1 else 1

    summary = {
        "rows": rows,
        "columns": columns,
        "missing_values": missing_values,
        "index_bits": index_bits,
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
