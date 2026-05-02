"""
Visualization module for creating plots from the Parkinson's voice dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import TARGET_COLUMN


def plot_status_distribution(df: pd.DataFrame, output_file: Path) -> None:
    """
    Create and save a bar plot showing healthy vs Parkinson's voice samples.

    Parameters:
        df (pd.DataFrame): Input dataset.
        output_file (Path): File path where the plot will be saved.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"'{TARGET_COLUMN}' column is required for plotting.")

    counts = df[TARGET_COLUMN].value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    plt.bar(["Healthy (0)", "Parkinson's (1)"], counts.values)
    plt.title("Distribution of Voice Samples")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
