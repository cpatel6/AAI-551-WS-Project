"""
Configuration file for Parkinson's Disease Detection project.

This file stores fixed paths and column names used by different modules.
"""

from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Input and output directories
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Dataset and output file paths
DATA_FILE = DATA_DIR / "parkinsons.csv"
SUMMARY_FILE = RESULTS_DIR / "dataset_summary.txt"

# ── Plot output paths ──────────────────────────────────────────────────────────
PLOT_FILE = RESULTS_DIR / "status_distribution.png"
CONFUSION_MATRIX_FILE = RESULTS_DIR / "confusion_matrix.png"
CORRELATION_HEATMAP_FILE = RESULTS_DIR / "feature_correlation_heatmap.png"
FEATURE_IMPORTANCE_FILE = RESULTS_DIR / "feature_importance.png"
ROC_CURVE_FILE = RESULTS_DIR / "roc_curve.png"
BOXPLOT_FILE = RESULTS_DIR / "feature_boxplots.png"

# Target column from the UCI Parkinson's dataset
TARGET_COLUMN = "status"

# Columns that should not be used as model features
NON_FEATURE_COLUMNS = ["name", TARGET_COLUMN]

# Random seed for reproducible train/test split
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Number of top features to show in the importance and box-plot charts
TOP_N_FEATURES = 10
