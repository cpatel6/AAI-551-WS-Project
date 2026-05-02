# Parkinson's Disease Detection Using Biomedical Voice Features

## Team Members
- Charmilkumar Vijaykumar Patel | SID: 20036845 | cpatel6@stevens.edu
- Yunyang Zhang | SID: 20043349 | yzhang102@stevens.edu

## Project Overview
This project uses biomedical voice measurements to detect whether a voice sample is related to Parkinson's disease. The project uses the Parkinson's dataset from the UCI Machine Learning Repository. The program loads the dataset, preprocesses the data, trains a machine learning model, evaluates model performance, and saves summary and visualization results.

## Project Objective
The main objective is to build a well-structured Python program that solves a real-world biomedical problem using object-oriented programming, data input/output, exception handling, Python libraries, and machine learning.

## Dependencies
Install required libraries using:

```bash
pip install -r requirements.txt
```

Required libraries:
- pandas
- numpy
- matplotlib
- scikit-learn
- pytest
- jupyter
- notebook

## File Structure

```text
AAI-551-WS-Project/
│
├── data/
│   └── parkinsons.csv              # UCI Parkinson's voice dataset
│
├── results/                        # Generated outputs (created at runtime)
│   ├── dataset_summary.txt
│   └── status_distribution.png
│
├── src/
│   ├── __init__.py                 # Package initialisation
│   ├── config.py                   # File paths and constants
│   ├── dataset.py                  # VoiceDataset class
│   ├── model.py                    # ParkinsonPredictor class
│   ├── utils.py                    # Helper functions
│   ├── visualization.py            # Plotting utilities
│   └── main.py                     # Standalone CLI entry point
│
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py             # Tests for VoiceDataset
│   ├── test_model.py               # Tests for ParkinsonPredictor
│   └── test_utils.py               # Tests for utility functions
│
├── parkinsons_detection.ipynb      # Main Jupyter Notebook (project entry point)
├── conftest.py                     # pytest path configuration
├── requirements.txt
└── README.md
```

## How to Use

### Option A – Jupyter Notebook (recommended)
```bash
jupyter notebook parkinsons_detection.ipynb
```
Run all cells from top to bottom.

### Option B – Command line
```bash
python src/main.py
```

## How to Run Tests
Run the following command from the project root:

```bash
pytest
```

## Python Requirements Covered

### Part 1
- Two meaningful classes: `VoiceDataset` and `ParkinsonPredictor`
- Class relationship: `ParkinsonPredictor` uses data prepared by `VoiceDataset` (composition)
- Multiple functions: `validate_file_path`, `validate_required_columns`, `select_numeric_features`, `save_dataset_summary`, `prediction_generator`, `plot_status_distribution`
- Advanced libraries: pandas, numpy, matplotlib, scikit-learn
- Exception handling: `FileNotFoundError` (missing CSV), `ValueError` (empty data / missing columns / single-class target), `RuntimeError` (predicting before training)
- Data I/O: reads `parkinsons.csv`; writes `results/dataset_summary.txt` and `results/status_distribution.png`
- Loops and if statements throughout all modules
- Mutable data types: `list`, `dict`, `pd.DataFrame`
- Immutable data types: `str`, `tuple`, `int`, `float`
- Operator overloads: `__str__`, `__len__` (VoiceDataset), `__str__`, `__add__` (ParkinsonPredictor)
- Docstrings and meaningful comments on every class and function

### Part 2
- Special function: `filter()` with `lambda` in `select_numeric_features()` (`utils.py`)
- Comprehension: list comprehension for feature column selection (`utils.py`, notebook)
- Built-in modules: `time` (timestamps) and `math` (log₂ index-bit calculation) in `utils.py`
- Generator function: `prediction_generator()` in `utils.py`
- Set operations: `validate_required_columns()` uses set difference (`utils.py`)
- `if __name__ == "__main__":` guard in `src/main.py`

## Team Contributions

### Charmilkumar Vijaykumar Patel
- Designed the overall project structure and module layout
- Implemented `VoiceDataset` class: data loading, column validation, train/test splitting
- Implemented exception handling scenarios (`FileNotFoundError`, `ValueError`)
- Wrote `validate_file_path`, `validate_required_columns`, and `save_dataset_summary` utility functions
- Wrote `tests/test_dataset.py` and `tests/test_utils.py`
- Created and maintained the Jupyter Notebook (`parkinsons_detection.ipynb`)

### Yunyang Zhang
- Implemented `ParkinsonPredictor` class: Random Forest training, prediction, evaluation
- Implemented `select_numeric_features` (filter + lambda + list comprehension)
- Implemented `prediction_generator` generator function
- Implemented `plot_status_distribution` visualization module
- Wrote `tests/test_model.py`
- Wrote `src/config.py` and `src/main.py`
- Updated README with complete documentation

