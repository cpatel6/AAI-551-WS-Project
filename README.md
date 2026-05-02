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

## File Structure

```text
parkinsons_voice_detection/
│
├── data/
│   └── parkinsons.csv
│
├── results/
│   ├── dataset_summary.txt
│   └── status_distribution.png
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── utils.py
│   ├── visualization.py
│   └── main.py
│
├── tests/
│   └── test_utils.py
│
├── requirements.txt
└── README.md
```

## How to Use
1. Download the Parkinson's dataset from the UCI Machine Learning Repository.
2. Rename the CSV file as `parkinsons.csv`.
3. Place the file inside the `data/` folder.
4. Run the program from the project root:

```bash
python src/main.py
```

## How to Run Tests
Run the following command from the project root:

```bash
pytest
```

## Python Requirements Covered
- Two meaningful classes: `VoiceDataset` and `ParkinsonPredictor`
- Class relationship: `ParkinsonPredictor` uses data prepared by `VoiceDataset`
- Multiple functions: validation, summary saving, feature selection, visualization
- Advanced libraries: pandas, numpy, matplotlib, scikit-learn
- Exception handling: missing file, empty data, invalid columns, untrained model
- Data I/O: reads CSV file and writes summary/plot results
- Loops and if statements included throughout the program
- Mutable data types: list, dictionary, DataFrame
- Immutable data types: string, tuple, integer, float
- Operator overloads: `__str__`, `__len__`, and `__add__`
- Special function: `filter()` with `lambda`
- Comprehension syntax: list comprehension for feature selection
- Built-in module: `time` and `pathlib`
- Generator function: `prediction_generator()`
- Set operations: required column validation
- Uses `if __name__ == "__main__":`
- Includes docstrings and meaningful comments

## Team Contributions
### Charmilkumar Vijaykumar Patel
- Designed the project structure
- Implemented data loading and preprocessing logic
- Added exception handling and dataset summary output

### Yunyang Zhang
- Implemented machine learning model training and evaluation
- Added visualization module
- Helped with testing, debugging, and README documentation
