"""
Main execution file for Parkinson's Disease Detection Using Biomedical Voice Features.

Run this file from the project root using:
    python src/main.py
"""

from config import DATA_FILE, PLOT_FILE, SUMMARY_FILE
from dataset import VoiceDataset
from model import ParkinsonPredictor
from utils import save_dataset_summary
from visualization import plot_status_distribution


def main() -> None:
    """
    Execute the full project pipeline:
    1. Load dataset
    2. Save dataset summary
    3. Create visualization
    4. Train model
    5. Evaluate model
    """
    try:
        dataset = VoiceDataset(DATA_FILE)
        df = dataset.load_data()

        print(dataset)
        print(f"Total samples using overloaded len(): {len(dataset)}")

        summary = save_dataset_summary(df, SUMMARY_FILE)
        print("Dataset summary saved:", summary)

        plot_status_distribution(df, PLOT_FILE)
        print(f"Status distribution plot saved to: {PLOT_FILE}")

        X_train, X_test, y_train, y_test = dataset.prepare_train_test_data()

        predictor = ParkinsonPredictor(n_estimators=100)
        predictor.train(X_train, y_train)

        print(predictor)

        results = predictor.evaluate(X_test, y_test)

        print("\nModel Accuracy:", round(results["accuracy"], 4))
        print("\nConfusion Matrix:\n", results["confusion_matrix"])
        print("\nClassification Report:\n", results["classification_report"])

        print("Sample Predictions:")
        for result in results["sample_predictions"]:
            print(result)

        # Demonstration of additional operator overloading.
        second_predictor = ParkinsonPredictor(n_estimators=50)
        print("Combined estimator count using overloaded + operator:", predictor + second_predictor)

    except FileNotFoundError as error:
        print("File Error:", error)
    except ValueError as error:
        print("Value Error:", error)
    except RuntimeError as error:
        print("Runtime Error:", error)


if __name__ == "__main__":
    main()
