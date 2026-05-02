"""
Main execution file for Parkinson's Disease Detection Using Biomedical Voice Features.

Run this file from the project root using:
    python src/main.py
"""

from config import (
    BOXPLOT_FILE,
    CONFUSION_MATRIX_FILE,
    CORRELATION_HEATMAP_FILE,
    DATA_FILE,
    FEATURE_IMPORTANCE_FILE,
    PLOT_FILE,
    ROC_CURVE_FILE,
    SUMMARY_FILE,
)
from dataset import VoiceDataset
from model import ParkinsonPredictor
from utils import save_dataset_summary
from visualization import (
    plot_confusion_matrix,
    plot_feature_boxplots,
    plot_feature_correlation_heatmap,
    plot_feature_importance,
    plot_roc_curve,
    plot_status_distribution,
)


def main() -> None:
    """
    Execute the full project pipeline:
    1. Load dataset
    2. Save dataset summary
    3. Create visualizations (class distribution, correlation heatmap, box plots)
    4. Train model
    5. Evaluate model
    6. Create post-model visualizations (confusion matrix, feature importance, ROC curve)
    """
    try:
        dataset = VoiceDataset(DATA_FILE)
        df = dataset.load_data()

        print(dataset)
        print(f"Total samples using overloaded len(): {len(dataset)}")

        summary = save_dataset_summary(df, SUMMARY_FILE)
        print("Dataset summary saved:", summary)

        # ── pre-model plots ────────────────────────────────────────────────────
        plot_status_distribution(df, PLOT_FILE)
        print(f"Status distribution plot saved to: {PLOT_FILE}")

        plot_feature_correlation_heatmap(df, CORRELATION_HEATMAP_FILE)
        print(f"Feature correlation heatmap saved to: {CORRELATION_HEATMAP_FILE}")

        # ── model training ─────────────────────────────────────────────────────
        X_train, X_test, y_train, y_test = dataset.prepare_train_test_data()

        predictor = ParkinsonPredictor(n_estimators=100)
        predictor.train(X_train, y_train)

        print(predictor)

        # Box plots use the original (unscaled) DataFrame for readability
        plot_feature_boxplots(df, dataset.feature_columns, BOXPLOT_FILE)
        print(f"Feature box plots saved to: {BOXPLOT_FILE}")

        # ── evaluation ────────────────────────────────────────────────────────
        results = predictor.evaluate(X_test, y_test)

        print("\nModel Accuracy:", round(results["accuracy"], 4))
        print("\nConfusion Matrix:\n", results["confusion_matrix"])
        print("\nClassification Report:\n", results["classification_report"])

        print("Sample Predictions:")
        for result in results["sample_predictions"]:
            print(result)

        # ── post-model plots ───────────────────────────────────────────────────
        plot_confusion_matrix(results["confusion_matrix"], CONFUSION_MATRIX_FILE)
        print(f"Confusion matrix plot saved to: {CONFUSION_MATRIX_FILE}")

        importances_dict = predictor.get_feature_importances(dataset.feature_columns)
        import numpy as np
        feat_names = list(importances_dict.keys())
        feat_vals = np.array(list(importances_dict.values()))
        plot_feature_importance(feat_names, feat_vals, FEATURE_IMPORTANCE_FILE)
        print(f"Feature importance plot saved to: {FEATURE_IMPORTANCE_FILE}")

        y_proba = predictor.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_proba, ROC_CURVE_FILE)
        print(f"ROC curve saved to: {ROC_CURVE_FILE}")

        # ── operator overload demo ─────────────────────────────────────────────
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
