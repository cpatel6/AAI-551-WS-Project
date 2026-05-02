"""
Visualization functions for the Parkinson's voice dataset analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

TARGET_COLUMN = "status"
TOP_N_FEATURES = 10


def plot_status_distribution(df, output_file):
    """Save a bar chart showing healthy vs Parkinson's voice samples."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"'{TARGET_COLUMN}' column is required for plotting.")

    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["Healthy (0)", "Parkinson's (1)"]
    colors = ["#4CAF50", "#F44336"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="black", width=0.5)

    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_title("Distribution of Voice Samples", fontsize=14, fontweight="bold")
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_ylim(0, max(counts.values) * 1.15)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, output_file, class_labels=None):
    """Save a styled confusion-matrix heatmap."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if class_labels is None:
        class_labels = ["Healthy (0)", "Parkinson's (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_labels,
        yticklabels=class_labels,
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    ax.title.set_fontsize(14)
    ax.title.set_fontweight("bold")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def plot_feature_correlation_heatmap(df, output_file, excluded_columns=None):
    """Save a Pearson correlation heatmap for all numeric features."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if excluded_columns is None:
        excluded_columns = ["name", TARGET_COLUMN]

    feature_df = df.select_dtypes(include=[np.number])
    feature_df = feature_df.drop(
        columns=[c for c in excluded_columns if c in feature_df.columns]
    )

    corr = feature_df.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Pearson r")

    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def plot_feature_importance(feature_names, importances, output_file, top_n=TOP_N_FEATURES):
    """Save a horizontal bar chart of the top feature importances."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    indices = np.argsort(importances)[::-1][:top_n]
    sorted_names = [feature_names[i] for i in indices][::-1]
    sorted_vals = importances[indices][::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_vals)))
    ax.barh(sorted_names, sorted_vals, color=colors, edgecolor="black")
    ax.set_xlabel("Mean Decrease in Impurity (importance)", fontsize=11)
    ax.set_title(
        f"Top {top_n} Feature Importances (Random Forest)",
        fontsize=14,
        fontweight="bold",
    )
    ax.axvline(x=0, color="black", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true, y_proba, output_file):
    """Save an ROC curve with AUC annotation."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1565C0", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1.2, linestyle="--", label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.10, color="#1565C0")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def plot_feature_boxplots(df, feature_names, output_file, top_n=TOP_N_FEATURES):
    """Save box plots comparing healthy vs Parkinson's for the top features."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"'{TARGET_COLUMN}' column is required for box plots.")

    cols_to_plot = feature_names[:top_n]
    n_cols = 2
    n_rows = (len(cols_to_plot) + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3.5))
    axes = axes.flatten()

    healthy = df[df[TARGET_COLUMN] == 0]
    parkinsons = df[df[TARGET_COLUMN] == 1]

    for idx, feature in enumerate(cols_to_plot):
        ax = axes[idx]
        data = [healthy[feature].dropna(), parkinsons[feature].dropna()]
        bp = ax.boxplot(
            data,
            labels=["Healthy (0)", "Parkinson's (1)"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        bp["boxes"][0].set_facecolor("#A5D6A7")
        bp["boxes"][1].set_facecolor("#EF9A9A")
        ax.set_title(feature, fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)

    for ax in axes[len(cols_to_plot):]:
        ax.set_visible(False)

    fig.suptitle(
        "Feature Distribution: Healthy vs Parkinson's",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
