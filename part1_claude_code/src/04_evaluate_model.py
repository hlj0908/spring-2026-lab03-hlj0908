"""Model Evaluation and Performance Analysis."""

import json
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

FEATURES_PATH: Path = Path("output/features/engineered_features.csv")
MODEL_PATH: Path = Path("output/models/xgboost_wine_classifier.pkl")
OUTPUT_DIR: Path = Path("output/evaluation")
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.2
CLASS_NAMES: list[str] = ["Class 0", "Class 1", "Class 2"]


def _create_output_directory() -> None:
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory created/verified: {OUTPUT_DIR}")


def _load_model_and_data() -> tuple:
    """Load trained model and test data.

    Returns
    -------
    tuple
        Model, X_test, y_test, feature_names
    """
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logging.info(f"Loaded trained model from {MODEL_PATH}")

    df = pl.read_csv(FEATURES_PATH)
    X = df.drop("target")
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    logging.info(f"Test set: {len(y_test)} samples")

    return model, X_test, y_test, X.columns


def _generate_predictions(
    model,
    X_test: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions and probabilities.

    Parameters
    ----------
    model
        Trained XGBoost model
    X_test : pl.DataFrame
        Test features

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Predictions and probabilities
    """
    X_test_np = X_test.to_numpy()

    y_pred = model.predict(X_test_np)
    y_pred_proba = model.predict_proba(X_test_np)

    logging.info("Generated predictions for test set")
    return y_pred, y_pred_proba


def _calculate_metrics(
    y_test: pl.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict:
    """Calculate evaluation metrics.

    Parameters
    ----------
    y_test : pl.Series
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_pred_proba : np.ndarray
        Prediction probabilities

    Returns
    -------
    dict
        Evaluation metrics
    """
    y_test_np = y_test.to_numpy()

    metrics = {
        "accuracy": float(accuracy_score(y_test_np, y_pred)),
        "precision_macro": float(precision_score(y_test_np, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_test_np, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_test_np, y_pred, average="macro")),
        "precision_weighted": float(precision_score(y_test_np, y_pred, average="weighted")),
        "recall_weighted": float(recall_score(y_test_np, y_pred, average="weighted")),
        "f1_weighted": float(f1_score(y_test_np, y_pred, average="weighted")),
    }

    # Calculate ROC AUC (one-vs-rest for multiclass)
    y_test_binarized = label_binarize(y_test_np, classes=[0, 1, 2])
    metrics["roc_auc_ovr"] = float(
        roc_auc_score(y_test_binarized, y_pred_proba, multi_class="ovr", average="macro")
    )

    # Per-class metrics
    class_report = classification_report(y_test_np, y_pred, output_dict=True)
    metrics["per_class"] = {}
    for i in range(3):
        metrics["per_class"][f"class_{i}"] = {
            "precision": float(class_report[str(i)]["precision"]),
            "recall": float(class_report[str(i)]["recall"]),
            "f1-score": float(class_report[str(i)]["f1-score"]),
            "support": int(class_report[str(i)]["support"]),
        }

    logging.info("Calculated evaluation metrics")
    logging.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logging.info(f"Test F1 (macro): {metrics['f1_macro']:.4f}")

    return metrics


def _plot_confusion_matrix(
    y_test: pl.Series,
    y_pred: np.ndarray,
) -> None:
    """Plot confusion matrix.

    Parameters
    ----------
    y_test : pl.Series
        True labels
    y_pred : np.ndarray
        Predicted labels
    """
    cm = confusion_matrix(y_test.to_numpy(), y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix", fontsize=14, pad=20)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved confusion matrix to {output_path}")


def _plot_roc_curves(
    y_test: pl.Series,
    y_pred_proba: np.ndarray,
) -> None:
    """Plot ROC curves for each class.

    Parameters
    ----------
    y_test : pl.Series
        True labels
    y_pred_proba : np.ndarray
        Prediction probabilities
    """
    y_test_np = y_test.to_numpy()
    y_test_binarized = label_binarize(y_test_np, classes=[0, 1, 2])

    plt.figure(figsize=(10, 8))

    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        auc = roc_auc_score(y_test_binarized[:, i], y_pred_proba[:, i])
        plt.plot(fpr, tpr, linewidth=2, label=f"{CLASS_NAMES[i]} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves (One-vs-Rest)", fontsize=14, pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "roc_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved ROC curves to {output_path}")


def _plot_feature_importance(
    model,
    feature_names: list[str],
) -> None:
    """Plot feature importance.

    Parameters
    ----------
    model
        Trained XGBoost model
    feature_names : list[str]
        List of feature names
    """
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:20]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importance[indices], color="steelblue", edgecolor="black")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=9)
    plt.xlabel("Importance", fontsize=12)
    plt.title("Top 20 Feature Importances", fontsize=14, pad=20)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "feature_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved feature importance plot to {output_path}")


def _plot_prediction_distribution(
    y_test: pl.Series,
    y_pred: np.ndarray,
) -> None:
    """Plot prediction distribution by class.

    Parameters
    ----------
    y_test : pl.Series
        True labels
    y_pred : np.ndarray
        Predicted labels
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # True distribution
    unique, counts = np.unique(y_test.to_numpy(), return_counts=True)
    axes[0].bar(unique, counts, color=["#1f77b4", "#ff7f0e", "#2ca02c"], edgecolor="black")
    axes[0].set_xlabel("Class", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("True Label Distribution", fontsize=12)
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(CLASS_NAMES)

    # Predicted distribution
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    axes[1].bar(
        unique_pred, counts_pred, color=["#1f77b4", "#ff7f0e", "#2ca02c"], edgecolor="black"
    )
    axes[1].set_xlabel("Class", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Predicted Label Distribution", fontsize=12)
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(CLASS_NAMES)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "prediction_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved prediction distribution to {output_path}")


def run_evaluation() -> None:
    """Run complete model evaluation."""
    logging.info("Starting model evaluation")

    _create_output_directory()

    model, X_test, y_test, feature_names = _load_model_and_data()

    y_pred, y_pred_proba = _generate_predictions(model, X_test)

    metrics = _calculate_metrics(y_test, y_pred, y_pred_proba)

    # Save metrics
    metrics_path = OUTPUT_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logging.info(f"Saved evaluation metrics to {metrics_path}")

    # Generate all plots
    _plot_confusion_matrix(y_test, y_pred)
    _plot_roc_curves(y_test, y_pred_proba)
    _plot_feature_importance(model, feature_names)
    _plot_prediction_distribution(y_test, y_pred)

    logging.info("Model evaluation completed successfully")


if __name__ == "__main__":
    run_evaluation()
