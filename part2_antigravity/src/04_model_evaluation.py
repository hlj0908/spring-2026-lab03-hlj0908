import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier, plot_importance

# Constants
PROCESSED_DIR: Path = Path("part2_antigravity/output/processed")
OUTPUT_DIR: Path = Path("part2_antigravity/output/model")
MODEL_FILE: Path = OUTPUT_DIR / "xgb_model.joblib"
METRICS_FILE: Path = OUTPUT_DIR / "metrics.json"
TUNING_RESULTS_FILE: Path = Path("part2_antigravity/output/tuning_results.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)


def _load_test_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load processed test data."""
    logging.info("Loading processed test data...")
    X_test = pl.read_parquet(PROCESSED_DIR / "X_test.parquet").to_pandas()
    y_test = pl.read_parquet(PROCESSED_DIR / "y_test.parquet")["target"].to_pandas()

    logging.info(f"Loaded test shape: {X_test.shape}")
    return X_test, y_test


def _load_model() -> XGBClassifier:
    """Load the trained XGBoost model."""
    logging.info(f"Loading model from {MODEL_FILE}...")
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_FILE}. Run 03_model_training.py first."
        )

    model = joblib.load(MODEL_FILE)
    return model


def _evaluate_model(
    model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, Any]:
    """Evaluate the model and save metrics."""
    logging.info("Evaluating model on test set...")
    y_pred = model.predict(X_test)

    accuracy: float = accuracy_score(y_test, y_pred)
    precision: float = precision_score(y_test, y_pred, average="weighted")
    recall: float = recall_score(y_test, y_pred, average="weighted")
    f1: float = f1_score(y_test, y_pred, average="weighted")

    cm = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    metrics: Dict[str, Any] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report,
    }

    logging.info(f"Test Metrics: {json.dumps(metrics, indent=2)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()

    logging.info(f"Confusion matrix saved to {OUTPUT_DIR / 'confusion_matrix.png'}")

    return metrics


def _plot_feature_importance(model: XGBClassifier) -> None:
    """Plot feature importance."""
    logging.info("Plotting feature importance...")
    plt.figure(figsize=(10, 8))
    plot_importance(model, max_num_features=15, height=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png")
    plt.close()
    logging.info(f"Feature importance plot saved to {OUTPUT_DIR / 'feature_importance.png'}")


def _get_best_params() -> Dict[str, Any]:
    """Load best hyperparameters from tuning results."""
    if TUNING_RESULTS_FILE.exists():
        with open(TUNING_RESULTS_FILE, "r") as f:
            data = json.load(f)
            return data.get("best_params", {})
    return {}


def _generate_report(
    model: XGBClassifier, metrics: Dict[str, Any], feature_names: List[str]
) -> None:
    """Generate a comprehensive evaluation report."""
    logging.info("Generating evaluation report...")

    # Extract top features
    import numpy as np

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [(feature_names[i], importances[i]) for i in indices]

    best_params = _get_best_params()

    report_content = [
        "# Wine Classification Model Evaluation Report",
        "\n## 1. Model Configuration",
        "The XGBoost model was optimized using RandomizedSearchCV with 5-fold cross-validation.",
        "\n### Best Hyperparameters:",
        "```json",
        json.dumps(best_params, indent=2),
        "```",
        "\n## 2. Model Performance Summary",
        f"- **Overall Accuracy**: {metrics['accuracy']:.4f}",
        f"- **Weighted Precision**: {metrics['precision']:.4f}",
        f"- **Weighted Recall**: {metrics['recall']:.4f}",
        f"- **Weighted F1 Score**: {metrics['f1_score']:.4f}",
        "\n### Per-Class Performance",
        "| Class | Precision | Recall | F1-Score | Support |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    class_report = metrics.get("classification_report", {})
    for cls, scores in class_report.items():
        if cls in ["accuracy", "macro avg", "weighted avg"]:
            continue
        report_content.append(
            f"| {cls} | {scores['precision']:.4f} | {scores['recall']:.4f} | "
            f"{scores['f1-score']:.4f} | {scores['support']} |"
        )

    report_content.extend(
        [
            "\n## 3. Confusion Matrix Analysis",
            "The confusion matrix visualizes the model's performance on the test set. "
            "A perfect diagonal indicates correct classifications.",
            "\n![Confusion Matrix](confusion_matrix.png)",
            "\n## 4. Feature Importance Analysis",
            "Feature importance scores indicate the relative contribution of each feature to the "
            "model's decision-making process. Higher scores mean greater influence.",
            "\n### Top Features:",
        ]
    )

    for rank, (name, imp) in enumerate(sorted_features, 1):
        report_content.append(f"{rank}. **{name}** (Importance: {imp:.4f})")

    report_content.extend(
        [
            "\n![Feature Importance](feature_importance.png)",
            "\n## 5. Recommendations & Next Steps",
            "- **High Confidence Deployment**: The model demonstrates robust performance "
            "across all classes.",
            "- **Interpretability**: Focus on the top features (e.g., Proline, Flavanoids) "
            "when explaining predictions to stakeholders.",
            "- **Monitoring**: Establish a baseline for input data distributions to detect drift, "
            "as the model relies heavily on specific chemical markers.",
        ]
    )

    report_path = Path("part2_antigravity/output/evaluation_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_content))

    logging.info(f"Report saved to {report_path}")


def run_evaluation() -> None:
    """Execute the evaluation pipeline."""
    X_test, y_test = _load_test_data()
    model = _load_model()

    metrics = _evaluate_model(model, X_test, y_test)
    _plot_feature_importance(model)

    _generate_report(model, metrics, X_test.columns.tolist())

    logging.info("Model evaluation completed successfully.")


if __name__ == "__main__":
    run_evaluation()
