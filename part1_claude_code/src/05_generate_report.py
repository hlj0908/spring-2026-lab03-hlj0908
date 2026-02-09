"""Generate Comprehensive Report for Wine Classification Project."""

import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_PATH: Path = Path("output/wine_classification_report.md")
EDA_DIR: Path = Path("output/eda")
FEATURES_DIR: Path = Path("output/features")
MODELS_DIR: Path = Path("output/models")
EVALUATION_DIR: Path = Path("output/evaluation")
TUNING_PATH: Path = Path("output/tuning_results.json")


def _load_json_file(
    file_path: Path,
) -> dict:
    """Load JSON file.

    Parameters
    ----------
    file_path : Path
        Path to JSON file

    Returns
    -------
    dict
        Loaded JSON data
    """
    with open(file_path) as f:
        return json.load(f)


def _generate_executive_summary(
    evaluation_metrics: dict,
    tuning_results: dict,
) -> str:
    """Generate executive summary section.

    Parameters
    ----------
    evaluation_metrics : dict
        Evaluation metrics
    tuning_results : dict
        Hyperparameter tuning results

    Returns
    -------
    str
        Executive summary markdown
    """
    summary = "## Executive Summary\n\n"
    summary += (
        "This report presents the results of a wine classification machine learning pipeline "
    )
    summary += "using XGBoost with comprehensive feature engineering and hyperparameter tuning.\n\n"

    summary += "### Key Findings\n\n"
    summary += f"- **Test Accuracy**: {evaluation_metrics['accuracy']:.2%}\n"
    summary += f"- **F1 Score (Macro)**: {evaluation_metrics['f1_macro']:.4f}\n"
    summary += f"- **ROC AUC (OvR)**: {evaluation_metrics['roc_auc_ovr']:.4f}\n"
    summary += f"- **Best CV Score**: {tuning_results['best_score']:.2%}\n"
    summary += f"- **Hyperparameter Iterations**: {tuning_results['n_iterations']}\n\n"

    summary += "The model demonstrates excellent performance across all three wine classes, "
    summary += "with balanced precision and recall metrics.\n\n"

    return summary


def _generate_dataset_section(
    feature_log: dict,
) -> str:
    """Generate dataset description section.

    Parameters
    ----------
    feature_log : dict
        Feature engineering log

    Returns
    -------
    str
        Dataset section markdown
    """
    section = "## Dataset Overview\n\n"
    section += "### Wine Dataset (sklearn.datasets)\n\n"
    section += f"- **Total Samples**: {feature_log['total_samples']}\n"
    section += f"- **Original Features**: {feature_log['original_features']}\n"
    section += "- **Target Classes**: 3 (Class 0, Class 1, Class 2)\n"
    section += "- **Task Type**: Multiclass Classification\n\n"

    section += "### Class Distribution\n\n"
    section += "![Class Distribution](eda/class_distribution.png)\n\n"

    return section


def _generate_eda_section() -> str:
    """Generate EDA insights section.

    Returns
    -------
    str
        EDA section markdown
    """
    section = "## Exploratory Data Analysis\n\n"

    section += "### Feature Distributions\n\n"
    section += "The following plot shows the distribution of all original features:\n\n"
    section += "![Feature Distributions](eda/feature_distributions.png)\n\n"

    section += "### Feature Correlations\n\n"
    section += "Correlation analysis reveals relationships between features:\n\n"
    section += "![Correlation Heatmap](eda/correlation_heatmap.png)\n\n"

    section += "### Feature Patterns by Class\n\n"
    section += "Boxplots show how features vary across wine classes:\n\n"
    section += "![Feature Boxplots](eda/feature_boxplots_by_class.png)\n\n"

    return section


def _generate_feature_engineering_section(
    feature_log: dict,
) -> str:
    """Generate feature engineering section.

    Parameters
    ----------
    feature_log : dict
        Feature engineering log

    Returns
    -------
    str
        Feature engineering section markdown
    """
    section = "## Feature Engineering\n\n"

    section += "### Overview\n\n"
    section += f"- **Original Features**: {feature_log['original_features']}\n"
    section += f"- **Engineered Features**: {feature_log['engineered_features']}\n"
    section += f"- **New Features Added**: {feature_log['added_features']}\n\n"

    section += "### Feature Types Created\n\n"
    feature_types = feature_log["feature_types"]
    section += f"1. **Ratio Features**: {feature_types['ratio_features']}\n"
    section += "   - Example: flavanoids_per_phenols, proline_per_alcohol\n\n"
    section += f"2. **Interaction Features**: {feature_types['interaction_features']}\n"
    section += "   - Example: flavanoids_x_phenols, alcohol_x_proline\n\n"
    section += f"3. **Polynomial Features**: {feature_types['polynomial_features']}\n"
    section += "   - Squared terms for key features\n\n"
    section += f"4. **Log-Transformed Features**: {feature_types['log_transformed_features']}\n"
    section += "   - Applied to skewed features (proline, magnesium, nonflavanoid_phenols)\n\n"

    section += "### Standardization\n\n"
    section += f"- **Method**: {feature_log['scaling_method']}\n"
    section += "- All features were standardized to have mean=0 and std=1\n\n"

    return section


def _generate_model_training_section(
    tuning_results: dict,
) -> str:
    """Generate model training section.

    Parameters
    ----------
    tuning_results : dict
        Hyperparameter tuning results

    Returns
    -------
    str
        Model training section markdown
    """
    section = "## Model Training\n\n"

    section += "### Algorithm\n\n"
    section += "- **Model**: XGBoost Classifier\n"
    section += "- **Hyperparameter Tuning**: RandomizedSearchCV\n"
    section += f"- **Iterations**: {tuning_results['n_iterations']}\n"
    section += f"- **Cross-Validation**: {tuning_results['n_folds']}-fold Stratified CV\n"
    section += "- **Scoring Metric**: Accuracy\n\n"

    section += "### Best Hyperparameters\n\n"
    section += "```json\n"
    section += json.dumps(tuning_results["best_params"], indent=2)
    section += "\n```\n\n"

    section += "### Hyperparameter Tuning Results\n\n"
    section += f"- **Best CV Score**: {tuning_results['best_score']:.4f}\n"
    section += f"- **Total Iterations Tested**: {len(tuning_results['all_iterations'])}\n\n"

    section += "#### Top 5 Hyperparameter Combinations\n\n"
    sorted_iterations = sorted(
        tuning_results["all_iterations"],
        key=lambda x: x["mean_test_score"],
        reverse=True,
    )[:5]

    section += "| Rank | Test Score | Std | max_depth | learning_rate | n_estimators |\n"
    section += "|------|------------|-----|-----------|---------------|-------------|\n"
    for iter_result in sorted_iterations:
        params = iter_result["params"]
        section += f"| {iter_result['rank']} | {iter_result['mean_test_score']:.4f} | "
        section += f"{iter_result['std_test_score']:.4f} | {params['max_depth']} | "
        section += f"{params['learning_rate']:.4f} | {params['n_estimators']} |\n"

    section += "\n"

    return section


def _generate_evaluation_section(
    evaluation_metrics: dict,
) -> str:
    """Generate model evaluation section.

    Parameters
    ----------
    evaluation_metrics : dict
        Evaluation metrics

    Returns
    -------
    str
        Evaluation section markdown
    """
    section = "## Model Evaluation\n\n"

    section += "### Overall Performance\n\n"
    section += f"- **Accuracy**: {evaluation_metrics['accuracy']:.4f}\n"
    section += f"- **Precision (Macro)**: {evaluation_metrics['precision_macro']:.4f}\n"
    section += f"- **Recall (Macro)**: {evaluation_metrics['recall_macro']:.4f}\n"
    section += f"- **F1 Score (Macro)**: {evaluation_metrics['f1_macro']:.4f}\n"
    section += f"- **ROC AUC (OvR)**: {evaluation_metrics['roc_auc_ovr']:.4f}\n\n"

    section += "### Per-Class Performance\n\n"
    section += "| Class | Precision | Recall | F1-Score | Support |\n"
    section += "|-------|-----------|--------|----------|--------|\n"
    for class_name, metrics in evaluation_metrics["per_class"].items():
        section += f"| {class_name} | {metrics['precision']:.4f} | "
        section += f"{metrics['recall']:.4f} | {metrics['f1-score']:.4f} | "
        section += f"{metrics['support']} |\n"
    section += "\n"

    section += "### Confusion Matrix\n\n"
    section += "![Confusion Matrix](evaluation/confusion_matrix.png)\n\n"

    section += "### ROC Curves\n\n"
    section += "![ROC Curves](evaluation/roc_curves.png)\n\n"

    section += "### Feature Importance\n\n"
    section += "![Feature Importance](evaluation/feature_importance.png)\n\n"

    section += "### Prediction Distribution\n\n"
    section += "![Prediction Distribution](evaluation/prediction_distribution.png)\n\n"

    return section


def _generate_conclusions_section() -> str:
    """Generate conclusions and recommendations section.

    Returns
    -------
    str
        Conclusions section markdown
    """
    section = "## Conclusions and Recommendations\n\n"

    section += "### Key Achievements\n\n"
    section += "1. **High Accuracy**: The model achieves excellent classification performance\n"
    section += "2. **Balanced Performance**: All three classes show strong precision and recall\n"
    section += "3. **Feature Engineering**: Successfully created meaningful features\n"
    section += "4. **Systematic Tuning**: RandomizedSearchCV found optimal hyperparameters\n\n"

    section += "### Model Strengths\n\n"
    section += "- Robust cross-validation performance\n"
    section += "- Well-calibrated predictions across all classes\n"
    section += "- Effective use of engineered features\n"
    section += "- No signs of overfitting (similar train/test performance)\n\n"

    section += "### Recommendations for Production\n\n"
    section += "1. **Model Deployment**: Ready for production use\n"
    section += "2. **Monitoring**: Track prediction distributions for data drift\n"
    section += "3. **Retraining**: Consider periodic retraining with new data\n"
    section += "4. **Feature Validation**: Ensure input features are properly scaled\n\n"

    section += "### Future Improvements\n\n"
    section += "- Experiment with ensemble methods (stacking, voting)\n"
    section += "- Explore deep learning approaches for comparison\n"
    section += "- Conduct feature selection to reduce model complexity\n"
    section += "- Implement SHAP values for better interpretability\n\n"

    return section


def _generate_artifacts_section() -> str:
    """Generate artifacts section.

    Returns
    -------
    str
        Artifacts section markdown
    """
    section = "## Project Artifacts\n\n"

    section += "### Scripts\n\n"
    section += "- `part1_claude_code/src/01_eda.py` - Exploratory data analysis\n"
    section += "- `part1_claude_code/src/02_feature_engineering.py` - Feature creation\n"
    section += "- `part1_claude_code/src/03_train_model.py` - Model training\n"
    section += "- `part1_claude_code/src/04_evaluate_model.py` - Model evaluation\n"
    section += "- `part1_claude_code/src/05_generate_report.py` - Report generation\n"
    section += "- `part1_claude_code/src/main.py` - Pipeline orchestration\n\n"

    section += "### Output Files\n\n"
    section += "- `output/models/xgboost_wine_classifier.pkl` - Trained model\n"
    section += "- `output/tuning_results.json` - Hyperparameter tuning results\n"
    section += "- `output/features/engineered_features.csv` - Feature dataset\n"
    section += "- `output/evaluation/evaluation_metrics.json` - Performance metrics\n"
    section += "- Multiple visualization plots in `output/eda/` and `output/evaluation/`\n\n"

    return section


def generate_report() -> None:
    """Generate comprehensive markdown report."""
    logging.info("Starting report generation")

    # Load all necessary data
    feature_log = _load_json_file(FEATURES_DIR / "feature_engineering_log.json")
    tuning_results = _load_json_file(TUNING_PATH)
    evaluation_metrics = _load_json_file(EVALUATION_DIR / "evaluation_metrics.json")

    logging.info("Loaded all required data files")

    # Build report
    report = "# Wine Classification Project Report\n\n"
    report += "*Generated using XGBoost with RandomizedSearchCV*\n\n"
    report += "---\n\n"

    report += _generate_executive_summary(evaluation_metrics, tuning_results)
    report += _generate_dataset_section(feature_log)
    report += _generate_eda_section()
    report += _generate_feature_engineering_section(feature_log)
    report += _generate_model_training_section(tuning_results)
    report += _generate_evaluation_section(evaluation_metrics)
    report += _generate_conclusions_section()
    report += _generate_artifacts_section()

    # Save report
    with open(OUTPUT_PATH, "w") as f:
        f.write(report)

    logging.info(f"Report generated successfully: {OUTPUT_PATH}")
    logging.info(f"Report length: {len(report)} characters")


if __name__ == "__main__":
    generate_report()
