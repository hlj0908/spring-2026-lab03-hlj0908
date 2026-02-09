"""Main Pipeline Orchestration Script for Wine Classification."""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIRS: list[Path] = [
    Path("output/eda"),
    Path("output/features"),
    Path("output/models"),
    Path("output/evaluation"),
]


def _create_output_directories() -> None:
    """Create all output directories."""
    for directory in OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
    logging.info("Created all output directories")


def _run_eda() -> None:
    """Run exploratory data analysis."""
    from importlib import import_module

    logging.info("=" * 80)
    logging.info("STEP 1: Running Exploratory Data Analysis")
    logging.info("=" * 80)

    module = import_module("01_eda")
    module.run_eda()

    logging.info("EDA completed successfully\n")


def _run_feature_engineering() -> None:
    """Run feature engineering."""
    from importlib import import_module

    logging.info("=" * 80)
    logging.info("STEP 2: Running Feature Engineering")
    logging.info("=" * 80)

    module = import_module("02_feature_engineering")
    module.run_feature_engineering()

    logging.info("Feature engineering completed successfully\n")


def _run_model_training() -> None:
    """Run model training with hyperparameter tuning."""
    from importlib import import_module

    logging.info("=" * 80)
    logging.info("STEP 3: Running Model Training")
    logging.info("=" * 80)

    module = import_module("03_train_model")
    module.run_model_training()

    logging.info("Model training completed successfully\n")


def _run_evaluation() -> None:
    """Run model evaluation."""
    from importlib import import_module

    logging.info("=" * 80)
    logging.info("STEP 4: Running Model Evaluation")
    logging.info("=" * 80)

    module = import_module("04_evaluate_model")
    module.run_evaluation()

    logging.info("Model evaluation completed successfully\n")


def _run_report_generation() -> None:
    """Run report generation."""
    from importlib import import_module

    logging.info("=" * 80)
    logging.info("STEP 5: Generating Comprehensive Report")
    logging.info("=" * 80)

    module = import_module("05_generate_report")
    module.generate_report()

    logging.info("Report generation completed successfully\n")


def run_full_pipeline() -> None:
    """Run the complete pipeline from start to finish."""
    logging.info("Starting Wine Classification Pipeline")
    logging.info("=" * 80)

    _create_output_directories()

    _run_eda()
    _run_feature_engineering()
    _run_model_training()
    _run_evaluation()
    _run_report_generation()

    logging.info("=" * 80)
    logging.info("PIPELINE COMPLETED SUCCESSFULLY")
    logging.info("=" * 80)
    logging.info("Check output/wine_classification_report.md for full results")


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Wine Classification ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full pipeline
  python main.py --step eda         # Run only EDA
  python main.py --step training    # Run only training
  python main.py --step all         # Run full pipeline (same as no args)
        """,
    )

    parser.add_argument(
        "--step",
        type=str,
        choices=["eda", "features", "training", "evaluation", "report", "all"],
        default="all",
        help="Specific step to run (default: all)",
    )

    args = parser.parse_args()

    _create_output_directories()

    try:
        if args.step == "eda":
            _run_eda()
        elif args.step == "features":
            _run_feature_engineering()
        elif args.step == "training":
            _run_model_training()
        elif args.step == "evaluation":
            _run_evaluation()
        elif args.step == "report":
            _run_report_generation()
        elif args.step == "all":
            run_full_pipeline()
        else:
            logging.error(f"Unknown step: {args.step}")
            sys.exit(1)

    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
