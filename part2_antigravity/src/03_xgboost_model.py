import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
import polars as pl
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

# Constants
RANDOM_STATE: int = 42
PROCESSED_DIR: Path = Path("part2_antigravity/output/processed")
OUTPUT_DIR: Path = Path("part2_antigravity/output/model")
TUNING_RESULTS_FILE: Path = Path("part2_antigravity/output/tuning_results.json")
MODEL_FILE: Path = OUTPUT_DIR / "xgb_model.joblib"
N_ITER: int = 20
CV_FOLDS: int = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)


def _load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load processed training data."""
    logging.info("Loading processed training data...")
    X_train = pl.read_parquet(PROCESSED_DIR / "X_train.parquet").to_pandas()
    y_train = pl.read_parquet(PROCESSED_DIR / "y_train.parquet")["target"].to_pandas()

    logging.info(f"Loaded train shape: {X_train.shape}")
    return X_train, y_train


def _tune_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """Perform hyperparameter tuning using RandomizedSearchCV."""
    logging.info("Starting hyperparameter tuning...")

    param_dist: Dict[str, Any] = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.5],
        "min_child_weight": [1, 3, 5, 7],
    }

    xgb = XGBClassifier(
        objective="multi:softmax", random_state=RANDOM_STATE, n_jobs=-1, eval_metric="mlogloss"
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring="accuracy",
        cv=cv,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)

    best_params: Dict[str, Any] = random_search.best_params_
    best_score: float = random_search.best_score_

    logging.info(f"Best CV Accuracy: {best_score:.4f}")
    logging.info(f"Best Parameters: {json.dumps(best_params, indent=2)}")

    # Save tuning results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {
        "best_params": best_params,
        "best_score": best_score,
        "cv_results_summary": f"Best of {N_ITER} iterations across {CV_FOLDS} folds",
    }

    with open(TUNING_RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    return best_params


def run_training() -> None:
    """Execute the training pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = _load_data()

    best_params = _tune_hyperparameters(X_train, y_train)

    logging.info("Retraining model with best parameters...")
    final_model = XGBClassifier(
        **best_params,
        objective="multi:softmax",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="mlogloss",
    )

    final_model.fit(X_train, y_train)

    logging.info(f"Saving model to {MODEL_FILE}...")
    joblib.dump(final_model, MODEL_FILE)

    logging.info("Model training completed successfully.")


if __name__ == "__main__":
    run_training()
