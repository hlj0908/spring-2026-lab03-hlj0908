"""Model Training with Hyperparameter Tuning using RandomizedSearchCV."""

import json
import logging
import pickle
from pathlib import Path

import polars as pl
from scipy.stats import uniform
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

FEATURES_PATH: Path = Path("output/features/engineered_features.csv")
OUTPUT_DIR: Path = Path("output/models")
TUNING_OUTPUT_PATH: Path = Path("output/tuning_results.json")
RANDOM_SEED: int = 42
N_ITER: int = 20
N_FOLDS: int = 5
TEST_SIZE: float = 0.2


def _create_output_directory() -> None:
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TUNING_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory created/verified: {OUTPUT_DIR}")


def _load_engineered_features() -> tuple[pl.DataFrame, pl.Series]:
    """Load engineered features from CSV.

    Returns
    -------
    tuple[pl.DataFrame, pl.Series]
        Features and target
    """
    df = pl.read_csv(FEATURES_PATH)
    X = df.drop("target")
    y = df["target"]

    logging.info(f"Loaded features: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def _create_hyperparameter_space() -> dict:
    """Create hyperparameter search space for RandomizedSearchCV.

    Returns
    -------
    dict
        Hyperparameter distributions
    """
    param_distributions = {
        "max_depth": [3, 5, 7, 9],
        "learning_rate": uniform(0.01, 0.29),
        "n_estimators": [50, 100, 150, 200, 250],
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
    }

    logging.info("Created hyperparameter search space")
    return param_distributions


def _perform_randomized_search(
    X_train: pl.DataFrame,
    y_train: pl.Series,
) -> tuple[RandomizedSearchCV, dict]:
    """Perform randomized search with cross-validation.

    Parameters
    ----------
    X_train : pl.DataFrame
        Training features
    y_train : pl.Series
        Training target

    Returns
    -------
    tuple[RandomizedSearchCV, dict]
        Fitted RandomizedSearchCV object and results dictionary
    """
    param_distributions = _create_hyperparameter_space()

    base_model = XGBClassifier(
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )

    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=1,
        return_train_score=True,
    )

    logging.info(f"Starting RandomizedSearchCV with {N_ITER} iterations and {N_FOLDS}-fold CV")

    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()

    random_search.fit(X_train_np, y_train_np)

    logging.info("RandomizedSearchCV completed")

    # Extract detailed results
    results = {
        "best_params": random_search.best_params_,
        "best_score": float(random_search.best_score_),
        "n_iterations": N_ITER,
        "n_folds": N_FOLDS,
        "all_iterations": [],
    }

    cv_results = random_search.cv_results_

    for i in range(len(cv_results["params"])):
        iteration_result = {
            "iteration": i + 1,
            "params": cv_results["params"][i],
            "mean_test_score": float(cv_results["mean_test_score"][i]),
            "std_test_score": float(cv_results["std_test_score"][i]),
            "mean_train_score": float(cv_results["mean_train_score"][i]),
            "std_train_score": float(cv_results["std_train_score"][i]),
            "rank": int(cv_results["rank_test_score"][i]),
        }
        results["all_iterations"].append(iteration_result)

    logging.info(f"Best CV score: {results['best_score']:.4f}")
    logging.info("Best parameters:\n" + json.dumps(results["best_params"], indent=2, default=str))

    return random_search, results


def _train_final_model(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    best_params: dict,
) -> XGBClassifier:
    """Train final model with best hyperparameters.

    Parameters
    ----------
    X_train : pl.DataFrame
        Training features
    y_train : pl.Series
        Training target
    best_params : dict
        Best hyperparameters from randomized search

    Returns
    -------
    XGBClassifier
        Trained model
    """
    model = XGBClassifier(
        **best_params,
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )

    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()

    model.fit(X_train_np, y_train_np)

    logging.info("Final model trained with best hyperparameters")
    return model


def run_model_training() -> None:
    """Run complete model training pipeline."""
    logging.info("Starting model training with hyperparameter tuning")

    _create_output_directory()

    X, y = _load_engineered_features()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    logging.info(f"Train set: {len(y_train)} samples")
    logging.info(f"Test set: {len(y_test)} samples")

    # Perform randomized search
    random_search, results = _perform_randomized_search(X_train, y_train)

    # Train final model
    final_model = _train_final_model(X_train, y_train, results["best_params"])

    # Save trained model
    model_path = OUTPUT_DIR / "xgboost_wine_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    logging.info(f"Saved trained model to {model_path}")

    # Save tuning results
    with open(TUNING_OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logging.info(f"Saved tuning results to {TUNING_OUTPUT_PATH}")

    # Save best parameters separately
    best_params_path = OUTPUT_DIR / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(results["best_params"], f, indent=2, default=str)
    logging.info(f"Saved best parameters to {best_params_path}")

    # Save train/test split indices for reproducibility
    split_info = {
        "train_size": len(y_train),
        "test_size": len(y_test),
        "test_size_ratio": TEST_SIZE,
        "random_seed": RANDOM_SEED,
        "stratified": True,
    }
    split_path = OUTPUT_DIR / "train_test_split.json"
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2, default=str)
    logging.info(f"Saved split information to {split_path}")

    logging.info("Model training completed successfully")


if __name__ == "__main__":
    run_model_training()
