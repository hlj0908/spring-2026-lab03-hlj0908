import logging
from pathlib import Path
from typing import Any, Tuple

import polars as pl
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Constants
RANDOM_STATE: int = 42
OUTPUT_DIR: Path = Path("part2_antigravity/output/processed")
TEST_SIZE: float = 0.2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)


def _load_data() -> Tuple[pl.DataFrame, pl.Series]:
    """Load Wine dataset and convert to Polars DataFrame."""
    logging.info("Loading Wine dataset...")
    wine_data: Any = load_wine(as_frame=True)
    df_pandas = wine_data.data
    target_pandas = wine_data.target

    X = pl.from_pandas(df_pandas)
    y = pl.from_pandas(target_pandas).alias("target")

    logging.info(f"Loaded dataset with shape: {X.shape}")
    return X, y


def _create_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create new features from existing ones."""
    logging.info("Creating derived features...")

    # Check if columns ensure existence (Wine dataset columns are standard but good to be safe)
    # columns: alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols,
    # flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity, hue,
    # od280/od315_of_diluted_wines, proline

    df_derived = df.with_columns(
        [
            (pl.col("proline") / pl.col("magnesium")).alias("proline_magnesium_ratio"),
            (pl.col("flavanoids") * pl.col("color_intensity")).alias("flavanoid_color_interaction"),
            (pl.col("alcohol") / pl.col("ash")).alias("alcohol_ash_ratio"),
        ]
    )

    logging.info(f"Added 3 derived features. New shape: {df_derived.shape}")
    return df_derived


def _split_and_scale_data(X: pl.DataFrame, y: pl.Series) -> None:
    """Split data into train/test and scale features."""
    logging.info("Splitting data into train/set sets...")

    # Conversion to numpy/pandas for sklearn split
    # (polars support in sklearn is experimental/limited for some versions)
    # Using to_pandas() is safe.
    X_train_pd, X_test_pd, y_train_pd, y_test_pd = train_test_split(
        X.to_pandas(),
        y.to_pandas(),
        test_size=TEST_SIZE,
        stratify=y.to_pandas(),
        random_state=RANDOM_STATE,
    )

    logging.info("Scaling features...")
    scaler = StandardScaler()

    # Fit on train, transform both
    X_train_scaled_np = scaler.fit_transform(X_train_pd)
    X_test_scaled_np = scaler.transform(X_test_pd)

    # Convert back to Polars, preserving column names
    columns = X.columns
    X_train = pl.DataFrame(X_train_scaled_np, schema=columns, orient="row")
    X_test = pl.DataFrame(X_test_scaled_np, schema=columns, orient="row")

    y_train = pl.Series(y_train_pd).alias("target")
    y_test = pl.Series(y_test_pd).alias("target")

    _save_processed_data(X_train, X_test, y_train, y_test)


def _save_processed_data(
    X_train: pl.DataFrame, X_test: pl.DataFrame, y_train: pl.Series, y_test: pl.Series
) -> None:
    """Save processed datasets to parquet files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Saving processed data to parquet...")
    X_train.write_parquet(OUTPUT_DIR / "X_train.parquet")
    X_test.write_parquet(OUTPUT_DIR / "X_test.parquet")
    pl.DataFrame(y_train).write_parquet(OUTPUT_DIR / "y_train.parquet")
    pl.DataFrame(y_test).write_parquet(OUTPUT_DIR / "y_test.parquet")

    logging.info(f"Saved processed files to {OUTPUT_DIR}")


def run_feature_engineering() -> None:
    """Execute feature engineering pipeline."""
    X, y = _load_data()
    X_derived = _create_derived_features(X)
    _split_and_scale_data(X_derived, y)
    logging.info("Feature engineering completed successfully.")


if __name__ == "__main__":
    run_feature_engineering()
