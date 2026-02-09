"""Feature Engineering for Wine Classification Dataset."""

import json
import logging
from pathlib import Path

import polars as pl
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: Path = Path("output/features")
RANDOM_SEED: int = 42


def _create_output_directory() -> None:
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory created/verified: {OUTPUT_DIR}")


def _load_wine_data() -> pl.DataFrame:
    """Load wine dataset and convert to polars DataFrame.

    Returns
    -------
    pl.DataFrame
        DataFrame with features and target
    """
    wine = load_wine()

    df = pl.DataFrame(
        wine.data,
        schema=wine.feature_names,
    )
    df = df.with_columns(pl.Series("target", wine.target))

    logging.info(f"Loaded wine dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    return df


def _create_ratio_features(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Create ratio features from existing features.

    Parameters
    ----------
    df : pl.DataFrame
        Original wine dataset

    Returns
    -------
    pl.DataFrame
        DataFrame with ratio features added
    """
    df = df.with_columns(
        [
            (pl.col("flavanoids") / (pl.col("total_phenols") + 1e-6)).alias(
                "flavanoids_per_phenols"
            ),
            (pl.col("od280/od315_of_diluted_wines") / (pl.col("color_intensity") + 1e-6)).alias(
                "od_ratio_per_color"
            ),
            (pl.col("proline") / (pl.col("alcohol") + 1e-6)).alias("proline_per_alcohol"),
            (pl.col("malic_acid") / (pl.col("ash") + 1e-6)).alias("malic_per_ash"),
        ]
    )

    logging.info("Created 4 ratio features")
    return df


def _create_interaction_features(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Create interaction features from highly correlated features.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with original features

    Returns
    -------
    pl.DataFrame
        DataFrame with interaction features added
    """
    df = df.with_columns(
        [
            (pl.col("flavanoids") * pl.col("total_phenols")).alias("flavanoids_x_phenols"),
            (pl.col("alcohol") * pl.col("proline")).alias("alcohol_x_proline"),
            (pl.col("od280/od315_of_diluted_wines") * pl.col("flavanoids")).alias(
                "od_x_flavanoids"
            ),
            (pl.col("color_intensity") * pl.col("hue")).alias("color_x_hue"),
        ]
    )

    logging.info("Created 4 interaction features")
    return df


def _create_polynomial_features(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Create polynomial features (degree 2) for key features.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with original features

    Returns
    -------
    pl.DataFrame
        DataFrame with polynomial features added
    """
    key_features = ["flavanoids", "proline", "alcohol", "od280/od315_of_diluted_wines"]

    for feature in key_features:
        df = df.with_columns(
            [
                (pl.col(feature) ** 2).alias(f"{feature}_squared"),
            ]
        )

    logging.info(f"Created polynomial features for {len(key_features)} features")
    return df


def _apply_log_transform(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Apply log transformation to skewed features.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with original features

    Returns
    -------
    pl.DataFrame
        DataFrame with log-transformed features added
    """
    skewed_features = ["proline", "magnesium", "nonflavanoid_phenols"]

    for feature in skewed_features:
        df = df.with_columns(
            [
                (pl.col(feature) + 1).log().alias(f"{feature}_log"),
            ]
        )

    logging.info(f"Applied log transformation to {len(skewed_features)} features")
    return df


def _standardize_features(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, dict[str, float]]]:
    """Standardize all features (except target).

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with all features

    Returns
    -------
    tuple[pl.DataFrame, dict[str, dict[str, float]]]
        Standardized DataFrame and scaling parameters
    """
    feature_cols = [col for col in df.columns if col != "target"]
    target = df["target"]

    # Convert to numpy for standardization
    feature_array = df.select(feature_cols).to_numpy()

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(feature_array)

    # Create new DataFrame with scaled features
    scaled_df = pl.DataFrame(
        scaled_array,
        schema=feature_cols,
    )
    scaled_df = scaled_df.with_columns(target.alias("target"))

    # Store scaling parameters
    scaling_params = {}
    for idx, col in enumerate(feature_cols):
        scaling_params[col] = {
            "mean": float(scaler.mean_[idx]),
            "std": float(scaler.scale_[idx]),
        }

    logging.info(f"Standardized {len(feature_cols)} features")
    return scaled_df, scaling_params


def run_feature_engineering() -> None:
    """Run complete feature engineering pipeline."""
    logging.info("Starting feature engineering")

    _create_output_directory()

    df = _load_wine_data()

    original_feature_count = len([col for col in df.columns if col != "target"])
    logging.info(f"Original feature count: {original_feature_count}")

    # Create new features
    df = _create_ratio_features(df)
    df = _create_interaction_features(df)
    df = _create_polynomial_features(df)
    df = _apply_log_transform(df)

    new_feature_count = len([col for col in df.columns if col != "target"])
    logging.info(f"Feature count after engineering: {new_feature_count}")
    logging.info(f"Added {new_feature_count - original_feature_count} new features")

    # Standardize features
    df_scaled, scaling_params = _standardize_features(df)

    # Save engineered features
    output_path = OUTPUT_DIR / "engineered_features.csv"
    df_scaled.write_csv(output_path)
    logging.info(f"Saved engineered features to {output_path}")

    # Save feature engineering log
    feature_log = {
        "original_features": original_feature_count,
        "engineered_features": new_feature_count,
        "added_features": new_feature_count - original_feature_count,
        "feature_types": {
            "ratio_features": 4,
            "interaction_features": 4,
            "polynomial_features": 4,
            "log_transformed_features": 3,
        },
        "scaling_method": "StandardScaler (z-score normalization)",
        "total_samples": df_scaled.shape[0],
    }

    log_path = OUTPUT_DIR / "feature_engineering_log.json"
    with open(log_path, "w") as f:
        json.dump(feature_log, f, indent=2, default=str)
    logging.info(f"Saved feature engineering log to {log_path}")

    # Save scaling parameters
    scaling_path = OUTPUT_DIR / "scaling_parameters.json"
    with open(scaling_path, "w") as f:
        json.dump(scaling_params, f, indent=2, default=str)
    logging.info(f"Saved scaling parameters to {scaling_path}")

    logging.info("Feature engineering completed successfully")


if __name__ == "__main__":
    run_feature_engineering()
