"""Exploratory Data Analysis for Wine Classification Dataset."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.datasets import load_wine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: Path = Path("output/eda")
RANDOM_SEED: int = 42


def _create_output_directory() -> None:
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory created/verified: {OUTPUT_DIR}")


def _load_wine_data() -> tuple[pl.DataFrame, list[str], list[str]]:
    """Load wine dataset and convert to polars DataFrame.

    Returns
    -------
    tuple[pl.DataFrame, list[str], list[str]]
        DataFrame with features and target, feature names, target names
    """
    wine = load_wine()

    # Create polars DataFrame
    df = pl.DataFrame(
        wine.data,
        schema=wine.feature_names,
    )
    df = df.with_columns(pl.Series("target", wine.target))

    logging.info(f"Loaded wine dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    return df, wine.feature_names, wine.target_names.tolist()


def _generate_summary_statistics(
    df: pl.DataFrame,
) -> dict[str, dict[str, float]]:
    """Generate summary statistics for all features.

    Parameters
    ----------
    df : pl.DataFrame
        Wine dataset

    Returns
    -------
    dict[str, dict[str, float]]
        Summary statistics for each feature
    """
    stats = {}
    feature_cols = [col for col in df.columns if col != "target"]

    for col in feature_cols:
        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "q25": float(df[col].quantile(0.25)),
            "median": float(df[col].median()),
            "q75": float(df[col].quantile(0.75)),
        }

    logging.info("Generated summary statistics for all features")
    return stats


def _plot_feature_distributions(
    df: pl.DataFrame,
    feature_names: list[str],
) -> None:
    """Plot distributions for all features.

    Parameters
    ----------
    df : pl.DataFrame
        Wine dataset
    feature_names : list[str]
        List of feature names
    """
    n_features = len(feature_names)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten()

    for idx, feature in enumerate(feature_names):
        data = df[feature].to_numpy()
        axes[idx].hist(data, bins=20, edgecolor="black", alpha=0.7)
        axes[idx].set_title(feature, fontsize=10)
        axes[idx].set_xlabel("Value")
        axes[idx].set_ylabel("Frequency")

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "feature_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved feature distributions plot to {output_path}")


def _plot_correlation_heatmap(
    df: pl.DataFrame,
    feature_names: list[str],
) -> None:
    """Plot correlation heatmap for features.

    Parameters
    ----------
    df : pl.DataFrame
        Wine dataset
    feature_names : list[str]
        List of feature names
    """
    # Calculate correlation matrix
    corr_matrix = df.select(feature_names).to_pandas().corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Heatmap", fontsize=14, pad=20)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "correlation_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved correlation heatmap to {output_path}")


def _plot_class_distribution(
    df: pl.DataFrame,
    target_names: list[str],
) -> None:
    """Plot class distribution.

    Parameters
    ----------
    df : pl.DataFrame
        Wine dataset
    target_names : list[str]
        List of target class names
    """
    class_counts = df.group_by("target").agg(pl.count()).sort("target")

    plt.figure(figsize=(8, 6))
    plt.bar(
        range(len(target_names)),
        class_counts["count"].to_list(),
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
        edgecolor="black",
    )
    plt.xlabel("Wine Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Class Distribution", fontsize=14)
    plt.xticks(range(len(target_names)), target_names)

    # Add count labels on bars
    for idx, count in enumerate(class_counts["count"].to_list()):
        plt.text(idx, count + 1, str(count), ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "class_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved class distribution plot to {output_path}")


def _plot_feature_boxplots_by_class(
    df: pl.DataFrame,
    feature_names: list[str],
) -> None:
    """Plot boxplots for features grouped by target class.

    Parameters
    ----------
    df : pl.DataFrame
        Wine dataset
    feature_names : list[str]
        List of feature names
    """
    n_features = len(feature_names)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten()

    # Convert to pandas for seaborn
    df_pandas = df.to_pandas()

    for idx, feature in enumerate(feature_names):
        sns.boxplot(data=df_pandas, x="target", y=feature, ax=axes[idx])
        axes[idx].set_title(feature, fontsize=10)
        axes[idx].set_xlabel("Wine Class")
        axes[idx].set_ylabel("Value")

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "feature_boxplots_by_class.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved feature boxplots by class to {output_path}")


def run_eda() -> None:
    """Run complete exploratory data analysis."""
    logging.info("Starting exploratory data analysis")

    _create_output_directory()

    df, feature_names, target_names = _load_wine_data()

    # Generate and log summary statistics
    stats = _generate_summary_statistics(df)
    logging.info("Summary statistics:\n" + json.dumps(stats, indent=2, default=str))

    # Save summary statistics
    stats_path = OUTPUT_DIR / "summary_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logging.info(f"Saved summary statistics to {stats_path}")

    # Create all visualizations
    _plot_feature_distributions(df, feature_names)
    _plot_correlation_heatmap(df, feature_names)
    _plot_class_distribution(df, target_names)
    _plot_feature_boxplots_by_class(df, feature_names)

    logging.info("Exploratory data analysis completed successfully")


if __name__ == "__main__":
    run_eda()
