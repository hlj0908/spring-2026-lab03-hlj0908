import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.datasets import load_wine

# Constants
RANDOM_STATE: int = 42
OUTPUT_DIR: Path = Path("part2_antigravity/output/eda")
SUMMARY_FILE: Path = OUTPUT_DIR / "summary_stats.md"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)


def _load_data() -> pl.DataFrame:
    """Load Wine dataset and convert to Polars DataFrame."""
    logging.info("Loading Wine dataset...")
    wine_data: Any = load_wine(as_frame=True)
    df_pandas = wine_data.frame
    df: pl.DataFrame = pl.from_pandas(df_pandas)
    logging.info(f"Loaded dataset with shape: {df.shape}")
    return df


def _save_summary_stats(df: pl.DataFrame) -> None:
    """Compute and save summary statistics."""
    logging.info("Computing summary statistics...")
    summary: pl.DataFrame = df.describe()

    with open(SUMMARY_FILE, "w") as f:
        f.write("# Wine Dataset Summary Statistics\n\n")
        f.write(str(summary))

    logging.info(f"Summary statistics saved to {SUMMARY_FILE}")


def _check_missing_values(df: pl.DataFrame) -> None:
    """Check for missing values and log results."""
    logging.info("Checking for missing values...")
    null_counts = df.null_count()
    if null_counts.sum_horizontal().item() > 0:
        logging.warning(f"Missing values found:\n{null_counts}")
    else:
        logging.info("No missing values found.")


def _plot_distributions(df: pl.DataFrame) -> None:
    """Plot distributions of features."""
    logging.info("Plotting feature distributions...")
    numeric_cols = [col for col in df.columns if col != "target"]

    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 5 * n_rows))
    for i, col in enumerate(numeric_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distributions.png")
    plt.close()
    logging.info(f"Distributions plot saved to {OUTPUT_DIR / 'distributions.png'}")


def _plot_correlation_heatmap(df: pl.DataFrame) -> None:
    """Plot correlation heatmap."""
    logging.info("Plotting correlation heatmap...")
    corr_matrix = df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
    plt.close()
    logging.info(f"Correlation heatmap saved to {OUTPUT_DIR / 'correlation_heatmap.png'}")


def _detect_outliers(df: pl.DataFrame) -> None:
    """Detect outliers using IQR method."""
    logging.info("Detecting outliers...")
    outlier_report = []

    numeric_cols = [col for col in df.columns if col != "target"]

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df.filter((pl.col(col) < lower_bound) | (pl.col(col) > upper_bound))
        count = outliers.height

        if count > 0:
            outlier_report.append(f"{col}: {count} outliers")

    with open(OUTPUT_DIR / "outliers.txt", "w") as f:
        f.write("\n".join(outlier_report))

    logging.info(f"Outlier detection complete. Found outliers in {len(outlier_report)} columns.")


def perform_eda() -> None:
    """Execute the EDA process."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_data()
    _save_summary_stats(df)
    _check_missing_values(df)
    _plot_distributions(df)
    _plot_correlation_heatmap(df)
    _detect_outliers(df)

    logging.info("EDA completed successfully.")


if __name__ == "__main__":
    perform_eda()
