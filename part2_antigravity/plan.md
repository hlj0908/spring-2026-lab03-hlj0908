# Wine Classification ML Pipeline Implementation Plan

This plan outlines the development of a complete Machine Learning pipeline for the Wine dataset using `sklearn`, `polars`, and `xgboost`.

## user review required

> [!IMPORTANT]
> - All scripts will be placed in `part2_antigravity/src/`.
> - Outputs will be saved to `part2_antigravity/output/` (or relative `output/` from execution context).
> - `uv` will be used for execution.
> - `polars` is mandated over `pandas`.

## Proposed Changes

### Directory Structure

```
part2_antigravity/
├── src/
│   ├── 01_eda.py
│   ├── 02_feature_engineering.py
│   └── 03_xgboost_model.py
├── output/
│   ├── eda/
│   ├── processed/
│   └── model/
└── plan.md
```

### 1. Exploratory Data Analysis (`01_eda.py`)

**Goal**: Understand data distribution, quality, and relationships.

**Steps**:
1.  **Load Data**: Use `sklearn.datasets.load_wine(as_frame=True)` but convert to Polars DataFrame immediately.
2.  **Summary Statistics**: Compute mean, median, std, min, max using Polars.
3.  **Missing Values & Types**: Check for nulls and data types.
4.  **Class Balance**: visualizing target distribution.
5.  **Distributions**: Plot histograms/KDE for features.
6.  **Correlation**: Heatmap of feature correlations.
7.  **Outlier Detection**: IQR method.
8.  **Output**: Save plots to `output/eda/` and summary stats to text/markdown files.

**Key Technical Details**:
- Use `polars` for all data manipulation.
- Use `seaborn`/`matplotlib` for plotting.
- constants: `RANDOM_STATE = 42`, `OUTPUT_DIR = Path("output/eda")`.

### 2. Feature Engineering (`02_feature_engineering.py`)

**Goal**: Prepare data for modeling and create new predictive features.

**Steps**:
1.  **Load Data**: Load raw data again (or from a saved intermediate if preferred, but loading fresh from sklearn is cleaner for this scale).
2.  **Derived Features**: Create at least 3 new features (e.g., interaction terms, ratios like `proline / magnesium`, etc.).
3.  **Split Data**: Stratified Train/Test split (80/20).
4.  **Scaling**: StandardScaler on features (fit on train, transform both).
5.  **Output**: Save processed train/test sets to `output/processed/` (e.g., parquet or csv).

**Key Technical Details**:
- Maintain Polars workflow.
- Use `sklearn.model_selection.train_test_split`.
- Use `sklearn.preprocessing.StandardScaler`.

### 3. Model Training (`03_model_training.py`) - **Agent 2**

**Goal**: Train XGBoost classifier with hyperparameter tuning and save the best model.

**Steps**:
1.  **Load Processed Data**: Read `X_train` and `y_train` from `output/processed/`.
2.  **Hyperparameter Tuning**:
    - Use `RandomizedSearchCV` (20 iter, 5-fold CV).
    - Save best params to `output/tuning_results.json`.
3.  **Model Training**: Retrain/Refit best model on full train set.
4.  **Save Model**: Save variable `final_model` to `output/model/xgb_model.joblib`.
5.  **Output**: `xgb_model.joblib`, `tuning_results.json`.

**Key Technical Details**:
- `joblib.dump` for model persistence.
- Logging of training progress.

### 4. Model Evaluation (`04_model_evaluation.py`) - **Agent 3**

**Goal**: Evaluate the trained model on test data and generate reports.

**Steps**:
1.  **Load Resources**: Load `xgb_model.joblib` and test data (`X_test`, `y_test`).
2.  **Evaluation**:
    - Compute Accuracy, Precision, Recall, F1.
    - Generate Confusion Matrix and Feature Importance plots.
3.  **Reporting**: 
    - Save metrics to `output/model/metrics.json`.
    - Save plots to `output/model/`.

**Key Technical Details**:
- `joblib.load` to retrieve model.
- Reuse plotting logic from original design.

## Verification Plan

### Automated Tests
- Run `uv run ruff check --fix part2_antigravity/src/`
- Run `uv run python -m py_compile part2_antigravity/src/*.py`
- Execute scripts in order:
    1. `uv run part2_antigravity/src/01_eda.py`
    2. `uv run part2_antigravity/src/02_feature_engineering.py`
    3. `uv run part2_antigravity/src/03_xgboost_model.py`
- specific check: ensure `output/` directory contains expected files.

### Manual Verification
- Review `output/eda/*.png` for legibility.
- Review `output/model/metrics.json` (or log output) for reasonable performance (Wine dataset should yield high accuracy, >90%).
