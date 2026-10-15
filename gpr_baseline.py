"""
Baseline Gaussian Process Regression on the UCI Energy Efficiency dataset.

This is a starting point, not the final model. For both targets (Y1 heating
load and Y2 cooling load) it:
  - loads the data produced by fetch_data.py
  - splits into train and test
  - standardizes the features
  - fits a GP with an RBF + white-noise kernel
  - reports RMSE and R2 and the mean predicted uncertainty

Run fetch_data.py first.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score

DATA_FILE = Path(__file__).parent / "data" / "energy_efficiency.csv"
RESULTS_DIR = Path(__file__).parent / "results"

# Both targets are evaluated: Y1 heating load, Y2 cooling load.
TARGETS = {"Y1": "Heating Load", "Y2": "Cooling Load"}

# X6 (orientation) and X8 (glazing area distribution) are categorical codes,
# not continuous quantities, their numeric values have no meaningful order
# or distance. They are one-hot encoded instead of standardized like the
# other six features.
CATEGORICAL_FEATURES = ["X6", "X8"]

# Shared column layout for all result CSVs, so tables for the report can be
# built by just concatenating files from results/ without extra glue code.
RESULT_FIELDS = [
    "model", "target", "rmse_mean", "rmse_std", "r2_mean", "r2_std",
    "coverage_68", "coverage_95",
]


def save_results_csv(rows, filename):
    """Write a list of result dicts to results/<filename>. Used by every
    script that runs cross-validation (gpr_model_selection, baselines,
    uncertainty, isotropic_comparison), so tables and plots for the report
    can be rebuilt from these files without repeating the expensive GP fits.
    Rows that do not have a value for a given column (e.g. coverage for
    models without predictive uncertainty) are left blank.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    out_file = RESULTS_DIR / filename
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in RESULT_FIELDS})
    print(f"Saved {out_file}")


def load_raw():
    """Load the CSV and return the feature matrix, the full dataframe, and
    the feature column names. Used directly by scripts that need their own
    train/validation splitting, e.g. for cross-validation.
    """
    df = pd.read_csv(DATA_FILE)
    # The UCI columns come as X1..X8, Y1, Y2. Fall back gracefully if names differ.
    feature_cols = [c for c in df.columns if c.lower().startswith("x")]
    if not feature_cols:
        feature_cols = list(df.columns[:8])
    X = df[feature_cols].to_numpy(dtype=float)
    return X, df, feature_cols


def build_preprocessor(feature_cols):
    """Build a ColumnTransformer that standardizes the continuous features
    and one-hot encodes the categorical ones (see CATEGORICAL_FEATURES).

    Used as the preprocessing step of every model, either as a Pipeline
    step (baselines.py) or fit manually per CV fold (gpr_model_selection.py,
    uncertainty.py, isotropic_comparison.py), always on the training fold
    only, the same rule already followed for the scaler alone before.
    """
    categorical_idx = [feature_cols.index(c) for c in CATEGORICAL_FEATURES if c in feature_cols]
    continuous_idx = [i for i in range(len(feature_cols)) if i not in categorical_idx]

    return ColumnTransformer([
        ("continuous", StandardScaler(), continuous_idx),
        ("categorical", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_idx),
    ])


def encoded_feature_names(feature_cols, df):
    """Human-readable names for the columns produced by build_preprocessor,
    in the same order it produces them: continuous feature names first,
    then one name per category level for each categorical feature.
    """
    categorical = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    continuous = [c for c in feature_cols if c not in categorical]

    names = list(continuous)
    for col in categorical:
        for level in sorted(df[col].unique()):
            names.append(f"{col}={level}")
    return names


def n_encoded_features(feature_cols, df):
    """Number of columns produced by build_preprocessor. Only used to size
    the GP kernel's per-feature length scales, not a leakage-sensitive
    statistic, it only depends on which category values exist anywhere in
    the dataset's schema, not on any fold-specific fit.
    """
    return len(encoded_feature_names(feature_cols, df))


def load_and_prepare(target, test_size=0.2, random_state=0):
    """Load the CSV, split into train/test, and preprocess the features
    (standardize the continuous ones, one-hot encode the categorical ones).

    Shared by both targets so the loading and preprocessing logic exists
    only once. Returns the preprocessed feature arrays, the raw target
    arrays, and the post-encoding feature names.
    """
    X, df, feature_cols = load_raw()
    y = df[target].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = build_preprocessor(feature_cols).fit(X_train)
    X_train_p = preprocessor.transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    return X_train_p, X_test_p, y_train, y_test, encoded_feature_names(feature_cols, df)


def build_model(n_features):
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(n_features)) + WhiteKernel(noise_level=1.0)
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=0,
    )


def evaluate_target(target):
    """Fit and evaluate the GP for one target column, e.g. "Y1"."""
    X_train_s, X_test_s, y_train, y_test, feature_cols = load_and_prepare(target)

    model = build_model(len(feature_cols))
    model.fit(X_train_s, y_train)

    y_pred, y_std = model.predict(X_test_s, return_std=True)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mean_std = y_std.mean()

    print(f"\nTarget: {target} ({TARGETS[target]})")
    print(f"Features: {feature_cols}")
    print(f"Learned kernel: {model.kernel_}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2:   {r2:.4f}")
    print(f"Mean predicted std (uncertainty): {mean_std:.3f}")

    return {"target": target, "rmse": rmse, "r2": r2, "mean_std": mean_std}


def main():
    results = [evaluate_target(target) for target in TARGETS]

    print("\nSummary:")
    print(f"{'Target':<8}{'RMSE':>10}{'R2':>10}{'Mean std':>12}")
    for r in results:
        print(f"{r['target']:<8}{r['rmse']:>10.3f}{r['r2']:>10.4f}{r['mean_std']:>12.3f}")


if __name__ == "__main__":
    main()
