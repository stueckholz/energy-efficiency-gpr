"""
Kernel comparison for Gaussian Process Regression on the UCI Energy
Efficiency dataset.

Extends the baseline in gpr_baseline.py, which fixes the kernel to RBF and
uses a single train/test split. This script instead:
  - compares three kernel candidates: RBF, Matern(nu=1.5), Matern(nu=2.5),
    each combined with a WhiteKernel for observation noise
  - uses k-fold cross-validation instead of one fixed split, for a more
    robust estimate of RMSE and R2 (mean +/- std across folds)
  - for both targets Y1 (heating) and Y2 (cooling), reports the best kernel
    and its learned per-feature length scales, which show how sensitive
    each target is to each of the 8 geometry features

The kernel's continuous hyperparameters (amplitude, length scales, noise
level) are already optimized automatically inside GaussianProcessRegressor
via marginal likelihood maximization. The kernel structure itself (RBF vs
Matern) is a modeling choice that is not learned automatically, so it is
compared here instead.

Run fetch_data.py first.
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score

from gpr_baseline import (
    load_raw, TARGETS, save_results_csv,
    build_preprocessor, encoded_feature_names, n_encoded_features,
)

N_FOLDS = 5
RANDOM_STATE = 0
# Cross-validation needs 3 kernels x 5 folds x 2 targets = 30 fits, so fewer
# optimizer restarts keep runtime reasonable. The final fit on the full
# dataset (used to report length scales) still uses more restarts, since
# only one fit per target is needed there.
CV_N_RESTARTS = 2
FINAL_N_RESTARTS = 5

# Best kernel per target, based on the CV comparison in select_best_kernel().
# Read by baselines.py, uncertainty.py and isotropic_comparison.py so the
# choice only lives in one place.
BEST_KERNEL_NAME = {"Y1": "Matern(nu=1.5)", "Y2": "Matern(nu=2.5)"}


def kernel_candidates(n_features):
    """Build the kernel candidates to compare for a model with the given
    number of features. Matern(nu=1.5) and Matern(nu=2.5) assume less smooth
    functions than RBF, which assumes infinitely smooth functions.
    """
    length_scale = np.ones(n_features)
    return {
        "RBF": ConstantKernel(1.0) * RBF(length_scale=length_scale) + WhiteKernel(1.0),
        "Matern(nu=1.5)": ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=1.5) + WhiteKernel(1.0),
        "Matern(nu=2.5)": ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=2.5) + WhiteKernel(1.0),
    }


def make_kfold():
    """Build the KFold splitter used for the kernel comparison below. Also
    imported by baselines.py so all models are compared on identical folds.
    """
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def cross_validate_kernel(X, y, kernel, feature_cols, n_restarts=CV_N_RESTARTS):
    """Run k-fold CV for one kernel and return per-fold RMSE and R2 arrays.

    Preprocessing (standardizing continuous features, one-hot encoding
    categorical ones) is fit on each training fold only, so no information
    from the validation fold leaks into it. n_restarts defaults to
    CV_N_RESTARTS but can be lowered by callers that already know good
    hyperparameters and just need a fast comparison (see
    isotropic_comparison.py).
    """
    kf = make_kfold()
    rmses, r2s = [], []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        preprocessor = build_preprocessor(feature_cols).fit(X_train)
        X_train_s = preprocessor.transform(X_train)
        X_val_s = preprocessor.transform(X_val)

        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=n_restarts,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_val_s)

        rmses.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        r2s.append(r2_score(y_val, y_pred))

    return np.array(rmses), np.array(r2s)


def extract_length_scales(kernel):
    """Pull the per-feature length scale array out of a fitted kernel,
    assuming the (Constant * RBF_or_Matern + White) structure built above.
    """
    base = kernel.k1.k2  # k1 = Constant * RBF_or_Matern, k2 of that = RBF_or_Matern
    return np.atleast_1d(base.length_scale)


def select_best_kernel(target):
    """Compare kernel candidates for one target via cross-validation, print
    a summary, and return the name and CV stats of the best one (lowest
    mean RMSE).
    """
    X, df, feature_cols = load_raw()
    y = df[target].to_numpy(dtype=float)
    candidates = kernel_candidates(n_encoded_features(feature_cols, df))

    print(f"\nTarget: {target} ({TARGETS[target]})", flush=True)
    results = {}
    for name, kernel in candidates.items():
        print(f"  running {name} ...", flush=True)
        rmses, r2s = cross_validate_kernel(X, y, kernel, feature_cols)
        results[name] = {
            "rmse_mean": rmses.mean(), "rmse_std": rmses.std(),
            "r2_mean": r2s.mean(), "r2_std": r2s.std(),
        }
        print(
            f"  {name:<16} RMSE = {rmses.mean():.3f} +/- {rmses.std():.3f}   "
            f"R2 = {r2s.mean():.4f} +/- {r2s.std():.4f}",
            flush=True,
        )

    best_name = min(results, key=lambda name: results[name]["rmse_mean"])
    print(f"  -> best kernel: {best_name}")

    # Fit the best kernel once on the full dataset to report the learned
    # per-feature length scales, useful for the report (feature sensitivity).
    preprocessor = build_preprocessor(feature_cols).fit(X)
    X_s = preprocessor.transform(X)
    model = GaussianProcessRegressor(
        kernel=candidates[best_name],
        normalize_y=True,
        n_restarts_optimizer=FINAL_N_RESTARTS,
        random_state=RANDOM_STATE,
    )
    model.fit(X_s, y)
    print(f"  learned kernel: {model.kernel_}")

    print("  length scales (smaller = target reacts more sensitively to this feature):")
    for col, ls in zip(encoded_feature_names(feature_cols, df), extract_length_scales(model.kernel_)):
        print(f"    {col}: {ls:.3f}")

    return best_name, results


def main():
    rows = []
    for target in TARGETS:
        best_name, results = select_best_kernel(target)
        for name, stats in results.items():
            rows.append({"model": name, "target": target, **stats})

    save_results_csv(rows, "kernel_comparison.csv")


if __name__ == "__main__":
    main()
