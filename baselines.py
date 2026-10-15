"""
Baseline model comparison for the UCI Energy Efficiency dataset.

Compares three regression models for both targets (Y1 heating load, Y2
cooling load), all evaluated with the same 5-fold cross-validation as
gpr_model_selection.py (identical folds, same random_state), so the
comparison is fair:
  - Linear Regression: simple linear baseline, no hyperparameters to tune
  - Kernel Ridge Regression (RBF kernel): alpha and gamma are tuned with
    GridSearchCV in an inner cross-validation on the training fold only
    (nested CV), so the outer validation fold never influences tuning
  - Gaussian Process Regression with the best kernel found in
    gpr_model_selection.py (Matern nu=1.5 for Y1, Matern nu=2.5 for Y2)

All three models are wrapped in an sklearn Pipeline with the shared
preprocessor from gpr_baseline.py (standardizes the continuous features,
one-hot encodes the categorical ones X6 and X8), always fit on the current
training fold only, never on the full dataset.

Also saves a bar chart of RMSE (with error bars for the std over folds) to
figures/model_comparison_rmse.png, one subplot per target.

Run fetch_data.py first.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor

from gpr_baseline import load_raw, TARGETS, save_results_csv, build_preprocessor, n_encoded_features
from gpr_model_selection import kernel_candidates, make_kfold, RANDOM_STATE, BEST_KERNEL_NAME
from plot_style import apply_style

apply_style()

FIG_DIR = Path(__file__).parent / "figures"

# The GP kernel structure and its typical hyperparameter values were already
# established in gpr_model_selection.py (n_restarts_optimizer=2 there). This
# comparison only needs a fair evaluation, not another hyperparameter search,
# so a single optimizer run keeps 5 folds x 2 targets = 10 GP fits fast.
GP_N_RESTARTS = 1

# Inner CV for the nested Kernel Ridge tuning. Deliberately smaller than the
# outer 5-fold CV, this only has to pick reasonable hyperparameters, not
# produce the final reported score.
INNER_FOLDS = 3
# alpha is widened down to 1e-8 (previously 1e-6). Test runs pushing it
# further (to 1e-9) still picked edge values for some folds with only a
# small, noisy RMSE change, that regime is where the RBF kernel matrix
# becomes near-degenerate and alpha stops being a numerically stable,
# meaningful choice, different folds pick different values across a flat
# region rather than converging to one point. The grid is capped at 1e-8
# rather than chased further into that regime. gamma is unaffected by this
# and kept at its previous range. A single global gamma still cannot
# exploit per-feature structure the way the GP's ARD kernel does, so a gap
# to the GP remains expected regardless of alpha.
KERNEL_RIDGE_GRID = {
    "model__alpha": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    "model__gamma": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
}

# Metrics collected for every model. cross_validate negates loss-type
# scorers internally, so RMSE is flipped back to positive after scoring.
SCORING = {"rmse": "neg_root_mean_squared_error", "r2": "r2"}


def make_inner_kfold():
    """Inner CV splitter for the nested Kernel Ridge hyperparameter search.
    Must be shuffled: the dataset is a structured factorial design (see
    eda.py), not a random sample, so consecutive rows share feature values
    (e.g. long runs of the same X6 orientation). An unshuffled KFold would
    give each inner fold a biased, unrepresentative slice of that structure.
    """
    return KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def build_models(feature_cols, df, target):
    """Return a dict of model name -> sklearn estimator for one target.
    Kernel Ridge is wrapped in GridSearchCV for nested hyperparameter
    tuning, the other two have no hyperparameters that need tuning here.
    """
    linear = Pipeline([
        ("preprocessor", build_preprocessor(feature_cols)),
        ("model", LinearRegression()),
    ])

    kernel_ridge = Pipeline([
        ("preprocessor", build_preprocessor(feature_cols)),
        ("model", KernelRidge(kernel="rbf")),
    ])
    kernel_ridge_tuned = GridSearchCV(
        kernel_ridge,
        param_grid=KERNEL_RIDGE_GRID,
        cv=make_inner_kfold(),
        scoring=SCORING["rmse"],
    )

    gp_kernel = kernel_candidates(n_encoded_features(feature_cols, df))[BEST_KERNEL_NAME[target]]
    gp = Pipeline([
        ("preprocessor", build_preprocessor(feature_cols)),
        ("model", GaussianProcessRegressor(
            kernel=gp_kernel,
            normalize_y=True,
            n_restarts_optimizer=GP_N_RESTARTS,
            random_state=RANDOM_STATE,
        )),
    ])

    return {
        "Linear Regression": linear,
        "Kernel Ridge (RBF)": kernel_ridge_tuned,
        f"GP ({BEST_KERNEL_NAME[target]})": gp,
    }


def check_grid_boundaries(fitted_grid_searches):
    """Warn if any outer fold's inner GridSearchCV picked alpha or gamma
    right at the edge of KERNEL_RIDGE_GRID. A value at the edge means the
    true optimum might lie outside the searched range, which would make the
    Kernel Ridge comparison unfair (an artificially narrow grid).
    """
    alpha_values = KERNEL_RIDGE_GRID["model__alpha"]
    gamma_values = KERNEL_RIDGE_GRID["model__gamma"]
    hit_edge = False

    for i, grid_search in enumerate(fitted_grid_searches):
        alpha = grid_search.best_params_["model__alpha"]
        gamma = grid_search.best_params_["model__gamma"]
        if alpha in (alpha_values[0], alpha_values[-1]):
            print(f"    WARNING: fold {i} picked alpha={alpha} at the grid edge")
            hit_edge = True
        if gamma in (gamma_values[0], gamma_values[-1]):
            print(f"    WARNING: fold {i} picked gamma={gamma} at the grid edge")
            hit_edge = True

    if not hit_edge:
        print("    all folds picked alpha/gamma inside the grid, no edge hits")


def evaluate_target(target):
    """Cross-validate all three models for one target, print a table of
    RMSE and R2 (mean +/- std over the outer folds), and return the stats.
    """
    X, df, feature_cols = load_raw()
    y = df[target].to_numpy(dtype=float)
    models = build_models(feature_cols, df, target)
    outer_cv = make_kfold()

    print(f"\nTarget: {target} ({TARGETS[target]})", flush=True)
    results = {}
    for name, estimator in models.items():
        print(f"  running {name} ...", flush=True)
        scores = cross_validate(
            estimator, X, y, cv=outer_cv, scoring=SCORING, return_estimator=True
        )
        rmse = -scores["test_rmse"]
        r2 = scores["test_r2"]
        results[name] = {
            "rmse_mean": rmse.mean(), "rmse_std": rmse.std(),
            "r2_mean": r2.mean(), "r2_std": r2.std(),
        }
        print(
            f"  {name:<22} RMSE = {rmse.mean():.3f} +/- {rmse.std():.3f}   "
            f"R2 = {r2.mean():.4f} +/- {r2.std():.4f}",
            flush=True,
        )
        if name == "Kernel Ridge (RBF)":
            check_grid_boundaries(scores["estimator"])

    return results


def plot_rmse_comparison(all_results):
    """Bar chart of RMSE with error bars (std over folds), one subplot per
    target, stacked vertically to fit a single narrow report column, saved
    to figures/model_comparison_rmse.png.
    """
    fig, axes = plt.subplots(2, 1, figsize=(6, 7))

    for ax, target in zip(axes, TARGETS):
        results = all_results[target]
        names = list(results.keys())
        means = [results[n]["rmse_mean"] for n in names]
        stds = [results[n]["rmse_std"] for n in names]

        ax.bar(names, means, yerr=stds, capsize=6, color="steelblue", edgecolor="black")
        ax.set_title(f"{target}: {TARGETS[target]}")
        ax.set_ylabel("RMSE (5-fold CV, mean +/- std)")
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Model Comparison: Cross-Validated RMSE", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    FIG_DIR.mkdir(exist_ok=True)
    out_file = FIG_DIR / "model_comparison_rmse.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"\nSaved {out_file}")


def main():
    all_results = {target: evaluate_target(target) for target in TARGETS}
    plot_rmse_comparison(all_results)

    rows = [
        {"model": name, "target": target, **stats}
        for target, results in all_results.items()
        for name, stats in results.items()
    ]
    save_results_csv(rows, "baseline_comparison.csv")


if __name__ == "__main__":
    main()
