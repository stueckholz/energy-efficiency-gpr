"""
Uncertainty evaluation for the Gaussian Process Regression models.

The main reason to use a GP instead of a plain regression model is that it
returns a predictive standard deviation, not just a point prediction. This
script checks whether that predicted uncertainty is trustworthy.

For both targets, using the same 5 CV folds and best kernel as
gpr_model_selection.py, it collects out-of-fold predictions (y_pred, y_std):
every data point is predicted exactly once, by a model that never saw it
during training. From these it evaluates:
  - coverage: the fraction of true values that actually fall inside the
    predicted 95% interval (y_pred +/- 1.96 * y_std) and 68% interval
    (y_pred +/- 1.0 * y_std). A well-calibrated model gives close to 0.95
    and 0.68.
  - whether the mean predicted std matches the actual RMSE, i.e. whether
    the model is over- or underconfident.

Also saves three figures to figures/:
  - calibration_curve.png: nominal confidence level vs. observed coverage,
    with the diagonal as the ideal line
  - predicted_vs_actual.png: predicted vs. true values with error bars from
    the predicted std, one subplot per target
  - predicted_vs_actual_zoom.png: the same plot, zoomed into a slice of the
    value range chosen to show both small and large error bars, since they
    are hard to tell apart on the full-range plot

Results (RMSE, R2, coverage) are also written to results/uncertainty.csv.

Run fetch_data.py first.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, r2_score

from gpr_baseline import load_raw, TARGETS, save_results_csv, build_preprocessor, n_encoded_features
from gpr_model_selection import kernel_candidates, make_kfold, RANDOM_STATE, BEST_KERNEL_NAME
from baselines import GP_N_RESTARTS
from plot_style import apply_style

apply_style()

FIG_DIR = Path(__file__).parent / "figures"

# Width of the true-value window used for the zoomed predicted-vs-actual
# plot, in load units.
ZOOM_WIDTH = 10.0

# Nominal confidence levels checked for the calibration curve.
CONFIDENCE_LEVELS = np.linspace(0.05, 0.99, 20)


def collect_oof_predictions(target):
    """Run 5-fold CV with the best kernel for this target. Returns the
    pooled out-of-fold true values, predictions, and predicted std (one
    entry per data point, covering the whole dataset), plus the per-fold
    RMSE and R2 arrays (same format as gpr_model_selection.py, for the CSV).
    """
    X, df, feature_cols = load_raw()
    y = df[target].to_numpy(dtype=float)
    kernel = kernel_candidates(n_encoded_features(feature_cols, df))[BEST_KERNEL_NAME[target]]
    kf = make_kfold()

    y_true_folds, y_pred_folds, y_std_folds = [], [], []
    fold_rmses, fold_r2s = [], []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        preprocessor = build_preprocessor(feature_cols).fit(X_train)
        X_train_s = preprocessor.transform(X_train)
        X_val_s = preprocessor.transform(X_val)

        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=GP_N_RESTARTS,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train_s, y_train)
        y_pred, y_std = model.predict(X_val_s, return_std=True)

        y_true_folds.append(y_val)
        y_pred_folds.append(y_pred)
        y_std_folds.append(y_std)
        fold_rmses.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        fold_r2s.append(r2_score(y_val, y_pred))

    y_true = np.concatenate(y_true_folds)
    y_pred = np.concatenate(y_pred_folds)
    y_std = np.concatenate(y_std_folds)
    return y_true, y_pred, y_std, np.array(fold_rmses), np.array(fold_r2s)


def coverage(y_true, y_pred, y_std, z):
    """Fraction of true values that fall inside y_pred +/- z * y_std."""
    lower = y_pred - z * y_std
    upper = y_pred + z * y_std
    return np.mean((y_true >= lower) & (y_true <= upper))


def evaluate_target(target):
    """Compute coverage and calibration stats for one target, print them,
    and return everything needed for the plots and the CSV row.
    """
    y_true, y_pred, y_std, fold_rmses, fold_r2s = collect_oof_predictions(target)

    mean_std = y_std.mean()
    cov_95 = coverage(y_true, y_pred, y_std, 1.96)
    cov_68 = coverage(y_true, y_pred, y_std, 1.0)

    print(f"\nTarget: {target} ({TARGETS[target]})", flush=True)
    print(f"  Kernel: {BEST_KERNEL_NAME[target]}")
    print(f"  RMSE (mean +/- std over folds): {fold_rmses.mean():.3f} +/- {fold_rmses.std():.3f}")
    print(f"  Mean predicted std:             {mean_std:.3f}")
    if mean_std > fold_rmses.mean():
        print("  -> predicted std is larger than the actual RMSE: model is a bit conservative (underconfident)")
    else:
        print("  -> predicted std is smaller than the actual RMSE: model is a bit overconfident")
    print(f"  Coverage 95% interval: {cov_95:.3f} (ideal 0.95)")
    print(f"  Coverage 68% interval: {cov_68:.3f} (ideal 0.68)")

    # Observed coverage at a range of nominal confidence levels, for the
    # calibration plot. z is the two-sided z-score for that confidence level.
    observed_coverage = np.array([
        coverage(y_true, y_pred, y_std, norm.ppf(0.5 + conf / 2))
        for conf in CONFIDENCE_LEVELS
    ])

    return {
        "y_true": y_true, "y_pred": y_pred, "y_std": y_std,
        "rmse_mean": fold_rmses.mean(), "rmse_std": fold_rmses.std(),
        "r2_mean": fold_r2s.mean(), "r2_std": fold_r2s.std(),
        "coverage_68": cov_68, "coverage_95": cov_95,
        "observed_coverage": observed_coverage,
    }


def plot_calibration_curve(all_results):
    """Nominal confidence vs. observed coverage, one line per target, with
    the diagonal as the ideal (perfectly calibrated) reference.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal calibration")

    colors = {"Y1": "indianred", "Y2": "seagreen"}
    for target in TARGETS:
        ax.plot(
            CONFIDENCE_LEVELS, all_results[target]["observed_coverage"],
            marker="o", markersize=4, color=colors[target],
            label=f"{target}: {TARGETS[target]}",
        )

    ax.set_xlabel("Nominal confidence level")
    ax.set_ylabel("Observed coverage")
    ax.set_title("GP Calibration: Predicted vs. Observed Coverage")
    ax.legend()
    fig.tight_layout()

    out_file = FIG_DIR / "calibration_curve.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"\nSaved {out_file}")


def find_zoom_window(y_true, y_std, width=ZOOM_WIDTH):
    """Slide a window of the given width along the true-value range and
    return the (lo, hi) bounds where the contained points show the largest
    spread in predicted std. Used to pick a zoomed-in slice that actually
    shows both small and large error bars, instead of an arbitrary one.
    """
    lo, hi = y_true.min(), y_true.max()
    step = width / 4
    best_window, best_spread = (lo, lo + width), -1.0

    start = lo
    while start + width <= hi:
        mask = (y_true >= start) & (y_true <= start + width)
        if mask.sum() >= 5:
            spread = y_std[mask].max() - y_std[mask].min()
            if spread > best_spread:
                best_spread, best_window = spread, (start, start + width)
        start += step

    return best_window


def plot_predicted_vs_actual(all_results, out_file, windows=None):
    """Predicted vs. true values with error bars from the predicted std,
    one subplot per target (stacked vertically to fit a single narrow
    report column), with the diagonal as the perfect-prediction line.

    windows, if given, is a dict target -> (lo, hi) used as both axis
    limits, to zoom into a slice of the value range. Otherwise the full
    data range is shown.
    """
    fig, axes = plt.subplots(2, 1, figsize=(6, 7))

    for ax, target in zip(axes, TARGETS):
        r = all_results[target]
        ax.errorbar(
            r["y_true"], r["y_pred"], yerr=r["y_std"],
            fmt="o", markersize=3, alpha=0.5,
            color="steelblue", ecolor="black", elinewidth=1.2, capsize=2,
        )

        if windows is not None:
            lo, hi = windows[target]
        else:
            lo = min(r["y_true"].min(), r["y_pred"].min())
            hi = max(r["y_true"].max(), r["y_pred"].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", label="Perfect prediction")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        ax.set_xlabel(f"True {TARGETS[target]}")
        ax.set_ylabel(f"Predicted {TARGETS[target]}")
        ax.set_title(f"{target}: {TARGETS[target]}")
        ax.legend()

    title = "Predicted vs. Actual (error bars = predicted std)"
    if windows is not None:
        title += ", zoomed"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    all_results = {target: evaluate_target(target) for target in TARGETS}

    plot_calibration_curve(all_results)
    plot_predicted_vs_actual(all_results, FIG_DIR / "predicted_vs_actual.png")

    zoom_windows = {
        target: find_zoom_window(all_results[target]["y_true"], all_results[target]["y_std"])
        for target in TARGETS
    }
    plot_predicted_vs_actual(
        all_results, FIG_DIR / "predicted_vs_actual_zoom.png", windows=zoom_windows
    )

    rows = [
        {"model": f"GP ({BEST_KERNEL_NAME[target]})", "target": target, **{
            k: v for k, v in results.items() if k not in ("y_true", "y_pred", "y_std", "observed_coverage")
        }}
        for target, results in all_results.items()
    ]
    save_results_csv(rows, "uncertainty.csv")


if __name__ == "__main__":
    main()