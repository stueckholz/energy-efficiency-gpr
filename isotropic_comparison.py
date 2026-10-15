"""
ARD vs isotropic kernel comparison for the Gaussian Process.

The kernels compared in gpr_model_selection.py all use ARD (Automatic
Relevance Determination): one length scale per feature. baselines.py shows
the GP clearly beats Kernel Ridge, but Kernel Ridge only has a single shared
gamma (isotropic), so part of the GP's advantage could simply be ARD rather
than the GP framework itself. This script isolates that: it fits the same
kernel family (Matern, same nu as the best kernel per target) with a single
shared length scale instead, on the identical 5-fold CV split, and compares
it to the ARD result.

The ARD numbers are read from results/kernel_comparison.csv if that file
already exists (written by gpr_model_selection.py), so the expensive ARD fit
is not repeated. If the file is missing, they are computed here instead.

Results are printed as a small table and written to
results/isotropic_comparison.csv.

Run fetch_data.py first, and ideally gpr_model_selection.py once so its CSV
can be reused here.
"""

import pandas as pd
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from gpr_baseline import load_raw, TARGETS, RESULTS_DIR, save_results_csv, n_encoded_features
from gpr_model_selection import kernel_candidates, cross_validate_kernel, BEST_KERNEL_NAME
from baselines import GP_N_RESTARTS

# nu for the best kernel per target, matches gpr_model_selection.py's choice.
BEST_NU = {"Y1": 1.5, "Y2": 2.5}


def isotropic_kernel(nu):
    """Same kernel family as the ARD version, but with a single length
    scale shared across all features instead of one per feature.
    """
    return ConstantKernel(1.0) * Matern(length_scale=1.0, nu=nu) + WhiteKernel(1.0)


def load_cached_ard_result(target):
    """Try to reuse the ARD result for the best kernel from
    results/kernel_comparison.csv, so this script does not have to repeat
    that expensive fit. Returns None if not available.
    """
    cache_file = RESULTS_DIR / "kernel_comparison.csv"
    if not cache_file.exists():
        return None

    df = pd.read_csv(cache_file)
    row = df[(df["target"] == target) & (df["model"] == BEST_KERNEL_NAME[target])]
    if row.empty:
        return None

    row = row.iloc[0]
    return {
        "rmse_mean": row["rmse_mean"], "rmse_std": row["rmse_std"],
        "r2_mean": row["r2_mean"], "r2_std": row["r2_std"],
    }


def main():
    X, df, feature_cols = load_raw()
    rows = []

    print(f"{'Target':<8}{'ARD RMSE':>18}{'Isotropic RMSE':>20}")
    for target in TARGETS:
        y = df[target].to_numpy(dtype=float)

        ard_result = load_cached_ard_result(target)
        if ard_result is None:
            print(f"  (no cached ARD result for {target}, computing it now)", flush=True)
            ard_kernel = kernel_candidates(n_encoded_features(feature_cols, df))[BEST_KERNEL_NAME[target]]
            ard_rmse, ard_r2 = cross_validate_kernel(X, y, ard_kernel, feature_cols, n_restarts=GP_N_RESTARTS)
            ard_result = {
                "rmse_mean": ard_rmse.mean(), "rmse_std": ard_rmse.std(),
                "r2_mean": ard_r2.mean(), "r2_std": ard_r2.std(),
            }

        print(f"  running isotropic Matern(nu={BEST_NU[target]}) for {target} ...", flush=True)
        iso_kernel = isotropic_kernel(BEST_NU[target])
        iso_rmse, iso_r2 = cross_validate_kernel(X, y, iso_kernel, feature_cols, n_restarts=GP_N_RESTARTS)
        iso_result = {
            "rmse_mean": iso_rmse.mean(), "rmse_std": iso_rmse.std(),
            "r2_mean": iso_r2.mean(), "r2_std": iso_r2.std(),
        }

        print(
            f"{target:<8}"
            f"{ard_result['rmse_mean']:>9.3f} +/- {ard_result['rmse_std']:<5.3f}"
            f"{iso_result['rmse_mean']:>11.3f} +/- {iso_result['rmse_std']:<5.3f}",
            flush=True,
        )

        rows.append({"model": f"GP ARD ({BEST_KERNEL_NAME[target]})", "target": target, **ard_result})
        rows.append({"model": f"GP isotropic (Matern nu={BEST_NU[target]})", "target": target, **iso_result})

    save_results_csv(rows, "isotropic_comparison.csv")


if __name__ == "__main__":
    main()