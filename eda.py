"""
Exploratory data analysis for the UCI Energy Efficiency dataset.

Loads the CSV produced by fetch_data.py and creates three figures used for
the data description in the report:
  - a 2x4 grid of histograms, one per feature (X1..X8)
  - a correlation heatmap of all 10 columns (X1..X8, Y1, Y2)
  - histograms of the two target variables (Y1 heating load, Y2 cooling load)

All figures are saved as PNG files in figures/ (created if missing).
Run fetch_data.py first.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_style import apply_style

apply_style()

DATA_FILE = Path(__file__).parent / "data" / "energy_efficiency.csv"
FIG_DIR = Path(__file__).parent / "figures"

# Readable names for axis labels and titles. The UCI dataset page does not
# give units for X1..X8 or Y1/Y2, so we only use descriptive names here.
FEATURE_LABELS = {
    "X1": "Relative Compactness",
    "X2": "Surface Area",
    "X3": "Wall Area",
    "X4": "Roof Area",
    "X5": "Overall Height",
    "X6": "Orientation",
    "X7": "Glazing Area",
    "X8": "Glazing Area Distribution",
}
TARGET_LABELS = {
    "Y1": "Heating Load",
    "Y2": "Cooling Load",
}


def load_data():
    return pd.read_csv(DATA_FILE)


def plot_feature_histograms(df):
    """4x2 grid of histograms, one subplot per feature X1..X8. 4 rows by 2
    columns instead of 2x4 so the figure fits a single narrow report column.
    """
    feature_cols = list(FEATURE_LABELS.keys())

    fig, axes = plt.subplots(4, 2, figsize=(6, 10))
    for ax, col in zip(axes.flat, feature_cols):
        ax.hist(df[col], bins=20, color="steelblue", edgecolor="black")
        ax.set_title(f"{col}: {FEATURE_LABELS[col]}")
        ax.set_xlabel(FEATURE_LABELS[col])
        ax.set_ylabel("Count")

    fig.suptitle("Distribution of Building Shape Features", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_file = FIG_DIR / "feature_histograms.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")


def plot_correlation_heatmap(df):
    """Correlation heatmap for all 8 features and both targets.

    Cells are annotated with one decimal place. Two decimals were too
    cramped once the annotation font size was increased for the report.
    """
    cols = list(FEATURE_LABELS.keys()) + list(TARGET_LABELS.keys())
    corr = df[cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)

    # Annotate each cell with its correlation coefficient.
    for i in range(len(cols)):
        for j in range(len(cols)):
            value = corr.iloc[i, j]
            # Avoid printing "-0.0" for values that round to zero.
            if abs(value) < 0.05:
                value = 0.0
            ax.text(
                j, i, f"{value:.1f}",
                ha="center", va="center", color="black", fontsize=10,
            )

    # Two-line title so it does not run into the colorbar.
    ax.set_title(
        "Correlation Matrix:\nFeatures (X1..X8) and Targets (Y1, Y2)"
    )
    fig.colorbar(im, ax=ax, label="Spearman correlation coefficient")
    fig.tight_layout()

    out_file = FIG_DIR / "correlation_heatmap.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")


def plot_target_histograms(df):
    """Histograms of the two target variables Y1 and Y2, stacked vertically
    so the figure fits a single narrow report column.
    """
    fig, axes = plt.subplots(2, 1, figsize=(6, 7))
    colors = {"Y1": "indianred", "Y2": "seagreen"}

    for ax, col in zip(axes, TARGET_LABELS.keys()):
        label = f"{col}: {TARGET_LABELS[col]}"
        ax.hist(df[col], bins=20, color=colors[col], edgecolor="black", label=label)
        ax.set_title(label)
        ax.set_xlabel(TARGET_LABELS[col])
        ax.set_ylabel("Count")
        ax.legend()

    fig.suptitle("Distribution of Heating and Cooling Load", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_file = FIG_DIR / "target_histograms.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")


def main():
    FIG_DIR.mkdir(exist_ok=True)

    df = load_data()
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {DATA_FILE}")

    plot_feature_histograms(df)
    plot_correlation_heatmap(df)
    plot_target_histograms(df)


if __name__ == "__main__":
    main()