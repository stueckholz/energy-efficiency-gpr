"""
Fetch the UCI Energy Efficiency dataset (Tsanas and Xifara, 2012) and save it
locally as a CSV. The course requires a fetch script that only downloads the
data and adds no further functionality.

Dataset: https://archive.ics.uci.edu/dataset/242/energy+efficiency
UCI id: 242

Features (X1..X8):
    X1  Relative Compactness
    X2  Surface Area
    X3  Wall Area
    X4  Roof Area
    X5  Overall Height
    X6  Orientation
    X7  Glazing Area
    X8  Glazing Area Distribution

Targets:
    Y1  Heating Load
    Y2  Cooling Load
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUT_FILE = DATA_DIR / "energy_efficiency.csv"


def main():
    DATA_DIR.mkdir(exist_ok=True)

    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=242)
    X = dataset.data.features
    y = dataset.data.targets

    df = X.join(y)
    df.to_csv(OUT_FILE, index=False)

    print(f"Saved {len(df)} rows and {df.shape[1]} columns to {OUT_FILE}")


if __name__ == "__main__":
    main()
