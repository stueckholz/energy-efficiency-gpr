"""
Shared matplotlib style for all figures in this project.

Figures are embedded in a two-column LaTeX layout, so each one is only
about 8 cm wide. Larger fonts and tight, minimal margins keep them readable
at that size. figure.dpi also sets the saved resolution, since none of the
plotting scripts pass an explicit dpi to savefig() anymore.
"""

import matplotlib.pyplot as plt

RC_PARAMS = {
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def apply_style():
    """Apply the shared rcParams. Call once near the top of each script
    that creates figures, before any plt.subplots() calls.
    """
    plt.rcParams.update(RC_PARAMS)
