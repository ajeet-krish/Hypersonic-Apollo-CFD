"""Professional academic visualization theme.

Clean white background with standard matplotlib colors for
publication-quality plots.
"""
import matplotlib.pyplot as plt

COLORS = {
    "bg_dark": "#FFFFFF",
    "bg_axes": "#FFFFFF",
    "grid": "#DDDDDD",
    "text": "#000000",
    "text_dim": "#444444",
    "accent_primary": "#1F77B4",
    "accent_secondary": "#FF7F0E",
    "accent_highlight": "#2CA02C",
    "wall": "#333333",
    "shock": "#D62728",
    "heating": "#E6550D",
    "sphere": "#1F77B4",
    "cone": "#FF7F0E",
    "data_good": "#2CA02C",
    "data_warn": "#FF7F0E",
    "data_bad": "#D62728",
}

RC_PARAMS = {
    "figure.facecolor": COLORS["bg_dark"],
    "axes.facecolor": COLORS["bg_axes"],
    "axes.edgecolor": COLORS["text"],
    "axes.labelcolor": COLORS["text"],
    "axes.grid": True,
    "grid.color": COLORS["grid"],
    "grid.alpha": 0.5,
    "text.color": COLORS["text"],
    "xtick.color": COLORS["text_dim"],
    "ytick.color": COLORS["text_dim"],
    "font.family": "sans-serif",
    "font.size": 11,
    "figure.dpi": 150,
    "savefig.facecolor": COLORS["bg_dark"],
    "savefig.edgecolor": COLORS["bg_dark"],
    "legend.facecolor": COLORS["bg_axes"],
    "legend.edgecolor": COLORS["grid"],
    "legend.labelcolor": COLORS["text"],
}


def apply_theme() -> None:
    """Apply the professional academic theme to matplotlib."""
    plt.rcParams.update(RC_PARAMS)
