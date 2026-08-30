"""Reentry orange/amber visualization theme.

Dark theme with warm colors for hypersonic reentry visualizations.
"""
import matplotlib.pyplot as plt

COLORS = {
    "bg_dark": "#1A0A00",
    "bg_axes": "#2D1600",
    "grid": "#4A2800",
    "text": "#FFE0B2",
    "text_dim": "#BF8A60",
    "accent_primary": "#FF8F00",
    "accent_secondary": "#E65100",
    "accent_highlight": "#FFB300",
    "wall": "#FF6D00",
    "shock": "#FF1744",
    "heating": "#FF3D00",
    "sphere": "#FFB300",
    "cone": "#FF8F00",
    "data_good": "#66BB6A",
    "data_warn": "#FFA726",
    "data_bad": "#EF5350",
}

RC_PARAMS = {
    "figure.facecolor": COLORS["bg_dark"],
    "axes.facecolor": COLORS["bg_axes"],
    "axes.edgecolor": COLORS["grid"],
    "axes.labelcolor": COLORS["text"],
    "axes.grid": True,
    "grid.color": COLORS["grid"],
    "grid.alpha": 0.4,
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
    """Apply the reentry orange/amber theme to matplotlib."""
    plt.rcParams.update(RC_PARAMS)
