"""Convergence history plotting for SU2 hypersonic simulations.

Plots RMS residual convergence history.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, RC_PARAMS


def plot_convergence(
    history: list[dict],
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Plot RMS residual convergence history.

    Extracts residual columns (RMS_DENSITY, RMS_MOMENTUM-X, etc.) from
    SU2 history data and plots them vs iteration number on a log scale.

    Args:
        history: List of parsed history.csv row dicts from SU2.
        output_path: Path to save the convergence plot PNG.
        dpi: Plot resolution.

    Returns:
        Path to the saved plot.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not history:
        # Empty plot placeholder
        plt.rcParams.update(RC_PARAMS)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(
            0.5, 0.5, "No convergence data available",
            ha="center", va="center", transform=ax.transAxes,
            color=COLORS["text"], fontsize=14,
        )
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    plt.rcParams.update(RC_PARAMS)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Identify residual columns
    residual_keys = []
    for key in history[0]:
        key_lower = key.lower()
        if "rms" in key_lower or "residual" in key_lower:
            residual_keys.append(key)

    if not residual_keys:
        # Fallback: try all numeric columns except ITER/Inner
        for key in history[0]:
            if key.lower() not in ("iter", "inner_iter", "outer_iter", "time(s)"):
                try:
                    float(history[0][key])
                    residual_keys.append(key)
                except (ValueError, KeyError):
                    continue

    if not residual_keys:
        ax.text(
            0.5, 0.5, "No residual columns found in history",
            ha="center", va="center", transform=ax.transAxes,
            color=COLORS["text"], fontsize=14,
        )
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # Get iteration numbers
    iter_key = None
    for candidate in ("Inner_Iter", "ITER", "Iter", "Iteration"):
        if candidate in history[0]:
            iter_key = candidate
            break

    iterations = []
    for i, row in enumerate(history):
        if iter_key:
            try:
                iterations.append(int(float(row[iter_key])))
            except (ValueError, KeyError):
                iterations.append(i + 1)
        else:
            iterations.append(i + 1)

    iterations = np.array(iterations)

    # Plot each residual
    colors = [
        COLORS["accent_primary"],
        COLORS["accent_secondary"],
        COLORS["accent_highlight"],
        COLORS["data_good"],
        COLORS["shock"],
    ]

    for i, key in enumerate(residual_keys):
        values = []
        for row in history:
            try:
                values.append(float(row[key]))
            except (ValueError, KeyError):
                values.append(float("nan"))
        values = np.array(values)

        # Filter out zeros/negatives for log plot
        valid = values > 0
        if valid.any():
            color = colors[i % len(colors)]
            label = key.replace("rms[", "").replace("]", "").replace("RMS_", "")
            ax.semilogy(iterations[valid], values[valid], color=color, linewidth=1.5, label=label)

    ax.set_xlabel("Iteration", color=COLORS["text"], fontsize=12)
    ax.set_ylabel("RMS Residual", color=COLORS["text"], fontsize=12)
    ax.set_title("SU2 Convergence History", color=COLORS["text"], fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(left=max(0, iterations[0] - 1))

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path
