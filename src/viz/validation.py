"""Validation and GCI visualization for hypersonic blunt body CFD.

Bar chart of SU2 vs analytical for each quantity with error % labels,
and GCI convergence plot showing quantity vs cell count with Richardson
extrapolation.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, apply_theme


def plot_validation_bars(
    validation_report: dict,
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Plot bar chart comparing SU2 vs analytical for each quantity.

    Shows paired bars (SU2 vs analytical) for each validated quantity,
    with error % labels and PASS/FAIL coloring using the orange/amber
    theme.

    Args:
        validation_report: Report dict from build_validation_report().
        output_path: Path to save the plot.
        dpi: Image resolution.

    Returns:
        Path to saved plot.
    """
    apply_theme()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = validation_report.get("results", [])
    if not results:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5, 0.5, "No validation results available",
            ha="center", va="center", transform=ax.transAxes,
            color=COLORS["text"], fontsize=14,
        )
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    n = len(results)
    quantities = [r["quantity"] for r in results]
    su2_vals = [r["su2_value"] for r in results]
    analytical_vals = [r["analytical_value"] for r in results]
    errors = [r["error_pct"] for r in results]
    statuses = [r["status"] for r in results]

    fig, ax = plt.subplots(figsize=(max(10, 4 * n), 7))

    x = np.arange(n)
    width = 0.35

    # Normalize values for display (relative to analytical)
    su2_rel = []
    ana_rel = []
    for sv, av in zip(su2_vals, analytical_vals):
        if av > 0:
            su2_rel.append(sv / av * 100.0)
            ana_rel.append(100.0)
        else:
            su2_rel.append(0.0)
            ana_rel.append(0.0)

    ax.bar(
        x - width / 2, su2_rel, width,
        label="SU2", color=COLORS["accent_primary"], alpha=0.85,
        edgecolor=COLORS["grid"], linewidth=0.8,
    )
    ax.bar(
        x + width / 2, ana_rel, width,
        label="Analytical", color=COLORS["accent_highlight"], alpha=0.85,
        edgecolor=COLORS["grid"], linewidth=0.8,
    )

    # Error % labels and PASS/FAIL coloring
    for i, (err, status) in enumerate(zip(errors, statuses)):
        if status == "PASS":
            color = COLORS["data_good"]
        elif err < 30:
            color = COLORS["data_warn"]
        else:
            color = COLORS["data_bad"]

        # Label above the taller bar
        max_h = max(su2_rel[i], ana_rel[i])
        ax.text(
            i, max_h + 2.0,
            f"{err:.1f}%",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=color,
        )
        ax.text(
            i, max_h + 6.0,
            status,
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=color,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(quantities, fontsize=10, color=COLORS["text"])
    ax.set_ylabel("Relative Value (%)", fontsize=12, color=COLORS["text"])
    ax.set_title(
        "Validation: SU2 vs Analytical Correlations",
        fontsize=14, color=COLORS["text"], fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)

    # Reference line at 100%
    ax.axhline(y=100, color=COLORS["text_dim"], linestyle="--", linewidth=0.8, alpha=0.6)

    # Case info
    case_name = validation_report.get("case", "")
    mach = validation_report.get("mach", "")
    alt = validation_report.get("altitude_m", 0)
    all_pass = validation_report.get("all_pass", False)
    status_text = "ALL PASSED" if all_pass else "SOME FAILED"
    status_color = COLORS["data_good"] if all_pass else COLORS["data_bad"]

    info_text = f"Case: {case_name}  |  M={mach}  |  Alt={alt/1000:.0f} km  |  {status_text}"
    ax.text(
        0.5, -0.15, info_text,
        ha="center", va="top", transform=ax.transAxes,
        fontsize=10, color=status_color, fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_gci_convergence(
    gci_result: dict,
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Plot GCI convergence: quantity vs cell count for 3 mesh levels.

    Shows the three mesh levels as data points, the Richardson-
    extrapolated value as a horizontal dashed line, and GCI error bars.

    Args:
        gci_result: GCI result dict from compute_gci() output.
        output_path: Path to save the plot.
        dpi: Image resolution.

    Returns:
        Path to saved plot.
    """
    apply_theme()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    levels = gci_result.get("levels", [])
    if len(levels) < 3:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5, 0.5, "Insufficient GCI data",
            ha="center", va="center", transform=ax.transAxes,
            color=COLORS["text"], fontsize=14,
        )
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    n_cells = [lv["n_cells"] for lv in levels]
    values = [lv["value"] for lv in levels]
    names = [lv["name"] for lv in levels]
    extrapolated = gci_result.get("extrapolated_value", values[-1])
    gci_fine_pct = gci_result.get("gci_fine_pct", 0.0)
    order = gci_result.get("apparent_order", 0.0)
    asymptotic = gci_result.get("asymptotic_ratio", 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mesh level data points
    ax.plot(
        n_cells, values, "o-",
        color=COLORS["accent_primary"], markersize=10,
        linewidth=2.0, label="Computed values", zorder=5,
    )

    # Mark individual levels with labels
    markers = ["s", "^", "D"]
    colors_tier = [COLORS["data_bad"], COLORS["data_warn"], COLORS["data_good"]]
    for i, (nc, v, name) in enumerate(zip(n_cells, values, names)):
        ax.plot(
            nc, v, markers[i],
            color=colors_tier[i], markersize=12,
            label=f"{name} ({nc:,} cells)",
            zorder=6,
        )

    # Richardson extrapolation line
    ax.axhline(
        y=extrapolated, color=COLORS["accent_highlight"],
        linestyle="--", linewidth=1.5,
        label=f"Richardson extrapolation: {extrapolated:.4f}",
    )

    # GCI error band around extrapolation
    gci_abs = abs(extrapolated * gci_fine_pct / 100.0)
    ax.fill_between(
        [n_cells[0] * 0.5, n_cells[-1] * 2],
        extrapolated - gci_abs, extrapolated + gci_abs,
        color=COLORS["accent_highlight"], alpha=0.1,
        label=f"GCI fine: {gci_fine_pct:.2f}%",
    )

    # Labels
    quantity = gci_result.get("quantity", "Quantity")
    ax.set_xlabel("Number of Cells", fontsize=12, color=COLORS["text"])
    ax.set_ylabel(quantity, fontsize=12, color=COLORS["text"])
    ax.set_title(
        f"GCI Mesh Convergence: {quantity}",
        fontsize=14, color=COLORS["text"], fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9)

    # Info box
    info = (
        f"Order p = {order:.2f}\n"
        f"GCI (fine) = {gci_fine_pct:.2f}%\n"
        f"Asymptotic ratio = {asymptotic:.2f}\n"
        f"Monotonic: {'Yes' if gci_result.get('monotonic', False) else 'No'}"
    )
    ax.text(
        0.02, 0.98, info,
        va="top", ha="left", transform=ax.transAxes,
        fontsize=9, color=COLORS["text"],
          bbox={
              "boxstyle": "round,pad=0.5",
              "facecolor": COLORS["bg_axes"], "edgecolor": COLORS["grid"],
          },
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path
