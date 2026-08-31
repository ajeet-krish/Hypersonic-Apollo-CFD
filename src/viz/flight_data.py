"""Flight data comparison visualization for Apollo CM CFD results.

Bar/scatter plot comparing SU2 stagnation heat flux against published
Apollo flight data points, with condition annotations and caveat note.
Uses the reentry orange/amber theme.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, apply_theme


def plot_flight_data_comparison(
    su2_q_stag: float,
    su2_conditions: dict,
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Plot SU2 stagnation heat flux vs published Apollo flight data.

    Creates a combined bar/scatter plot showing the SU2 result alongside
    published Apollo CM flight data points. Includes a caveat note about
    the condition mismatch (different altitude/Mach).

    Args:
        su2_q_stag: SU2 stagnation heat flux (W/m^2).
        su2_conditions: Dictionary with 'mach', 'altitude_m', 'R_nose'.
        output_path: Path to save the plot.
        dpi: Image resolution.

    Returns:
        Path to saved plot.
    """
    apply_theme()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from validation.flight_data import apollo_flight_data

    flight_data = apollo_flight_data()

    # Filter to heat flux data points only
    heat_flux_points = [dp for dp in flight_data if "Heat Flux" in dp.quantity]

    # Build labels and values
    labels = []
    values = []
    sources = []
    missions = []

    for dp in heat_flux_points:
        # Abbreviate mission name for labels
        short_name = dp.mission.split("(")[0].strip() if "(" in dp.mission else dp.mission
        if "Sutton-Graves" in dp.mission:
            short_name = "Sutton-Graves\n(predicted)"
        labels.append(short_name)
        values.append(dp.value)
        sources.append(dp.source)
        missions.append(dp.mission)

    # Add SU2 result
    su2_q_w_cm2 = su2_q_stag / 10000.0
    labels.append("SU2 CFD\n(this work)")
    values.append(su2_q_w_cm2)
    sources.append("SU2 RANS")
    missions.append("SU2 CFD")

    n = len(labels)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(12, 3 * n), 8))

    # Color bars: flight data in accent colors, SU2 in highlight
    bar_colors = []
    for i, m in enumerate(missions):
        if "SU2" in m:
            bar_colors.append(COLORS["accent_highlight"])
        elif "Sutton-Graves" in m:
            bar_colors.append(COLORS["accent_secondary"])
        else:
            bar_colors.append(COLORS["accent_primary"])

    bars = ax.bar(
        x, values, 0.6,
        color=bar_colors, alpha=0.85,
        edgecolor=COLORS["grid"], linewidth=0.8,
    )

    # Value labels on each bar
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5.0,
            f"{val:.0f}",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            color=COLORS["text"],
        )

    # Error bar on SU2 bar showing the range of flight data
    flight_vals = [dp.value for dp in heat_flux_points
                   if "Sutton-Graves" not in dp.mission]
    if flight_vals:
        flight_min = min(flight_vals)
        flight_max = max(flight_vals)
        su2_idx = n - 1  # last bar is SU2
        ax.plot(
            [su2_idx - 0.15, su2_idx + 0.15],
            [flight_min, flight_min],
            color=COLORS["data_warn"], linewidth=1.5, linestyle="--",
        )
        ax.plot(
            [su2_idx - 0.15, su2_idx + 0.15],
            [flight_max, flight_max],
            color=COLORS["data_warn"], linewidth=1.5, linestyle="--",
        )
        ax.fill_between(
            [su2_idx - 0.3, su2_idx + 0.3],
            flight_min, flight_max,
            color=COLORS["data_warn"], alpha=0.1,
            label=f"Apollo flight range ({flight_min:.0f}-{flight_max:.0f} W/cm$^2$)",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color=COLORS["text"])
    ax.set_ylabel("Stagnation Heat Flux (W/cm$^2$)", fontsize=12, color=COLORS["text"])
    ax.set_title(
        "Apollo CM: Stagnation Heat Flux -- SU2 CFD vs Flight Data",
        fontsize=14, color=COLORS["text"], fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)

    # Condition annotation
    mach = su2_conditions.get("mach", 0.0)
    alt_km = su2_conditions.get("altitude_m", 0.0) / 1000.0
    r_nose = su2_conditions.get("R_nose", 0.0)

    info_text = (
        f"SU2 conditions: M={mach}, Alt={alt_km:.0f} km, R_nose={r_nose:.3f} m\n"
        f"Flight data at peak heating: M~35, Alt~55 km"
    )
    ax.text(
        0.02, 0.98, info_text,
        va="top", ha="left", transform=ax.transAxes,
        fontsize=9, color=COLORS["text"],
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLORS["bg_axes"],
            "edgecolor": COLORS["grid"],
        },
    )

    # Caveat note
    caveat = (
        "CAVEAT: Flight data are at peak heating (M~35, ~55 km). "
        "SU2 is at M=12, 30 km. Direct comparison is indicative only."
    )
    ax.text(
        0.5, -0.12, caveat,
        ha="center", va="top", transform=ax.transAxes,
        fontsize=8, color=COLORS["data_warn"], style="italic",
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path
