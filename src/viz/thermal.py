"""Thermal analysis visualization for 2D heat shield results.

Provides three plot types:
1. Wall temperature distribution along body surface
2. Through-wall temperature profiles at selected locations
3. 2D temperature contour through the wall
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, apply_theme


def plot_wall_temperature(
    result: "ThermalResult2D",
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Plot wall temperature distribution along body surface.

    Shows T_wall(s) at final time: temperature at the hot face versus
    surface coordinate arc length from nose.

    Args:
        result: ThermalResult2D from the 2D thermal solver.
        output_path: Path to save the plot.
        dpi: Image resolution.

    Returns:
        Path to saved plot.
    """
    apply_theme()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    s = result.s
    T_wall = result.T_final[:, 0]

    # Wall temperature vs surface coordinate
    ax.plot(
        s, T_wall,
        color=COLORS["heating"], linewidth=2.0, label="T_wall (hot face)",
    )

    # Fill under curve
    ax.fill_between(
        s, result.config.get("cold_wall_temp", 300.0) if hasattr(result, "config") else 300.0,
        T_wall,
        color=COLORS["heating"], alpha=0.15,
    )

    # Mark maximum temperature
    idx_max = int(np.argmax(T_wall))
    T_max = T_wall[idx_max]
    s_max = s[idx_max]

    ax.plot(
        s_max, T_max, "o",
        color=COLORS["data_bad"], markersize=10, zorder=5,
        label=f"Peak: {T_max:.0f} K at s={s_max:.4f} m",
    )

    ax.annotate(
        f"T_max = {T_max:.0f} K",
        xy=(s_max, T_max),
        xytext=(s_max + (s[-1] - s[0]) * 0.05, T_max * 0.95),
        fontsize=10, color=COLORS["data_bad"], fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": COLORS["data_bad"], "lw": 1.2},
    )

    # Styling
    ax.set_xlabel("Arc Length s (m)", fontsize=12, color=COLORS["text"])
    ax.set_ylabel("Temperature T (K)", fontsize=12, color=COLORS["text"])
    ax.set_title(
        f"Wall Temperature Distribution ({result.material})",
        fontsize=14, color=COLORS["text"], fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(left=max(0, s[0]))

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_through_wall_profiles(
    result: "ThermalResult2D",
    output_path: Path,
    n_profiles: int = 5,
    dpi: int = 150,
) -> Path:
    """Plot temperature profiles through wall thickness at selected locations.

    Shows T(z) at n_profiles evenly spaced surface locations, illustrating
    how heat penetrates through the wall at different points along the body.

    Args:
        result: ThermalResult2D from the 2D thermal solver.
        output_path: Path to save the plot.
        n_profiles: Number of profiles to show (evenly spaced along s).
        dpi: Image resolution.

    Returns:
        Path to saved plot.
    """
    apply_theme()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    n_s = len(result.s)
    z = result.z

    # Select evenly spaced surface indices
    if n_profiles >= n_s:
        indices = list(range(n_s))
    else:
        indices = np.linspace(0, n_s - 1, n_profiles, dtype=int)

    # Color map from hot (stagnation) to cool (downstream)
    cmap = plt.cm.RdYlBu_r

    for k, i in enumerate(indices):
        fraction = k / max(len(indices) - 1, 1)
        color = cmap(0.2 + 0.6 * fraction)  # Avoid extreme ends of colormap
        s_val = result.s[i]
        T_profile = result.T_final[i, :]

        ax.plot(
            z * 1000, T_profile,  # Convert to mm
            color=color, linewidth=2.0,
            label=f"s={s_val:.4f} m",
        )

    # Styling
    ax.set_xlabel("Wall Thickness z (mm)", fontsize=12, color=COLORS["text"])
    ax.set_ylabel("Temperature T (K)", fontsize=12, color=COLORS["text"])
    ax.set_title(
        f"Through-Wall Temperature Profiles ({result.material})",
        fontsize=14, color=COLORS["text"], fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9, title="Surface location")
    ax.set_xlim(left=0.0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_temperature_contour(
    result: "ThermalResult2D",
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Plot 2D temperature contour T(s,z) through the wall.

    Shows the full temperature field as a 2D color map with surface
    coordinate on the x-axis and wall thickness on the y-axis.

    Args:
        result: ThermalResult2D from the 2D thermal solver.
        output_path: Path to save the plot.
        dpi: Image resolution.

    Returns:
        Path to saved plot.
    """
    apply_theme()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    s = result.s
    z = result.z * 1000  # Convert to mm
    T = result.T_final

    # Create meshgrid for contour plot
    S, Z = np.meshgrid(s, z, indexing="ij")

    # Plot temperature contour
    T_min = float(np.min(T))
    T_max = float(np.max(T))
    levels = np.linspace(T_min, T_max, 30)

    contour = ax.contourf(
        S, Z, T,
        levels=levels, cmap="RdYlBu_r", extend="both",
    )

    # Add contour lines
    ax.contour(
        S, Z, T,
        levels=levels[::3], colors="k", linewidths=0.3, alpha=0.4,
    )

    # Colorbar
    cbar = fig.colorbar(contour, ax=ax, label="Temperature (K)", pad=0.02)

    # Mark hot and cold faces
    ax.axhline(y=0, color=COLORS["heating"], linewidth=2.0, linestyle="--", alpha=0.7)
    ax.axhline(
        y=result.wall_thickness * 1000,
        color=COLORS["accent_primary"], linewidth=2.0, linestyle="--", alpha=0.7,
    )

    # Annotations
    ax.text(
        s[-1] * 0.02, result.wall_thickness * 1000 * 0.05,
        "Hot face", fontsize=9, color=COLORS["heating"], fontweight="bold",
    )
    ax.text(
        s[-1] * 0.02, result.wall_thickness * 1000 * 0.92,
        "Cold face", fontsize=9, color=COLORS["accent_primary"], fontweight="bold",
    )

    # Styling
    ax.set_xlabel("Surface Coordinate s (m)", fontsize=12, color=COLORS["text"])
    ax.set_ylabel("Wall Thickness z (mm)", fontsize=12, color=COLORS["text"])
    ax.set_title(
        f"Temperature Contour Through Wall ({result.material})",
        fontsize=14, color=COLORS["text"], fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_char_layer(
    result: "AblationResult1D | AblationResult2D",
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Plot char layer formation through wall thickness.

    Shows the density profile from virgin to charred material, with the
    char front marked. For 1D results, plots a single profile. For 2D
    results, plots the stagnation-point profile.

    Args:
        result: AblationResult1D or AblationResult2D from the thermal solver.
        output_path: Path to save the plot.
        dpi: Image resolution.

    Returns:
        Path to saved plot.
    """
    apply_theme()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    z = result.z * 1000  # Convert to mm

    # Get density profiles
    rho_initial = result.rho_initial
    rho_final = result.rho_final

    # For 2D results, use stagnation point (first surface index)
    if rho_initial.ndim == 2:
        rho_initial = rho_initial[0, :]
        rho_final = rho_final[0, :]

    # Plot virgin and charred profiles
    ax.plot(
        z, rho_initial,
        color=COLORS["accent_primary"], linewidth=2.0, linestyle="--",
        label="Virgin (initial)",
    )
    ax.plot(
        z, rho_final,
        color=COLORS["heating"], linewidth=2.5,
        label="Final (after heating)",
    )

    # Fill the charred region
    rho_v = float(np.max(rho_initial))
    rho_c = float(np.min(rho_final))
    char_mask = rho_final < (rho_v + rho_c) / 2.0
    if np.any(char_mask):
        ax.fill_between(
            z, rho_c * 0.9, rho_final,
            where=char_mask,
            color=COLORS["data_bad"], alpha=0.15,
            label="Char layer",
        )

    # Mark char front
    char_threshold = (rho_v + rho_c) / 2.0
    char_idx = np.argmax(rho_final < char_threshold)
    if char_idx > 0:
        ax.axvline(
            x=z[char_idx], color=COLORS["data_warn"],
            linewidth=1.5, linestyle=":", alpha=0.8,
            label=f"Char front: {z[char_idx]:.2f} mm",
        )

    # Styling
    ax.set_xlabel("Wall Thickness z (mm)", fontsize=12, color=COLORS["text"])
    ax.set_ylabel("Density rho (kg/m^3)", fontsize=12, color=COLORS["text"])
    ax.set_title(
        f"Char Layer Formation ({result.material})",
        fontsize=14, color=COLORS["text"], fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_ablation_rate(
    result: "AblationResult1D | AblationResult2D",
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Plot surface recession rate along body surface.

    Shows the ablation rate (mm/s) at each surface point. For 1D results,
    shows a single value. For 2D results, shows variation along the surface.

    Args:
        result: AblationResult1D or AblationResult2D from the thermal solver.
        output_path: Path to save the plot.
        dpi: Image resolution.

    Returns:
        Path to saved plot.
    """
    apply_theme()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    rate = result.ablation_rate_mm_s

    if np.isscalar(rate) or (hasattr(rate, "ndim") and rate.ndim == 0):
        # 1D result: single value, show as bar
        ax.bar(
            [0], [float(rate)],
            color=COLORS["heating"], width=0.5,
            label=f"Ablation rate: {float(rate):.4f} mm/s",
        )
        ax.set_xlim(-1, 1)
    else:
        # 2D result: plot along surface
        s = result.s
        ax.plot(
            s, rate,
            color=COLORS["heating"], linewidth=2.5,
            label="Ablation rate",
        )
        ax.fill_between(
            s, 0, rate,
            color=COLORS["heating"], alpha=0.15,
        )

        # Mark maximum
        idx_max = int(np.argmax(rate))
        rate_max = rate[idx_max]
        s_max = s[idx_max]
        ax.plot(
            s_max, rate_max, "o",
            color=COLORS["data_bad"], markersize=10, zorder=5,
            label=f"Peak: {rate_max:.4f} mm/s",
        )

    # Styling
    if np.isscalar(rate) or (hasattr(rate, "ndim") and rate.ndim == 0):
        ax.set_xlabel("Surface Point", fontsize=12, color=COLORS["text"])
    else:
        ax.set_xlabel("Arc Length s (m)", fontsize=12, color=COLORS["text"])
    ax.set_ylabel("Ablation Rate (mm/s)", fontsize=12, color=COLORS["text"])
    ax.set_title(
        f"Surface Recession Rate ({result.material})",
        fontsize=14, color=COLORS["text"], fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.set_ylim(bottom=0.0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_mass_loss(
    result: "AblationResult1D | AblationResult2D",
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Plot mass loss history over time.

    Shows the cumulative mass loss per unit area (kg/m^2) as a function
    of time, derived from the density history.

    Args:
        result: AblationResult1D or AblationResult2D from the thermal solver.
        output_path: Path to save the plot.
        dpi: Image resolution.

    Returns:
        Path to saved plot.
    """
    apply_theme()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    t = result.t_history
    rho_history = result.rho_history

    if rho_history is not None and len(t) > 1:
        # Determine virgin density from the material name
        if "AVCOAT" in result.material:
            rho_v = 512.0
        elif "PICA" in result.material:
            rho_v = 240.0
        else:
            rho_v = 512.0

        if rho_history.ndim == 2 and rho_history.shape[0] > 1:
            # Compute mass loss over time by integrating through wall
            mass_loss = np.array([
                np.trapezoid(rho_v - rho_history[k, :], result.z)
                for k in range(len(t))
            ])

            ax.plot(
                t, mass_loss,
                color=COLORS["heating"], linewidth=2.5,
                label="Mass loss",
            )
            ax.fill_between(
                t, 0, mass_loss,
                color=COLORS["heating"], alpha=0.15,
            )

            # Mark final value
            ax.annotate(
                f"{mass_loss[-1]:.4f} kg/m^2",
                xy=(t[-1], mass_loss[-1]),
                xytext=(t[-1] * 0.8, mass_loss[-1] * 0.85),
                fontsize=10, color=COLORS["heating"], fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": COLORS["heating"], "lw": 1.2},
            )

    # Styling
    ax.set_xlabel("Time t (s)", fontsize=12, color=COLORS["text"])
    ax.set_ylabel("Cumulative Mass Loss (kg/m^2)", fontsize=12, color=COLORS["text"])
    ax.set_title(
        f"Mass Loss History ({result.material})",
        fontsize=14, color=COLORS["text"], fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path
