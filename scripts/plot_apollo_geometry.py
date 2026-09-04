"""Generate publication-quality geometry plots for the Apollo CM."""

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geometry.presets import apollo_cm
from geometry.blunt_body import generate_contour


def identify_sections(config):
    """Return index ranges for each body section in the contour."""
    if config.R_fillet > 0:
        n_sphere = int(config.num_points * 0.25)
        n_fillet = int(config.num_points * 0.15)
        n_cone = int(config.num_points * 0.45)
        Rbf = config.base_fillet_radius
        if Rbf > 0:
            n_base_fillet = config.num_points - n_sphere - n_fillet - n_cone
        else:
            n_base_fillet = 0
            n_cone = config.num_points - n_sphere - n_fillet
        return {
            "sphere": (0, n_sphere),
            "fillet": (n_sphere, n_sphere + n_fillet),
            "cone": (n_sphere + n_fillet, n_sphere + n_fillet + n_cone),
            "base_fillet": (n_sphere + n_fillet + n_cone, n_sphere + n_fillet + n_cone + n_base_fillet),
        }
    else:
        n_sphere = int(config.num_points * 0.4)
        Rbf = config.base_fillet_radius
        if Rbf > 0:
            n_cone = int(config.num_points * 0.45)
            n_base_fillet = config.num_points - n_sphere - n_cone
        else:
            n_cone = config.num_points - n_sphere
            n_base_fillet = 0
        return {
            "sphere": (0, n_sphere),
            "cone": (n_sphere, n_sphere + n_cone),
            "base_fillet": (n_sphere + n_cone, n_sphere + n_cone + n_base_fillet),
        }


def plot_2d_geometry(config, x, r, sections, out_path):
    """Create 2D annotated geometry plot matching Image 1 style."""
    L = config.computed_body_length
    max_r = config.max_radius

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # --- Body sections ---
    s = sections

    # Sphere (blue)
    ax.plot(x[s["sphere"][0]:s["sphere"][1]], r[s["sphere"][0]:s["sphere"][1]],
            color="#1565C0", linewidth=2.5, label="Spherical heat shield", zorder=5)
    ax.plot(x[s["sphere"][0]:s["sphere"][1]], -r[s["sphere"][0]:s["sphere"][1]],
            color="#1565C0", linewidth=2.5, zorder=5)

    # Shoulder fillet (purple)
    if "fillet" in s:
        ax.plot(x[s["fillet"][0]:s["fillet"][1]], r[s["fillet"][0]:s["fillet"][1]],
                color="#6A1B9A", linewidth=2.5, label="Shoulder fillet", zorder=5)
        ax.plot(x[s["fillet"][0]:s["fillet"][1]], -r[s["fillet"][0]:s["fillet"][1]],
                color="#6A1B9A", linewidth=2.5, zorder=5)

    # Cone (orange)
    ax.plot(x[s["cone"][0]:s["cone"][1]], r[s["cone"][0]:s["cone"][1]],
            color="#E65100", linewidth=2.5, label="Conical afterbody", zorder=5)
    ax.plot(x[s["cone"][0]:s["cone"][1]], -r[s["cone"][0]:s["cone"][1]],
            color="#E65100", linewidth=2.5, zorder=5)

    # Base fillet (red)
    if "base_fillet" in s and s["base_fillet"][1] > s["base_fillet"][0]:
        ax.plot(x[s["base_fillet"][0]:s["base_fillet"][1]], r[s["base_fillet"][0]:s["base_fillet"][1]],
                color="#C62828", linewidth=2.5, label="Base fillet", zorder=5)
        ax.plot(x[s["base_fillet"][0]:s["base_fillet"][1]], -r[s["base_fillet"][0]:s["base_fillet"][1]],
                color="#C62828", linewidth=2.5, zorder=5)

    # --- Axis of symmetry ---
    ax.axhline(y=0, color="#888888", linestyle="--", linewidth=0.8, zorder=3)

    # --- Junction markers (black dots) ---
    # Sphere-fillet junction
    if "fillet" in s:
        jx = x[s["fillet"][0]]
        jr = r[s["fillet"][0]]
        ax.plot(jx, jr, "o", color="black", markersize=6, zorder=6)
        ax.plot(jx, -jr, "o", color="black", markersize=6, zorder=6)

    # Fillet-cone junction
    if "fillet" in s:
        jx2 = x[s["cone"][0]]
        jr2 = r[s["cone"][0]]
        ax.plot(jx2, jr2, "o", color="black", markersize=6, zorder=6)
        ax.plot(jx2, -jr2, "o", color="black", markersize=6, zorder=6)

    # Cone-base junction
    jx3 = x[s["cone"][1] - 1]
    jr3 = r[s["cone"][1] - 1]
    ax.plot(jx3, jr3, "o", color="black", markersize=6, zorder=6)
    ax.plot(jx3, -jr3, "o", color="black", markersize=6, zorder=6)

    # --- Dimension annotations ---
    # R_max (vertical arrow at x=0)
    ax.annotate("", xy=(0, max_r), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<->", color="#2E7D32", lw=1.5))
    ax.text(-0.15, max_r / 2, f"$R_{{max}}$ = {max_r:.3f} m",
            fontsize=10, color="#2E7D32", fontweight="bold",
            ha="right", va="center", rotation=90)

    # R_base (vertical arrow at base)
    base_r = config.base_radius
    ax.annotate("", xy=(L, base_r), xytext=(L, 0),
                arrowprops=dict(arrowstyle="<->", color="#C62828", lw=1.2))
    ax.text(L + 0.15, base_r / 2, f"$R_{{base}}$ = {base_r:.3f} m",
            fontsize=9, color="#C62828", fontweight="bold",
            ha="left", va="center", rotation=90)

    # L (horizontal arrow along axis)
    ax.annotate("", xy=(0, -max_r - 0.3), xytext=(L, -max_r - 0.3),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.2))
    ax.text(L / 2, -max_r - 0.5, f"$L$ = {L:.3f} m",
            fontsize=10, color="#333333", fontweight="bold", ha="center")

    # R_shield (annotation at sphere center)
    R_s = config.R_shield
    ax.annotate(f"$R_{{shield}}$ = {R_s:.3f} m",
                xy=(0.3, max_r * 0.4), fontsize=10, color="#1565C0", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#1565C0", alpha=0.9))

    # Theta (cone angle)
    ax.annotate(f"$\\theta$ = {config.cone_half_angle:.0f}$^\\circ$",
                xy=(L / 2, -max_r - 0.8), fontsize=11, color="#E65100", fontweight="bold",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#E65100", alpha=0.9))

    # --- Styling ---
    ax.set_xlabel("Axial coordinate $x$ (m)", fontsize=12)
    ax.set_ylabel("Radial coordinate $r$ (m)", fontsize=12)
    ax.set_title("Apollo Command Module -- DXF-Verified Geometry",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 4.2)
    ax.set_ylim(-max_r - 1.0, max_r + 0.5)
    ax.tick_params(colors="black", labelsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved 2D plot: {out_path}")


def plot_3d_geometry(config, x, r, sections, out_path):
    """Create 3D revolved surface plot with plasma colormap."""
    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    n_rev = 120
    theta = np.linspace(0, 2 * np.pi, n_rev)
    X = np.repeat(x[:, np.newaxis], n_rev, axis=1)
    Y = r[:, np.newaxis] * np.cos(theta)
    Z = r[:, np.newaxis] * np.sin(theta)

    norm = plt.Normalize(X.min(), X.max())
    colors = plt.cm.plasma(norm(X))

    ax.plot_surface(X, Y, Z, facecolors=colors, shade=True, alpha=0.92)

    ax.set_xlabel("Axial (m)", fontsize=10, labelpad=8)
    ax.set_ylabel("Radial (m)", fontsize=10, labelpad=8)
    ax.set_zlabel("Radial (m)", fontsize=10, labelpad=8)
    ax.set_title("Apollo CM -- 3D Revolved Geometry", fontsize=12, fontweight="bold", pad=15)

    max_range = max(config.computed_body_length, config.max_radius) * 0.7
    mid_x = config.computed_body_length / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.view_init(elev=-170.0, azim=-15.0)

    mappable = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
    mappable.set_array(X)
    fig.colorbar(mappable, ax=ax, shrink=0.6, label="Axial position (m)")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved 3D plot: {out_path}")


def main():
    config = apollo_cm()
    x, r = generate_contour(config)
    sections = identify_sections(config)

    out_dir = Path("docs/assets/images/apollo-cm")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_2d_geometry(config, x, r, sections, out_dir / "geometry.png")
    plot_3d_geometry(config, x, r, sections, out_dir / "body_3d.png")


if __name__ == "__main__":
    main()
