"""Generate publication-quality geometry plots for the Apollo CM."""

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on path
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
            "base_fillet": (
                n_sphere + n_fillet + n_cone,
                n_sphere + n_fillet + n_cone + n_base_fillet,
            ),
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
            "fillet": (n_sphere, n_sphere),  # empty
            "cone": (n_sphere, n_sphere + n_cone),
            "base_fillet": (n_sphere + n_cone, n_sphere + n_cone + n_base_fillet),
        }


def compute_section_boundaries(config, sections):
    """Compute junction x/r coordinates for each section boundary."""
    R = config.R_shield
    Rf = config.R_fillet
    theta = config.half_angle_rad
    max_r = config.max_radius
    Rbf = config.base_fillet_radius

    if Rf > 0:
        # Sphere-fillet junction
        cos_phi = (R + Rf - max_r) / (R + Rf)
        phi_sf = math.acos(max(-1.0, min(1.0, cos_phi)))
        x_sf = R * (1.0 - math.cos(phi_sf))
        r_sf = R * math.sin(phi_sf)

        # Fillet center
        x_f = R - (R + Rf) * math.cos(phi_sf)
        r_f = (R + Rf) * math.sin(phi_sf)

        # Fillet-cone junction
        x_tc = x_f + Rf * math.sin(theta)
        r_tc = r_f + Rf * math.cos(theta)

        return {
            "nose": (0.0, 0.0),
            "sphere_fillet": (x_sf, r_sf),
            "fillet_cone": (x_tc, r_tc),
            "base": (config.computed_body_length, config.base_radius),
        }
    else:
        phi_j = math.asin(max_r / R)
        x_j = R * (1.0 - math.cos(phi_j))
        r_j = R * math.sin(phi_j)
        return {
            "nose": (0.0, 0.0),
            "sphere_cone": (x_j, r_j),
            "base": (config.computed_body_length, config.base_radius),
        }


def plot_2d_geometry(config, x, r, sections, boundaries, out_path):
    """Create publication-quality 2D geometry plot with annotations."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Unpack section ranges
    i_sph = slice(*sections["sphere"])
    i_fil = slice(*sections["fillet"])
    i_con = slice(*sections["cone"])
    i_bf = slice(*sections["base_fillet"])

    # Plot color-coded upper profile
    ax.plot(x[i_sph], r[i_sph], color="#2563EB", linewidth=2.2, label="Spherical heat shield")
    if sections["fillet"][1] > sections["fillet"][0]:
        ax.plot(x[i_fil], r[i_fil], color="#7C3AED", linewidth=2.2, label="Shoulder fillet")
    ax.plot(x[i_con], r[i_con], color="#F97316", linewidth=2.2, label="Conical afterbody")
    if sections["base_fillet"][1] > sections["base_fillet"][0]:
        ax.plot(x[i_bf], r[i_bf], color="#DC2626", linewidth=2.2, label="Base fillet")

    # Mirror below axis
    ax.plot(x[i_sph], -r[i_sph], color="#2563EB", linewidth=2.2)
    if sections["fillet"][1] > sections["fillet"][0]:
        ax.plot(x[i_fil], -r[i_fil], color="#7C3AED", linewidth=2.2)
    ax.plot(x[i_con], -r[i_con], color="#F97316", linewidth=2.2)
    if sections["base_fillet"][1] > sections["base_fillet"][0]:
        ax.plot(x[i_bf], -r[i_bf], color="#DC2626", linewidth=2.2)

    # Symmetry axis
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

    # Junction markers
    for name, (jx, jr) in boundaries.items():
        if name == "nose":
            continue
        ax.plot(jx, jr, "ko", markersize=5, zorder=5)
        ax.plot(jx, -jr, "ko", markersize=5, zorder=5)

    # --- Dimension annotations ---
    L = config.computed_body_length
    R_shield = config.R_shield
    max_r = config.max_radius
    R_base = config.base_radius
    theta_deg = config.cone_half_angle

    # R_shield: arrow from sphere center to surface
    # Sphere center is at (R_shield, 0) for concave sphere
    sph_center_x = R_shield
    # Mark sphere center
    ax.plot(sph_center_x, 0, "b+", markersize=10, markeredgewidth=1.5, zorder=5)
    # Arrow from center to a point on the sphere (e.g., at 60% of sphere arc)
    phi_ann = 0.6 * math.asin(max_r / R_shield)
    x_ann = R_shield * (1.0 - math.cos(phi_ann))
    r_ann = R_shield * math.sin(phi_ann)
    ax.annotate(
        "", xy=(x_ann, r_ann), xytext=(sph_center_x, 0),
        arrowprops=dict(arrowstyle="<->", color="#2563EB", lw=1.5),
    )
    ax.text(
        (x_ann + sph_center_x) / 2 - 0.15, r_ann / 2 + 0.05,
        f"$R_{{shield}}$ = {R_shield:.3f} m",
        fontsize=10, color="#2563EB", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#2563EB", alpha=0.9),
    )

    # R_max: horizontal line at max radius
    ax.annotate(
        "", xy=(0, max_r), xytext=(0, 0),
        arrowprops=dict(arrowstyle="<->", color="#10B981", lw=1.5),
    )
    ax.text(
        -0.08, max_r / 2,
        f"$R_{{max}}$ = {max_r:.3f} m",
        fontsize=10, color="#10B981", fontweight="bold", rotation=90,
        ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#10B981", alpha=0.9),
    )

    # R_base: horizontal line at base
    ax.annotate(
        "", xy=(L, R_base), xytext=(L, 0),
        arrowprops=dict(arrowstyle="<->", color="#DC2626", lw=1.5),
    )
    ax.text(
        L + 0.03, R_base / 2,
        f"$R_{{base}}$ = {R_base:.3f} m",
        fontsize=9, color="#DC2626", fontweight="bold", rotation=90,
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#DC2626", alpha=0.9),
    )

    # L: horizontal dimension at bottom
    y_dim = -max_r - 0.15
    ax.annotate(
        "", xy=(L, y_dim), xytext=(0, y_dim),
        arrowprops=dict(arrowstyle="<->", color="#6B7280", lw=1.5),
    )
    ax.text(
        L / 2, y_dim - 0.08,
        f"$L$ = {L:.3f} m",
        fontsize=10, color="#6B7280", fontweight="bold",
        ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#6B7280", alpha=0.9),
    )

    # theta: cone half-angle arc at fillet-cone junction
    if "fillet_cone" in boundaries:
        jx, jr = boundaries["fillet_cone"]
    elif "sphere_cone" in boundaries:
        jx, jr = boundaries["sphere_cone"]
    else:
        jx, jr = boundaries.get("sphere_fillet", (0, 0))

    # Draw angle arc
    arc_r = 0.25
    theta_rad = math.radians(theta_deg)
    arc_phi = np.linspace(-theta_rad / 2, theta_rad / 2, 40)
    # The cone starts at the junction; angle is measured from the cone axis
    # Place the arc at the junction, showing the cone opening
    arc_x = jx + arc_r * np.cos(np.linspace(math.pi - theta_rad, math.pi, 40))
    arc_y = jr + arc_r * np.sin(np.linspace(math.pi - theta_rad, math.pi, 40))
    ax.plot(arc_x, arc_y, color="#F97316", linewidth=1.5)
    # Label
    ax.text(
        jx - 0.12, jr + 0.12,
        f"$\\theta$ = {theta_deg:.0f}$^\\circ$",
        fontsize=10, color="#F97316", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#F97316", alpha=0.9),
    )

    # Formatting
    ax.set_xlabel("Axial coordinate $x$ (m)", fontsize=12)
    ax.set_ylabel("Radial coordinate $r$ (m)", fontsize=12)
    ax.set_title("Apollo Command Module -- DXF-Verified Geometry", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.2, L + 0.5)
    ax.set_ylim(-max_r - 0.45, max_r + 0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved 2D plot: {out_path}")


def plot_3d_geometry(config, x, r, sections, out_path):
    """Create 3D revolved surface plot with plasma colormap."""
    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Revolve around x-axis
    theta_arr = np.linspace(0, 2 * np.pi, 120)
    X, THETA = np.meshgrid(x, theta_arr)
    R_mesh = np.tile(r, (len(theta_arr), 1))
    Y = R_mesh * np.cos(THETA)
    Z = R_mesh * np.sin(THETA)

    # Plot surface
    surf = ax.plot_surface(
        X, Y, Z,
        cmap="plasma",
        linewidth=0,
        antialiased=True,
        alpha=0.95,
        rstride=1,
        cstride=1,
    )

    # Formatting
    ax.set_xlabel("$x$ (m)", fontsize=11, labelpad=8)
    ax.set_ylabel("$y$ (m)", fontsize=11, labelpad=8)
    ax.set_zlabel("$z$ (m)", fontsize=11, labelpad=8)
    ax.set_title("Apollo CM -- 3D Revolved Geometry", fontsize=13, fontweight="bold", pad=15)

    # Equal aspect
    max_range = max(config.computed_body_length, 2 * config.max_radius) / 2
    mid_x = config.computed_body_length / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, label="$x$ (m)", pad=0.1)

    # View angle
    ax.view_init(elev=20, azim=-60)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved 3D plot: {out_path}")


def main():
    config = apollo_cm()
    x, r = generate_contour(config)

    sections = identify_sections(config)
    boundaries = compute_section_boundaries(config, sections)

    out_dir = Path(__file__).resolve().parent.parent / "docs" / "assets" / "images" / "apollo-cm"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_2d_geometry(config, x, r, sections, boundaries, out_dir / "geometry.png")
    plot_3d_geometry(config, x, r, sections, out_dir / "body_3d.png")


if __name__ == "__main__":
    main()
