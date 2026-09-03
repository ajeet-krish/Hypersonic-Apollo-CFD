"""Surface heat flux plotting for hypersonic blunt body CFD results.

Line plot of heat flux vs arc length along the body surface, with the
stagnation point peak marked.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, apply_theme


def plot_surface_heat_flux(
    s_profile: np.ndarray,
    q_profile: np.ndarray,
    output_path: Path,
    title: str = "Surface Heat Flux",
    dpi: int = 150,
) -> Path:
    """Plot surface heat flux vs arc length.

    Shows the heat flux distribution along the body surface with the
    stagnation point peak highlighted.

    Args:
        s_profile: Arc length along the body surface (m).
        q_profile: Heat flux values (W/m^2).
        output_path: Path to save the plot.
        title: Plot title.
        dpi: Image resolution.

    Returns:
        Path to saved plot.
    """
    apply_theme()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    if len(s_profile) == 0 or len(q_profile) == 0:
        ax.text(
            0.5, 0.5, "No heat flux data available",
            ha="center", va="center", transform=ax.transAxes,
            color=COLORS["text"], fontsize=14,
        )
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # Plot heat flux vs arc length
    ax.plot(
        s_profile, q_profile / 1000.0,
        color=COLORS["heating"], linewidth=2.0, label="Heat flux",
    )

    # Fill under curve
    ax.fill_between(
        s_profile, 0, q_profile / 1000.0,
        color=COLORS["heating"], alpha=0.15,
    )

    # Mark stagnation point (maximum heat flux)
    idx_max = int(np.argmax(q_profile))
    q_max = q_profile[idx_max] / 1000.0
    s_max = s_profile[idx_max]

    ax.plot(
        s_max, q_max, "o",
        color=COLORS["accent_highlight"], markersize=10, zorder=5,
        label=f"Peak: {q_max:.1f} kW/m^2 at s={s_max:.4f} m",
    )

    # Annotate the peak
    ax.annotate(
        f"q_max = {q_max:.1f} kW/m^2",
        xy=(s_max, q_max),
        xytext=(s_max + (s_profile[-1] - s_profile[0]) * 0.05, q_max * 0.9),
        fontsize=10, color=COLORS["accent_highlight"], fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": COLORS["accent_highlight"], "lw": 1.2},
    )

    # Styling
    ax.set_xlabel("Arc Length s (m)", fontsize=12, color=COLORS["text"])
    ax.set_ylabel("Heat Flux q (kW/m^2)", fontsize=12, color=COLORS["text"])
    ax.set_title(title, fontsize=14, color=COLORS["text"], fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(left=max(0, s_profile[0]))

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path
