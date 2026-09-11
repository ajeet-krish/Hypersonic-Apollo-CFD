"""Heat flux extraction and profile generation.

Provides functions to extract wall heat flux from SU2 VTU solutions
and generate parameterized heat flux distributions for thermal analysis.
"""
from pathlib import Path

import numpy as np


def extract_heat_flux_from_vtu(
    vtu_path: Path,
    body_contour: tuple[np.ndarray, np.ndarray],
) -> dict:
    """Extract wall heat flux from SU2 VTU solution.

    Parses the VTU file, finds the nearest node to each body contour
    point, and reads the Heat_Flux field. Computes surface arc length
    and identifies the stagnation location.

    Args:
        vtu_path: Path to the SU2 VTU solution file.
        body_contour: Tuple of (x, r) arrays defining the body contour.

    Returns:
        Dict with keys:
            s: Surface coordinate array (m), arc length from nose.
            q: Heat flux array (W/m^2).
            q_max: Peak heat flux (W/m^2).
            s_stagnation: Arc length at stagnation point (m).
    """
    from cfd.vtu_parser import parse_vtu

    data = parse_vtu(vtu_path)
    x_contour, r_contour = body_contour
    coords = data.coordinates
    heat_flux_field = data.point_data.get("Heat_Flux")

    n_pts = len(x_contour)
    s = np.zeros(n_pts)
    q = np.zeros(n_pts)

    if heat_flux_field is None:
        return {"s": s, "q": q, "q_max": 0.0, "s_stagnation": 0.0}

    # Build KD-tree for nearest-neighbor lookup
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)

    for i in range(n_pts):
        _dist, idx = tree.query([x_contour[i], r_contour[i], 0.0])
        q[i] = float(heat_flux_field[idx])

        # Compute arc length
        if i > 0:
            dx = x_contour[i] - x_contour[i - 1]
            dr = r_contour[i] - r_contour[i - 1]
            s[i] = s[i - 1] + np.sqrt(dx**2 + dr**2)

    q_max = float(np.max(q))
    s_stagnation = float(s[np.argmax(q)]) if q_max > 0 else 0.0

    return {"s": s, "q": q, "q_max": q_max, "s_stagnation": s_stagnation}


def heat_flux_distribution(
    s: np.ndarray,
    q_max: float,
    distribution: str = "sinusoidal",
    L: float = 1.0,
) -> np.ndarray:
    """Generate heat flux distribution along body surface.

    Args:
        s: Surface coordinate array (normalized 0 to 1, or physical coords).
        q_max: Peak heat flux at stagnation point (W/m^2).
        distribution: Distribution type: 'uniform', 'sinusoidal', or 'cosine'.
        L: Body length for scaling (m). Default 1.0.

    Returns:
        Heat flux array (W/m^2) same shape as s.

    Raises:
        ValueError: If distribution type is not recognized.
    """
    s_norm = np.asarray(s, dtype=float) / L
    s_norm = np.clip(s_norm, 0.0, 1.0)

    if distribution == "uniform":
        return np.full_like(s_norm, q_max)

    if distribution == "sinusoidal":
        # Sinusoidal decay from stagnation: q(s) = q_max * cos(pi*s/2L)
        return q_max * np.cos(0.5 * np.pi * s_norm)

    if distribution == "cosine":
        # Cosine-squared decay: q(s) = q_max * cos^2(pi*s/2L)
        return q_max * np.cos(0.5 * np.pi * s_norm) ** 2

    raise ValueError(
        f"Unknown distribution '{distribution}'. "
        f"Valid options: 'uniform', 'sinusoidal', 'cosine'."
    )
