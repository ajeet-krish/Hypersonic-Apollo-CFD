"""Post-processing functions for SU2 hypersonic solution data.

Extracts physics from VTU solution files: surface profiles, shock standoff,
total heating, and real-gas corrections.
"""
import json
import warnings
from pathlib import Path

import numpy as np

from .vtu_parser import VTUData, extract_stagnation_values


def extract_surface_profiles(
    data: VTUData,
    body_contour: tuple[np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    """Extract surface heat flux, pressure, and Cp along the body contour.

    For each body contour point, finds the nearest VTU node and reads the
    flow quantities. Computes arc length along the surface and local surface
    angle theta for Newtonian comparison.

    Args:
        data: Parsed VTU solution data.
        body_contour: Tuple of (x, r) arrays defining the body contour.

    Returns:
        Dictionary with keys:
            s: arc length along body surface (m)
            q: heat flux (W/m^2), 0 if not available
            p: static pressure (Pa)
            cp: pressure coefficient
            theta: local surface angle (degrees, 0 at stagnation)
    """
    x_contour, r_contour = body_contour
    coords = data.coordinates

    # Extract fields
    heat_flux = data.point_data.get("Heat_Flux")
    pressure_arr = data.pressure
    cp_arr = data.point_data.get("Pressure_Coefficient")

    n_pts = len(x_contour)
    s = np.zeros(n_pts)
    q = np.zeros(n_pts)
    p = np.zeros(n_pts)
    cp = np.zeros(n_pts)
    theta = np.zeros(n_pts)

    # Build KD-tree for nearest-neighbor lookup
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)

    for i in range(n_pts):
        # Find nearest VTU node to this contour point
        _dist, idx = tree.query([x_contour[i], r_contour[i], 0.0])

        # Read values from nearest node
        if pressure_arr is not None:
            p[i] = float(pressure_arr[idx])
        if heat_flux is not None:
            q[i] = float(heat_flux[idx])
        if cp_arr is not None:
            cp[i] = float(cp_arr[idx])

        # Compute arc length
        if i > 0:
            dx = x_contour[i] - x_contour[i - 1]
            dr = r_contour[i] - r_contour[i - 1]
            s[i] = s[i - 1] + np.sqrt(dx**2 + dr**2)

    # Compute local surface angle theta (0 at stagnation/nose, 90 at base)
    for i in range(n_pts):
        if i == 0:
            # Forward difference at nose
            if n_pts > 1:
                dx = x_contour[1] - x_contour[0]
                dr = r_contour[1] - r_contour[0]
                theta[i] = np.degrees(np.arctan2(abs(dr), abs(dx)))
            else:
                theta[i] = 0.0
        elif i == n_pts - 1:
            # Backward difference at base
            dx = x_contour[i] - x_contour[i - 1]
            dr = r_contour[i] - r_contour[i - 1]
            theta[i] = np.degrees(np.arctan2(abs(dr), abs(dx)))
        else:
            # Central difference
            dx = x_contour[i + 1] - x_contour[i - 1]
            dr = r_contour[i + 1] - r_contour[i - 1]
            theta[i] = np.degrees(np.arctan2(abs(dr), abs(dx)))

    return {"s": s, "q": q, "p": p, "cp": cp, "theta": theta}


def measure_shock_standoff(data: VTUData, R_nose: float, config_mach: float | None = None) -> float:
    """Measure shock standoff distance from the density gradient.

    Finds the shock location along the stagnation streamline (r ~ 0) by
    locating the maximum density gradient ahead of the body nose. Computes
    delta = x_shock - x_body_nose.

    The body nose is assumed to be at x=0 (standard for blunt body CFD
    where the nose tip is at the origin).

    To avoid bias from non-uniform node spacing along the stagnation
    streamline, the density profile is interpolated onto a uniform 1D grid
    (1000 points) before computing the gradient. The gradient is then
    computed on the uniform grid using np.gradient, which assumes constant
    spacing.

    Args:
        data: Parsed VTU solution data.
        R_nose: Nose sphere radius (m).
        config_mach: Freestream Mach number from config. If provided, the
            function checks for solver divergence (max_mach > 3 * config_mach)
            and returns 0.0 with a warning if detected.

    Returns:
        Shock standoff distance delta (m). Returns 0.0 if density data
        is not available, shock cannot be detected, or solution appears
        diverged.
    """
    # Divergence guard: if max Mach in field exceeds 3x the freestream,
    # the SU2 solution has almost certainly blown up (e.g., M=15.6 producing
    # max_mach=161). Return 0.0 to avoid post-processing a diverged field.
    if config_mach is not None and data.mach is not None:
        max_mach = float(data.mach.max())
        if max_mach > 3.0 * config_mach:
            warnings.warn(
                f"Solution appears diverged: max Mach {max_mach:.2f} exceeds "
                f"3x freestream Mach {config_mach:.2f}. Returning shock "
                f"standoff = 0.0.",
                stacklevel=2,
            )
            return 0.0
    if data.density is None:
        return 0.0

    coords = data.coordinates
    density = data.density
    x = coords[:, 0]
    r = coords[:, 1] if coords.shape[1] > 1 else np.zeros(len(x))

    # Body nose is at x=0 (standard blunt body convention)
    x_body_nose = 0.0

    # Select nodes along the stagnation streamline (r close to 0)
    streamline_tolerance = R_nose * 0.05  # 5% of nose radius
    mask_streamline = np.abs(r) < streamline_tolerance

    if not mask_streamline.any():
        # Fallback: use all nodes near the axis
        mask_streamline = np.abs(r) < R_nose * 0.15

    if not mask_streamline.any():
        return 0.0

    x_line = x[mask_streamline]
    rho_line = density[mask_streamline]

    # Sort by x coordinate
    order = np.argsort(x_line)
    x_line = x_line[order]
    rho_line = rho_line[order]

    # Only look at nodes upstream of the body nose (shock is always at x < 0).
    # Exclude the body surface region where the stagnation-to-wake density
    # gradient would dominate the shock gradient.
    upstream = (x_line >= x_body_nose - R_nose * 5.0) & (x_line < x_body_nose)

    if upstream.sum() < 3:
        # Fallback: include a small buffer past the nose
        upstream = (x_line >= x_body_nose - R_nose * 5.0) & (x_line <= x_body_nose + R_nose * 0.1)

    if upstream.sum() < 3:
        return 0.0

    x_up = x_line[upstream]
    rho_up = rho_line[upstream]

    if len(x_up) < 3:
        return 0.0

    # Interpolate onto a uniform grid to eliminate non-uniform spacing bias.
    # np.gradient assumes constant spacing; applying it to non-uniform
    # node distributions (denser near body/shock) produces spurious gradients.
    n_uniform = 1000
    x_uniform = np.linspace(x_up[0], x_up[-1], n_uniform)
    rho_uniform = np.interp(x_uniform, x_up, rho_up)

    # Compute density gradient on the uniform grid
    drho_dx = np.gradient(rho_uniform, x_uniform)

    # Find maximum density gradient (shock location)
    idx_shock = int(np.argmax(np.abs(drho_dx)))
    x_shock = float(x_uniform[idx_shock])

    # Shock standoff: distance from body nose to shock.
    # The shock is upstream of the body (x_shock < x_body_nose),
    # so delta = x_body_nose - x_shock (positive value).
    delta = x_body_nose - x_shock
    return max(delta, 0.0)


def compute_total_heating(
    q_profile: np.ndarray,
    s_profile: np.ndarray,
) -> float:
    """Integrate heat flux over the body surface (per-unit-circumference).

    Computes Q = integral(q * ds) using trapezoidal integration. This
    returns heating per unit circumferential length (W/m). For the full
    axisymmetric integral Q = integral(q * 2*pi*r * ds), use
    ``compute_total_heating_axisymmetric`` instead.

    Args:
        q_profile: Heat flux along the surface (W/m^2).
        s_profile: Arc length along the surface (m).

    Returns:
        Total heating per unit circumference (W/m).
    """
    if len(q_profile) < 2 or len(s_profile) < 2:
        return 0.0

    # Compute ds increments
    ds = np.diff(s_profile)
    q_avg = 0.5 * (q_profile[:-1] + q_profile[1:])

    # Simple trapezoidal integration: integral(q * ds)
    integral = float(np.sum(q_avg * ds))
    return integral


def compute_total_heating_axisymmetric(
    q_profile: np.ndarray,
    s_profile: np.ndarray,
    r_profile: np.ndarray,
) -> float:
    """Integrate heat flux over an axisymmetric body surface.

    Q = integral(q * 2 * pi * r * ds)

    Args:
        q_profile: Heat flux along the surface (W/m^2).
        s_profile: Arc length along the surface (m).
        r_profile: Radial coordinate along the surface (m).

    Returns:
        Total heating load (W).
    """
    if len(q_profile) < 2 or len(s_profile) < 2:
        return 0.0

    ds = np.diff(s_profile)
    q_avg = 0.5 * (q_profile[:-1] + q_profile[1:])
    r_avg = 0.5 * (r_profile[:-1] + r_profile[1:])

    # Q = integral(q * 2 * pi * r * ds)
    integral = float(np.sum(q_avg * 2.0 * np.pi * r_avg * ds))
    return integral


def apply_real_gas_correction(
    q_stag: float,
    T_stag: float,
) -> tuple[float, float]:
    """Apply real-gas correction to stagnation heat flux.

    Uses gamma_correction_factor from src/physics/real_gas.py to compute
    the correction factor sqrt(gamma_real / gamma_ref). The corrected
    heat flux is q_corrected = q_stag * correction_factor.

    Args:
        q_stag: Stagnation heat flux (W/m^2).
        T_stag: Stagnation temperature (K).

    Returns:
        Tuple of (q_corrected, correction_factor).
    """
    from physics.real_gas import gamma_correction_factor

    correction = gamma_correction_factor(T_stag)
    q_corrected = q_stag * correction
    return q_corrected, correction


def build_results_summary(
    data: VTUData,
    config: object,
    body_contour: tuple[np.ndarray, np.ndarray],
) -> dict:
    """Assemble all derived quantities into a JSON-serializable dict.

    Computes stagnation properties, shock standoff, total heating,
    real-gas correction, and global field extrema.

    Args:
        data: Parsed VTU solution data.
        config: CaseConfig or similar with mach, altitude attributes.
        body_contour: Tuple of (x, r) body contour arrays.

    Returns:
        Dictionary with all post-processing results.
    """
    # Stagnation values (from wall node with max pressure)
    stag = extract_stagnation_values(data)

    # Surface profiles
    profiles = extract_surface_profiles(data, body_contour)

    # Body contour arrays
    x_contour, r_contour = body_contour

    # Nose radius estimation
    r_nose_measured = measure_shock_standoff_r_nose(x_contour, r_contour)

    shock_standoff = measure_shock_standoff(
        data, r_nose_measured, config_mach=getattr(config, "mach", None),
    )

    # Total heating (axisymmetric)
    total_heating = compute_total_heating_axisymmetric(
        profiles["q"], profiles["s"], r_contour,
    )

    # Stagnation heat flux: prefer the value from the identified stagnation
    # wall node (extract_stagnation_values). Fall back to surface profile max
    # only if the stagnation node has no Heat_Flux data.
    T_stag = stag.get("Temperature", 0.0)
    q_stag = stag.get("Heat_Flux")
    if q_stag is not None and q_stag > 0.0:
        # Stagnation node has valid Heat_Flux -- use it directly.
        pass
    elif profiles["q"] is not None and len(profiles["q"]) > 0:
        # Fallback: use the maximum heat flux from the surface profile,
        # which should be at or near the stagnation point.
        q_stag = float(np.max(profiles["q"]))
    else:
        q_stag = 0.0

    if T_stag > 0 and q_stag > 0:
        q_corrected, rg_correction = apply_real_gas_correction(q_stag, T_stag)
    else:
        q_corrected = q_stag
        rg_correction = 1.0

    # Global field extrema
    max_mach = float(data.mach.max()) if data.mach is not None else None
    cp_max = float(data.point_data.get("Pressure_Coefficient", np.array([0.0])).max()) if "Pressure_Coefficient" in data.point_data else None

    summary = {
        "stagnation": {
            "pressure_Pa": round(stag.get("Pressure", 0.0), 2),
            "temperature_K": round(T_stag, 2),
            "density_kg_m3": round(stag.get("Density", 0.0), 6),
            "mach": round(stag.get("Mach", 0.0), 4),
            "heat_flux_W_m2": round(q_stag, 2) if q_stag is not None else None,
        },
        "shock_standoff": {
            "delta_m": round(shock_standoff, 6),
            "R_nose_m": round(r_nose_measured, 6),
            "delta_over_R": round(
                shock_standoff / r_nose_measured, 6
            ) if r_nose_measured > 0 else None,
        },
        "total_heating": {
            "Q_total_W": round(total_heating, 2),
        },
        "real_gas_correction": {
            "T_stagnation_K": round(T_stag, 2),
            "correction_factor": round(rg_correction, 6),
            "q_corrected_W_m2": round(q_corrected, 2),
        },
        "field_extrema": {
            "max_mach": round(max_mach, 4) if max_mach is not None else None,
            "cp_max": round(cp_max, 6) if cp_max is not None else None,
            "max_pressure_Pa": round(
                float(data.pressure.max()), 2
            ) if data.pressure is not None else None,
            "max_temperature_K": round(
                float(data.temperature.max()), 2
            ) if data.temperature is not None else None,
        },
        "surface_profiles": {
            "n_points": len(profiles["s"]),
            "arc_length_m": round(float(profiles["s"][-1]), 6) if len(profiles["s"]) > 0 else 0.0,
        },
    }

    return summary


def measure_shock_standoff_r_nose(
    x_contour: np.ndarray,
    r_contour: np.ndarray,
) -> float:
    """Estimate nose radius from the body contour.

    For a spherical nose centered at (0, R_nose) in the (x, r) plane,
    the sphere equation is x^2 + (r - R)^2 = R^2. Given a point (x_j, r_j)
    on the sphere, R_nose = (x_j^2 + r_j^2) / (2 * r_j).

    Uses the last point of the spherical nose section (near the junction).

    Args:
        x_contour: Axial coordinates of body contour.
        r_contour: Radial coordinates of body contour.

    Returns:
        Estimated nose radius (m).
    """
    if len(x_contour) < 5:
        return 0.1  # default fallback

    # Use first 10 points (on the spherical nose) to fit
    n_fit = min(10, len(x_contour))
    x_pts = x_contour[:n_fit]
    r_pts = r_contour[:n_fit]

    # For a sphere centered at (0, R) with radius R:
    #   x^2 + (r - R)^2 = R^2
    # Given a point (x_j, r_j):
    #   R = (x_j^2 + r_j^2) / (2 * r_j)
    # Use the last point of the sphere section for accuracy
    x_j = float(x_pts[-1])
    r_j = float(r_pts[-1])

    if abs(r_j) < 1e-12:
        return 0.1

    R_nose = (x_j**2 + r_j**2) / (2.0 * r_j)

    # Sanity check: R_nose should be positive and reasonable
    if R_nose <= 0 or R_nose > 100.0:
        return 0.1

    return R_nose


def save_postprocess_results(
    summary: dict,
    output_path: Path,
) -> Path:
    """Save post-processing results to JSON.

    Args:
        summary: Results dictionary from build_results_summary.
        output_path: Path to the output JSON file.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    return output_path
