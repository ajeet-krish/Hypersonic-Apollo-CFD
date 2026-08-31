"""Triple validation: compare SU2 results against analytical correlations.

Compares stagnation heat flux (Sutton-Graves), shock standoff (Billig),
and surface Cp (modified Newtonian) against SU2 CFD results.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .billig import billig_blunted_cone
from .fay_riddell import sutton_graves
from .newtonian import modified_newtonian_cp


@dataclass
class ValidationResult:
    """Single validation comparison result.

    Attributes:
        quantity: Name of the compared quantity.
        su2_value: Value from SU2 CFD simulation.
        analytical_value: Value from analytical correlation.
        error_pct: Absolute percentage error.
        target_pct: Target accuracy threshold (pass if error < target).
        status: PASS or FAIL.
        notes: Human-readable summary.
    """
    quantity: str
    su2_value: float
    analytical_value: float
    error_pct: float
    target_pct: float
    status: str
    notes: str


def compare_stagnation_heat_flux(
    su2_q_stag: float,
    rho_inf: float,
    V_inf: float,
    R_nose: float,
    target_pct: float = 20.0,
) -> ValidationResult:
    """Compare SU2 stagnation heat flux against Sutton-Graves correlation.

    Uses the Sutton-Graves form: q = C * sqrt(rho/R) * V^3

    Args:
        su2_q_stag: Stagnation heat flux from SU2 (W/m^2).
        rho_inf: Freestream density (kg/m^3).
        V_inf: Freestream velocity (m/s).
        R_nose: Nose sphere radius (m).
        target_pct: Target accuracy threshold (default 20%).

    Returns:
        ValidationResult with comparison metrics.
    """
    sg = sutton_graves(rho_inf, V_inf, R_nose)
    analytical = sg.q_stag

    if analytical > 0:
        error_pct = abs(su2_q_stag - analytical) / analytical * 100.0
    else:
        error_pct = float("inf")

    status = "PASS" if error_pct < target_pct else "FAIL"
    notes = (
        f"Sutton-Graves: {analytical:.0f} W/m^2, "
        f"SU2: {su2_q_stag:.0f} W/m^2, "
        f"error: {error_pct:.1f}% (target < {target_pct}%)"
    )

    return ValidationResult(
        quantity="Stagnation Heat Flux",
        su2_value=su2_q_stag,
        analytical_value=analytical,
        error_pct=error_pct,
        target_pct=target_pct,
        status=status,
        notes=notes,
    )


def compare_shock_standoff(
    su2_delta_over_R: float,
    R_nose: float,
    M: float,
    target_pct: float = 10.0,
) -> ValidationResult:
    """Compare SU2 shock standoff against Billig correlation.

    Uses the Billig blunted-cone correlation: delta/R = 0.143 * exp(3.24/M^2)

    Args:
        su2_delta_over_R: Shock standoff ratio from SU2 (delta/R).
        R_nose: Nose sphere radius (m).
        M: Freestream Mach number.
        target_pct: Target accuracy threshold (default 10%).

    Returns:
        ValidationResult with comparison metrics.
    """
    billig = billig_blunted_cone(R_nose, M)
    analytical = billig.delta_over_R

    if analytical > 0:
        error_pct = abs(su2_delta_over_R - analytical) / analytical * 100.0
    else:
        error_pct = float("inf")

    status = "PASS" if error_pct < target_pct else "FAIL"
    notes = (
        f"Billig: {analytical:.4f}, "
        f"SU2: {su2_delta_over_R:.4f}, "
        f"error: {error_pct:.1f}% (target < {target_pct}%)"
    )

    return ValidationResult(
        quantity="Shock Standoff (delta/R)",
        su2_value=su2_delta_over_R,
        analytical_value=analytical,
        error_pct=error_pct,
        target_pct=target_pct,
        status=status,
        notes=notes,
    )


def compare_surface_cp(
    su2_cp_profile: np.ndarray,
    theta_profile: np.ndarray,
    M: float,
    target_pct: float = 15.0,
) -> ValidationResult:
    """Compare SU2 surface Cp against modified Newtonian theory.

    Computes mean absolute percentage error on the CONE section only
    (theta > 30 degrees), excluding the sphere nose region where
    Newtonian theory is less accurate.

    Args:
        su2_cp_profile: SU2 Cp values along the surface.
        theta_profile: Surface angles in degrees (0 at stagnation).
        M: Freestream Mach number.
        target_pct: Target accuracy threshold (default 15%).

    Returns:
        ValidationResult with comparison metrics.
    """
    # Select cone section: theta > 30 degrees (exclude sphere nose)
    cone_mask = theta_profile > 30.0
    n_cone = int(np.sum(cone_mask))

    if n_cone < 3:
        # Fallback: use all points if cone section too small
        cone_mask = np.ones(len(theta_profile), dtype=bool)
        n_cone = len(theta_profile)

    theta_cone_rad = np.deg2rad(theta_profile[cone_mask])
    su2_cp_cone = su2_cp_profile[cone_mask]

    # Modified Newtonian prediction on the cone section
    cp_newton = modified_newtonian_cp(theta_cone_rad, M)

    # Mean absolute percentage error (avoid division by zero)
    nonzero_mask = np.abs(cp_newton) > 1e-10
    if nonzero_mask.any():
        pct_errors = np.abs(su2_cp_cone[nonzero_mask] - cp_newton[nonzero_mask]) / np.abs(cp_newton[nonzero_mask]) * 100.0
        mean_error = float(np.mean(pct_errors))
    else:
        mean_error = float("inf")

    status = "PASS" if mean_error < target_pct else "FAIL"
    notes = (
        f"Modified Newtonian Cp on cone (n={n_cone} points): "
        f"mean error {mean_error:.1f}% (target < {target_pct}%)"
    )

    return ValidationResult(
        quantity="Surface Cp (cone section)",
        su2_value=float(np.mean(su2_cp_cone)),
        analytical_value=float(np.mean(cp_newton)),
        error_pct=mean_error,
        target_pct=target_pct,
        status=status,
        notes=notes,
    )


def build_validation_report(
    su2_results: dict,
    freestream: dict,
    geometry: dict,
) -> dict:
    """Assemble all validation comparisons into a JSON-serializable report.

    Args:
        su2_results: Post-processed SU2 results (from postprocess.json).
        freestream: Freestream conditions (rho_inf, V_inf, M, altitude).
        geometry: Geometry parameters (R_nose, half_angle, etc.).

    Returns:
        Dictionary with validation results and summary table.
    """
    R_nose = geometry["R_nose"]
    M = freestream["M"]
    rho_inf = freestream["rho_inf"]
    V_inf = freestream["V_inf"]

    # 1. Stagnation heat flux
    q_stag = su2_results["stagnation"]["heat_flux_W_m2"]
    if q_stag is None or q_stag == 0:
        # Use real-gas corrected value if raw is unavailable
        q_stag = su2_results.get("real_gas_correction", {}).get("q_corrected_W_m2", 0.0)
    hf_result = compare_stagnation_heat_flux(q_stag, rho_inf, V_inf, R_nose)

    # 2. Shock standoff
    delta_over_R = su2_results["shock_standoff"]["delta_over_R"]
    if delta_over_R is None:
        delta_over_R = 0.0
    ss_result = compare_shock_standoff(delta_over_R, R_nose, M)

    # 3. Surface Cp (if profiles available)
    # Build from Cp_max vs stagnation Cp comparison when full profiles
    # are not available in the summary JSON
    cp_max_su2 = su2_results["field_extrema"]["cp_max"]
    if cp_max_su2 is not None and cp_max_su2 > 0:
        # Compare Cp_max (stagnation point, theta=90 deg)
        from .newtonian import stagnation_cp
        cp_max_newton = stagnation_cp(M)
        if cp_max_newton > 0:
            cp_error = abs(cp_max_su2 - cp_max_newton) / cp_max_newton * 100.0
        else:
            cp_error = float("inf")
        cp_status = "PASS" if cp_error < 15.0 else "FAIL"
        cp_result = ValidationResult(
            quantity="Surface Cp (stagnation point)",
            su2_value=cp_max_su2,
            analytical_value=cp_max_newton,
            error_pct=cp_error,
            target_pct=15.0,
            status=cp_status,
            notes=(
                f"Modified Newtonian Cp_max: {cp_max_newton:.4f}, "
                f"SU2 Cp_max: {cp_max_su2:.4f}, "
                f"error: {cp_error:.1f}% (target < 15%)"
            ),
        )
    else:
        cp_result = ValidationResult(
            quantity="Surface Cp (stagnation point)",
            su2_value=0.0,
            analytical_value=0.0,
            error_pct=float("inf"),
            target_pct=15.0,
            status="FAIL",
            notes="Cp data not available from SU2 results",
        )

    results = [hf_result, ss_result, cp_result]
    all_pass = all(r.status == "PASS" for r in results)

    report = {
        "case": su2_results.get("case", "unknown"),
        "mach": M,
        "altitude_m": freestream.get("altitude", 0.0),
        "all_pass": all_pass,
        "results": [asdict(r) for r in results],
        "summary_table": [
            {
                "quantity": r.quantity,
                "su2": f"{r.su2_value:.4f}",
                "analytical": f"{r.analytical_value:.4f}",
                "error_pct": f"{r.error_pct:.1f}%",
                "target": f"< {r.target_pct:.0f}%",
                "status": r.status,
            }
            for r in results
        ],
    }

    return report


def save_validation_report(
    report: dict,
    output_path: Path,
) -> Path:
    """Save validation report to JSON.

    Args:
        report: Validation report dictionary.
        output_path: Path to the output JSON file.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return output_path
