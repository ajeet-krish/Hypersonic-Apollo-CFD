"""Thermal analysis validation against analytical correlations.

Compares thermal results (from thermal_results.json) against physics-based
checks: Sutton-Graves heat flux comparison, physical temperature bounds,
and heat flux sign sanity.
"""
from __future__ import annotations

from .fay_riddell import sutton_graves


def validate_thermal_results(
    thermal_results: dict,
    freestream: dict,
    geometry: dict,
    material: str = "avcoat",
) -> dict:
    """Validate thermal results against analytical correlations.

    Args:
        thermal_results: Dict from thermal_results.json.
        freestream: Dict with M, rho_inf, V_inf, altitude.
        geometry: Dict with R_nose.
        material: Material name for context.

    Returns:
        Validation report dict with checks and overall status.
    """
    report: dict = {"checks": [], "all_pass": True, "material": material}

    # Check 1: Sutton-Graves stagnation heating comparison
    sg = sutton_graves(freestream["rho_inf"], freestream["V_inf"], geometry["R_nose"])
    q_stag_analytical = sg.q_stag

    # Get max wall temperature and estimate peak heat flux from thermal results
    T_max = thermal_results.get("results", {}).get("T_max_wall_K", 0.0)
    t_end = thermal_results.get("config", {}).get("t_end_s", 100.0)

    # Estimate average heat flux from total heat / time (J/m² / s = W/m²)
    q_total = thermal_results.get("results", {}).get("q_total_J_m2", 0.0)
    q_peak_estimated = q_total / t_end if t_end > 0 else 0.0

    # Order of magnitude check (within factor of 10)
    if q_peak_estimated > 0 and q_stag_analytical > 0:
        ratio = q_peak_estimated / q_stag_analytical
        status = "PASS" if 0.1 < ratio < 10.0 else "FAIL"
    else:
        status = "FAIL"
        ratio = 0.0

    report["checks"].append({
        "name": "Sutton-Graves comparison",
        "analytical_q_stag_W_m2": round(q_stag_analytical, 2),
        "estimated_q_peak_W_m2": round(q_peak_estimated, 2),
        "ratio": round(ratio, 4),
        "status": status,
    })

    # Check 2: Physical temperature bounds (material sublimation limit)
    T_max = thermal_results.get("results", {}).get("T_max_wall_K", 0.0)
    if material.lower() in ("avcoat", "avcoat 56"):
        sublimation_limit = 3500.0
    elif material.lower() in ("pica", "pica-x"):
        sublimation_limit = 3800.0
    else:
        sublimation_limit = 3500.0

    temp_status = "PASS" if T_max <= sublimation_limit else "FAIL"
    report["checks"].append({
        "name": "T_max_sublimation_limit",
        "T_max_wall_K": round(T_max, 2),
        "sublimation_limit_K": sublimation_limit,
        "status": temp_status,
    })
    if temp_status == "FAIL":
        report["all_pass"] = False

    # Check 3: Heat flux sign sanity
    q_total = thermal_results.get("results", {}).get("q_total_J_m2", 0.0)
    q_status = "PASS" if q_total >= 0 else "FAIL"
    report["checks"].append({
        "name": "q_positive",
        "q_total_J_m2": round(q_total, 2),
        "status": q_status,
    })
    if q_status == "FAIL":
        report["all_pass"] = False

    # Check 4: Cold side temperature reasonable
    T_max_back = thermal_results.get("results", {}).get("T_max_back_K", 0.0)
    cold_status = "PASS" if 200.0 <= T_max_back <= 1500.0 else "FAIL"
    report["checks"].append({
        "name": "cold_side_temperature",
        "T_max_back_K": round(T_max_back, 2),
        "status": cold_status,
    })
    if cold_status == "FAIL":
        report["all_pass"] = False

    # Check 5: Overall Sutton-Graves status
    if status == "FAIL":
        report["all_pass"] = False

    return report
