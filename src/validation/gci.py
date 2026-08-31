"""Grid Convergence Index computation (ASME V&V 20-2009).

Computes the GCI for stagnation heat flux and shock standoff across
three mesh refinement levels (draft, standard, high). Reports order
of accuracy, Richardson extrapolation, and asymptotic convergence ratio.
"""
import math
from dataclasses import dataclass


@dataclass
class GCIMeshLevel:
    """Single mesh level for GCI study.

    Attributes:
        name: Mesh level name (e.g. 'draft', 'standard', 'high').
        n_cells: Number of cells in the mesh.
        value: Quantity value at this mesh level (e.g. stagnation heat flux).
    """
    name: str
    n_cells: int
    value: float


@dataclass
class GCIResult:
    """Grid Convergence Index result for a single quantity.

    Attributes:
        quantity: Name of the quantity analyzed.
        levels: Input mesh levels.
        refinement_ratio: Effective mesh refinement ratio.
        apparent_order: Apparent order of convergence (Richardson).
        extrapolated_value: Richardson-extrapolated value.
        gci_fine_pct: GCI on finest mesh (percentage of fine value).
        gci_coarse_pct: GCI on coarsest mesh (percentage of coarse value).
        asymptotic_ratio: Asymptotic convergence ratio check.
        monotonic: Whether convergence is monotonic.
        passed: True if asymptotic_ratio in (0.5, 2.0).
        notes: Human-readable summary.
    """
    quantity: str
    levels: list[GCIMeshLevel]
    refinement_ratio: float
    apparent_order: float
    extrapolated_value: float
    gci_fine_pct: float
    gci_coarse_pct: float
    asymptotic_ratio: float
    monotonic: bool
    passed: bool
    notes: str


def compute_gci(
    levels: list[GCIMeshLevel],
    safety_factor: float = 1.25,
) -> GCIResult:
    """Compute Grid Convergence Index per ASME V&V 20-2009.

    Uses three mesh levels (fine, medium, coarse) to estimate the
    apparent order of convergence p, the Richardson-extrapolated
    exact solution, and the GCI on the finest mesh.

    Args:
        levels: List of 3 GCIMeshLevel objects (sorted coarsest to finest
                or finest to coarsest; order is determined by n_cells).
        safety_factor: Safety factor for GCI (default 1.25 per ASME).

    Returns:
        GCIResult with all convergence metrics.
    """
    if len(levels) != 3:
        raise ValueError(f"Need exactly 3 mesh levels, got {len(levels)}")

    # Sort by cell count: [coarse, medium, fine]
    sorted_levels = sorted(levels, key=lambda lv: lv.n_cells)
    coarse, medium, fine = sorted_levels

    # Effective refinement ratio (geometric mean of pairwise ratios)
    r21 = fine.n_cells / medium.n_cells
    r32 = medium.n_cells / coarse.n_cells
    r = math.sqrt(r21 * r32) if r21 > 0 and r32 > 0 else 2.0

    # Solution values
    f1 = fine.value       # finest
    f2 = medium.value     # medium
    f3 = coarse.value     # coarsest

    # Apparent order of convergence
    if abs(f1 - f2) < 1e-15 or abs(f2 - f3) < 1e-15:
        p = 1.0
    else:
        p = math.log(abs(f3 - f2) / abs(f2 - f1)) / math.log(r)

    # Richardson extrapolation to zero网格size
    if abs(r**p - 1.0) > 1e-15:
        f_exact = f1 + (f1 - f2) / (r**p - 1.0)
    else:
        f_exact = f1

    # GCI on finest mesh: GCI_fine = F_s * |epsilon| / (r^p - 1)
    epsilon_fine = abs(f1 - f2)
    if p > 0 and abs(r**p - 1.0) > 1e-15:
        gci_fine = safety_factor * epsilon_fine / (r**p - 1.0)
    else:
        gci_fine = safety_factor * epsilon_fine

    # GCI on coarsest mesh
    epsilon_coarse = abs(f2 - f3)
    if p > 0 and abs(r**p - 1.0) > 1e-15:
        gci_coarse = safety_factor * epsilon_coarse / (r**p - 1.0)
    else:
        gci_coarse = safety_factor * epsilon_coarse

    # Asymptotic ratio: GCI_coarse / (r^p * GCI_fine)
    if gci_fine > 0 and p > 0:
        asymptotic_ratio = gci_coarse / (r**p * gci_fine)
    else:
        asymptotic_ratio = 1.0

    # Percentage of extrapolated value
    if abs(f_exact) > 1e-15:
        gci_fine_pct = abs(gci_fine / f_exact) * 100.0
        gci_coarse_pct = abs(gci_coarse / f_exact) * 100.0
    else:
        gci_fine_pct = 0.0
        gci_coarse_pct = 0.0

    # Monotonicity check
    monotonic = (f3 < f2 < f1) or (f3 > f2 > f1)

    # Pass: asymptotic ratio in (0.5, 2.0)
    passed = 0.5 < asymptotic_ratio < 2.0

    notes = (
        f"Order={p:.2f}, GCI_fine={gci_fine_pct:.2f}%, "
        f"asymptotic={asymptotic_ratio:.2f}, "
        f"{'PASSED' if passed else 'FAILED'}"
    )

    return GCIResult(
        quantity="",
        levels=sorted_levels,
        refinement_ratio=r,
        apparent_order=p,
        extrapolated_value=f_exact,
        gci_fine_pct=gci_fine_pct,
        gci_coarse_pct=gci_coarse_pct,
        asymptotic_ratio=asymptotic_ratio,
        monotonic=monotonic,
        passed=passed,
        notes=notes,
    )


def run_gci_study(
    case_config: object,
    quantities: list[str] | None = None,
) -> dict:
    """Run a GCI mesh convergence study on three mesh tiers.

    Runs the reference case on draft, standard, and high tiers using
    the euler-rans strategy, then computes GCI for stagnation heat flux
    and shock standoff.

    Args:
        case_config: CaseConfig with name, preset_fn, mach, altitude, etc.
        quantities: List of quantity names to study (default:
                    ['stagnation_heat_flux', 'shock_standoff']).

    Returns:
        Dictionary with GCI results for each quantity.
    """
    import json
    from pathlib import Path

    from pipeline.case_config import CaseConfig
    from pipeline.stages import (
        run_mesh_stage,
        run_postprocess_stage,
        run_su2_stage,
    )

    if quantities is None:
        quantities = ["stagnation_heat_flux", "shock_standoff"]

    config: CaseConfig = case_config
    tiers = ["draft", "standard", "high"]
    tier_results: dict[str, dict] = {}

    for tier in tiers:
        tier_config = CaseConfig(
            name=config.name,
            label=f"{config.label} ({tier})",
            preset_fn=config.preset_fn,
            mach=config.mach,
            altitude=config.altitude,
            mesh_tier=tier,
            su2_strategy=config.su2_strategy,
            su2_euler_iterations=config.su2_euler_iterations,
            su2_rans_iterations=config.su2_rans_iterations,
            su2_cfl=config.su2_cfl,
        )

        # Run mesh + SU2 + postprocess for this tier
        mesh_ok = run_mesh_stage(tier_config)
        if mesh_ok != 0:
            print(f"  WARNING: Mesh failed for tier {tier}, skipping")
            continue

        su2_ok = run_su2_stage(tier_config)
        if su2_ok != 0:
            print(f"  WARNING: SU2 failed for tier {tier}, skipping")
            continue

        post_ok = run_postprocess_stage(tier_config)
        if post_ok != 0:
            print(f"  WARNING: Postprocess failed for tier {tier}, skipping")
            continue

        # Load results
        post_path = Path(tier_config.output_dir) / "postprocess" / "postprocess.json"
        if post_path.exists():
            with open(post_path) as f:
                tier_results[tier] = json.load(f)

    if len(tier_results) < 3:
        return {"error": f"Only {len(tier_results)}/3 tiers succeeded, GCI needs all 3"}

    # Build GCI results for each quantity
    gci_results: dict[str, dict] = {}

    for qty in quantities:
        if qty == "stagnation_heat_flux":
            levels = [
                GCIMeshLevel(
                    name=tier,
                    n_cells=_get_cell_count(config, tier),
                    value=tier_results[tier]["stagnation"]["heat_flux_W_m2"],
                )
                for tier in tiers
            ]
            gci = compute_gci(levels)
            gci.quantity = "Stagnation Heat Flux (W/m^2)"
        elif qty == "shock_standoff":
            levels = [
                GCIMeshLevel(
                    name=tier,
                    n_cells=_get_cell_count(config, tier),
                    value=tier_results[tier]["shock_standoff"]["delta_over_R"],
                )
                for tier in tiers
            ]
            gci = compute_gci(levels)
            gci.quantity = "Shock Standoff (delta/R)"
        else:
            continue

        gci_results[qty] = {
            "quantity": gci.quantity,
            "refinement_ratio": round(gci.refinement_ratio, 4),
            "apparent_order": round(gci.apparent_order, 4),
            "extrapolated_value": round(gci.extrapolated_value, 6),
            "gci_fine_pct": round(gci.gci_fine_pct, 4),
            "gci_coarse_pct": round(gci.gci_coarse_pct, 4),
            "asymptotic_ratio": round(gci.asymptotic_ratio, 4),
            "monotonic": gci.monotonic,
            "passed": gci.passed,
            "notes": gci.notes,
            "levels": [
                {"name": lv.name, "n_cells": lv.n_cells, "value": round(lv.value, 6)}
                for lv in gci.levels
            ],
        }

    return gci_results


def _get_cell_count(config: object, tier: str) -> int:
    """Get the estimated cell count for a given tier."""
    from cfd.mesh_config import MeshConfig
    mc = MeshConfig.for_tier(tier)
    return mc.estimated_cell_count
