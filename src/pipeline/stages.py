"""Pipeline stage functions for per-case aerothermodynamics analysis.

KEY RULE: Plots go to docs/assets/images/{case}/, artifacts go to output/{case}/.
"""
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from geometry.blunt_body import generate_contour
from physics.atmosphere import standard_atmosphere
from physics.real_gas import gamma_correction_factor, gamma_curve_fit
from validation.billig import billig_blunted_cone
from validation.fay_riddell import sutton_graves
from validation.newtonian import stagnation_cp
from validation.shock_relations import normal_shock

from .case_config import CaseConfig, PipelineStage


def run_geometry_stage(config: CaseConfig) -> int:
    """Generate geometry and compute analytical predictions.

    Produces:
        - output/{name}/geometry/contour.json: body contour coordinates
        - output/{name}/geometry/analytical.json: analytical predictions
        - docs/assets/images/{name}/geometry.png: annotated geometry plot

    Returns:
        0 on success, 1 on failure.
    """
    print(f"\n[{config.label}] Geometry stage")

    body_config = config.preset_fn()
    x, r = generate_contour(body_config)

    # Compute atmosphere at altitude
    atm = standard_atmosphere(config.altitude)
    V_inf = atm.speed_of_sound * config.mach

    # Sutton-Graves heating
    heating = sutton_graves(atm.density, V_inf, body_config.R_nose)

    # Billig standoff
    standoff = billig_blunted_cone(body_config.R_nose, config.mach)

    # Modified Newtonian Cp
    cp_stag = stagnation_cp(config.mach, config.gamma)

    # Normal shock
    shock = normal_shock(config.mach, config.gamma)

    # Real-gas correction
    T_stag = atm.temperature * (1.0 + (config.gamma - 1.0) / 2.0 * config.mach**2)
    gamma_real = gamma_curve_fit(T_stag).gamma
    rg_correction = gamma_correction_factor(T_stag, config.gamma)

    # Save contour JSON
    out_dir = Path(config.output_dir) / "geometry"
    out_dir.mkdir(parents=True, exist_ok=True)

    contour_data = {
        "case": config.name,
        "mach": config.mach,
        "altitude": config.altitude,
        "x": x.tolist(),
        "r": r.tolist(),
    }
    with open(out_dir / "contour.json", "w") as f:
        json.dump(contour_data, f, indent=2)
    print(f"  Contour: {out_dir / 'contour.json'} ({len(x)} points)")

    # Save analytical results
    analytical = {
        "case": config.name,
        "mach": config.mach,
        "altitude_m": config.altitude,
        "atmosphere": {
            "temperature_K": round(atm.temperature, 4),
            "pressure_Pa": round(atm.pressure, 2),
            "density_kg_m3": round(atm.density, 6),
            "speed_of_sound_ms": round(atm.speed_of_sound, 4),
        },
        "freestream_velocity_ms": round(V_inf, 2),
        "stagnation_heating": {
            "q_stag_W_m2": round(heating.q_stag, 2),
            "q_stag_kW_m2": round(heating.q_stag_kw, 4),
            "formula": heating.formula,
        },
        "shock_standoff": {
            "delta_over_R": round(standoff.delta_over_R, 6),
            "delta_m": round(standoff.delta, 6),
            "formula": standoff.formula,
        },
        "newtonian_cp": {
            "cp_stagnation": round(cp_stag, 6),
        },
        "normal_shock": {
            "M2": round(shock.M2, 6),
            "p_ratio": round(shock.p_ratio, 6),
            "T_ratio": round(shock.T_ratio, 6),
            "rho_ratio": round(shock.rho_ratio, 6),
            "p0_ratio": round(shock.p0_ratio, 6),
        },
        "real_gas": {
            "T_stagnation_K": round(T_stag, 2),
            "gamma_real": round(gamma_real, 6),
            "gamma_correction_factor": round(rg_correction, 6),
        },
        "geometry": {
            "R_nose": body_config.R_nose,
            "half_angle_deg": body_config.half_angle,
            "base_radius": body_config.base_radius,
            "body_length": round(body_config.computed_body_length, 6),
            "junction_x": round(body_config.junction_x, 6),
            "junction_r": round(body_config.junction_r, 6),
        },
    }
    with open(out_dir / "analytical.json", "w") as f:
        json.dump(analytical, f, indent=2)
    print(f"  Analytical: {out_dir / 'analytical.json'}")

    # Generate plot
    images_dir = Path(config.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        from viz.geometry import plot_annotated_geometry
        plot_path = images_dir / "geometry.png"
        plot_annotated_geometry(
            body_config, x, r, plot_path,
            dpi=200, show_dimensions=True, show_junction=True,
            case_name=config.label,
        )
        print(f"  Plot: {plot_path}")
    except (OSError, RuntimeError) as exc:
        print(f"  Plot FAILED: {exc}")
        return 1

    # Print summary
    print(f"  Mach: {config.mach}, Altitude: {config.altitude/1000:.0f} km")
    print(f"  q_stag: {heating.q_stag_kw:.2f} kW/m^2")
    print(f"  Standoff delta/R: {standoff.delta_over_R:.4f}")
    print(f"  Cp_max: {cp_stag:.4f}")
    print(f"  Gamma (real-gas): {gamma_real:.4f} (correction: {rg_correction:.4f})")

    return 0


def run_mesh_stage(config: CaseConfig) -> int:
    """Generate a Gmsh shock-aligned mesh for the blunt body.

    Produces:
        - output/{name}/mesh/{name}.su2: SU2 mesh file
        - output/{name}/mesh/mesh_quality.json: mesh quality metrics
        - docs/assets/images/{name}/mesh.png: mesh visualization

    Returns:
        0 on success, 1 on failure.
    """
    print(f"\n[{config.label}] Mesh stage")

    from cfd.mesh import generate_body_mesh
    from cfd.mesh_config import MeshConfig
    from cfd.mesh_quality import check_mesh_quality, validate_su2_mesh
    from viz.mesh import plot_mesh

    body_config = config.preset_fn()
    mesh_config = MeshConfig.for_tier(config.mesh_tier)

    mesh_dir = Path(config.output_dir) / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = mesh_dir / f"{config.name}.su2"

    print(f"  Tier: {config.mesh_tier}")
    print(f"  Target cells: ~{mesh_config.estimated_cell_count:,}")
    print(f"  Shock refinement: {mesh_config.shock_refinement}")

    try:
        generate_body_mesh(
            body_config, mesh_config, config.mach, mesh_path,
        )
    except (RuntimeError, OSError) as exc:
        print(f"  Mesh generation FAILED: {exc}")
        return 1

    print(f"  Mesh: {mesh_path} ({mesh_path.stat().st_size:,} bytes)")

    # Validate mesh
    is_valid = validate_su2_mesh(mesh_path)
    print(f"  SU2 valid: {is_valid}")

    # Quality check
    quality = check_mesh_quality(mesh_path)
    quality_path = mesh_dir / "mesh_quality.json"
    with open(quality_path, "w") as f:
        json.dump(quality, f, indent=2)
    print(f"  Quality: {quality_path}")
    print(f"    Cells: {quality['n_cells']:,}")
    print(f"    Min quality: {quality['min_quality']:.4f}")
    print(f"    Mean quality: {quality['mean_quality']:.4f}")
    print(f"    Bad cells: {quality['pct_bad_cells']:.1f}%")

    # Plot mesh
    images_dir = Path(config.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    try:
        plot_path = plot_mesh(mesh_path, images_dir / "mesh.png")
        print(f"  Plot: {plot_path}")
    except (OSError, RuntimeError) as exc:
        print(f"  Plot FAILED: {exc}")

    return 0 if is_valid else 1


STAGE_FUNCTIONS: dict[PipelineStage, Callable[[CaseConfig], int]] = {
    PipelineStage.GEOMETRY: run_geometry_stage,
    PipelineStage.MESH: run_mesh_stage,
}


def run_full_pipeline(
    config: CaseConfig,
    stages: list[PipelineStage] | None = None,
) -> int:
    """Run all (or selected) pipeline stages.

    Args:
        config: Case configuration
        stages: List of stages to run. None = all available stages.

    Returns:
        0 if all stages pass, 1 if any fail.
    """
    if stages is None:
        stages = list(STAGE_FUNCTIONS.keys())

    print("=" * 60)
    print(f"  Pipeline: {config.label} ({config.name})")
    print(f"  Stages: {[s.value for s in stages]}")
    print("=" * 60)

    t0 = time.time()
    results: dict[PipelineStage, int] = {}

    for stage in stages:
        fn = STAGE_FUNCTIONS.get(stage)
        if fn is None:
            print(f"\n[{config.label}] Stage '{stage.value}' not implemented yet, skipping")
            results[stage] = 0
        else:
            results[stage] = fn(config)

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"  Summary: {config.label}")
    print(f"{'=' * 60}")
    print(f"  Total time: {elapsed:.1f}s\n")

    for stage, code in results.items():
        status = "PASSED" if code == 0 else "FAILED"
        print(f"  {stage.value:16s} {status}")

    n_pass = sum(1 for v in results.values() if v == 0)
    print(f"\n  {n_pass}/{len(results)} stages passed")
    print(f"{'=' * 60}")

    return 0 if all(v == 0 for v in results.values()) else 1
