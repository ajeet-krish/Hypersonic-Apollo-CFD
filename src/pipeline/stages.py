"""Pipeline stage functions for per-case aerothermodynamics analysis.

KEY RULE: Plots go to docs/assets/images/{case}/, artifacts go to output/{case}/.
"""
from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent))

from geometry.blunt_body import generate_contour
from physics.atmosphere import standard_atmosphere
from physics.real_gas import gamma_correction_factor, gamma_curve_fit
from validation.billig import billig_blunted_cone
from validation.fay_riddell import sutton_graves
from validation.newtonian import stagnation_cp
from validation.shock_relations import normal_shock

from .case_config import CaseConfig, PipelineStage

if TYPE_CHECKING:
    from cfd.config import SU2HypersonicConfig
    from cfd.solver import SU2Results, SU2Solver


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
    if config.aoa != 0.0:
        print(f"  Angle of attack: {config.aoa} deg (full2d forced)")

    try:
        generate_body_mesh(
            body_config, mesh_config, config.mach, mesh_path,
            aoa=config.aoa,
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


def run_su2_stage(config: CaseConfig) -> int:
    """Run SU2 CFD simulation for the blunt body.

    Supports three convergence strategies:
    - direct: Single RANS run (legacy, often fails at hypersonic Mach numbers)
    - euler-rans: Euler first to establish bow shock, then RANS restart
    - mach-ramp: Start at lower Mach, restart at target Mach

    Produces:
        - output/{name}/su2/config.cfg: SU2 configuration file
        - output/{name}/su2/history.csv: convergence history
        - output/{name}/su2/flow.vtu: solution field data
        - output/{name}/su2/results.json: summary of simulation results
        - docs/assets/images/{name}/convergence.png: convergence plot

    Returns:
        0 on success, 1 on failure.
    """
    print(f"\n[{config.label}] SU2 stage (strategy={config.su2_strategy})")

    from cfd.config import SU2HypersonicConfig
    from cfd.solver import SU2Solver
    from physics.atmosphere import standard_atmosphere

    # Compute atmosphere for freestream conditions
    atm = standard_atmosphere(config.altitude)
    V_inf = atm.speed_of_sound * config.mach
    reynolds_number = atm.density * V_inf * 1.0 / atm.dynamic_viscosity

    # Base SU2 config
    su2_config = SU2HypersonicConfig(
        mach=config.mach,
        freestream_pressure=atm.pressure,
        freestream_temperature=atm.temperature,
        freestream_density=atm.density,
        freestream_viscosity=atm.dynamic_viscosity,
        reynolds_number=reynolds_number,
        cfl_number=config.su2_cfl,
        iterations=config.su2_iterations,
    )

    # Apply angle of attack and full2d when aoa is nonzero
    if config.aoa != 0.0:
        su2_config = su2_config.as_full2d().with_aoa(config.aoa)
        print(f"  AoA: {config.aoa} deg, full2d mode")

    # Output directory
    su2_dir = Path(config.output_dir) / "su2"
    su2_dir.mkdir(parents=True, exist_ok=True)

    # Mesh file from Phase 2
    mesh_path = Path(config.output_dir) / "mesh" / f"{config.name}.su2"
    if not mesh_path.exists():
        print(f"  ERROR: Mesh not found at {mesh_path}. Run mesh stage first.")
        return 1

    # Copy mesh to SU2 working directory (SU2 looks for mesh in cwd)
    mesh_dest = su2_dir / mesh_path.name
    if not mesh_dest.exists() or mesh_path.stat().st_size != mesh_dest.stat().st_size:
        import shutil
        shutil.copy2(mesh_path, mesh_dest)
    mesh_filename = mesh_path.name

    solver = SU2Solver()

    from cfd.convergence import ConvergenceStrategy

    strategy = config.convergence_strategy or ConvergenceStrategy.for_mach(config.mach)
    print(f"  Strategy: {len(strategy.stages)} stages")
    for i, stage in enumerate(strategy.stages):
        print(f"    {i + 1}. {stage.name} (M={stage.mach}, "
              f"CFL={stage.cfl}, iters={stage.iterations})")
    results = solver.run_stages(strategy, su2_config, su2_dir, mesh_filename)
    return _report_and_save(results, su2_config, su2_dir, config)


def _run_direct(
    su2_config: SU2HypersonicConfig,
    solver: SU2Solver,
    su2_dir: Path,
    mesh_filename: str,
    config: CaseConfig,
) -> int:
    """Run a single direct RANS solve (legacy approach)."""
    cfg_path = su2_config.write(su2_dir, mesh_filename=mesh_filename)
    print(f"  Config: {cfg_path}")
    print(f"  Running SU2 RANS (M={config.mach}, max iter={config.su2_iterations})...")

    results = solver.run(cfg_path, su2_dir, timeout=7200)
    return _report_and_save(results, su2_config, su2_dir, config)


def _run_euler_rans(
    su2_config: SU2HypersonicConfig,
    solver: SU2Solver,
    su2_dir: Path,
    mesh_filename: str,
    config: CaseConfig,
) -> int:
    """Two-stage RANS convergence: first-order then second-order restart.

    Stage 1: First-order RANS (MUSCL disabled) with conservative settings
             to establish the shock structure and turbulence field.
    Stage 2: Second-order RANS restart from the first-order solution.

    Note: Direct Euler-to-RANS restart fails in SU2 v8.4 because Euler
    produces 4-field restarts (rho, rhoU, rhoV, rhoE) but RANS expects 5
    fields (rho, rhoU, rhoV, rhoE, nu_turb). First-order RANS from scratch
    with proper SA turbulence initialization is the robust approach.
    """
    # --- Stage 1: First-order RANS ---
    print(f"\n  === Stage 1: First-order RANS ({config.su2_euler_iterations} iters) ===")
    fo_config = su2_config.as_first_order_rans()
    fo_config.iterations = config.su2_euler_iterations
    # Use very conservative CFL for initial stability
    fo_cfl = min(config.su2_cfl, 0.01)
    fo_config = fo_config.with_cfl(fo_cfl)
    fo_config = fo_config.with_cfl_adapt(
        cfl_min=0.005, cfl_max=2.0, decrease=0.5, increase=10.0,
    )
    # Use BCGSTAB linear solver (more robust than FGMRES for hypersonic)
    # and relax linear solver tolerance for initial stability
    fo_config.linear_solver = "BCGSTAB"
    fo_config.linear_solver_error = 1e-2
    fo_config.linear_solver_iter = 20
    from cfd.config import SU2HypersonicConfig as _Cfg
    fo_output = _Cfg(**fo_config.__dict__)
    fo_output.output_files = ("RESTART",)

    cfg_path = fo_output.write(su2_dir, mesh_filename=mesh_filename)
    print(f"  Config: {cfg_path}")
    print(f"  CFL: {fo_cfl}, First-order: YES, BCGSTAB linear solver")

    fo_results = solver.run(cfg_path, su2_dir, timeout=3600)
    fo_drop = fo_results.residual_drop
    print(f"  First-order complete: {fo_results.iterations} iters, "
          f"drop={fo_drop:.2f} orders")

    if fo_drop < 2.0:
        print("  WARNING: First-order RANS did not converge well. "
              "Second-order restart may be poor.")

    # Find restart file (SU2 v8.x writes restart.dat, copies to solution.dat)
    restart_file = _find_restart_file(su2_dir)
    if restart_file is None:
        print("  ERROR: No first-order restart file found. "
              "Falling back to direct RANS.")
        return _run_direct(su2_config, solver, su2_dir, mesh_filename, config)

    # Copy restart.dat to solution.dat (SU2 v8.4 reads solution.dat by default)
    import shutil
    solution_path = su2_dir / "solution.dat"
    shutil.copy2(restart_file, solution_path)
    print(f"  Restart file: {restart_file.name} -> solution.dat")

    # --- Stage 2: Continued first-order RANS restart ---
    # MUSCL second-order diverges at hypersonic Mach numbers due to
    # oscillations near the bow shock.  Stay first-order with higher CFL
    # for robust convergence.
    print(f"\n  === Stage 2: Continued RANS restart "
          f"({config.su2_rans_iterations} iters) ===")
    rans_config = su2_config.with_restart(Path("solution.dat"))
    rans_config.muscl = False  # Stay first-order for stability
    rans_config.iterations = config.su2_rans_iterations
    # Keep same CFL as stage 1 to avoid restart divergence
    rans_cfl = min(config.su2_cfl, 0.01)
    rans_config = rans_config.with_cfl(rans_cfl)
    rans_config = rans_config.with_cfl_adapt(
        cfl_min=0.005, cfl_max=1.0, decrease=0.5, increase=3.0,
    )
    rans_config.linear_solver = "BCGSTAB"
    rans_config.linear_solver_error = 1e-4
    rans_config.linear_solver_iter = 40

    cfg_path = rans_config.write(su2_dir, mesh_filename=mesh_filename)
    print(f"  Config: {cfg_path}")
    print(f"  CFL: {rans_cfl}, First-order: YES, continued from Stage 1")

    rans_results = solver.run(cfg_path, su2_dir, timeout=7200)

    # Save results and plot
    return _report_and_save(rans_results, rans_config, su2_dir, config)


def _run_mach_ramp(
    su2_config: SU2HypersonicConfig,
    solver: SU2Solver,
    su2_dir: Path,
    mesh_filename: str,
    config: CaseConfig,
) -> int:
    """Mach ramping strategy using ConvergenceStrategy for staged solves.

    Uses ConvergenceStrategy.for_mach() to automatically determine the
    appropriate number of ramp stages, then executes them sequentially
    via solver.run_stages().
    """
    from cfd.convergence import ConvergenceStrategy

    strategy = ConvergenceStrategy.for_mach(config.mach)
    print(f"  Strategy: {len(strategy.stages)} stages")
    for i, stage in enumerate(strategy.stages):
        print(f"    {i + 1}. {stage.name} (M={stage.mach}, "
              f"CFL={stage.cfl}, iters={stage.iterations})")

    results = solver.run_stages(strategy, su2_config, su2_dir, mesh_filename)
    return _report_and_save(results, su2_config, su2_dir, config)


def _find_restart_file(su2_dir: Path) -> Path | None:
    """Find the latest SU2 restart file in the working directory.

    SU2 v8.x names restart files as 'restart.dat' (or flow_restart_XXX.dat
    in older versions).

    Args:
        su2_dir: Directory to search.

    Returns:
        Path to the restart file, or None if not found.
    """
    import re

    # Try SU2 v8.x naming first: restart.dat
    restart_v8 = su2_dir / "restart.dat"
    if restart_v8.exists():
        return restart_v8

    # Fallback: flow_restart_000XXX.dat (older SU2 versions)
    restart_files = list(su2_dir.glob("flow_restart_*.dat"))
    if not restart_files:
        return None

    # Sort by iteration number (the numeric part of the filename)
    def _iter_num(p: Path) -> int:
        match = re.search(r"flow_restart_(\d+)\.dat", p.name)
        return int(match.group(1)) if match else 0

    restart_files.sort(key=_iter_num)
    return restart_files[-1]


def _report_and_save(
    results: SU2Results,
    su2_config: SU2HypersonicConfig,
    su2_dir: Path,
    config: CaseConfig,
) -> int:
    """Save SU2 results JSON and convergence plot.

    Args:
        results: Parsed SU2 results.
        su2_config: SU2 configuration used.
        su2_dir: SU2 output directory.
        config: Pipeline case config.

    Returns:
        0 on success, 1 if not converged.
    """
    from viz.convergence import plot_convergence

    results_dict = {
        "case": config.name,
        "mach": config.mach,
        "altitude_m": config.altitude,
        "strategy": config.su2_strategy,
        "converged": results.converged,
        "iterations": results.iterations,
        "residual_drop": round(results.residual_drop, 4),
        "final_residual_log10": round(results.final_residual, 4),
        "stagnation_pressure_Pa": (
            round(results.stagnation_pressure, 2)
            if results.stagnation_pressure is not None else None
        ),
        "max_mach": (
            round(results.max_mach, 4)
            if results.max_mach is not None else None
        ),
        "wall_temperature_K": su2_config.wall_temperature,
        "cfl_number": su2_config.cfl_number,
    }
    results_path = su2_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"  Results: {results_path}")

    # Plot convergence
    images_dir = Path(config.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    if results.history:
        plot_path = plot_convergence(results.history, images_dir / "convergence.png")
        print(f"  Convergence plot: {plot_path}")

    # Print summary
    status = "CONVERGED" if results.converged else "DID NOT CONVERGE"
    print(f"\n  === Final Status: {status} ===")
    print(f"  Iterations: {results.iterations}")
    print(f"  Residual drop: {results.residual_drop:.2f} orders")
    print(f"  Final rms[Rho]: 10^{results.final_residual:.2f}")
    if results.stagnation_pressure is not None:
        print(f"  Stagnation pressure: {results.stagnation_pressure:.1f} Pa")
    if results.max_mach is not None:
        print(f"  Max Mach: {results.max_mach:.2f}")

    return 0 if results.converged else 1


def run_postprocess_stage(config: CaseConfig) -> int:
    """Post-process SU2 solution: extract physics, generate plots.

    Produces:
        - output/{name}/postprocess/postprocess.json: derived quantities
        - docs/assets/images/{name}/mach_contour.png: Mach contour
        - docs/assets/images/{name}/pressure_contour.png: pressure contour
        - docs/assets/images/{name}/temperature_contour.png: temperature contour
        - docs/assets/images/{name}/heat_flux.png: surface heat flux
        - docs/assets/images/{name}/shock_structure.png: shock structure
        - docs/assets/images/{name}/shock_standoff.png: standoff measurement
        - docs/assets/images/{name}/body_3d.png: 3D revolved body

    Returns:
        0 on success, 1 on failure.
    """
    print(f"\n[{config.label}] Post-processing stage")

    from cfd.postprocess import (
        build_results_summary,
        extract_surface_profiles,
        save_postprocess_results,
    )
    from cfd.vtu_parser import parse_vtu
    from geometry.blunt_body import generate_contour
    from viz.contour import (
        plot_mach_contour,
        plot_pressure_contour,
        plot_temperature_contour,
    )
    from viz.geometry_3d import plot_body_3d
    from viz.heat_flux import plot_surface_heat_flux
    from viz.shock import plot_shock_standoff_measurement, plot_shock_structure

    # Load body contour
    body_config = config.preset_fn()
    x_body, r_body = generate_contour(body_config)

    # Load VTU solution
    vtu_path = Path(config.output_dir) / "su2" / "flow.vtu"
    if not vtu_path.exists():
        print(f"  ERROR: VTU file not found at {vtu_path}. Run SU2 stage first.")
        return 1

    print(f"  Loading: {vtu_path}")
    data = parse_vtu(vtu_path)

    # Extract surface profiles
    profiles = extract_surface_profiles(data, (x_body, r_body))
    print(f"  Surface profiles: {len(profiles['s'])} points")

    # Build results summary
    summary = build_results_summary(data, config, (x_body, r_body))

    # Save postprocess JSON
    post_dir = Path(config.output_dir) / "postprocess"
    post_dir.mkdir(parents=True, exist_ok=True)
    save_postprocess_results(summary, post_dir / "postprocess.json")
    print(f"  Results: {post_dir / 'postprocess.json'}")

    # Images directory
    images_dir = Path(config.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    n_plots = 0
    n_ok = 0

    # 1. Mach contour
    n_plots += 1
    try:
        path = plot_mach_contour(
            data, images_dir / "mach_contour.png",
            body_contour=(x_body, r_body),
        )
        print(f"  Plot: {path}")
        n_ok += 1
    except (OSError, RuntimeError) as exc:
        print(f"  Mach contour FAILED: {exc}")

    # 2. Pressure contour
    n_plots += 1
    try:
        path = plot_pressure_contour(
            data, images_dir / "pressure_contour.png",
            body_contour=(x_body, r_body),
        )
        print(f"  Plot: {path}")
        n_ok += 1
    except (OSError, RuntimeError) as exc:
        print(f"  Pressure contour FAILED: {exc}")

    # 3. Temperature contour
    n_plots += 1
    try:
        path = plot_temperature_contour(
            data, images_dir / "temperature_contour.png",
            body_contour=(x_body, r_body),
        )
        print(f"  Plot: {path}")
        n_ok += 1
    except (OSError, RuntimeError) as exc:
        print(f"  Temperature contour FAILED: {exc}")

    # 4. Heat flux
    n_plots += 1
    try:
        path = plot_surface_heat_flux(
            profiles["s"], profiles["q"],
            images_dir / "heat_flux.png",
        )
        print(f"  Plot: {path}")
        n_ok += 1
    except (OSError, RuntimeError) as exc:
        print(f"  Heat flux plot FAILED: {exc}")

    # 5. Shock structure
    n_plots += 1
    try:
        path = plot_shock_structure(
            data, images_dir / "shock_structure.png",
            body_contour=(x_body, r_body),
        )
        print(f"  Plot: {path}")
        n_ok += 1
    except (OSError, RuntimeError) as exc:
        print(f"  Shock structure FAILED: {exc}")

    # 6. Shock standoff measurement
    n_plots += 1
    try:
        from cfd.postprocess import measure_shock_standoff_r_nose
        r_nose = measure_shock_standoff_r_nose(x_body, r_body)
        path = plot_shock_standoff_measurement(
            data, images_dir / "shock_standoff.png", R_nose=r_nose,
        )
        print(f"  Plot: {path}")
        n_ok += 1
    except (OSError, RuntimeError) as exc:
        print(f"  Shock standoff plot FAILED: {exc}")

    # 7. 3D body
    n_plots += 1
    try:
        path = plot_body_3d(body_config, images_dir / "body_3d.png")
        print(f"  Plot: {path}")
        n_ok += 1
    except (OSError, RuntimeError) as exc:
        print(f"  3D body plot FAILED: {exc}")

    # Print summary
    stag = summary["stagnation"]
    shock = summary["shock_standoff"]
    heating = summary["total_heating"]
    rg = summary["real_gas_correction"]

    print("\n  --- Post-Processing Summary ---")
    print(f"  Stagnation pressure: {stag['pressure_Pa']:.1f} Pa")
    print(f"  Stagnation temperature: {stag['temperature_K']:.1f} K")
    print(f"  Stagnation heat flux: {stag['heat_flux_W_m2']:.1f} W/m^2" if stag['heat_flux_W_m2'] else "  Stagnation heat flux: N/A")
    print(f"  Shock standoff: {shock['delta_m']*1000:.2f} mm (delta/R = {shock['delta_over_R']:.4f})" if shock['delta_over_R'] else "  Shock standoff: N/A")
    print(f"  Total heating: {heating['Q_total_W']:.1f} W")
    print(f"  Real-gas correction: {rg['correction_factor']:.4f}")
    print(f"  Max Mach: {summary['field_extrema']['max_mach']:.2f}" if summary['field_extrema']['max_mach'] else "  Max Mach: N/A")
    print(f"  Plots: {n_ok}/{n_plots} generated")

    return 0 if n_ok == n_plots else 1


def run_validation_stage(config: CaseConfig) -> int:
    """Run triple validation: Sutton-Graves, Billig, Newtonian comparisons.

    Loads postprocess.json, computes freestream from atmosphere, geometry
    from preset, and runs all three analytical-vs-CFD comparisons.

    Produces:
        - output/{name}/validation/validation.json: validation report
        - docs/assets/images/{name}/validation.png: validation bar plot

    Returns:
        0 on success, 1 on failure.
    """
    print(f"\n[{config.label}] Validation stage")

    from validation.compare import build_validation_report, save_validation_report

    # Load postprocess results
    post_path = Path(config.output_dir) / "postprocess" / "postprocess.json"
    if not post_path.exists():
        print(f"  ERROR: postprocess.json not found at {post_path}. Run postprocess first.")
        return 1

    with open(post_path) as f:
        su2_results = json.load(f)

    # Compute freestream conditions
    atm = standard_atmosphere(config.altitude)
    V_inf = atm.speed_of_sound * config.mach
    freestream = {
        "M": config.mach,
        "rho_inf": atm.density,
        "V_inf": V_inf,
        "altitude": config.altitude,
        "temperature": atm.temperature,
        "pressure": atm.pressure,
    }

    # Compute geometry parameters
    body_config = config.preset_fn()
    geometry = {
        "R_nose": body_config.R_nose,
        "half_angle": body_config.half_angle,
        "base_radius": body_config.base_radius,
    }

    # Build validation report
    report = build_validation_report(su2_results, freestream, geometry)

    # Save validation JSON
    val_dir = Path(config.output_dir) / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    save_validation_report(report, val_dir / "validation.json")
    print(f"  Report: {val_dir / 'validation.json'}")

    # Generate validation bar plot
    images_dir = Path(config.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        from viz.validation import plot_validation_bars
        plot_path = plot_validation_bars(report, images_dir / "validation.png")
        print(f"  Plot: {plot_path}")
    except (OSError, RuntimeError) as exc:
        print(f"  Validation plot FAILED: {exc}")

    # Print summary
    print("\n  --- Validation Summary ---")
    for entry in report["summary_table"]:
        status_mark = "PASS" if entry["status"] == "PASS" else "FAIL"
        print(f"  {entry['quantity']:30s}  SU2={entry['su2']:>12s}  "
              f"Analytical={entry['analytical']:>12s}  "
              f"Error={entry['error_pct']:>6s}  [{status_mark}]")

    overall = "ALL PASSED" if report["all_pass"] else "SOME FAILED"
    print(f"\n  Overall: {overall}")

    return 0 if report["all_pass"] else 1


def run_gci_stage(config: CaseConfig) -> int:
    """Run GCI mesh convergence study on three tiers.

    Runs the reference case on draft, standard, and high tiers using
    the euler-rans strategy, computes GCI for stagnation heat flux
    and shock standoff.

    Produces:
        - output/{name}/gci/gci.json: GCI results
        - docs/assets/images/{name}/gci_convergence.png: GCI plot

    Returns:
        0 on success, 1 on failure.
    """
    print(f"\n[{config.label}] GCI stage")

    from validation.gci import run_gci_study

    gci_results = run_gci_study(
        config,
        quantities=["stagnation_heat_flux", "shock_standoff"],
    )

    if "error" in gci_results:
        print(f"  ERROR: {gci_results['error']}")
        return 1

    # Save GCI JSON
    gci_dir = Path(config.output_dir) / "gci"
    gci_dir.mkdir(parents=True, exist_ok=True)
    gci_path = gci_dir / "gci.json"
    with open(gci_path, "w") as f:
        json.dump(gci_results, f, indent=2)
    print(f"  Results: {gci_path}")

    # Generate GCI convergence plots
    images_dir = Path(config.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        from viz.validation import plot_gci_convergence
        for qty_name, qty_data in gci_results.items():
            plot_path = plot_gci_convergence(
                qty_data, images_dir / f"gci_{qty_name}.png",
            )
            print(f"  Plot: {plot_path}")
    except (OSError, RuntimeError) as exc:
        print(f"  GCI plot FAILED: {exc}")

    # Print summary
    print("\n  --- GCI Summary ---")
    for qty_name, qty_data in gci_results.items():
        print(f"  {qty_data['quantity']}:")
        print(f"    Order of accuracy: {qty_data['apparent_order']:.2f}")
        print(f"    Extrapolated value: {qty_data['extrapolated_value']:.6f}")
        print(f"    GCI (fine): {qty_data['gci_fine_pct']:.2f}%")
        print(f"    Asymptotic ratio: {qty_data['asymptotic_ratio']:.2f}")
        print(f"    Monotonic: {qty_data['monotonic']}")
        print(f"    Status: {'PASSED' if qty_data['passed'] else 'FAILED'}")

    all_pass = all(q["passed"] for q in gci_results.values())
    return 0 if all_pass else 1


def run_apollo_stage(config: CaseConfig) -> int:
    """Run the Apollo CM headline case: mach-ramp M=5 -> M=12, then compare to flight data.

    Executes the full Apollo CM pipeline:
        1. Geometry (Apollo preset: R_nose=0.196m, 50-deg cone, base R=1.955m)
        2. Mesh (standard tier)
        3. SU2 (mach-ramp strategy: M=5 first-order, then M=12 second-order)
        4. Post-process (heat flux, contours, shock standoff)
        5. Validate (Sutton-Graves, Billig, Newtonian)
        6. Flight data comparison (Apollo 4/6/11 data)

    Produces:
        - output/apollo-cm/apollo_results.json: combined results
        - docs/assets/images/apollo-cm/flight_data_comparison.png

    Args:
        config: Case configuration (must use apollo_cm preset).

    Returns:
        0 on success, 1 on failure.
    """
    print(f"\n{'='*60}")
    print("  APOLLO CM HEADLINE CASE")
    print(f"  Mach {config.mach} at {config.altitude/1000:.0f} km altitude")
    print(f"  Strategy: mach-ramp M={config.su2_mach_ramp_start} -> M={config.mach}")
    print(f"{'='*60}")

    from validation.flight_data import compare_to_flight_data
    from viz.flight_data import plot_flight_data_comparison

    # Step 1: Geometry
    geo_result = run_geometry_stage(config)
    if geo_result != 0:
        print("  ERROR: Geometry stage failed.")
        return 1

    # Step 2: Mesh
    mesh_result = run_mesh_stage(config)
    if mesh_result != 0:
        print("  ERROR: Mesh stage failed.")
        return 1

    # Step 3: SU2 (mach-ramp)
    su2_result = run_su2_stage(config)
    if su2_result != 0:
        print("  WARNING: SU2 did not fully converge. Continuing with partial results.")

    # Step 4: Post-process
    post_result = run_postprocess_stage(config)
    if post_result != 0:
        print("  WARNING: Post-processing had issues. Continuing with available data.")

    # Step 5: Validate
    run_validation_stage(config)

    # Step 6: Flight data comparison
    print(f"\n[{config.label}] Flight data comparison stage")

    # Load postprocess results for SU2 heat flux
    post_path = Path(config.output_dir) / "postprocess" / "postprocess.json"
    su2_q_stag = 0.0
    if post_path.exists():
        with open(post_path) as f:
            post_data = json.load(f)
        su2_q_stag = post_data.get("stagnation", {}).get("heat_flux_W_m2", 0.0)

    # Build conditions dict
    su2_conditions = {
        "mach": config.mach,
        "altitude_m": config.altitude,
        "R_nose": config.preset_fn().R_nose,
    }

    # Run flight data comparison
    comparison = compare_to_flight_data(su2_q_stag, su2_conditions)

    # Save Apollo results JSON
    apollo_results = {
        "case": config.name,
        "mach": config.mach,
        "altitude_m": config.altitude,
        "strategy": config.su2_strategy,
        "flight_data_comparison": comparison,
        "su2_stagnation": {
            "heat_flux_W_m2": su2_q_stag,
            "heat_flux_W_cm2": round(su2_q_stag / 10000.0, 2),
        },
    }

    apollo_dir = Path(config.output_dir)
    apollo_dir.mkdir(parents=True, exist_ok=True)
    apollo_path = apollo_dir / "apollo_results.json"
    with open(apollo_path, "w") as f:
        json.dump(apollo_results, f, indent=2)
    print(f"  Apollo results: {apollo_path}")

    # Generate flight data comparison plot
    images_dir = Path(config.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    try:
        plot_path = plot_flight_data_comparison(
            su2_q_stag, su2_conditions, images_dir / "flight_data_comparison.png",
        )
        print(f"  Flight data plot: {plot_path}")
    except (OSError, RuntimeError) as exc:
        print(f"  Flight data plot FAILED: {exc}")

    # Print summary
    print("\n  --- Apollo CM Summary ---")
    print(f"  Mach: {config.mach}, Altitude: {config.altitude/1000:.0f} km")
    print(f"  SU2 stagnation heat flux: {su2_q_stag/10000:.1f} W/cm^2")
    print(f"  Strategy: {config.su2_strategy}")
    print(f"  Flight data comparison: {comparison['summary']}")
    print(f"  Caveats: {len(comparison['caveats'])} noted")

    return 0


STAGE_FUNCTIONS: dict[PipelineStage, Callable[[CaseConfig], int]] = {
    PipelineStage.GEOMETRY: run_geometry_stage,
    PipelineStage.MESH: run_mesh_stage,
    PipelineStage.SU2: run_su2_stage,
    PipelineStage.POSTPROCESS: run_postprocess_stage,
    PipelineStage.VALIDATION: run_validation_stage,
    PipelineStage.GCI: run_gci_stage,
    PipelineStage.APOLLO: run_apollo_stage,
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
