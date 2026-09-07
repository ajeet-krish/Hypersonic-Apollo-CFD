"""Unified CLI for the Hypersonic Body CFD pipeline.

Replaces: run_geometry.py, run_mesh.py, run_postprocess.py,
          run_reference.py, run_apollo.py, run_import_geometry.py,
          run_validation.py

Usage:
    python run.py --case apollo-cm --step all
    python run.py --case apollo-cm --step su2 --mach 15.6
    python run.py --case generic --step geometry --mach 8.0
    python run.py --case apollo-cm --step apollo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from geometry.presets import apollo_cm, generic
from pipeline.case_config import CaseConfig, PipelineStage
from pipeline.stages import run_full_pipeline

PRESETS = {
    "generic": generic,
    "apollo-cm": apollo_cm,
}

CONVERGENCE_STRATEGIES = ["direct", "euler-rans", "mach-ramp"]


def main() -> int:
    """Run hypersonic blunt body CFD pipeline."""
    parser = argparse.ArgumentParser(
        description="Hypersonic blunt body aerothermodynamics pipeline",
    )
    parser.add_argument(
        "--case",
        choices=list(PRESETS.keys()),
        default="generic",
        help="Blunt body case (default: generic)",
    )
    parser.add_argument(
        "--mach",
        type=float,
        default=8.0,
        help="Freestream Mach number (default: 8.0)",
    )
    parser.add_argument(
        "--altitude",
        type=float,
        default=30000.0,
        help="Flight altitude in meters (default: 30000)",
    )
    parser.add_argument(
        "--aoa",
        type=float,
        default=0.0,
        help="Angle of attack in degrees (default: 0.0)",
    )
    parser.add_argument(
        "--tier",
        choices=["draft", "standard", "high"],
        default="standard",
        help="Mesh refinement tier (default: standard)",
    )
    parser.add_argument(
        "--step",
        choices=[s.value for s in PipelineStage],
        default=None,
        help="Run a single pipeline step (default: all)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="SU2 max iterations for RANS stage (default: 10000)",
    )
    parser.add_argument(
        "--cfl",
        type=float,
        default=0.1,
        help="SU2 CFL number (default: 0.1)",
    )
    parser.add_argument(
        "--strategy",
        choices=CONVERGENCE_STRATEGIES,
        default="euler-rans",
        help="Convergence strategy (default: euler-rans)",
    )
    parser.add_argument(
        "--euler-iterations",
        type=int,
        default=3000,
        help="Iterations for Euler stage (default: 3000)",
    )
    parser.add_argument(
        "--rans-iterations",
        type=int,
        default=10000,
        help="Iterations for RANS stage (default: 10000)",
    )
    parser.add_argument(
        "--mach-ramp-start",
        type=float,
        default=5.0,
        help="Starting Mach for mach-ramp strategy (default: 5.0)",
    )
    args = parser.parse_args()

    config = CaseConfig(
        name=args.case,
        label=args.case.replace("-", " ").title(),
        preset_fn=PRESETS[args.case],
        mach=args.mach,
        altitude=args.altitude,
        aoa=args.aoa,
        mesh_tier=args.tier,
        su2_iterations=args.iterations,
        su2_cfl=args.cfl,
        su2_strategy=args.strategy,
        su2_euler_iterations=args.euler_iterations,
        su2_rans_iterations=args.rans_iterations,
        su2_mach_ramp_start=args.mach_ramp_start,
    )

    if args.step:
        stages = [PipelineStage(args.step)]
    else:
        stages = None

    return run_full_pipeline(config, stages)


if __name__ == "__main__":
    sys.exit(main())
