"""Hypersonic blunt body validation runner.

Runs triple validation (Sutton-Graves, Billig, Newtonian) and/or
GCI mesh convergence study for the reference case.
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
    """Run validation pipeline stages."""
    parser = argparse.ArgumentParser(
        description="Hypersonic blunt body validation and GCI study",
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
        "--step",
        choices=["validation", "gci", "all"],
        default="validation",
        help="Step to run: validation, gci, or all (default: validation)",
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
        "--cfl",
        type=float,
        default=0.1,
        help="SU2 CFL number (default: 0.1)",
    )
    args = parser.parse_args()

    config = CaseConfig(
        name=args.case,
        label=args.case.replace("-", " ").title(),
        preset_fn=PRESETS[args.case],
        mach=args.mach,
        altitude=args.altitude,
        su2_strategy=args.strategy,
        su2_euler_iterations=args.euler_iterations,
        su2_rans_iterations=args.rans_iterations,
        su2_cfl=args.cfl,
    )

    if args.step == "all":
        stages = [PipelineStage.VALIDATION, PipelineStage.GCI]
    elif args.step == "validation":
        stages = [PipelineStage.VALIDATION]
    else:
        stages = [PipelineStage.GCI]

    return run_full_pipeline(config, stages)


if __name__ == "__main__":
    sys.exit(main())
