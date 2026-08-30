"""Hypersonic blunt body reference case runner.

Runs the full pipeline (geometry -> mesh -> SU2) for the M=8
blunt body aerothermodynamics reference case.
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


def main() -> int:
    """Run hypersonic blunt body reference pipeline."""
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
        help="SU2 max iterations (default: 10000)",
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
        mesh_tier=args.tier,
        su2_iterations=args.iterations,
        su2_cfl=args.cfl,
    )

    if args.step:
        stages = [PipelineStage(args.step)]
    else:
        stages = None

    return run_full_pipeline(config, stages)


if __name__ == "__main__":
    sys.exit(main())
