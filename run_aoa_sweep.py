"""Apollo CM angle of attack sweep runner.

Runs the Apollo CM pipeline at AoA = 0, 5, 10, 15, 20, 25, 30 degrees
at M=10 (moderate hypersonic for convergence).

Usage:
    python run_aoa_sweep.py
    python run_aoa_sweep.py --mach 12 --aoa 0 10 20 30
    python run_aoa_sweep.py --tier draft --step mesh
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from geometry.presets import apollo_cm
from pipeline.case_config import CaseConfig, PipelineStage
from pipeline.stages import run_full_pipeline


def main() -> int:
    """Run Apollo CM AoA sweep."""
    parser = argparse.ArgumentParser(
        description="Apollo CM angle of attack sweep",
    )
    parser.add_argument(
        "--mach",
        type=float,
        default=10.0,
        help="Freestream Mach number (default: 10.0)",
    )
    parser.add_argument(
        "--aoa",
        type=float,
        nargs="+",
        default=[0, 5, 10, 15, 20],
        help="Angles of attack to sweep (default: 0 5 10 15 20)",
    )
    parser.add_argument(
        "--tier",
        default="draft",
        choices=["draft", "standard", "high"],
        help="Mesh refinement tier (default: draft)",
    )
    parser.add_argument(
        "--step",
        choices=["all", "geometry", "mesh", "su2", "postprocess", "validation"],
        default="all",
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--cfl",
        type=float,
        default=0.01,
        help="SU2 CFL number (default: 0.01)",
    )
    args = parser.parse_args()

    # Map step argument to pipeline stages
    if args.step == "all":
        stages = [
            PipelineStage.GEOMETRY,
            PipelineStage.MESH,
            PipelineStage.SU2,
            PipelineStage.POSTPROCESS,
        ]
    else:
        stages = [PipelineStage(args.step)]

    overall_result = 0

    for aoa in args.aoa:
        print(f"\n{'=' * 60}")
        print(f"  AoA = {aoa} degrees, M = {args.mach}")
        print(f"{'=' * 60}")

        config = CaseConfig(
            name=f"apollo-cm-aoa{aoa}",
            label=f"Apollo CM AoA={aoa}",
            preset_fn=apollo_cm,
            mach=args.mach,
            altitude=54600.0,
            aoa=aoa,
            mesh_tier=args.tier,
            su2_strategy="euler-rans",
            su2_cfl=args.cfl,
        )

        result = run_full_pipeline(config, stages)
        if result != 0:
            overall_result = 1
            print(f"\n  WARNING: AoA={aoa} sweep case failed.")

    return overall_result


if __name__ == "__main__":
    sys.exit(main())
