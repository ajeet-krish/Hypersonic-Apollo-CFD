"""Apollo Command Module headline case runner.

Runs the full Apollo CM pipeline: geometry -> mesh -> SU2 (mach-ramp
M=5 -> M=15.6) -> postprocess -> validation -> flight data comparison.

Apollo CM geometry: spherical heat shield R=4.694m + toroidal shoulder
fillet R=0.196m + 33-deg conical afterbody. Max diameter 3.91m.

AS-202 Case 3 conditions: M=15.6 at 54.6 km altitude (IRJET 2017,
Shafeeque et al.).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from geometry.presets import apollo_cm
from pipeline.case_config import CaseConfig, PipelineStage
from pipeline.stages import run_full_pipeline


def main() -> int:
    """Run Apollo CM headline case pipeline."""
    parser = argparse.ArgumentParser(
        description="Apollo CM aerothermodynamics headline case",
    )
    parser.add_argument(
        "--step",
        choices=["all", "geometry", "mesh", "su2", "postprocess",
                 "validation", "apollo"],
        default="all",
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--tier",
        choices=["draft", "standard", "high"],
        default="standard",
        help="Mesh refinement tier (default: standard)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20000,
        help="SU2 max iterations for RANS stage (default: 20000)",
    )
    parser.add_argument(
        "--euler-iterations",
        type=int,
        default=5000,
        help="Iterations for first-order stage in mach-ramp (default: 5000)",
    )
    parser.add_argument(
        "--cfl",
        type=float,
        default=0.01,
        help="SU2 CFL number (default: 0.01 for stability)",
    )
    parser.add_argument(
        "--mach-ramp-start",
        type=float,
        default=5.0,
        help="Starting Mach for mach-ramp strategy (default: 5.0)",
    )
    args = parser.parse_args()

    # Apollo CM: M=15.6, 54.6 km, AS-202 Case 3 (IRJET 2017)
    config = CaseConfig(
        name="apollo-cm",
        label="Apollo CM",
        preset_fn=apollo_cm,
        mach=15.6,
        altitude=54600.0,
        gamma=1.4,
        mesh_tier=args.tier,
        su2_iterations=args.iterations,
        su2_cfl=args.cfl,
        su2_strategy="mach-ramp",
        su2_euler_iterations=args.euler_iterations,
        su2_rans_iterations=args.iterations,
        su2_mach_ramp_start=args.mach_ramp_start,
    )

    # Map step argument to pipeline stages
    if args.step == "all":
        stages = [
            PipelineStage.GEOMETRY,
            PipelineStage.MESH,
            PipelineStage.SU2,
            PipelineStage.POSTPROCESS,
            PipelineStage.VALIDATION,
            PipelineStage.APOLLO,
        ]
    elif args.step == "apollo":
        stages = [PipelineStage.APOLLO]
    else:
        stages = [PipelineStage(args.step)]

    return run_full_pipeline(config, stages)


if __name__ == "__main__":
    sys.exit(main())
