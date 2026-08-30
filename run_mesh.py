"""Hypersonic blunt body mesh generation runner."""
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
    """Run blunt body mesh generation pipeline."""
    parser = argparse.ArgumentParser(
        description="Hypersonic blunt body mesh generation",
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
        default="draft",
        help="Mesh refinement tier (default: draft)",
    )
    parser.add_argument(
        "--step",
        choices=[s.value for s in PipelineStage],
        default=None,
        help="Run a single pipeline step (default: all)",
    )
    args = parser.parse_args()

    config = CaseConfig(
        name=args.case,
        label=args.case.replace("-", " ").title(),
        preset_fn=PRESETS[args.case],
        mach=args.mach,
        altitude=args.altitude,
        mesh_tier=args.tier,
    )

    if args.step:
        stages = [PipelineStage(args.step)]
    else:
        stages = None

    return run_full_pipeline(config, stages)


if __name__ == "__main__":
    sys.exit(main())
