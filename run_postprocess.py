"""Post-processing runner for hypersonic blunt body CFD results.

Extracts physics from SU2 solution, generates contour plots, and
computes derived quantities.
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
    """Run post-processing stage."""
    parser = argparse.ArgumentParser(
        description="Post-process hypersonic blunt body CFD results",
    )
    parser.add_argument(
        "--case",
        choices=list(PRESETS.keys()),
        default="generic",
        help="Blunt body case (default: generic)",
    )
    parser.add_argument(
        "--step",
        choices=[s.value for s in PipelineStage],
        default=None,
        help="Run a single pipeline step (default: postprocess)",
    )
    args = parser.parse_args()

    config = CaseConfig(
        name=args.case,
        label=args.case.replace("-", " ").title(),
        preset_fn=PRESETS[args.case],
    )

    stages = [PipelineStage.POSTPROCESS]
    if args.step:
        stages = [PipelineStage(args.step)]

    return run_full_pipeline(config, stages)


if __name__ == "__main__":
    sys.exit(main())
