"""Load and visualize external CAD geometry.

Usage:
    python run_import_geometry.py path/to/geometry.step
    python run_import_geometry.py path/to/geometry.stl
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from geometry.external import load_geometry
from viz.geometry_annotated import plot_apollo_geometry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import external CAD geometry for CFD analysis",
    )
    parser.add_argument(
        "geometry_file",
        type=Path,
        help="Path to CAD file (.step, .stp, .stl, .iges)",
    )
    parser.add_argument(
        "--points", "-n",
        type=int,
        default=600,
        help="Number of contour points (default: 600)",
    )
    parser.add_argument(
        "--axis",
        choices=["x", "y"],
        default="x",
        help="Axis of symmetry (default: x)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output path for geometry plot",
    )
    args = parser.parse_args()

    # Load geometry
    print(f"Loading geometry from: {args.geometry_file}")
    geom = load_geometry(args.geometry_file, args.points, args.axis)

    print(f"  Format: {geom.format}")
    print(f"  Points: {geom.n_points}")
    print(f"  Body length: {geom.body_length:.4f} m")
    print(f"  Max radius: {geom.max_radius:.4f} m")
    print(f"  Base radius: {geom.base_radius:.4f} m")

    # Generate plot
    from geometry.config import BluntBodyConfig

    config = BluntBodyConfig(
        R_shield=geom.max_radius,  # Approximate for visualization
        cone_half_angle=0,
        max_radius=geom.max_radius,
        base_radius=geom.base_radius,
        body_length=geom.body_length,
        num_points=geom.n_points,
    )

    output_path = args.output or Path(
        f"docs/assets/images/imported/{args.geometry_file.stem}_geometry.png"
    )

    plot_apollo_geometry(
        config,
        output_path,
        case_name=args.geometry_file.stem,
    )
    print(f"Plot saved to: {output_path}")

    # Save contour data
    import json

    contour_path = output_path.with_suffix(".json")
    contour_data = {
        "source_file": str(geom.source_file),
        "format": geom.format,
        "x": geom.x.tolist(),
        "r": geom.r.tolist(),
        "body_length": geom.body_length,
        "max_radius": geom.max_radius,
        "base_radius": geom.base_radius,
    }
    contour_path.write_text(json.dumps(contour_data, indent=2))
    print(f"Contour data saved to: {contour_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
