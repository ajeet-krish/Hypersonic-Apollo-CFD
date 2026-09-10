#!/usr/bin/env python3
"""Generate 3D cylindrical wind tunnel meshes for Apollo CM at different quality tiers.

Creates meshes from the 2D body contour by revolving around the x-axis,
then boolean subtracting from a box domain.

Usage:
    uv run python scripts/generate_3d_meshes.py [--tier draft|standard|high|all]
"""
from pathlib import Path
import sys
import time
import math

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import gmsh

from geometry.presets import apollo_cm
from geometry.blunt_body import generate_contour


# Quality tier definitions
TIERS = {
    "draft": {
        "min_element_size": 0.05,
        "max_element_size": 5.0,
        "description": "~2-5M elements",
    },
    "standard": {
        "min_element_size": 0.02,
        "max_element_size": 2.0,
        "description": "~8-15M elements",
    },
    "high": {
        "min_element_size": 0.008,
        "max_element_size": 1.0,
        "description": "~20-30M elements",
    },
}


def create_body_from_contour(
    x_contour: np.ndarray,
    r_contour: np.ndarray,
    n_spline_pts: int = 60,
) -> int:
    """Create a 3D body of revolution from a 2D contour.

    Args:
        x_contour: Axial coordinates (mm).
        r_contour: Radial coordinates (mm).
        n_spline_pts: Number of points for the spline (reduces complexity).

    Returns:
        Volume tag of the created body.
    """
    n_pts = len(x_contour)

    # Downsample for spline
    if n_pts > n_spline_pts:
        indices = np.linspace(0, n_pts - 1, n_spline_pts, dtype=int)
        x_ds = x_contour[indices]
        r_ds = r_contour[indices]
    else:
        x_ds = x_contour
        r_ds = r_contour

    n = len(x_ds)

    # Create points
    pts = []
    for i in range(n):
        tag = gmsh.model.occ.addPoint(float(x_ds[i]), float(r_ds[i]), 0.0)
        pts.append(tag)

    # Axis closure points
    pts.append(gmsh.model.occ.addPoint(float(x_ds[-1]), 0.0, 0.0))  # axis at base
    pts.append(gmsh.model.occ.addPoint(float(x_ds[0]), 0.0, 0.0))   # axis at nose

    gmsh.model.occ.synchronize()

    # Spline along contour
    spline = gmsh.model.occ.addSpline(pts[:n])

    # Lines for closure
    line_base = gmsh.model.occ.addLine(pts[n - 1], pts[n])
    line_axis = gmsh.model.occ.addLine(pts[n], pts[n + 1])

    gmsh.model.occ.synchronize()

    # Wire: skip nose line if first point is on axis
    if abs(r_ds[0]) > 1e-10:
        line_nose = gmsh.model.occ.addLine(pts[n + 1], pts[0])
        gmsh.model.occ.synchronize()
        wire = gmsh.model.occ.addWire([spline, line_base, line_axis, line_nose])
    else:
        wire = gmsh.model.occ.addWire([spline, line_base, line_axis])

    surface = gmsh.model.occ.addPlaneSurface([wire])
    gmsh.model.occ.synchronize()

    # Revolve around x-axis (almost full circle to avoid periodic surfaces)
    angle = 2.0 * math.pi - 0.01  # Leave a tiny gap to break periodicity
    gmsh.model.occ.revolve(
        [(2, surface)],
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        angle,
    )
    gmsh.model.occ.synchronize()

    vols = [tag for dim, tag in gmsh.model.getEntities(3) if dim == 3]
    if not vols:
        raise RuntimeError("Revolve failed: no volume created")

    return vols[0]


def generate_mesh(
    tier: str,
    output_path: Path,
    R_nose: float = 0.196,
    body_diameter: float = 3.848,
) -> dict:
    """Generate a 3D wind tunnel mesh at the specified quality tier.

    Args:
        tier: Quality tier name (draft, standard, high).
        output_path: Output .su2 mesh path.
        R_nose: Nose radius (m).
        body_diameter: Body diameter (m).

    Returns:
        Dict with mesh statistics.
    """
    tier_config = TIERS[tier]
    min_sz = tier_config["min_element_size"]
    max_sz = tier_config["max_element_size"]

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add(f"wind_tunnel_3d_{tier}")

    try:
        # Get body contour
        body_config = apollo_cm()
        x_contour, r_contour = generate_contour(body_config)

        # Convert to mm
        x_mm = x_contour * 1000.0
        r_mm = r_contour * 1000.0

        R_nose_mm = R_nose * 1000.0
        body_diameter_mm = body_diameter * 1000.0

        # Create body
        print(f"  Creating body geometry...")
        body_vol = create_body_from_contour(x_mm, r_mm)
        print(f"  Body volume: {body_vol}")

        # Domain dimensions
        x_body_min = float(x_mm.min())
        x_body_max = float(x_mm.max())

        upstream = 10.0 * R_nose_mm      # 10x nose radius upstream
        downstream = 20.0 * body_diameter_mm  # 20x body diameter downstream
        radius = 8.0 * R_nose_mm          # 8x nose radius lateral

        x_min = x_body_min - upstream
        x_max = x_body_max + downstream
        cyl_length = x_max - x_min

        print(f"  Domain: x=[{x_min:.0f}, {x_max:.0f}], radius={radius:.0f} mm")

        # Create box domain
        box_tag = gmsh.model.occ.addBox(
            x_min, -radius, -radius,
            cyl_length, 2.0 * radius, 2.0 * radius,
        )
        gmsh.model.occ.synchronize()

        # Boolean subtract: box - body = fluid domain
        print("  Performing boolean subtraction...")
        body_dimtag = [(3, body_vol)]
        box_dimtag = [(3, box_tag)]

        fluid_dimtags, _map = gmsh.model.occ.cut(box_dimtag, body_dimtag)
        gmsh.model.occ.synchronize()

        fluid_vols = [tag for dim, tag in gmsh.model.getEntities(3) if dim == 3]
        if not fluid_vols:
            raise RuntimeError("Boolean subtraction failed: no fluid volume")

        # Classify surfaces
        all_surfaces = [tag for dim, tag in gmsh.model.getEntities(2) if dim == 2]

        body_bbox = (x_body_min, -float(r_mm.max()), -float(r_mm.max()),
                     x_body_max, float(r_mm.max()), float(r_mm.max()))

        body_surfs = []
        farfield_surfs = []
        for surf_tag in all_surfaces:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, surf_tag)
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            cz = (zmin + zmax) / 2.0

            # Body surfaces are inside the body bounding box
            in_body = (
                body_bbox[0] - 10 <= cx <= body_bbox[3] + 10 and
                body_bbox[1] - 10 <= cy <= body_bbox[4] + 10 and
                body_bbox[2] - 10 <= cz <= body_bbox[5] + 10
            )

            if in_body:
                body_surfs.append(surf_tag)
            else:
                farfield_surfs.append(surf_tag)

        print(f"  Surfaces: body={len(body_surfs)}, farfield={len(farfield_surfs)}")

        # Physical groups
        if body_surfs:
            gmsh.model.geo.addPhysicalGroup(2, body_surfs, name="body")
        if farfield_surfs:
            gmsh.model.geo.addPhysicalGroup(2, farfield_surfs, name="farfield")
        gmsh.model.geo.addPhysicalGroup(3, fluid_vols, name="fluid")
        gmsh.model.geo.synchronize()

        # Size fields
        distance_tag = 100
        gmsh.model.mesh.field.add("Distance", distance_tag)
        if body_surfs:
            gmsh.model.mesh.field.setNumbers(distance_tag, "SurfacesList", body_surfs)

        # MathEval: exponential ramp from min_size near body to max_size in farfield
        size_tag = 200
        gmsh.model.mesh.field.add("MathEval", size_tag)
        ramp_dist = radius
        gmsh.model.mesh.field.setString(
            size_tag, "F",
            f"{min_sz} * Exp(Min(F{distance_tag}, {ramp_dist}) "
            f"* Log({max_sz}/{min_sz}) / {ramp_dist})",
        )

        # Background field
        bg_tag = 300
        gmsh.model.mesh.field.add("Constant", bg_tag)
        gmsh.model.mesh.field.setNumber(bg_tag, "VIn", max_sz)
        gmsh.model.mesh.field.setNumber(bg_tag, "VOut", max_sz)

        # Combine
        min_tag = 999
        gmsh.model.mesh.field.add("Min", min_tag)
        gmsh.model.mesh.field.setNumbers(min_tag, "FieldsList", [size_tag, bg_tag])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_tag)

        # Mesh generation
        print(f"  Generating mesh (min={min_sz}, max={max_sz})...")
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
        gmsh.option.setNumber("Mesh.Smoothing", 20)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.3)

        gmsh.model.mesh.generate(3)

        n_nodes = gmsh.model.mesh.getNumNodes()
        n_elements = gmsh.model.mesh.getNumNumberOfElements()
        print(f"  Generated: {n_nodes} nodes, {n_elements} elements")

        # Optimize
        if tier != "draft":
            print("  Optimizing...")
            gmsh.model.mesh.optimize("Netgen")
            gmsh.model.mesh.optimize("Laplace3D")

        # Export
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(output_path))
        print(f"  Saved: {output_path} ({output_path.stat().st_size:,} bytes)")

        return {
            "tier": tier,
            "nodes": n_nodes,
            "elements": n_elements,
            "file_size": output_path.stat().st_size,
        }

    finally:
        gmsh.finalize()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate 3D meshes for Apollo CM")
    parser.add_argument("--tier", choices=["draft", "standard", "high", "all"],
                        default="all", help="Quality tier")
    args = parser.parse_args()

    base_dir = Path("output/apollo-cm-3d-fixed/su2/m15_6")
    base_dir.mkdir(parents=True, exist_ok=True)

    tiers = list(TIERS.keys()) if args.tier == "all" else [args.tier]

    results = {}
    for tier in tiers:
        print(f"\n{'='*60}")
        print(f"  Generating {tier} tier mesh")
        print(f"  {TIERS[tier]['description']}")
        print(f"{'='*60}")

        output_path = base_dir / f"mesh_3d_{tier}.su2"
        t0 = time.time()

        try:
            stats = generate_mesh(tier, output_path)
            stats["elapsed"] = time.time() - t0
            results[tier] = stats
            print(f"  Completed in {stats['elapsed']:.1f}s")
        except Exception as exc:
            import traceback
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            results[tier] = {"error": str(exc)}

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for tier, stats in results.items():
        if "error" in stats:
            print(f"  {tier:12s}: FAILED - {stats['error']}")
        else:
            print(
                f"  {tier:12s}: {stats['elements']:>10,} elements, "
                f"{stats['nodes']:>8,} nodes, "
                f"{stats['file_size']/1e6:.1f} MB, "
                f"{stats['elapsed']:.1f}s"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
