"""3D cylindrical wind tunnel mesh generation for hypersonic blunt body CFD.

Generates tetrahedral meshes in a cylindrical domain around the body
geometry using Gmsh OpenCASCADE. Features:
- Boolean subtraction: cylinder - body = fluid domain
- Fallback: revolve 2D body contour if STEP boolean fails
- Distance-based size field refinement near body wall
- Shock refinement at Billig standoff distance
- Wake refinement downstream of body
- Junction refinement at sphere-cone junction
- Post-generation optimization (Netgen + Laplace3D)
- Bad-cell check with additional optimization pass if needed

Note: Prismatic boundary layer requires extrudeBoundaryLayer API
(not yet implemented). Current implementation uses tetrahedral
elements with distance-based refinement near the body wall.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from geometry.step_loader import StepGeometry

from .mesh3d_config import Mesh3DConfig


def _classify_surfaces_with_normals(
    all_surfaces: list[int],
    body_bbox: tuple[float, float, float, float, float, float],
    sym_tolerance: float = 2.0,
) -> tuple[list[int], list[int], list[int]]:
    """Classify surfaces as body, farfield, or symmetry using bounding box and normals.

    Primary classification uses the body bounding box (surfaces whose centroid
    falls within the expanded body bbox are classified as body surfaces).
    Secondary classification uses surface normal direction: body surfaces have
    normals pointing inward (toward the fluid domain), farfield surfaces have
    normals pointing outward (away from the fluid domain).

    Symmetry surfaces are those on the y=0 plane (thin extent in y, centered
    near y=0).

    Args:
        all_surfaces: List of surface tags from Gmsh model.
        body_bbox: (x_min, y_min, z_min, x_max, y_max, z_max) of original body.
        sym_tolerance: Tolerance for symmetry plane detection (mm).

    Returns:
        Tuple of (body_surfs, farfield_surfs, sym_surfs).
    """
    import gmsh

    body_x_min, body_y_min, body_z_min, body_x_max, body_y_max, body_z_max = body_bbox

    # Expand body bbox by 10mm tolerance for classification
    tol = 10.0
    bx_range = (body_x_min - tol, body_x_max + tol)
    by_range = (body_y_min - tol, body_y_max + tol)
    bz_range = (body_z_min - tol, body_z_max + tol)

    # Fluid domain centroid (approximate center of the cylindrical domain)
    fc_x = (body_x_min + body_x_max) / 2.0
    fc_y = 0.0
    fc_z = 0.0

    body_surfs: list[int] = []
    farfield_surfs: list[int] = []
    sym_surfs: list[int] = []

    for surf_tag in all_surfaces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, surf_tag)
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        cz = (zmin + zmax) / 2.0

        # Symmetry plane: thin surface centered near y=0
        is_sym = abs(cy) < sym_tolerance and (ymax - ymin) < 2.0 * sym_tolerance

        # Bounding box classification
        in_body_bbox = (
            bx_range[0] <= cx <= bx_range[1] and
            by_range[0] <= cy <= by_range[1] and
            bz_range[0] <= cz <= bz_range[1]
        )

        if is_sym:
            sym_surfs.append(surf_tag)
        elif in_body_bbox:
            # Secondary check: surface normal direction
            # Body surfaces should have normals pointing inward (toward fluid)
            # Farfield surfaces should have normals pointing outward (away from fluid)
            try:
                # Compute surface normal at centroid using parametric coordinates
                # For planar/cylindrical surfaces, (0.5, 0.5) is near the center
                normal = gmsh.model.getNormal(surf_tag, [0.5, 0.5])
                # Vector from surface center to fluid domain center
                to_center = np.array([fc_x - cx, fc_y - cy, fc_z - cz])
                to_center_norm = np.linalg.norm(to_center)
                if to_center_norm > 1e-10:
                    to_center_unit = to_center / to_center_norm
                    # dot > 0: normal points toward center (body surface)
                    # dot < 0: normal points away from center (farfield surface)
                    dot = float(np.dot(normal, to_center_unit))
                    if dot < -0.3:
                        # Normal clearly points away from center -> farfield
                        farfield_surfs.append(surf_tag)
                    else:
                        # Normal points toward center or ambiguous -> body
                        body_surfs.append(surf_tag)
                else:
                    body_surfs.append(surf_tag)
            except Exception:
                # If normal computation fails, fall back to bbox classification
                body_surfs.append(surf_tag)
        else:
            farfield_surfs.append(surf_tag)

    return body_surfs, farfield_surfs, sym_surfs


def _revolve_body_contour(
    x_contour: np.ndarray,
    r_contour: np.ndarray,
) -> None:
    """Create a 3D solid of revolution from a 2D axisymmetric body contour.

    Revolves the (x, r) contour around the x-axis to create a solid body
    suitable for boolean subtraction from the cylindrical domain.

    The contour is expected to go from the nose tip (x=0, r=0) along the
    body surface to the base.  The revolution creates a closed wire by
    adding a line from the last point back to the first point along the
    axis (r=0).

    Args:
        x_contour: Axial coordinates of the body contour (mm).
        r_contour: Radial coordinates of the body contour (mm).
    """
    import gmsh

    n_pts = len(x_contour)
    if n_pts < 3:
        raise RuntimeError("Body contour too short for revolution (need >= 3 points).")

    # Downsample contour if too many points (Gmsh OCC struggles with >100 points)
    max_spline_pts = 80
    if n_pts > max_spline_pts:
        # Keep first, last, and evenly spaced points
        indices = np.linspace(0, n_pts - 1, max_spline_pts, dtype=int)
        x_contour = x_contour[indices]
        r_contour = r_contour[indices]
        n_pts = len(x_contour)

    # Create points along the contour (including axis points for closure)
    point_tags: list[int] = []
    for i in range(n_pts):
        tag = gmsh.model.occ.addPoint(float(x_contour[i]), float(r_contour[i]), 0.0)
        point_tags.append(tag)

    # Close the wire: add line from last contour point back to first along axis
    # Last point -> axis at same x -> axis at first x -> first point
    tag_axis_last = gmsh.model.occ.addPoint(float(x_contour[-1]), 0.0, 0.0)
    tag_axis_first = gmsh.model.occ.addPoint(float(x_contour[0]), 0.0, 0.0)
    point_tags.append(tag_axis_last)
    point_tags.append(tag_axis_first)

    gmsh.model.occ.synchronize()

    # Create spline along the contour
    contour_spline = gmsh.model.occ.addSpline(point_tags[:n_pts])

    # Create lines for the closure
    # Base: last contour point -> axis at same x
    line_base = gmsh.model.occ.addLine(point_tags[n_pts - 1], point_tags[n_pts])
    # Axis: axis at last x -> axis at first x
    line_axis = gmsh.model.occ.addLine(point_tags[n_pts], point_tags[n_pts + 1])

    gmsh.model.occ.synchronize()

    # Nose line: axis at first x -> first contour point
    # Skip if first point is already on axis (r ~= 0) to avoid zero-length line
    if abs(r_contour[0]) > 1e-10:
        line_nose = gmsh.model.occ.addLine(point_tags[n_pts + 1], point_tags[0])
        gmsh.model.occ.synchronize()
        wire = gmsh.model.occ.addWire([contour_spline, line_base, line_axis, line_nose])
    else:
        # First point is on axis, just close with base and axis lines
        wire = gmsh.model.occ.addWire([contour_spline, line_base, line_axis])
    surface = gmsh.model.occ.addPlaneSurface([wire])
    gmsh.model.occ.synchronize()

    # Revolve 360 degrees around x-axis
    gmsh.model.occ.revolve(
        [(2, surface)],
        0.0, 0.0, 0.0,  # Point on axis
        1.0, 0.0, 0.0,  # Axis direction (x-axis)
        2.0 * math.pi,   # Full revolution
    )
    gmsh.model.occ.synchronize()


def generate_3d_mesh(
    geometry: StepGeometry,
    mesh_config: Mesh3DConfig,
    output_path: Path,
    R_nose: float = 4.694,
    body_diameter: float = 3.848,
    contour_x: np.ndarray | None = None,
    contour_r: np.ndarray | None = None,
) -> Path:
    """Generate a 3D cylindrical wind tunnel mesh.

    Creates a cylindrical domain around the body with:
        - Tetrahedral volume mesh in the farfield
        - Distance-based size field refinement near body wall
        - Smooth size transition from BL to farfield
        - SU2-compatible physical group markers

    The cylindrical domain is oriented with the axis along x (flow direction),
    centered on the body centroid.

    If the STEP boolean subtract fails and contour_x/contour_r are provided,
    falls back to revolving the 2D body contour to create the 3D body geometry.

    Args:
        geometry: Loaded STEP geometry.
        mesh_config: 3D mesh configuration.
        output_path: Output .su2 mesh file path.
        R_nose: Nose sphere radius (m) for domain sizing.
        body_diameter: Body diameter (m) for domain sizing.
        contour_x: Optional 2D body contour x-coordinates (mm) for fallback.
        contour_r: Optional 2D body contour r-coordinates (mm) for fallback.

    Returns:
        Path to the generated .su2 mesh file.

    Raises:
        RuntimeError: If mesh generation fails.
    """
    import gmsh

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("wind_tunnel_3d")

        # --- Load body geometry (with fallback) ---
        # Try STEP first, fall back to revolve if it fails
        boolean_failed = False
        try:
            gmsh.model.occ.importShapes(str(geometry.path))
            gmsh.model.occ.synchronize()
        except Exception as exc:
            print(f"  WARNING: STEP import failed ({exc})")
            if contour_x is not None and contour_r is not None:
                boolean_failed = True
            else:
                raise RuntimeError(
                    f"STEP import failed and no fallback contour provided: {exc}"
                ) from exc

        # Fragment all geometry to break periodic surfaces from STEP
        # Only do this for STEP imports (multiple volumes), not for revolve
        if not boolean_failed:
            print("  Fragmenting geometry to break periodic surfaces...")
            all_vols = [(dim, tag) for dim, tag in gmsh.model.getEntities() if dim == 3]
            if len(all_vols) > 1:
                try:
                    # Fuse volumes to break periodicity (self-fuse)
                    gmsh.model.occ.fuse(all_vols, [])
                    gmsh.model.occ.synchronize()
                except Exception as exc:
                    print(f"  WARNING: Self-fuse failed ({exc}), continuing")
            elif len(all_vols) == 1:
                print(f"  Single volume found, skipping self-fuse")

        body_volumes = [tag for dim, tag in gmsh.model.getEntities(3) if dim == 3]

        # Check if STEP geometry is usable (has valid volumes for boolean)
        # If the STEP has surfaces but no proper volumes, or if it's a 2D surface
        # exported as 3D, use the revolve fallback
        if not body_volumes and not boolean_failed:
            if contour_x is not None and contour_r is not None:
                print("  No volumes in STEP, attempting revolve fallback...")
                boolean_failed = True
            else:
                raise RuntimeError("No volumes found in STEP geometry.")

        if boolean_failed:
            gmsh.model.remove()
            gmsh.model.add("wind_tunnel_3d")
            print("  Revolving 2D body contour to create 3D geometry...")
            _revolve_body_contour(contour_x, contour_r)
            body_volumes = [tag for dim, tag in gmsh.model.getEntities(3) if dim == 3]
            if not body_volumes:
                raise RuntimeError("Revolve fallback failed: no body volume created.")

            # Fragment revolved body to break periodic surfaces
            print("  Fragmenting revolved body to break periodic surfaces...")
            all_vols = [(dim, tag) for dim, tag in gmsh.model.getEntities(3) if dim == 3]
            if len(all_vols) > 0:
                try:
                    gmsh.model.occ.fragment(all_vols, [])
                    gmsh.model.occ.synchronize()
                except Exception as exc:
                    print(f"  WARNING: Fragment failed ({exc}), continuing")

        # --- Compute cylindrical domain dimensions ---
        bbox = geometry.bbox
        cx = (bbox.x_min + bbox.x_max) / 2.0  # Body centroid x

        # Convert R_nose from meters to millimeters (STEP file units)
        R_nose_mm = R_nose * 1000.0
        body_diameter_mm = body_diameter * 1000.0

        upstream = mesh_config.upstream_factor * R_nose_mm
        downstream = mesh_config.downstream_factor * body_diameter_mm
        radius = mesh_config.lateral_factor * R_nose_mm

        x_min = bbox.x_min - upstream
        x_max = bbox.x_max + downstream
        cylinder_length = x_max - x_min

        print(f"  Domain: x=[{x_min:.2f}, {x_max:.2f}], radius={radius:.2f}")
        print(f"  Upstream: {upstream:.2f} mm, Downstream: {downstream:.2f} mm")

        # --- Create cylinder (axis along x) ---
        # Use box-shaped domain to avoid periodic surface issues with cylinders
        cyl_tag = gmsh.model.occ.addBox(
            x_min,
            -radius,
            -radius,
            cylinder_length,
            2.0 * radius,
            2.0 * radius,
        )
        gmsh.model.occ.synchronize()

        # --- Boolean subtract: box - body = fluid domain ---
        print("  Performing boolean subtraction...")
        try:
            fluid_dimtags, _map = gmsh.model.occ.cut(
                [(3, cyl_tag)],
                [(3, v) for v in body_volumes],
            )
            gmsh.model.occ.synchronize()
        except Exception as exc:
            print(f"  WARNING: Boolean subtract failed ({exc})")
            if contour_x is not None and contour_r is not None:
                print("  Falling back to revolve geometry...")
                boolean_failed = True
            else:
                raise RuntimeError(
                    f"Boolean subtraction failed: {exc}. "
                    "Try providing fallback contour (contour_x, contour_r)."
                ) from exc

        # If boolean failed, restart with revolve geometry
        if boolean_failed:
            gmsh.model.remove()
            gmsh.model.add("wind_tunnel_3d")
            print("  Revolving 2D body contour to create 3D geometry...")
            _revolve_body_contour(contour_x, contour_r)
            body_volumes = [tag for dim, tag in gmsh.model.getEntities(3) if dim == 3]
            if not body_volumes:
                raise RuntimeError("Revolve fallback failed: no body volume created.")

            # Fragment revolved body to break periodic surfaces
            print("  Fragmenting revolved body to break periodic surfaces...")
            all_vols = [(dim, tag) for dim, tag in gmsh.model.getEntities(3) if dim == 3]
            if len(all_vols) > 0:
                try:
                    # Cut body with a thin box to break periodicity
                    # Place a thin cutting plane at y=epsilon to break the revolution symmetry
                    bbox_body = gmsh.model.occ.getBoundingBox(3, all_vols[0][1])
                    cut_box = gmsh.model.occ.addBox(
                        bbox_body[0] - 1.0,  # x_min - margin
                        -0.001,               # y_min (thin slice)
                        bbox_body[2] - 1.0,   # z_min - margin
                        bbox_body[3] - bbox_body[0] + 2.0,  # x_size
                        0.002,                # y_size (thin)
                        bbox_body[5] - bbox_body[2] + 2.0,  # z_size
                    )
                    gmsh.model.occ.synchronize()
                    gmsh.model.occ.cut(all_vols, [(3, cut_box)])
                    gmsh.model.occ.synchronize()
                except Exception as exc:
                    print(f"  WARNING: Cut to break periodicity failed ({exc}), continuing")

            body_volumes = [tag for dim, tag in gmsh.model.getEntities(3) if dim == 3]

            # Recreate cylinder and subtract
            bbox = geometry.bbox
            R_nose_mm = R_nose * 1000.0
            body_diameter_mm = body_diameter * 1000.0
            upstream = mesh_config.upstream_factor * R_nose_mm
            downstream = mesh_config.downstream_factor * body_diameter_mm
            radius = mesh_config.lateral_factor * R_nose_mm
            x_min = bbox.x_min - upstream
            x_max = bbox.x_max + downstream
            cylinder_length = x_max - x_min

            cyl_tag = gmsh.model.occ.addBox(
                x_min, -radius, -radius,
                cylinder_length, 2.0 * radius, 2.0 * radius,
            )
            gmsh.model.occ.synchronize()

            fluid_dimtags, _map = gmsh.model.occ.cut(
                [(3, cyl_tag)],
                [(3, v) for v in body_volumes],
            )
            gmsh.model.occ.synchronize()

        # Get fluid volume
        fluid_volumes = [tag for dim, tag in gmsh.model.getEntities(3) if dim == 3]
        if not fluid_volumes:
            raise RuntimeError("Boolean subtraction failed: no fluid volume created.")

        # --- Classify surfaces after boolean ---
        all_surfaces = [tag for dim, tag in gmsh.model.getEntities(2) if dim == 2]

        body_bbox = (bbox.x_min, bbox.y_min, bbox.z_min, bbox.x_max, bbox.y_max, bbox.z_max)
        body_surfs_after, farfield_surfs, sym_surfs = _classify_surfaces_with_normals(
            all_surfaces, body_bbox
        )

        print(
            f"  Surfaces: body={len(body_surfs_after)}, "
            f"farfield={len(farfield_surfs)}, sym={len(sym_surfs)}"
        )

        # --- Physical groups ---
        if body_surfs_after:
            gmsh.model.geo.addPhysicalGroup(2, body_surfs_after, name="body")
        else:
            # Fallback: use all non-farfield, non-sym surfaces as body
            body_surfs_after = [
                s for s in all_surfaces
                if s not in farfield_surfs and s not in sym_surfs
            ]
            if body_surfs_after:
                gmsh.model.geo.addPhysicalGroup(2, body_surfs_after, name="body")
                print(f"  Using fallback body surfaces: {len(body_surfs_after)}")

        if farfield_surfs:
            gmsh.model.geo.addPhysicalGroup(2, farfield_surfs, name="farfield")

        if mesh_config.use_symmetry and sym_surfs:
            gmsh.model.geo.addPhysicalGroup(2, sym_surfs, name="sym")

        gmsh.model.geo.addPhysicalGroup(3, fluid_volumes, name="fluid")

        gmsh.model.geo.synchronize()

        # --- Size fields ---
        # Distance field from body surfaces for smooth size transition
        distance_tag = 100
        gmsh.model.mesh.field.add("Distance", distance_tag)
        if body_surfs_after:
            gmsh.model.mesh.field.setNumbers(distance_tag, "SurfacesList", body_surfs_after)

        # Shock refinement: Ball field at expected shock location
        shock_tag = 400
        gmsh.model.mesh.field.add("Ball", shock_tag)
        # Shock standoff: delta/R = 0.143 * exp(3.24/M^2) (Billig correlation)
        mach = mesh_config.mach
        standoff_ratio = 0.143 * math.exp(3.24 / (mach ** 2))
        shock_x = bbox.x_min - standoff_ratio * R_nose_mm  # Upstream of nose
        gmsh.model.mesh.field.setNumber(shock_tag, "XCenter", shock_x)
        gmsh.model.mesh.field.setNumber(shock_tag, "YCenter", 0.0)
        gmsh.model.mesh.field.setNumber(shock_tag, "ZCenter", 0.0)
        gmsh.model.mesh.field.setNumber(shock_tag, "VIn", mesh_config.min_element_size * 2)
        gmsh.model.mesh.field.setNumber(shock_tag, "VOut", mesh_config.max_element_size * 0.5)
        gmsh.model.mesh.field.setNumber(shock_tag, "Radius", standoff_ratio * R_nose_mm * 2)

        # Wake refinement: Box field downstream of body
        wake_tag = 500
        gmsh.model.mesh.field.add("Box", wake_tag)
        gmsh.model.mesh.field.setNumber(wake_tag, "XMin", bbox.x_max)
        gmsh.model.mesh.field.setNumber(wake_tag, "XMax", bbox.x_max + 10.0 * R_nose_mm)
        gmsh.model.mesh.field.setNumber(wake_tag, "YMin", -bbox.y_max * 0.5)
        gmsh.model.mesh.field.setNumber(wake_tag, "YMax", bbox.y_max * 0.5)
        gmsh.model.mesh.field.setNumber(wake_tag, "ZMin", -bbox.z_max * 0.5)
        gmsh.model.mesh.field.setNumber(wake_tag, "ZMax", bbox.z_max * 0.5)
        gmsh.model.mesh.field.setNumber(wake_tag, "VIn", mesh_config.min_element_size * 3)
        gmsh.model.mesh.field.setNumber(wake_tag, "VOut", mesh_config.max_element_size * 0.7)

        # Junction refinement: Ball field at sphere-cone junction
        junc_tag = 450
        gmsh.model.mesh.field.add("Ball", junc_tag)
        junc_x = bbox.x_min + R_nose_mm  # Approximate junction location
        gmsh.model.mesh.field.setNumber(junc_tag, "XCenter", junc_x)
        gmsh.model.mesh.field.setNumber(junc_tag, "YCenter", 0.0)
        gmsh.model.mesh.field.setNumber(junc_tag, "ZCenter", 0.0)
        gmsh.model.mesh.field.setNumber(junc_tag, "VIn", mesh_config.min_element_size * 1.5)
        gmsh.model.mesh.field.setNumber(junc_tag, "VOut", mesh_config.max_element_size * 0.3)
        gmsh.model.mesh.field.setNumber(junc_tag, "Radius", 1.5 * R_nose_mm)

        # MathEval: ramp from min_size near body to max_size in farfield
        size_tag = 200
        gmsh.model.mesh.field.add("MathEval", size_tag)
        min_sz = mesh_config.min_element_size
        max_sz = mesh_config.max_element_size
        ramp_dist = radius  # Use cylinder radius as ramp distance
        gmsh.model.mesh.field.setString(
            size_tag,
            "F",
            f"{min_sz} * Exp(Min(F{distance_tag}, {ramp_dist}) "
            f"* Log({max_sz}/{min_sz}) / {ramp_dist})",
        )

        # Background mesh (constant size in farfield)
        bg_tag = 300
        gmsh.model.mesh.field.add("Constant", bg_tag)
        gmsh.model.mesh.field.setNumber(bg_tag, "VIn", mesh_config.max_element_size)
        gmsh.model.mesh.field.setNumber(bg_tag, "VOut", mesh_config.max_element_size)

        # Combine fields with Min
        min_tag = 999
        gmsh.model.mesh.field.add("Min", min_tag)
        gmsh.model.mesh.field.setNumbers(
            min_tag, "FieldsList",
            [size_tag, shock_tag, wake_tag, junc_tag, bg_tag],
        )
        gmsh.model.mesh.field.setAsBackgroundMesh(min_tag)

        # --- Mesh generation ---
        print("  Generating 3D tetrahedral mesh...")
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
        gmsh.option.setNumber("Mesh.Smoothing", 20)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.3)
        # Moderate RandomFactor for numerical stability
        gmsh.option.setNumber("Mesh.RandomFactor", 1e-3)
        gmsh.option.setNumber("Mesh.RandomSeed", 42)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", 4)
        gmsh.model.mesh.generate(3)

        # Post-generation optimization passes (skip for draft speed)
        if mesh_config.mesh_tier != "draft":
            print("  Running post-generation optimization...")
            gmsh.model.mesh.optimize("Netgen")
            gmsh.model.mesh.optimize("Laplace3D")

        # --- Bad-cell check and additional optimization (skip for draft) ---
        if mesh_config.mesh_tier != "draft":
            n_tets = 0
            n_bad = 0
            for elem_dim, elem_tag in gmsh.model.getEntities(3):
                elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3, elem_tag)
                for etype, tags, nodes in zip(elem_types, elem_tags, elem_node_tags):
                    if etype == 4:  # Tetrahedron
                        n_nodes_per = 4
                        n_elems = len(tags)
                        n_tets += n_elems
                        for i in range(n_elems):
                            node_ids = nodes[i * n_nodes_per:(i + 1) * n_nodes_per]
                            coords = np.array([
                                gmsh.model.mesh.getNode(int(nid))[:3]
                                for nid in node_ids
                            ])
                            # Simple quality check: volume-based
                            v1 = coords[1] - coords[0]
                            v2 = coords[2] - coords[0]
                            v3 = coords[3] - coords[0]
                            det = np.dot(v1, np.cross(v2, v3))
                            vol = abs(det) / 6.0
                            if vol < 1e-30:
                                n_bad += 1

            if n_tets > 0:
                pct_bad = n_bad / n_tets * 100.0
                print(f"  Quality check: {n_bad}/{n_tets} bad cells ({pct_bad:.1f}%)")
                if pct_bad > 15.0:
                    print("  Bad cell ratio > 15%, running additional optimization pass...")
                    gmsh.model.mesh.optimize("Netgen")
                    gmsh.model.mesh.optimize("Laplace3D")

        # --- Export ---
        gmsh.write(str(output_path))
        print(f"  Mesh exported: {output_path}")

    except Exception as exc:
        raise RuntimeError(f"3D mesh generation failed: {exc}") from exc
    finally:
        try:
            gmsh.finalize()
        except OSError:
            pass

    return output_path
