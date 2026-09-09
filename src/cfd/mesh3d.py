"""3D cylindrical wind tunnel mesh generation for hypersonic blunt body CFD.

Generates tetrahedral meshes in a cylindrical domain around the body
geometry using Gmsh OpenCASCADE. Features:
- Boolean subtraction: cylinder - body = fluid domain
- Distance-based size field refinement near body wall
- Shock refinement at Billig standoff distance
- Wake refinement downstream of body
- Junction refinement at sphere-cone junction
- Post-generation optimization (Netgen + Laplace3D)

Note: Prismatic boundary layer requires extrudeBoundaryLayer API
(not yet implemented). Current implementation uses tetrahedral
elements with distance-based refinement near the body wall.
"""
from __future__ import annotations

from pathlib import Path

from geometry.step_loader import StepGeometry

from .mesh3d_config import Mesh3DConfig


def generate_3d_mesh(
    geometry: StepGeometry,
    mesh_config: Mesh3DConfig,
    output_path: Path,
    R_nose: float = 4.694,
    body_diameter: float = 3.848,
) -> Path:
    """Generate a 3D cylindrical wind tunnel mesh.

    Creates a cylindrical domain around the body with:
        - Tetrahedral volume mesh in the farfield
        - Prismatic boundary layer at the body wall
        - Smooth size transition from BL to farfield
        - SU2-compatible physical group markers

    The cylindrical domain is oriented with the axis along x (flow direction),
    centered on the body centroid.

    Args:
        geometry: Loaded STEP geometry.
        mesh_config: 3D mesh configuration.
        output_path: Output .su2 mesh file path.
        R_nose: Nose sphere radius (m) for domain sizing.
        body_diameter: Body diameter (m) for domain sizing.

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
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("wind_tunnel_3d")

        # --- Load body geometry ---
        gmsh.model.occ.importShapes(str(geometry.path))
        gmsh.model.occ.synchronize()

        # Get body surfaces and volumes
        body_surfaces = [tag for dim, tag in gmsh.model.getEntities(2) if dim == 2]
        body_volumes = [tag for dim, tag in gmsh.model.getEntities(3) if dim == 3]

        if not body_volumes:
            raise RuntimeError("No volumes found in STEP geometry.")

        # --- Compute cylindrical domain dimensions ---
        bbox = geometry.bbox
        cx = (bbox.x_min + bbox.x_max) / 2.0  # Body centroid x
        cy = 0.0
        cz = 0.0

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
        # OCC addCylinder: (x, y, z) center of bottom face, (dx, dy, dz) axis vector, radius
        cyl_tag = gmsh.model.occ.addCylinder(
            x_min, cy, cz,  # Center of upstream face
            cylinder_length, 0, 0,  # Axis vector (along x)
            radius,
        )
        gmsh.model.occ.synchronize()

        # --- Boolean subtract: cylinder - body = fluid domain ---
        print("  Performing boolean subtraction...")
        fluid_dimtags, map_ = gmsh.model.occ.cut(
            [(3, cyl_tag)],
            [(3, v) for v in body_volumes],
        )
        gmsh.model.occ.synchronize()

        # Get fluid volume
        fluid_volumes = [tag for dim, tag in gmsh.model.getEntities(3) if dim == 3]
        if not fluid_volumes:
            raise RuntimeError("Boolean subtraction failed: no fluid volume created.")

        # --- Classify surfaces after boolean ---
        # After boolean subtraction, we need to identify:
        # - Body surfaces (the cavity walls) - these are inside the original body bbox
        # - Farfield surfaces (cylinder faces) - these are outside the body bbox
        all_surfaces = [tag for dim, tag in gmsh.model.getEntities(2) if dim == 2]

        body_surfs_after = []
        farfield_surfs = []
        sym_surfs = []

        # Body surfaces are those that are CLOSE to the original body surfaces
        # (within a tolerance of the body bounding box)
        body_x_range = (bbox.x_min - 10, bbox.x_max + 10)
        body_y_range = (bbox.y_min - 10, bbox.y_max + 10)
        body_z_range = (bbox.z_min - 10, bbox.z_max + 10)

        for surf_tag in all_surfaces:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, surf_tag)

            # Check if surface center is within body bounding box
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            cz = (zmin + zmax) / 2.0

            is_body = (
                body_x_range[0] <= cx <= body_x_range[1] and
                body_y_range[0] <= cy <= body_y_range[1] and
                body_z_range[0] <= cz <= body_z_range[1]
            )

            # Check if surface is on symmetry plane (y=0)
            # Symmetry surfaces are thin (small extent in y) and centered at y=0
            is_sym = abs(cy) < 1.0 and (ymax - ymin) < 2.0

            if is_body:
                body_surfs_after.append(surf_tag)
            elif is_sym:
                sym_surfs.append(surf_tag)
            else:
                farfield_surfs.append(surf_tag)

        print(f"  Surfaces: body={len(body_surfs_after)}, farfield={len(farfield_surfs)}, sym={len(sym_surfs)}")

        # --- Physical groups ---
        # Body surfaces (isothermal wall)
        if body_surfs_after:
            gmsh.model.geo.addPhysicalGroup(2, body_surfs_after, name="body")
        else:
            # Fallback: use all non-farfield, non-sym surfaces as body
            body_surfs_after = [s for s in all_surfaces if s not in farfield_surfs and s not in sym_surfs]
            if body_surfs_after:
                gmsh.model.geo.addPhysicalGroup(2, body_surfs_after, name="body")
                print(f"  Using fallback body surfaces: {len(body_surfs_after)}")

        # Farfield (cylinder surface + inflow + outflow faces)
        if farfield_surfs:
            gmsh.model.geo.addPhysicalGroup(2, farfield_surfs, name="farfield")

        # Symmetry plane (optional, for 0 deg AoA)
        if mesh_config.use_symmetry and sym_surfs:
            gmsh.model.geo.addPhysicalGroup(2, sym_surfs, name="sym")

        # Fluid volume
        gmsh.model.geo.addPhysicalGroup(3, fluid_volumes, name="fluid")

        gmsh.model.geo.synchronize()

        # --- Size fields ---
        # Note: BoundaryLayer requires extrudeBoundaryLayer API (more complex)
        # For now, use size fields to approximate BL refinement near body
        # Distance field from body surfaces for smooth size transition
        distance_tag = 100
        gmsh.model.mesh.field.add("Distance", distance_tag)
        if body_surfs_after:
            gmsh.model.mesh.field.setNumbers(distance_tag, "SurfacesList", body_surfs_after)

        # Shock refinement: Ball field at expected shock location
        shock_tag = 400
        gmsh.model.mesh.field.add("Ball", shock_tag)
        # Shock standoff: delta/R = 0.143 * exp(3.24/M^2) (Billig correlation)
        import math
        mach = 15.6  # Default, should be passed as parameter
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
        # Exponential ramp for smooth size transitions (matching 2D pattern)
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
        gmsh.model.mesh.field.setNumbers(min_tag, "FieldsList", [size_tag, shock_tag, wake_tag, junc_tag, bg_tag])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_tag)

        # --- Mesh generation ---
        print("  Generating 3D tetrahedral mesh...")
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
        gmsh.option.setNumber("Mesh.Smoothing", 50)  # Increased from 10
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.3)  # Optimize cells with quality < 0.3
        gmsh.model.mesh.generate(3)

        # Post-generation optimization passes
        print("  Running post-generation optimization...")
        gmsh.model.mesh.optimize("Netgen")  # Optimize worst cells
        gmsh.model.mesh.optimize("Laplace3D")  # Smooth node positions

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
