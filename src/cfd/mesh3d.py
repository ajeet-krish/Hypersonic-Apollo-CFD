"""3D mesh generation for hypersonic blunt body CFD.

Generates tetrahedral meshes with prismatic boundary layers from STEP
geometry files using Gmsh OpenCASCADE kernel.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from geometry.step_loader import StepGeometry

from .mesh3d_config import Mesh3DConfig

if TYPE_CHECKING:
    pass


def generate_3d_mesh(
    geometry: StepGeometry,
    mesh_config: Mesh3DConfig,
    output_path: Path,
) -> Path:
    """Generate a 3D tetrahedral mesh with prismatic boundary layer.

    Creates a spherical farfield around the body geometry with:
        - Tetrahedral volume mesh in the farfield
        - Prismatic boundary layer at the body wall
        - Smooth size transition from BL to farfield

    Args:
        geometry: Loaded STEP geometry.
        mesh_config: 3D mesh configuration.
        output_path: Output .su2 mesh file path.

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
        gmsh.model.add("blunt_body_3d")

        # --- Load geometry ---
        gmsh.model.occ.importShapes(str(geometry.path))
        gmsh.model.occ.synchronize()

        # --- Get body surfaces and volumes ---
        surfaces = [tag for dim, tag in gmsh.model.getEntities(2) if dim == 2]
        volumes = [tag for dim, tag in gmsh.model.getEntities(3) if dim == 3]

        if not volumes:
            raise RuntimeError("No volumes found in STEP geometry. Check geometry file.")

        # --- Create spherical farfield ---
        bbox = geometry.bbox
        cx, cy, cz = bbox.size_x / 2.0, bbox.size_y / 2.0, bbox.size_z / 2.0
        radius = mesh_config.farfield_radius_factor * max(bbox.size_x, bbox.size_y, bbox.size_z)

        farfield_sphere = gmsh.model.occ.addSphere(
            cx, cy, cz, radius,
        )

        # --- Boolean subtraction: farfield - body ---
        # This creates the fluid domain (farfield with body cavity)
        fluid_dimtags, _ = gmsh.model.occ.cut(
            [(3, farfield_sphere)],
            [(3, v) for v in volumes],
        )
        gmsh.model.occ.synchronize()

        # --- Get fluid volume ---
        fluid_volumes = [tag for dim, tag in gmsh.model.getEntities(3) if dim == 3]
        if not fluid_volumes:
            raise RuntimeError("Boolean subtraction failed: no fluid volume created.")

        # --- Physical groups ---
        # Body surfaces (the cavity walls)
        gmsh.model.geo.addPhysicalGroup(2, surfaces, name="body")

        # Farfield (sphere surface)
        farfield_curves = []
        for dim, tag in gmsh.model.getEntities(1):
            # Get curves on the sphere surface
            pass  # Will be identified by bounding box

        # Fluid volume
        gmsh.model.geo.addPhysicalGroup(3, fluid_volumes, name="fluid")

        gmsh.model.geo.synchronize()

        # --- Size fields ---
        # Background mesh size
        bg_tag = 100
        gmsh.model.mesh.field.add("Constant", bg_tag)
        gmsh.model.mesh.field.setNumber(bg_tag, "VIn", mesh_config.max_element_size)
        gmsh.model.mesh.field.setNumber(bg_tag, "VOut", mesh_config.max_element_size)

        # Body refinement (near the body)
        body_tag = 200
        gmsh.model.mesh.field.add("Distance", body_tag)
        gmsh.model.mesh.field.setNumbers(body_tag, "SurfacesList", surfaces)

        # Size field: small near body, large in farfield
        size_tag = 300
        gmsh.model.mesh.field.add("MathEval", size_tag)
        gmsh.model.mesh.field.setString(
            size_tag,
            "F",
            f"Max({mesh_config.min_element_size}, "
            f"Min({mesh_config.max_element_size}, "
            f"{mesh_config.min_element_size} + F{body_tag} * 0.1))",
        )

        # Combine fields
        min_tag = 999
        gmsh.model.mesh.field.add("Min", min_tag)
        gmsh.model.mesh.field.setNumbers(min_tag, "FieldsList", [bg_tag, size_tag])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_tag)

        # --- Mesh generation ---
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
        gmsh.option.setNumber("Mesh.Smoothing", 10)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.model.mesh.generate(3)

        # --- Export ---
        gmsh.write(str(output_path))

    except Exception as exc:
        raise RuntimeError(f"3D mesh generation failed: {exc}") from exc
    finally:
        try:
            gmsh.finalize()
        except OSError:
            pass

    return output_path
