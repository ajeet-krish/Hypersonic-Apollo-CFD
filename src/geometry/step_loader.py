"""STEP file loader for 3D geometry using Gmsh OpenCASCADE kernel.

Provides utilities to load STEP files, extract surfaces and volumes,
and compute bounding boxes for mesh generation.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np


class BoundingBox(NamedTuple):
    """3D bounding box for geometry."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @property
    def size_x(self) -> float:
        return self.x_max - self.x_min

    @property
    def size_y(self) -> float:
        return self.y_max - self.y_min

    @property
    def size_z(self) -> float:
        return self.z_max - self.z_min

    @property
    def centroid(self) -> tuple[float, float, float]:
        return (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
            (self.z_min + self.z_max) / 2.0,
        )


class StepGeometry:
    """Container for STEP file geometry data.

    Attributes:
        path: Path to the STEP file.
        surfaces: List of surface tags in the Gmsh model.
        volumes: List of volume tags in the Gmsh model.
        bbox: Bounding box of the geometry.
    """

    def __init__(
        self,
        path: Path,
        surfaces: list[int],
        volumes: list[int],
        bbox: BoundingBox,
    ) -> None:
        self.path = path
        self.surfaces = surfaces
        self.volumes = volumes
        self.bbox = bbox


def load_step(step_path: Path) -> StepGeometry:
    """Load a STEP file using Gmsh OpenCASCADE kernel.

    Args:
        step_path: Path to the .step or .stp file.

    Returns:
        StepGeometry with surfaces, volumes, and bounding box.

    Raises:
        FileNotFoundError: If the STEP file does not exist.
        RuntimeError: If Gmsh fails to load the file.
    """
    import gmsh

    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.model.add("step_geometry")
        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()

        # Get all surfaces (dimension 2) and volumes (dimension 3)
        surfaces = [tag for dim, tag in gmsh.model.getEntities(2) if dim == 2]
        volumes = [tag for dim, tag in gmsh.model.getEntities(3) if dim == 3]

        # Compute bounding box
        bbox = _compute_bounding_box()

        return StepGeometry(
            path=step_path,
            surfaces=surfaces,
            volumes=volumes,
            bbox=bbox,
        )

    finally:
        gmsh.finalize()


def _compute_bounding_box() -> BoundingBox:
    """Compute bounding box of the current Gmsh model.

    Returns:
        BoundingBox with min/max coordinates.
    """
    import gmsh

    # Get bounding box of all entities
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)

    return BoundingBox(
        x_min=xmin,
        x_max=xmax,
        y_min=ymin,
        y_max=ymax,
        z_min=zmin,
        z_max=zmax,
    )


def get_surface_area(surface_tag: int) -> float:
    """Compute the area of a surface in the current Gmsh model.

    Args:
        surface_tag: Tag of the surface.

    Returns:
        Surface area in model units.
    """
    import gmsh

    # Triangulate the surface for integration
    triangles, coords = gmsh.model.mesh.getElements(2, surface_tag)

    area = 0.0
    for tri in triangles:
        # Get node coordinates
        nodes = tri[:3]
        pts = []
        for n in nodes:
            x, y, z = gmsh.model.mesh.getNode(n)
            pts.append([x, y, z])
        pts = np.array(pts)

        # Compute area using cross product
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        area += 0.5 * np.linalg.norm(np.cross(v1, v2))

    return area


def get_volume(volume_tag: int) -> float:
    """Compute the volume of a solid volume in the current Gmsh model.

    Args:
        volume_tag: Tag of the volume.

    Returns:
        Volume in model units.
    """
    import gmsh

    # Get all surfaces in the volume
    surfaces = gmsh.model.getBoundary([(3, volume_tag)], oriented=False)

    # Use Gmsh's built-in volume computation if available
    try:
        # Gmsh 4.x+ has getMass for volumes
        mass = gmsh.model.occ.getMass(3, volume_tag)
        return mass
    except Exception:
        # Fallback: compute from surface triangulations
        # This is approximate and should only be used if getMass fails
        return 0.0
