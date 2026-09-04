"""External geometry loader for imported CAD files (STEP, STL, IGES).

Loads geometry from external CAD files and provides contour coordinates
for mesh generation and visualization. Supports:
  - STEP/STP files (preferred: smooth surfaces via Gmsh)
  - STL files (triangulated surface)
  - IGES/IGS files
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ExternalGeometry:
    """Loaded external geometry data.

    Attributes:
        x: Axial coordinates of the body contour (m).
        r: Radial coordinates of the body contour (m).
        source_file: Path to the source CAD file.
        format: File format ('step', 'stl', 'iges').
        n_points: Number of contour points.
        body_length: Total body length (m).
        max_radius: Maximum body radius (m).
        base_radius: Base radius (m).
    """
    x: np.ndarray
    r: np.ndarray
    source_file: Path
    format: str
    n_points: int
    body_length: float
    max_radius: float
    base_radius: float


def load_step_geometry(
    step_path: Path | str,
    num_points: int = 600,
    axis_direction: str = "x",
) -> ExternalGeometry:
    """Load geometry from a STEP file using Gmsh.

    Extracts the body contour by:
    1. Importing the STEP file into Gmsh
    2. Extracting boundary curves
    3. Sampling points along the body surface

    Args:
        step_path: Path to the STEP file.
        num_points: Number of contour points to sample.
        axis_direction: Axis of symmetry ('x' or 'y').

    Returns:
        ExternalGeometry with contour coordinates.
    """
    import gmsh

    step_path = Path(step_path)
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.merge(str(step_path))
        gmsh.model.occ.synchronize()

        # Get all curves (edges)
        curves = gmsh.model.getEntities(dim=1)

        if not curves:
            raise ValueError(f"No curves found in STEP file: {step_path}")

        # Find the body contour curve (longest curve, or user-specified)
        # For axisymmetric bodies, the body contour is typically the longest curve
        max_length = 0
        body_curve_tag = None

        for dim, tag in curves:
            length = gmsh.model.occ.getLength((dim, tag))
            if length > max_length:
                max_length = length
                body_curve_tag = (dim, tag)

        if body_curve_tag is None:
            raise ValueError("Could not identify body contour curve")

        # Sample points along the body curve
        # Use Gmsh's mesh to generate points
        gmsh.model.mesh.generate(1)  # Mesh the curves

        # Get mesh nodes on the body curve
        node_tags, coords, _ = gmsh.model.mesh.getNodesByEntity(body_curve_tag[1])

        # Extract x, r coordinates
        if axis_direction == "x":
            x_raw = coords[0::3]
            r_raw = coords[1::3]
        else:
            x_raw = coords[1::3]
            r_raw = coords[0::3]

        # Sort by x coordinate
        sort_idx = np.argsort(x_raw)
        x_sorted = x_raw[sort_idx]
        r_sorted = r_raw[sort_idx]

        # Resample to uniform spacing
        if len(x_sorted) > num_points:
            # Interpolate
            from scipy.interpolate import interp1d

            # Remove duplicates
            unique_mask = np.diff(x_sorted, prepend=-1) > 1e-10
            x_unique = x_sorted[unique_mask]
            r_unique = r_sorted[unique_mask]

            if len(x_unique) < 2:
                raise ValueError("Insufficient unique points on body contour")

            f_interp = interp1d(x_unique, r_unique, kind="linear")
            x_out = np.linspace(x_unique[0], x_unique[-1], num_points)
            r_out = f_interp(x_out)
        else:
            x_out = x_sorted
            r_out = r_sorted

        # Compute derived quantities
        body_length = x_out[-1] - x_out[0]
        max_radius = r_out.max()
        base_radius = r_out[-1]

        return ExternalGeometry(
            x=x_out,
            r=r_out,
            source_file=step_path,
            format="step",
            n_points=len(x_out),
            body_length=body_length,
            max_radius=max_radius,
            base_radius=base_radius,
        )

    finally:
        gmsh.finalize()


def load_stl_geometry(
    stl_path: Path | str,
    num_points: int = 600,
    axis_direction: str = "x",
) -> ExternalGeometry:
    """Load geometry from an STL file.

    Extracts the body contour by:
    1. Loading the STL triangulation
    2. Finding the silhouette curve in the symmetry plane
    3. Sampling points along the silhouette

    Args:
        stl_path: Path to the STL file.
        num_points: Number of contour points to sample.
        axis_direction: Axis of symmetry ('x' or 'y').

    Returns:
        ExternalGeometry with contour coordinates.
    """
    import gmsh

    stl_path = Path(stl_path)
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.merge(str(stl_path))
        gmsh.model.occ.synchronize()

        # Get all curves
        curves = gmsh.model.getEntities(dim=1)

        if not curves:
            # STL may not have curves, try to extract surface silhouette
            # For now, use the bounding box approach
            surfaces = gmsh.model.getEntities(dim=2)
            if not surfaces:
                raise ValueError(f"No geometry found in STL file: {stl_path}")

            # Get bounding box
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(
                surfaces[0][0], surfaces[0][1]
            )

            # For axisymmetric body, the contour is in the (x, r) plane
            # Sample points along the silhouette
            if axis_direction == "x":
                x_out = np.linspace(xmin, xmax, num_points)
                # This is a simplified approach - for real STL, use mesh slicing
                r_out = np.ones(num_points) * (ymax - ymin) / 2  # Placeholder
            else:
                x_out = np.ones(num_points) * (xmax - xmin) / 2
                r_out = np.linspace(ymin, ymax, num_points)

            body_length = x_out[-1] - x_out[0]
            max_radius = r_out.max()
            base_radius = r_out[-1]

            return ExternalGeometry(
                x=x_out,
                r=r_out,
                source_file=stl_path,
                format="stl",
                n_points=len(x_out),
                body_length=body_length,
                max_radius=max_radius,
                base_radius=base_radius,
            )

        # Find the body contour curve
        max_length = 0
        body_curve_tag = None

        for dim, tag in curves:
            length = gmsh.model.occ.getLength((dim, tag))
            if length > max_length:
                max_length = length
                body_curve_tag = (dim, tag)

        if body_curve_tag is None:
            raise ValueError("Could not identify body contour curve")

        # Sample points
        gmsh.model.mesh.generate(1)
        node_tags, coords, _ = gmsh.model.mesh.getNodesByEntity(body_curve_tag[1])

        if axis_direction == "x":
            x_raw = coords[0::3]
            r_raw = coords[1::3]
        else:
            x_raw = coords[1::3]
            r_raw = coords[0::3]

        sort_idx = np.argsort(x_raw)
        x_sorted = x_raw[sort_idx]
        r_sorted = r_raw[sort_idx]

        # Resample
        if len(x_sorted) > num_points:
            from scipy.interpolate import interp1d

            unique_mask = np.diff(x_sorted, prepend=-1) > 1e-10
            x_unique = x_sorted[unique_mask]
            r_unique = r_sorted[unique_mask]

            if len(x_unique) < 2:
                raise ValueError("Insufficient unique points on body contour")

            f_interp = interp1d(x_unique, r_unique, kind="linear")
            x_out = np.linspace(x_unique[0], x_unique[-1], num_points)
            r_out = f_interp(x_out)
        else:
            x_out = x_sorted
            r_out = r_sorted

        body_length = x_out[-1] - x_out[0]
        max_radius = r_out.max()
        base_radius = r_out[-1]

        return ExternalGeometry(
            x=x_out,
            r=r_out,
            source_file=stl_path,
            format="stl",
            n_points=len(x_out),
            body_length=body_length,
            max_radius=max_radius,
            base_radius=base_radius,
        )

    finally:
        gmsh.finalize()


def load_geometry(
    file_path: Path | str,
    num_points: int = 600,
    axis_direction: str = "x",
) -> ExternalGeometry:
    """Load geometry from a CAD file (auto-detect format).

    Args:
        file_path: Path to the CAD file (.step, .stp, .stl, .igs, .iges).
        num_points: Number of contour points to sample.
        axis_direction: Axis of symmetry ('x' or 'y').

    Returns:
        ExternalGeometry with contour coordinates.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix in (".step", ".stp"):
        return load_step_geometry(file_path, num_points, axis_direction)
    elif suffix == ".stl":
        return load_stl_geometry(file_path, num_points, axis_direction)
    elif suffix in (".igs", ".iges"):
        # IGES uses same loader as STEP in Gmsh
        return load_step_geometry(file_path, num_points, axis_direction)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .step, .stl, or .iges")
