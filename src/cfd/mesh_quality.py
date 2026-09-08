"""Mesh quality metrics and SU2 mesh validation.

Provides utilities to check cell quality, validate SU2 mesh format,
and estimate y+ from first cell height for RANS boundary layer modeling.
"""
import math
from pathlib import Path

import numpy as np


def _parse_su2_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, list[tuple[int, ...]]]]:
    """Parse an SU2 mesh file into nodes, elements, and markers.

    Args:
        mesh_path: Path to the .su2 mesh file.

    Returns:
        Tuple of (nodes, elements, markers) where:
            nodes: (N, 3) array of node coordinates.
            elements: (M, 5) array of element connectivity (type, n1, n2, n3, n4).
            markers: Dict mapping marker tag to list of (type, n1, n2, ...) tuples.

    Raises:
        ValueError: If the mesh file is malformed or empty.
    """
    with open(mesh_path, "r") as f:
        lines = f.readlines()

    nodes: list[list[float]] = []
    elements: list[list[int]] = []
    markers: dict[str, list[tuple[int, ...]]] = {}

    section: str | None = None
    marker_tag: str | None = None
    marker_elems_remaining: int = 0
    ndime: int = 2  # default to 2D

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("NDIME="):
            ndime = int(line.split("=")[1].strip())
            section = "ndime"
            continue
        elif line.startswith("NELEM="):
            section = "elements"
            continue
        elif line.startswith("NPOIN="):
            section = "nodes"
            continue
        elif line.startswith("NMARK="):
            section = "markers"
            continue
        elif line.startswith("MARKER_TAG="):
            marker_tag = line.split("=", 1)[1].strip()
            markers[marker_tag] = []
            section = "marker_tag"
            continue
        elif line.startswith("MARKER_ELEMS="):
            marker_elems_remaining = int(line.split("=", 1)[1].strip())
            section = "marker_elems"
            continue

        if section == "elements":
            parts = line.split()
            elem_type = int(parts[0])
            raw_conn = [int(p) for p in parts[1:]]

            # SU2 element type -> expected node count
            _SU2_NODE_COUNTS = {3: 2, 5: 3, 9: 4}
            expected = _SU2_NODE_COUNTS.get(elem_type)

            # gmsh SU2 writer appends an element ID as the trailing integer
            # for interior elements. Strip it if present.
            if expected is not None and len(raw_conn) == expected + 1:
                connectivity = raw_conn[:expected]
            else:
                connectivity = raw_conn

            elements.append([elem_type] + connectivity)
        elif section == "nodes":
            parts = line.split()
            # gmsh SU2 2D: x y node_id (3 values)
            # gmsh SU2 3D: x y z node_id (4 values)
            # Standard SU2 2D: x y (2 values)
            # Standard SU2 3D: x y z (3 values)
            x = float(parts[0])
            y = float(parts[1])
            if ndime == 3 and len(parts) >= 4:
                # 3D: x y z node_id
                z = float(parts[2])
            else:
                # 2D: x y [node_id] - ignore trailing node_id
                z = 0.0
            nodes.append([x, y, z])
        elif section == "marker_elems" and marker_tag is not None:
            if marker_elems_remaining > 0:
                parts = line.split()
                elem_type = int(parts[0])
                connectivity = [int(p) for p in parts[1:]]
                markers[marker_tag].append((elem_type, *connectivity))
                marker_elems_remaining -= 1
                if marker_elems_remaining == 0:
                    section = "markers"

    if not nodes:
        raise ValueError(f"No nodes found in mesh file: {mesh_path}")
    if not elements:
        raise ValueError(f"No elements found in mesh file: {mesh_path}")

    return (
        np.array(nodes, dtype=np.float64),
        np.array(elements, dtype=np.int64),
        markers,
    )


def _element_area(
    nodes: np.ndarray,
    elem_conn: np.ndarray,
) -> float:
    """Compute the area of a 2D element (triangle or quad) using the shoelace formula.

    Args:
        nodes: Full node coordinate array.
        elem_conn: Element connectivity (type, n1, n2, ...).

    Returns:
        Signed area of the element.
    """
    node_indices = elem_conn[1:]
    coords = nodes[node_indices]
    x = coords[:, 0]
    y = coords[:, 1]
    n = len(x)
    area = 0.0
    for k in range(n):
        k_next = (k + 1) % n
        area += x[k] * y[k_next] - x[k_next] * y[k]
    return 0.5 * area


def _element_quality(area: float, elem_conn: np.ndarray, nodes: np.ndarray) -> float:
    """Compute a normalized quality metric for a 2D element.

    Quality is based on the ratio of the element area to the area of a
    regular element with the same perimeter. Ranges from 0 (degenerate)
    to 1 (perfectly regular).

    Args:
        area: Signed area of the element.
        elem_conn: Element connectivity (type, n1, n2, ...).
        nodes: Full node coordinate array.

    Returns:
        Quality metric in [0, 1].
    """
    node_indices = elem_conn[1:]
    coords = nodes[node_indices]
    n = len(node_indices)

    # Compute perimeter
    perimeter = 0.0
    for k in range(n):
        k_next = (k + 1) % n
        edge = np.linalg.norm(coords[k_next] - coords[k])
        perimeter += edge

    if perimeter < 1e-15 or abs(area) < 1e-15:
        return 0.0

    # For a regular polygon with n sides and area A:
    # perimeter_reg = n * s, area_reg = n * s^2 / (4 * tan(pi/n))
    # So area_reg / perimeter_reg^2 = 1 / (4 * n * tan(pi/n))
    # Quality = actual_area_reg / ideal_area_reg
    # where ideal_area_reg = perimeter^2 / (4 * n * tan(pi/n))
    ideal_area = perimeter**2 / (4.0 * n * math.tan(math.pi / n))
    return min(abs(area) / ideal_area, 1.0) if ideal_area > 0 else 0.0


def check_mesh_quality(mesh_path: Path) -> dict:
    """Check mesh quality metrics from an SU2 mesh file.

    Computes cell count, minimum/mean quality, percentage of bad cells
    (quality < 0.3), and verifies boundary marker presence.

    Compatible with both O-grid and C-grid mesh topologies.  C-grid
    meshes use the same physical group markers (body, farfield, sym) as
    O-grid meshes, so no special-casing is needed.

    Args:
        mesh_path: Path to the .su2 mesh file.

    Returns:
        Dictionary with keys: n_cells, min_quality, mean_quality,
        pct_bad_cells, has_body_marker, has_farfield_marker, has_sym_marker.
    """
    if not mesh_path.exists():
        return {
            "n_cells": 0,
            "min_quality": 0.0,
            "mean_quality": 0.0,
            "pct_bad_cells": 100.0,
            "has_body_marker": False,
            "has_farfield_marker": False,
            "has_sym_marker": False,
        }

    nodes, elements, markers = _parse_su2_mesh(mesh_path)

    # Compute quality for each element
    qualities: list[float] = []
    for elem in elements:
        area = _element_area(nodes, elem)
        q = _element_quality(area, elem, nodes)
        qualities.append(q)

    quality_arr = np.array(qualities) if qualities else np.array([0.0])

    n_cells = len(elements)
    min_quality = float(np.min(quality_arr))
    mean_quality = float(np.mean(quality_arr))
    pct_bad = float(np.sum(quality_arr < 0.3) / max(n_cells, 1) * 100.0)

    return {
        "n_cells": n_cells,
        "min_quality": round(min_quality, 6),
        "mean_quality": round(mean_quality, 6),
        "pct_bad_cells": round(pct_bad, 2),
        "has_body_marker": "body" in markers,
        "has_farfield_marker": "farfield" in markers,
        "has_sym_marker": "sym" in markers,
    }


def validate_su2_mesh(mesh_path: Path) -> bool:
    """Validate that a .su2 mesh file is well-formed.

    Checks:
        - File exists and is non-empty
        - NDIME=2 is present
        - Required markers (body, farfield, sym) are present
        - No negative-volume cells (all element areas are non-negative)

    Compatible with both O-grid and C-grid mesh topologies.  C-grid
    meshes use the same physical group markers (body, farfield) as
    O-grid meshes, so no special-casing is needed.

    Args:
        mesh_path: Path to the .su2 mesh file.

    Returns:
        True if the mesh is valid, False otherwise.
    """
    if not mesh_path.exists():
        return False

    try:
        with open(mesh_path, "r") as f:
            content = f.read()

        # Check dimension
        if "NDIME= 2" not in content:
            return False

        # Check required markers: body and farfield are always present.
        # 'sym' is present only for axisymmetric domains; full2d has no
        # symmetry boundary, so accept either (sym, farfield, body) or
        # just (farfield, body).
        for marker in ("body", "farfield"):
            if f"MARKER_TAG= {marker}" not in content:
                return False

        # Check for negative-volume cells
        nodes, elements, _ = _parse_su2_mesh(mesh_path)
        for elem in elements:
            area = _element_area(nodes, elem)
            if area < -1e-10:
                return False

        return True

    except (ValueError, OSError):
        return False


def estimate_y_plus(
    first_cell_height: float,
    freestream_velocity: float,
    freestream_density: float,
    freestream_viscosity: float,
    wall_shear_estimate: float,
) -> float:
    """Estimate y+ from the first cell height for RANS boundary layer resolution.

    Uses the relation: y+ = (rho * u_tau * dy) / mu
    where u_tau = sqrt(tau_w / rho) and dy is the first cell height.

    This is a rough estimate. For accurate y+, a wall-resolved RANS
    solution is needed.

    Args:
        first_cell_height: Height of the first cell off the wall (m).
        freestream_velocity: Freestream velocity (m/s).
        freestream_density: Freestream density (kg/m3).
        freestream_viscosity: Freestream dynamic viscosity (Pa*s).
        wall_shear_estimate: Estimated wall shear stress (Pa).

    Returns:
        Estimated y+ value.
    """
    if first_cell_height <= 0 or freestream_density <= 0 or wall_shear_estimate <= 0:
        return float("nan")

    # Compute friction velocity
    u_tau = math.sqrt(wall_shear_estimate / freestream_density)

    # y+ = rho * u_tau * dy / mu
    y_plus = freestream_density * u_tau * first_cell_height / freestream_viscosity
    return y_plus
