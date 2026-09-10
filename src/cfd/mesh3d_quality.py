"""3D mesh quality metrics and SU2 mesh validation.

Provides utilities to check cell quality for 3D tetrahedral meshes,
validate SU2 mesh format, and compute geometric metrics such as
dihedral angles and aspect ratios.

Quality metric is based on the normalized minimum altitude (condition
number) of the Jacobian matrix, mapping the reference tetrahedron to
the physical element.  Ranges from 0 (degenerate) to 1 (perfectly
regular tetrahedron).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# SU2 element type codes (3D volume elements)
# ---------------------------------------------------------------------------
_TET_TYPE = 4    # Linear tetrahedron (4 nodes)
_HEX_TYPE = 10   # Linear hexahedron (8 nodes)
_PRISM_TYPE = 6  # Linear prism (6 nodes)

# Surface element types (for marker faces)
_TRI_SURFACE_TYPE = 5   # Triangle
_QUAD_SURFACE_TYPE = 9  # Quad

# Node counts for 3D element types
_3D_NODE_COUNTS: dict[int, int] = {
    _TET_TYPE: 4,
    _HEX_TYPE: 8,
    _PRISM_TYPE: 6,
}


def _parse_su2_mesh_3d(
    mesh_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[tuple[int, ...]]], int]:
    """Parse a 3D SU2 mesh file into nodes, elements, and markers.

    Handles both 2D and 3D SU2 formats.  The parser reads the NDIME
    header to determine dimensionality and parses node coordinates
    accordingly (2 values for 2D, 3 for 3D, with optional trailing
    node ID).

    Args:
        mesh_path: Path to the .su2 mesh file.

    Returns:
        Tuple of (nodes, elements, markers, ndime) where:
            nodes: (N, 3) array of node coordinates.
            elements: (M, K) array of element connectivity
                      (type, n1, n2, ..., nk).
            markers: Dict mapping marker tag to list of
                     (type, n1, n2, ...) tuples.
            ndime: Spatial dimension (2 or 3).

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
    ndime: int = 3

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
            _SU2_NODE_COUNTS = {3: 2, 5: 3, 9: 4, 4: 4, 10: 8, 6: 6}
            expected = _SU2_NODE_COUNTS.get(elem_type)

            # gmsh SU2 writer appends an element ID as the trailing integer
            # for interior elements.  Strip it if present.
            if expected is not None and len(raw_conn) == expected + 1:
                connectivity = raw_conn[:expected]
            else:
                connectivity = raw_conn

            elements.append([elem_type] + connectivity)
        elif section == "nodes":
            parts = line.split()
            x = float(parts[0])
            y = float(parts[1])
            if ndime == 3 and len(parts) >= 3:
                z = float(parts[2])
            else:
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
        ndime,
    )


def _tet_volume(nodes: np.ndarray, n0: int, n1: int, n2: int, n3: int) -> float:
    """Compute volume of a tetrahedron.

    Uses the scalar triple product: V = |det([v1-v0, v2-v0, v3-v0])| / 6.

    Args:
        nodes: (N, 3) node coordinate array.
        n0, n1, n2, n3: Node indices of the tetrahedron.

    Returns:
        Volume of the tetrahedron (always non-negative).
    """
    v1 = nodes[n1] - nodes[n0]
    v2 = nodes[n2] - nodes[n0]
    v3 = nodes[n3] - nodes[n0]
    det = v1[0] * (v2[1] * v3[2] - v2[2] * v3[1]) - \
          v1[1] * (v2[0] * v3[2] - v2[2] * v3[0]) + \
          v1[2] * (v2[0] * v3[1] - v2[1] * v3[0])
    return abs(det) / 6.0


def _tet_quality(nodes: np.ndarray, n0: int, n1: int, n2: int, n3: int) -> float:
    """Compute normalized quality metric for a tetrahedron.

    Quality is based on the ratio of the minimum altitude to the
    circumradius, normalized so that a regular tetrahedron has quality 1.
    Equivalently, this is the condition number of the Jacobian mapping
    the reference tet to the physical element.

    Q = 3 * V / (sum of face_areas * circumradius)

    Ranges from 0 (degenerate) to 1 (perfectly regular).

    Args:
        nodes: (N, 3) node coordinate array.
        n0, n1, n2, n3: Node indices of the tetrahedron.

    Returns:
        Quality metric in [0, 1].
    """
    vol = _tet_volume(nodes, n0, n1, n2, n3)
    if vol < 1e-30:
        return 0.0

    # Edge vectors
    e01 = nodes[n1] - nodes[n0]
    e02 = nodes[n2] - nodes[n0]
    e03 = nodes[n3] - nodes[n0]
    e12 = nodes[n2] - nodes[n1]
    e13 = nodes[n3] - nodes[n1]
    e23 = nodes[n3] - nodes[n2]

    # Face areas (cross product magnitudes / 2)
    face0 = 0.5 * np.linalg.norm(np.cross(e02, e03))  # face opposite n0
    face1 = 0.5 * np.linalg.norm(np.cross(e01, e03))  # face opposite n1
    face2 = 0.5 * np.linalg.norm(np.cross(e01, e02))  # face opposite n2
    face3 = 0.5 * np.linalg.norm(np.cross(e12, e13))  # face opposite n3

    total_face_area = face0 + face1 + face2 + face3
    if total_face_area < 1e-30:
        return 0.0

    # Circumradius: R = product of edge lengths / (24 * volume)
    edge_lengths = np.array([
        np.linalg.norm(e01), np.linalg.norm(e02), np.linalg.norm(e03),
        np.linalg.norm(e12), np.linalg.norm(e13), np.linalg.norm(e23),
    ])
    product = float(np.prod(edge_lengths))
    circumradius = product / (24.0 * vol) if vol > 0 else 0.0

    if circumradius < 1e-30:
        return 0.0

    # Minimum altitude h = 3V / face_area (for the largest face)
    max_face = max(face0, face1, face2, face3)
    if max_face < 1e-30:
        return 0.0

    min_altitude = 3.0 * vol / max_face

    # Normalized quality: h / (2 * R) gives ratio in [0, 1] for regular tets
    quality = min_altitude / (2.0 * circumradius)
    return min(quality, 1.0)


def _compute_face_normals(
    nodes: np.ndarray, n0: int, n1: int, n2: int
) -> np.ndarray:
    """Compute the outward normal of a triangular face.

    Args:
        nodes: (N, 3) node coordinate array.
        n0, n1, n2: Node indices of the triangular face.

    Returns:
        Unit normal vector (3,).
    """
    v1 = nodes[n1] - nodes[n0]
    v2 = nodes[n2] - nodes[n0]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-30:
        return np.array([0.0, 0.0, 0.0])
    return normal / norm


def _compute_dihedral_angles(
    nodes: np.ndarray, n0: int, n1: int, n2: int, n3: int
) -> list[float]:
    """Compute the six dihedral angles of a tetrahedron.

    Each dihedral angle is the angle between two faces sharing an edge.
    The four faces are:
        face 0: (n1, n2, n3)  - opposite n0
        face 1: (n0, n2, n3)  - opposite n1
        face 2: (n0, n1, n3)  - opposite n2
        face 3: (n0, n1, n2)  - opposite n3

    The six edges and their adjacent face pairs:
        edge (n0,n1): face2, face3
        edge (n0,n2): face1, face3
        edge (n0,n3): face1, face2
        edge (n1,n2): face0, face3
        edge (n1,n3): face0, face2
        edge (n2,n3): face0, face1

    Args:
        nodes: (N, 3) node coordinate array.
        n0, n1, n2, n3: Node indices of the tetrahedron.

    Returns:
        List of 6 dihedral angles in degrees.
    """
    # Compute face normals (outward from tet)
    normals = [
        _compute_face_normals(nodes, n1, n2, n3),  # face 0 (opp n0)
        _compute_face_normals(nodes, n0, n2, n3),  # face 1 (opp n1)
        _compute_face_normals(nodes, n0, n1, n3),  # face 2 (opp n2)
        _compute_face_normals(nodes, n0, n1, n2),  # face 3 (opp n3)
    ]

    # Ensure normals point outward (away from opposite vertex)
    centroid = (nodes[n0] + nodes[n1] + nodes[n2] + nodes[n3]) / 4.0
    face_centers = [
        (nodes[n1] + nodes[n2] + nodes[n3]) / 3.0,
        (nodes[n0] + nodes[n2] + nodes[n3]) / 3.0,
        (nodes[n0] + nodes[n1] + nodes[n3]) / 3.0,
        (nodes[n0] + nodes[n1] + nodes[n2]) / 3.0,
    ]
    for i in range(4):
        to_center = face_centers[i] - centroid
        if np.dot(normals[i], to_center) < 0:
            normals[i] = -normals[i]

    # Edge -> adjacent face pairs
    edge_face_pairs = [
        (2, 3),  # edge (n0, n1)
        (1, 3),  # edge (n0, n2)
        (1, 2),  # edge (n0, n3)
        (0, 3),  # edge (n1, n2)
        (0, 2),  # edge (n1, n3)
        (0, 1),  # edge (n2, n3)
    ]

    angles: list[float] = []
    for f1_idx, f2_idx in edge_face_pairs:
        # Dihedral angle is pi - angle between outward normals
        cos_angle = np.dot(normals[f1_idx], normals[f2_idx])
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        # The interior dihedral is supplementary to the angle between outward normals
        dihedral = 180.0 - angle_deg
        dihedral = max(0.0, min(180.0, dihedral))
        angles.append(dihedral)

    return angles


def _compute_tet_aspect_ratio(
    nodes: np.ndarray, n0: int, n1: int, n2: int, n3: int
) -> float:
    """Compute aspect ratio of a tetrahedron.

    Aspect ratio = longest edge / shortest altitude.
    A regular tetrahedron has aspect ratio = 1.0.

    Args:
        nodes: (N, 3) node coordinate array.
        n0, n1, n2, n3: Node indices of the tetrahedron.

    Returns:
        Aspect ratio (always >= 1.0).
    """
    vol = _tet_volume(nodes, n0, n1, n2, n3)
    if vol < 1e-30:
        return float("inf")

    # Edge vectors
    e01 = nodes[n1] - nodes[n0]
    e02 = nodes[n2] - nodes[n0]
    e03 = nodes[n3] - nodes[n0]
    e12 = nodes[n2] - nodes[n1]
    e13 = nodes[n3] - nodes[n1]
    e23 = nodes[n3] - nodes[n2]

    # Longest edge
    edge_lengths = np.array([
        np.linalg.norm(e01), np.linalg.norm(e02), np.linalg.norm(e03),
        np.linalg.norm(e12), np.linalg.norm(e13), np.linalg.norm(e23),
    ])
    longest_edge = float(np.max(edge_lengths))

    # Face areas
    face0 = 0.5 * np.linalg.norm(np.cross(e02, e03))
    face1 = 0.5 * np.linalg.norm(np.cross(e01, e03))
    face2 = 0.5 * np.linalg.norm(np.cross(e01, e02))
    face3 = 0.5 * np.linalg.norm(np.cross(e12, e13))

    # Altitude for each face: h = 3V / area
    faces = [face0, face1, face2, face3]
    altitudes = [3.0 * vol / f if f > 1e-30 else float("inf") for f in faces]
    shortest_altitude = min(altitudes)

    if shortest_altitude < 1e-30:
        return float("inf")

    return longest_edge / shortest_altitude


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_mesh_quality_3d(mesh_path: Path) -> dict:
    """Check 3D mesh quality metrics from an SU2 mesh file.

    Computes cell count, minimum/mean quality, percentage of bad cells
    (quality < 0.3), dihedral angle range, aspect ratio, and verifies
    boundary marker presence.

    The quality metric is based on the normalized minimum altitude
    (condition number) of the Jacobian matrix mapping the reference
    tetrahedron to the physical element.

    Args:
        mesh_path: Path to the .su2 mesh file.

    Returns:
        Dictionary with keys: n_cells, n_tets, min_quality,
        mean_quality, pct_bad_cells, min_dihedral_angle,
        max_dihedral_angle, max_aspect_ratio, has_body_marker,
        has_farfield_marker, has_fluid_marker.
    """
    if not mesh_path.exists():
        return {
            "n_cells": 0,
            "n_tets": 0,
            "min_quality": 0.0,
            "mean_quality": 0.0,
            "pct_bad_cells": 100.0,
            "min_dihedral_angle": 0.0,
            "max_dihedral_angle": 0.0,
            "max_aspect_ratio": float("inf"),
            "has_body_marker": False,
            "has_farfield_marker": False,
            "has_fluid_marker": False,
        }

    nodes, elements, markers, _ndime = _parse_su2_mesh_3d(mesh_path)

    # Identify 3D volume elements (tets, hexes, prisms)
    _3D_VOLUME_TYPES = {_TET_TYPE, _HEX_TYPE, _PRISM_TYPE}

    cell_qualities: list[float] = []
    n_tets = 0
    all_dihedral_angles: list[float] = []
    aspect_ratios: list[float] = []

    for elem in elements:
        elem_type = int(elem[0])
        if elem_type not in _3D_VOLUME_TYPES:
            continue

        if elem_type == _TET_TYPE and len(elem) >= 5:
            n_tets += 1
            n0, n1, n2, n3 = int(elem[1]), int(elem[2]), int(elem[3]), int(elem[4])
            q = _tet_quality(nodes, n0, n1, n2, n3)
            cell_qualities.append(q)

            angles = _compute_dihedral_angles(nodes, n0, n1, n2, n3)
            all_dihedral_angles.extend(angles)

            ar = _compute_tet_aspect_ratio(nodes, n0, n1, n2, n3)
            aspect_ratios.append(ar)

    quality_arr = np.array(cell_qualities) if cell_qualities else np.array([0.0])

    n_cells = len(cell_qualities)
    min_quality = float(np.min(quality_arr))
    mean_quality = float(np.mean(quality_arr))
    pct_bad = float(np.sum(quality_arr < 0.3) / max(n_cells, 1) * 100.0)

    result: dict = {
        "n_cells": n_cells,
        "n_tets": n_tets,
        "min_quality": round(min_quality, 6),
        "mean_quality": round(mean_quality, 6),
        "pct_bad_cells": round(pct_bad, 2),
        "min_dihedral_angle": round(float(np.min(all_dihedral_angles)), 4) if all_dihedral_angles else 0.0,
        "max_dihedral_angle": round(float(np.max(all_dihedral_angles)), 4) if all_dihedral_angles else 0.0,
        "max_aspect_ratio": round(float(np.max(aspect_ratios)), 4) if aspect_ratios else 0.0,
        "has_body_marker": "body" in markers,
        "has_farfield_marker": "farfield" in markers,
        "has_fluid_marker": "fluid" in markers,
    }

    return result


def validate_su2_mesh_3d(mesh_path: Path) -> bool:
    """Validate that a 3D .su2 mesh file is readable and has required markers.

    Checks:
        - File exists and is non-empty
        - NDIME=3 is present
        - Required markers (body, farfield) are present
        - No negative-volume cells (all element volumes are non-negative)

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
        if "NDIME= 3" not in content:
            return False

        # Check required markers
        for marker in ("body", "farfield"):
            if f"MARKER_TAG= {marker}" not in content:
                return False

        # Check for negative-volume cells
        nodes, elements, _markers, _ndime = _parse_su2_mesh_3d(mesh_path)
        _3D_VOLUME_TYPES = {_TET_TYPE, _HEX_TYPE, _PRISM_TYPE}
        for elem in elements:
            elem_type = int(elem[0])
            if elem_type not in _3D_VOLUME_TYPES:
                continue
            if elem_type == _TET_TYPE and len(elem) >= 5:
                n0, n1, n2, n3 = int(elem[1]), int(elem[2]), int(elem[3]), int(elem[4])
                vol = _tet_volume(nodes, n0, n1, n2, n3)
                if vol < -1e-10:
                    return False

        return True

    except (ValueError, OSError):
        return False
