"""Mesh visualization for hypersonic blunt body CFD.

Renders a coarse view of the 2D mesh using matplotlib triplot.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation

from .style import COLORS, apply_theme


def _parse_su2_for_plot(
    mesh_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse SU2 mesh file for plotting (points and triangular elements).

    For mixed tri/quad meshes, quads are split into two triangles.

    Args:
        mesh_path: Path to .su2 mesh file.

    Returns:
        Tuple of (points, triangles) arrays for matplotlib triplot.
    """
    points: list[list[float]] = []
    triangles: list[list[int]] = []

    with open(mesh_path, "r") as f:
        lines = f.readlines()

    section: str | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("NDIME="):
            continue
        elif line.startswith("NELEM="):
            section = "elements"
            continue
        elif line.startswith("NPOIN="):
            section = "nodes"
            continue
        elif line.startswith(("NMARK=", "MARKER_")):
            section = "markers"
            continue

        if section == "elements":
            parts = line.split()
            elem_type = int(parts[0])
            raw_ids = [int(p) for p in parts[1:]]

            # gmsh SU2: type 5 = triangle (3 nodes), type 9 = quad (4 nodes)
            # gmsh appends element ID as trailing integer, so strip it
            _SU2_NODES = {5: 3, 9: 4}
            expected = _SU2_NODES.get(elem_type)
            if expected is not None and len(raw_ids) == expected + 1:
                node_ids = raw_ids[:expected]
            else:
                node_ids = raw_ids

            if elem_type == 5 and len(node_ids) == 3:
                triangles.append(node_ids)
            elif elem_type == 9 and len(node_ids) == 4:
                # Split quad into 2 triangles
                triangles.append([node_ids[0], node_ids[1], node_ids[2]])
                triangles.append([node_ids[0], node_ids[2], node_ids[3]])

        elif section == "nodes":
            parts = line.split()
            x = float(parts[0])
            y = float(parts[1])
            points.append([x, y])

    return np.array(points), np.array(triangles, dtype=np.int32)


def plot_mesh(
    mesh_path: Path,
    output_path: Path,
    max_cells: int = 20000,
) -> Path:
    """Render a coarse view of the 2D mesh using matplotlib triplot.

    Displays a subsampled view of the mesh elements. The body contour
    is overlaid as a thick line.

    Args:
        mesh_path: Path to the .su2 mesh file.
        output_path: Path to save the rendered image.
        max_cells: Maximum number of cells to display (subsample if larger).

    Returns:
        Path to the saved image.
    """
    apply_theme()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points, triangles = _parse_su2_for_plot(mesh_path)

    # Subsample if too many cells
    if len(triangles) > max_cells:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(triangles), max_cells, replace=False)
        triangles_plot = triangles[indices]
    else:
        triangles_plot = triangles

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot mesh elements
    if len(triangles_plot) > 0 and len(points) > 0:
        triang = Triangulation(points[:, 0], points[:, 1], triangles_plot)
        ax.triplot(
            triang,
            color=COLORS["grid"],
            linewidth=0.3,
            alpha=0.6,
        )

    # Overlay body contour (parsed from markers or geometry)
    body_points = _extract_body_boundary_points(mesh_path, points)
    if body_points is not None and len(body_points) > 1:
        ax.plot(
            body_points[:, 0], body_points[:, 1],
            color=COLORS["wall"], linewidth=2.0, label="Body wall",
        )

    # Axis of symmetry
    ax.axhline(
        y=0, color=COLORS["text_dim"], linestyle="--", linewidth=0.8, alpha=0.5,
    )

    # Styling
    ax.set_xlabel("Axial Distance x (m)", fontsize=12, color=COLORS["text"])
    ax.set_ylabel("Radial Distance r (m)", fontsize=12, color=COLORS["text"])
    ax.set_title(
        f"Mesh View ({len(triangles):,} cells)",
        fontsize=14, color=COLORS["text"], fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path


def _extract_body_boundary_points(
    mesh_path: Path,
    all_points: np.ndarray,
) -> np.ndarray | None:
    """Extract body boundary node coordinates from SU2 markers.

    Args:
        mesh_path: Path to the .su2 mesh file.
        all_points: All node coordinates from the mesh.

    Returns:
        (N, 2) array of body boundary points sorted by x, or None.
    """
    with open(mesh_path, "r") as f:
        lines = f.readlines()

    body_node_ids: set[int] = set()
    in_body = False

    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("MARKER_TAG= body"):
            in_body = True
            continue
        elif line.startswith("MARKER_TAG="):
            in_body = False
            continue
        elif line.startswith("MARKER_ELEMS="):
            continue
        elif line.startswith(("NMARK=", "NDIME=", "NELEM=", "NPOIN=")):
            in_body = False
            continue

        if in_body:
            parts = line.split()
            if len(parts) >= 3:
                # Element type + node IDs
                node_ids = [int(p) for p in parts[1:]]
                for nid in node_ids:
                    body_node_ids.add(nid)

    if not body_node_ids:
        return None

    body_pts = all_points[sorted(body_node_ids)]
    # Sort by x coordinate
    order = np.argsort(body_pts[:, 0])
    return body_pts[order]
