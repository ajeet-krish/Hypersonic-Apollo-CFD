"""DXF geometry loader for axisymmetric body contours.

Parses DXF files containing 2D cross-sections of axisymmetric bodies and
extracts the upper-half body contour (x, r) for use in CFD mesh generation.

The loader handles:
  - LINE entities (cone edges, base edges)
  - ARC entities (sphere nose, shoulder fillets, base fillets)
  - Automatic filtering of construction/reference lines (vertical, horizontal)
  - Upper-half extraction for axisymmetric bodies
  - Uniform arc-length resampling

Typical DXF structure for a blunt body:
  - Sphere arc centered on the axis (nose heat shield)
  - Toroidal fillet arc blending sphere to cone
  - Conical section (straight line in 2D profile)
  - Base fillet arc at the aft edge
  - Construction lines for dimensions and references
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from .external import ExternalGeometry


def _sample_arc(
    cx: float,
    cy: float,
    radius: float,
    start_angle: float,
    end_angle: float,
    n_samples: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample points along a DXF arc.

    DXF arcs are defined by center, radius, and angles measured
    counterclockwise from the positive X axis (in degrees).

    Args:
        cx: Center X coordinate.
        cy: Center Y coordinate.
        radius: Arc radius.
        start_angle: Start angle in degrees (CCW from +X).
        end_angle: End angle in degrees (CCW from +X).
        n_samples: Number of sample points.

    Returns:
        x: Array of X coordinates.
        y: Array of Y coordinates.
    """
    start_rad = math.radians(start_angle)
    end_rad = math.radians(end_angle)

    # Handle wrap-around (e.g., 300 to 60 means going CCW through 360/0)
    if end_rad < start_rad:
        end_rad += 2.0 * math.pi

    angles = np.linspace(start_rad, end_rad, n_samples)
    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)
    return x, y


def _is_construction_line(entity) -> bool:
    """Check if a LINE entity is a construction/reference line.

    Construction lines are purely vertical or horizontal lines used for
    dimensions, axes, or references.  Body contour lines are diagonal
    (both x and y components differ between endpoints).

    Args:
        entity: A DXF LINE entity.

    Returns:
        True if the line is vertical or horizontal (construction line).
    """
    sx, sy = entity.dxf.start.x, entity.dxf.start.y
    ex, ey = entity.dxf.end.x, entity.dxf.end.y

    tol = 1e-10
    is_vertical = abs(sx - ex) < tol
    is_horizontal = abs(sy - ey) < tol

    return is_vertical or is_horizontal


def _deduplicate_contour(
    x: np.ndarray,
    y: np.ndarray,
    x_tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove near-duplicate x values, keeping the point with maximum y.

    When multiple entities contribute points at nearly the same x coordinate
    (e.g., at junction points between arcs and lines), this function keeps
    only the outermost point (maximum y), which corresponds to the body
    surface.

    Args:
        x: X coordinates (assumed sorted).
        y: Y coordinates.
        x_tol: Tolerance for considering x values as duplicates.

    Returns:
        x_unique: Deduplicated X coordinates.
        y_unique: Deduplicated Y coordinates.
    """
    if len(x) == 0:
        return x, y

    unique_x: list[float] = []
    unique_y: list[float] = []

    i = 0
    while i < len(x):
        j = i
        while j < len(x) and abs(x[j] - x[i]) < x_tol:
            j += 1
        # Keep the point with maximum y in this x-bin
        best = np.argmax(y[i:j])
        unique_x.append(float(x[i + best]))
        unique_y.append(float(y[i + best]))
        i = j

    return np.array(unique_x), np.array(unique_y)


def load_dxf_geometry(
    dxf_path: Path | str,
    num_points: int = 600,
) -> ExternalGeometry:
    """Load axisymmetric body contour from a DXF file.

    Extracts LINE and ARC entities from the DXF model space, filters out
    construction/reference lines, samples arcs densely, and builds a
    continuous upper-half body contour sorted by axial coordinate.

    The algorithm:
      1. Parse all LINE and ARC entities from the DXF
      2. Filter out vertical/horizontal construction lines
      3. For ARC entities: sample densely and keep points with y >= 0
      4. For LINE entities: keep diagonal lines (body contour segments)
      5. Concatenate all points, sort by x, deduplicate (keep max y)
      6. Resample to uniform arc-length spacing

    Args:
        dxf_path: Path to the DXF file.
        num_points: Number of contour points in the output.

    Returns:
        ExternalGeometry with body contour coordinates (x, r).

    Raises:
        FileNotFoundError: If the DXF file does not exist.
        ValueError: If no body contour entities are found.
    """
    dxf_path = Path(dxf_path)
    if not dxf_path.exists():
        raise FileNotFoundError(f"DXF file not found: {dxf_path}")

    import ezdxf

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    # Collect sample points from all body contour entities
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for entity in msp:
        etype = entity.dxftype()

        if etype == "ARC":
            cx = entity.dxf.center.x
            cy = entity.dxf.center.y
            radius = entity.dxf.radius
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle

            # Sample the arc densely
            x_arc, y_arc = _sample_arc(cx, cy, radius, start_angle, end_angle)

            # Keep only upper half (y >= 0) for axisymmetric contour
            mask = y_arc >= -1e-6
            if mask.any():
                all_x.append(x_arc[mask])
                all_y.append(np.maximum(y_arc[mask], 0.0))

        elif etype == "LINE":
            # Skip construction lines (vertical/horizontal reference lines)
            if _is_construction_line(entity):
                continue

            # Keep diagonal lines (body contour segments like cone edges)
            sx = entity.dxf.start.x
            sy = entity.dxf.start.y
            ex = entity.dxf.end.x
            ey = entity.dxf.end.y

            # Only keep if at least partially in upper half
            if sy >= -1e-6 or ey >= -1e-6:
                # Sample the line at intermediate points for better resolution
                line_len = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
                n_line_samples = max(int(line_len / 0.01), 2)
                x_line = np.linspace(sx, ex, n_line_samples)
                y_line = np.linspace(sy, ey, n_line_samples)

                # Clip to upper half
                mask = y_line >= -1e-6
                if mask.any():
                    all_x.append(x_line[mask])
                    all_y.append(np.maximum(y_line[mask], 0.0))

    if not all_x:
        raise ValueError(f"No body contour entities found in {dxf_path}")

    # Concatenate all points
    x = np.concatenate(all_x)
    y = np.concatenate(all_y)

    # Ensure y >= 0
    y = np.maximum(y, 0.0)

    # Sort by x coordinate
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Remove near-duplicate x values (keep outermost point)
    x, y = _deduplicate_contour(x, y)

    # Ensure the contour starts at the nose tip (x=0, r=0).
    # DXF arc sampling may not hit exactly 180 degrees, leaving the
    # first point slightly off the axis.  Insert the exact nose tip
    # if needed so the body profile begins on the symmetry axis.
    if len(x) > 0 and x[0] > 1e-6:
        x = np.concatenate([[0.0], x])
        y = np.concatenate([[0.0], y])
    elif len(x) > 0 and y[0] > 1e-6:
        y[0] = 0.0

    # Resample to uniform arc-length spacing
    if len(x) > 1:
        ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        total_length = s[-1]

        if total_length > 1e-12:
            f_x = interp1d(s, x, kind="linear")
            f_y = interp1d(s, y, kind="linear")

            s_uniform = np.linspace(0.0, total_length, num_points)
            x_out = f_x(s_uniform)
            y_out = f_y(s_uniform)
        else:
            x_out = x
            y_out = y
    else:
        x_out = x
        y_out = y

    # Compute derived quantities
    body_length = float(x_out[-1] - x_out[0])
    max_radius = float(y_out.max())
    base_radius = float(y_out[-1])

    return ExternalGeometry(
        x=x_out,
        r=y_out,
        source_file=dxf_path,
        format="dxf",
        n_points=len(x_out),
        body_length=body_length,
        max_radius=max_radius,
        base_radius=base_radius,
    )


def extract_dxf_dimensions(dxf_path: Path | str) -> dict[str, float]:
    """Extract geometric dimensions from a DXF file.

    Parses the DXF to identify key geometric parameters of the axisymmetric
    body: sphere radius, fillet radii, cone half-angle, body length, and
    maximum radius.

    Args:
        dxf_path: Path to the DXF file.

    Returns:
        Dictionary of dimension names to values (meters / degrees):
          - body_length: Total axial length (m)
          - max_radius: Maximum body radius (m)
          - base_radius: Base/aft radius (m)
          - R_sphere: Nose sphere radius (m)
          - cone_half_angle: Cone half-angle (degrees)
          - R_fillet: Shoulder fillet radius (m)
          - R_base_fillet: Base fillet radius (m)
    """
    dxf_path = Path(dxf_path)

    import ezdxf

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    arcs: list[dict] = []
    lines: list[dict] = []

    for entity in msp:
        etype = entity.dxftype()

        if etype == "ARC":
            arcs.append(
                {
                    "cx": entity.dxf.center.x,
                    "cy": entity.dxf.center.y,
                    "radius": entity.dxf.radius,
                    "start_angle": entity.dxf.start_angle,
                    "end_angle": entity.dxf.end_angle,
                }
            )
        elif etype == "LINE":
            lines.append(
                {
                    "sx": entity.dxf.start.x,
                    "sy": entity.dxf.start.y,
                    "ex": entity.dxf.end.x,
                    "ey": entity.dxf.end.y,
                }
            )

    # Get contour for overall dimensions
    geom = load_dxf_geometry(dxf_path, num_points=1000)

    dims: dict[str, float] = {
        "body_length": geom.body_length,
        "max_radius": geom.max_radius,
        "base_radius": geom.base_radius,
    }

    # Identify sphere radius (largest arc centered on axis)
    max_arc_radius = 0.0
    for arc in arcs:
        if abs(arc["cy"]) < 1e-6 and arc["radius"] > max_arc_radius:
            max_arc_radius = arc["radius"]
    dims["R_sphere"] = max_arc_radius

    # Identify fillet radii (smaller arcs)
    fillet_arcs = [
        a for a in arcs if abs(a["cy"]) > 1e-6 and a["radius"] < max_arc_radius
    ]
    if fillet_arcs:
        # The shoulder fillet is closer to the nose (smaller cx)
        # The base fillet is closer to the base (larger cx)
        fillet_arcs.sort(key=lambda a: a["cx"])
        dims["R_fillet"] = fillet_arcs[0]["radius"]
        dims["R_base_fillet"] = fillet_arcs[-1]["radius"]
    else:
        dims["R_fillet"] = 0.0
        dims["R_base_fillet"] = 0.0

    # Compute cone half-angle from diagonal LINE entities
    for line in lines:
        sx, sy = line["sx"], line["sy"]
        ex, ey = line["ex"], line["ey"]

        # Skip construction lines
        if abs(sx - ex) < 1e-10 or abs(sy - ey) < 1e-10:
            continue

        # Take the upper-half diagonal line (cone edge)
        if sy > 0 and ey > 0:
            dx = ex - sx
            dy = ey - sy
            angle = math.degrees(abs(math.atan2(abs(dy), abs(dx))))
            # Cone half-angle is measured from the axis
            # A line going from (x1, r1) to (x2, r2) with r decreasing
            # has half-angle = atan(|dr|/dx)
            dims["cone_half_angle"] = angle
            break

    return dims
