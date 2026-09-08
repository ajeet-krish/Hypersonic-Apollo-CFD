"""Gmsh mesh generation for spherically-blunted cones (O-grid and C-grid topologies).

Produces a 2D axisymmetric mesh in the (x, r) plane with:
  - Body-conforming elliptical (O-grid) farfield boundary
  - Structured boundary-layer cells at the body wall (manual quad layers)
  - Shock-region refinement based on Billig standoff correlation
  - Sphere-cone junction refinement
  - Wake refinement downstream of the body base
  - SU2-compatible physical group markers

The domain is an ellipse centered on the body's axial midpoint, with
semi-axes computed from configurable upstream/downstream/lateral factors.
This replaces the rectangular farfield with a body-conforming boundary
that is 87% smaller than the previous rectangular domain.

The boundary layer is built by explicit node placement: for each body
contour point, additional nodes are placed along the outward normal at
geometrically-spaced distances.  Triangles connect consecutive layers.

When ``aoa != 0`` the domain is forced to full2d.
Angle of attack is handled by SU2 (freestream rotation), not body rotation.

References:
    Billig, F. S. (1967), "Shock-Wave Shapes Around Unswept- and
    Swept-Nose Bodies," J. Spacecraft & Rockets, 4(6), 822-823.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from geometry.blunt_body import generate_contour
from geometry.config import BluntBodyConfig
from validation.billig import billig_blunted_cone

from .mesh_config import MeshConfig, CGridDomain, OGridDomain

if TYPE_CHECKING:
    pass

# Mesh size multipliers per tier (inverse of cell-count multipliers).
# Draft uses 2x larger cells (fewer total), high uses 0.5x (more total).
_TIER_SIZE_MULTIPLIERS: dict[str, float] = {
    "draft": 2.0,
    "standard": 1.0,
    "high": 0.5,
}


def _build_ogrid_domain(
    config: BluntBodyConfig,
    mesh_config: MeshConfig,
) -> OGridDomain:
    """Compute elliptical O-grid domain parameters.

    The ellipse is centered at the axial midpoint of the body, with
    semi-axes determined by upstream/downstream/lateral factors applied
    to the body geometry (R_nose, body_diameter, body_length).

    Args:
        config: Blunt body geometry parameters.
        mesh_config: Mesh configuration with domain factor multipliers.

    Returns:
        OGridDomain with all elliptical boundary parameters.
    """
    R_nose = config.R_nose
    body_length = config.computed_body_length
    body_diameter = 2.0 * config.max_radius

    upstream = mesh_config.upstream_factor * R_nose
    downstream = mesh_config.downstream_factor * body_diameter
    lateral = mesh_config.lateral_factor * R_nose

    semi_major = (upstream + body_length + downstream) / 2.0
    semi_minor = lateral
    center_x = 0.0 + upstream + body_length / 2.0

    return OGridDomain(
        center_x=center_x,
        center_r=0.0,
        semi_major=semi_major,
        semi_minor=semi_minor,
        x_nose=0.0,
        x_base=body_length,
        r_base=config.base_radius,
        x_min=center_x - semi_major,
        x_max=center_x + semi_major,
        r_max=semi_minor,
    )


def _generate_ellipse_points(
    cx: float,
    cr: float,
    a: float,
    b: float,
    n: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate points on an ellipse for Gmsh spline.

    Args:
        cx: Center x-coordinate.
        cr: Center r-coordinate.
        a: Semi-major axis (axial).
        b: Semi-minor axis (radial).
        n: Number of points (default 40).

    Returns:
        (x, r) arrays of ellipse points, ordered counterclockwise
        starting from the rightmost point (theta=0).
    """
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = cx + a * np.cos(theta)
    r = cr + b * np.sin(theta)
    return x, r


def _compute_offset_contour(
    x_body: np.ndarray,
    r_body: np.ndarray,
    bl_thickness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute offset contour for the boundary-layer outer boundary.

    Offsets each body contour point outward along the surface normal
    by *bl_thickness*.  The outward normal is the left-hand normal of
    the body curve (which points away from the symmetry axis for a
    left-to-right oriented contour).

    Args:
        x_body: Axial coordinates of the body contour (m).
        r_body: Radial coordinates of the body contour (m).
        bl_thickness: Total boundary-layer thickness (m).

    Returns:
        (x_offset, r_offset) arrays of the same shape as the inputs.
    """
    n_pts = len(x_body)
    x_off = np.empty(n_pts)
    r_off = np.empty(n_pts)

    for i in range(n_pts):
        if i == 0:
            dx = x_body[1] - x_body[0]
            dr = r_body[1] - r_body[0]
        elif i == n_pts - 1:
            dx = x_body[-1] - x_body[-2]
            dr = r_body[-1] - r_body[-2]
        else:
            dx = x_body[i + 1] - x_body[i - 1]
            dr = r_body[i + 1] - r_body[i - 1]

        mag = np.hypot(dx, dr)
        if mag < 1e-15:
            nx, nr = 0.0, 1.0
        else:
            nx = -dr / mag
            nr = dx / mag

        if nr < 0.0:
            nx, nr = -nx, -nr

        x_off[i] = x_body[i] + nx * bl_thickness
        r_off[i] = r_body[i] + nr * bl_thickness

    return x_off, r_off


def _add_shock_refinement(
    config: BluntBodyConfig,
    mach: float,
    mesh_config: MeshConfig,
    field_tag: int,
    tier_mult: float = 1.0,
) -> None:
    """Add shock-region refinement using Ball field at expected shock location.

    Creates a Ball field centered at the expected bow shock location
    (based on Billig standoff correlation) to concentrate cells near
    the shock.

    Args:
        config: Blunt body geometry parameters.
        mach: Freestream Mach number.
        mesh_config: Mesh configuration.
        field_tag: Starting field tag for the refinement fields.
        tier_mult: Mesh size multiplier for the tier (larger = coarser).
    """
    import gmsh

    standoff = billig_blunted_cone(config.R_nose, mach)
    delta = standoff.delta
    R_nose = config.R_nose

    ball_tag = field_tag
    gmsh.model.mesh.field.add("Ball", ball_tag)
    gmsh.model.mesh.field.setNumber(ball_tag, "XCenter", delta)
    gmsh.model.mesh.field.setNumber(ball_tag, "YCenter", 0.0)
    gmsh.model.mesh.field.setNumber(ball_tag, "ZCenter", 0.0)
    gmsh.model.mesh.field.setNumber(ball_tag, "VIn", 0.1 * R_nose * tier_mult)
    gmsh.model.mesh.field.setNumber(ball_tag, "VOut", 0.3 * R_nose * tier_mult)
    gmsh.model.mesh.field.setNumber(
        ball_tag, "Radius", mesh_config.shock_standoff_factor * delta,
    )


def _add_junction_refinement(
    config: BluntBodyConfig,
    mesh_config: MeshConfig,
    field_tag: int,
    tier_mult: float = 1.0,
) -> None:
    """Add refinement near the sphere-cone junction using a Ball field.

    Args:
        config: Blunt body geometry parameters.
        mesh_config: Mesh configuration.
        field_tag: Field tag for the junction refinement ball field.
        tier_mult: Mesh size multiplier for the tier (larger = coarser).
    """
    import gmsh

    R_nose = config.R_nose

    gmsh.model.mesh.field.add("Ball", field_tag)
    gmsh.model.mesh.field.setNumber(field_tag, "XCenter", config.junction_x)
    gmsh.model.mesh.field.setNumber(field_tag, "YCenter", config.junction_r)
    gmsh.model.mesh.field.setNumber(field_tag, "ZCenter", 0.0)
    gmsh.model.mesh.field.setNumber(field_tag, "VIn", 0.1 * R_nose * tier_mult)
    gmsh.model.mesh.field.setNumber(field_tag, "VOut", 0.3 * R_nose * tier_mult)
    gmsh.model.mesh.field.setNumber(field_tag, "Radius", 1.5 * R_nose)


def _add_wake_refinement(
    config: BluntBodyConfig,
    mesh_config: MeshConfig,
    field_tag: int,
    tier_mult: float = 1.0,
) -> None:
    """Add refinement in the wake region downstream of the body base.

    Creates a Box field covering the region immediately behind the body
    base to capture wake structures (recirculation zone, shear layers).

    Args:
        config: Blunt body geometry parameters.
        mesh_config: Mesh configuration.
        field_tag: Field tag for the wake refinement box field.
        tier_mult: Mesh size multiplier for the tier (larger = coarser).
    """
    import gmsh

    body_length = config.computed_body_length
    R_nose = config.R_nose

    gmsh.model.mesh.field.add("Box", field_tag)
    gmsh.model.mesh.field.setNumber(field_tag, "XMin", body_length)
    gmsh.model.mesh.field.setNumber(field_tag, "XMax", body_length + 10.0 * R_nose)
    gmsh.model.mesh.field.setNumber(field_tag, "YMin", -config.max_radius * 2.0)
    gmsh.model.mesh.field.setNumber(field_tag, "YMax", config.max_radius * 2.0)
    gmsh.model.mesh.field.setNumber(field_tag, "VIn", 0.1 * R_nose * tier_mult)
    gmsh.model.mesh.field.setNumber(field_tag, "VOut", 0.3 * R_nose * tier_mult)


def _build_cgrid_domain(
    config: BluntBodyConfig,
    mesh_config: MeshConfig,
) -> CGridDomain:
    """Compute C-grid domain parameters from body geometry.

    The C-grid domain wraps the blunt body with an upper boundary
    that is an elliptical arc, a nose-cap closure upstream, and
    extends to an outflow boundary downstream.

    Args:
        config: Blunt body geometry parameters.
        mesh_config: Mesh configuration with domain factor multipliers.

    Returns:
        CGridDomain with all C-grid boundary parameters.
    """
    R_nose = config.R_nose
    body_length = config.computed_body_length
    max_radius = config.max_radius

    x_inflow = -mesh_config.upstream_factor * R_nose
    x_outflow = body_length + mesh_config.downstream_factor * (2.0 * max_radius)
    r_upper = mesh_config.lateral_factor * R_nose

    # Upper boundary: elliptical arc centered at midpoint of inflow-outflow
    upper_center_x = (x_inflow + x_outflow) / 2.0
    upper_center_r = 0.0
    upper_semi_major = (x_outflow - x_inflow) / 2.0
    upper_semi_minor = r_upper

    # Nose cap: straight line from (x_inflow, 0) to body nose (0, R_nose)
    # Use midpoint of the closure line as the nose cap center
    nose_cap_x = (x_inflow + 0.0) / 2.0
    nose_cap_r = (0.0 + R_nose) / 2.0

    return CGridDomain(
        x_inflow=x_inflow,
        r_inflow_upper=upper_semi_minor,
        x_outflow=x_outflow,
        r_outflow_upper=upper_semi_minor,
        upper_center_x=upper_center_x,
        upper_center_r=upper_center_r,
        upper_semi_major=upper_semi_major,
        upper_semi_minor=upper_semi_minor,
        nose_cap_x=nose_cap_x,
        nose_cap_r=nose_cap_r,
        x_nose=0.0,
        x_base=body_length,
        r_base=config.base_radius,
    )


def _generate_cgrid_upper_boundary_points(
    cx: float,
    cr: float,
    a: float,
    b: float,
    n: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate points on the upper half of an ellipse for C-grid upper boundary.

    Produces points on the upper half of an ellipse (theta from 0 to pi),
    suitable for use as a Gmsh spline. This is the C-grid equivalent of
    _generate_ellipse_points but returns only the upper half.

    Args:
        cx: Center x-coordinate.
        cr: Center r-coordinate.
        a: Semi-major axis (axial).
        b: Semi-minor axis (radial).
        n: Number of points (default 40).

    Returns:
        (x, r) arrays of upper-half ellipse points, ordered from
        theta=0 (rightmost) to theta=pi (leftmost).
    """
    theta = np.linspace(0, np.pi, n)
    x = cx + a * np.cos(theta)
    r = cr + b * np.sin(theta)
    return x, r


def _generate_cgrid_nose_closure_points(
    x_inflow: float,
    x_nose: float,
    r_nose: float,
    n: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate points along a straight line for C-grid nose closure.

    Produces n points evenly spaced along a straight line from
    (x_inflow, 0) to (x_nose, r_nose). These points form the nose
    closure of the C-grid topology.

    Args:
        x_inflow: Axial coordinate of the inflow boundary (m).
        x_nose: Axial coordinate of the body nose (m).
        r_nose: Radial coordinate of the body nose (m).
        n: Number of points (default 20).

    Returns:
        (x, r) arrays of the nose closure line points, ordered from
        (x_inflow, 0) to (x_nose, r_nose).
    """
    x = np.linspace(x_inflow, x_nose, n)
    r = np.linspace(0.0, r_nose, n)
    return x, r


def generate_cgrid_mesh(
    config: BluntBodyConfig,
    mesh_config: MeshConfig,
    mach: float,
    output_path: Path,
    aoa: float = 0.0,
) -> Path:
    """Generate a 2D C-grid Gmsh mesh for a spherically-blunted cone.

    Creates the computational domain with:
        - Body contour (sphere + cone) from geometry generation
        - C-grid farfield: upper elliptical arc, nose closure, outflow
        - Symmetry axis along r=0 (axisymmetric) or full domain (full2d)
        - Structured boundary-layer cells at the body wall
        - Shock-region refinement from Billig correlation
        - Sphere-cone junction refinement
        - Wake refinement downstream of body base

    The C-grid wraps around the body nose and extends far downstream,
    providing distinct inflow and outflow boundaries that improve
    hypersonic boundary condition treatment.

    When *aoa* != 0 the domain is forced to full2d regardless of the
    ``mesh_config.domain_type`` setting.

    Args:
        config: Blunt body geometry configuration.
        mesh_config: Mesh refinement configuration.
        mach: Freestream Mach number (for shock refinement).
        output_path: Output .su2 mesh file path.
        aoa: Angle of attack in degrees (default 0.0).

    Returns:
        Path to the generated .su2 mesh file.

    Raises:
        RuntimeError: If mesh generation fails.
    """
    import gmsh

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Angle of attack handling ---
    if aoa != 0.0:
        if mesh_config.domain_type == "axisymmetric":
            print(
                f"  WARNING: axisymmetric domain requested but aoa={aoa} deg; "
                "forcing full2d"
            )
        mesh_config = MeshConfig.for_tier(
            mesh_config.mesh_tier,
            domain_type="full2d",
        )

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("blunt_body_cgrid")

        # --- Body contour ---
        x_body, r_body = generate_contour(config)
        R_nose = config.R_nose
        n_body = len(x_body)

        # --- C-grid domain computation ---
        domain = _build_cgrid_domain(config, mesh_config)

        # --- Boundary layer geometry (identical to O-grid) ---
        first_h = mesh_config.resolve_first_cell_height(R_nose)
        n_bl = mesh_config.n_bl
        ratio = mesh_config.bl_growth_ratio

        # Compute outward normals at each body point.
        normals = np.empty((n_body, 2))
        for i in range(n_body):
            if i == 0:
                dx = x_body[1] - x_body[0]
                dr = r_body[1] - r_body[0]
            elif i == n_body - 1:
                dx = x_body[-1] - x_body[-2]
                dr = r_body[-1] - r_body[-2]
            else:
                dx = x_body[i + 1] - x_body[i - 1]
                dr = r_body[i + 1] - r_body[i - 1]
            mag = np.hypot(dx, dr)
            if mag < 1e-15:
                normals[i] = [0.0, 1.0]
            else:
                n = np.array([-dr / mag, dx / mag])
                if n[1] < 0:
                    n = -n
                normals[i] = n

        # --- Create BL layer points ---
        bl_nodes: list[list[int]] = []
        bl_node_coords: list[list[tuple[float, float]]] = []
        cumulative_h = np.zeros(n_bl + 1)
        for k in range(1, n_bl + 1):
            cumulative_h[k] = cumulative_h[k - 1] + first_h * ratio ** (k - 1)

        for i in range(n_body):
            layer_pts: list[int] = []
            layer_coords: list[tuple[float, float]] = []
            for k in range(n_bl + 1):
                x = x_body[i] + normals[i, 0] * cumulative_h[k]
                r = r_body[i] + normals[i, 1] * cumulative_h[k]
                pt = gmsh.model.geo.addPoint(float(x), float(r), 0)
                layer_pts.append(pt)
                layer_coords.append((float(x), float(r)))
            bl_nodes.append(layer_pts)
            bl_node_coords.append(layer_coords)

        # --- Create BL triangular surfaces (identical to O-grid) ---
        bl_surfaces: list[int] = []
        for i in range(n_body - 1):
            for k in range(n_bl):
                bl = bl_nodes[i][k]
                br = bl_nodes[i + 1][k]
                tr = bl_nodes[i + 1][k + 1]
                tl = bl_nodes[i][k + 1]
                loop1 = gmsh.model.geo.addCurveLoop([
                    gmsh.model.geo.addLine(bl, br),
                    gmsh.model.geo.addLine(br, tr),
                    gmsh.model.geo.addLine(tr, bl),
                ])
                surf1 = gmsh.model.geo.addPlaneSurface([loop1])
                bl_surfaces.append(surf1)

                loop2 = gmsh.model.geo.addCurveLoop([
                    gmsh.model.geo.addLine(bl, tr),
                    gmsh.model.geo.addLine(tr, tl),
                    gmsh.model.geo.addLine(tl, bl),
                ])
                surf2 = gmsh.model.geo.addPlaneSurface([loop2])
                bl_surfaces.append(surf2)

        # --- Full 2D: mirror body and BL to lower half ---
        is_full2d = mesh_config.domain_type == "full2d"

        if is_full2d:
            x_lower = x_body[::-1]
            r_lower = -r_body[::-1]
            n_lower = len(x_lower)

            lower_bl_nodes: list[list[int]] = []
            for i in range(n_body):
                layer_pts: list[int] = []
                for k in range(n_bl + 1):
                    x = float(bl_node_coords[i][k][0])
                    r = float(-bl_node_coords[i][k][1])
                    pt = gmsh.model.geo.addPoint(x, r, 0)
                    layer_pts.append(pt)
                lower_bl_nodes.append(layer_pts)

            for i in range(n_body - 1):
                for k in range(n_bl):
                    bl = lower_bl_nodes[i][k]
                    br = lower_bl_nodes[i + 1][k]
                    tr = lower_bl_nodes[i + 1][k + 1]
                    tl = lower_bl_nodes[i][k + 1]
                    loop1 = gmsh.model.geo.addCurveLoop([
                        gmsh.model.geo.addLine(bl, tr),
                        gmsh.model.geo.addLine(tr, br),
                        gmsh.model.geo.addLine(br, bl),
                    ])
                    surf1 = gmsh.model.geo.addPlaneSurface([loop1])
                    bl_surfaces.append(surf1)

                    loop2 = gmsh.model.geo.addCurveLoop([
                        gmsh.model.geo.addLine(bl, tl),
                        gmsh.model.geo.addLine(tl, tr),
                        gmsh.model.geo.addLine(tr, bl),
                    ])
                    surf2 = gmsh.model.geo.addPlaneSurface([loop2])
                    bl_surfaces.append(surf2)

        # ============================================================
        #  C-GRID BOUNDARY + PHYSICAL GROUPS
        # ============================================================
        offset_curves: list[int]

        if is_full2d:
            # ---- Full 2D C-grid: mirror upper boundary to lower half ----
            # Follows the SAME annular surface pattern as the O-grid full2d:
            # - Inner boundary: BL offset (closed curve with nose/base connections)
            # - Outer boundary: C-shaped farfield (closed curve wrapping around nose)
            # - Single surface with BL hole
            #
            # Upper boundary: elliptical arc from outflow to inflow (upper half)
            n_upper = 41
            upper_x, upper_r = _generate_cgrid_upper_boundary_points(
                domain.upper_center_x,
                domain.upper_center_r,
                domain.upper_semi_major,
                domain.upper_semi_minor,
                n=n_upper,
            )

            upper_pts: list[int] = []
            for j in range(n_upper):
                pt = gmsh.model.geo.addPoint(
                    float(upper_x[j]), float(upper_r[j]), 0,
                )
                upper_pts.append(pt)
            upper_spline = gmsh.model.geo.addSpline(upper_pts)

            # Lower boundary: mirror of upper (negate r, reverse order)
            lower_pts: list[int] = []
            for j in range(n_upper - 1, -1, -1):
                pt = gmsh.model.geo.addPoint(
                    float(upper_x[j]), float(-upper_r[j]), 0,
                )
                lower_pts.append(pt)
            lower_spline = gmsh.model.geo.addSpline(lower_pts)

            # Nose closure: upper half (inflow to body nose upper)
            nose_x, nose_r = _generate_cgrid_nose_closure_points(
                domain.x_inflow,
                domain.x_nose,
                R_nose,
                n=20,
            )
            nose_upper_pts: list[int] = []
            for j in range(len(nose_x)):
                pt = gmsh.model.geo.addPoint(
                    float(nose_x[j]), float(nose_r[j]), 0,
                )
                nose_upper_pts.append(pt)
            nose_upper_spline = gmsh.model.geo.addSpline(nose_upper_pts)

            # Nose closure: lower half (body nose lower to inflow, reversed)
            nose_lower_pts: list[int] = []
            for j in range(len(nose_x) - 1, -1, -1):
                pt = gmsh.model.geo.addPoint(
                    float(nose_x[j]), float(-nose_r[j]), 0,
                )
                nose_lower_pts.append(pt)
            nose_lower_spline = gmsh.model.geo.addSpline(nose_lower_pts)

            # Downstream closure: upper half (outflow upper to body base upper)
            body_base_upper_pt = gmsh.model.geo.addPoint(
                float(domain.x_base), float(domain.r_base), 0,
            )
            outflow_upper_pt = upper_pts[0]
            down_upper_line = gmsh.model.geo.addLine(
                outflow_upper_pt, body_base_upper_pt,
            )

            # Downstream closure: lower half (body base lower to outflow lower)
            body_base_lower_pt = gmsh.model.geo.addPoint(
                float(domain.x_base), float(-domain.r_base), 0,
            )
            outflow_lower_pt = lower_pts[-1]
            down_lower_line = gmsh.model.geo.addLine(
                body_base_lower_pt, outflow_lower_pt,
            )

            # Base connection: body base upper to body base lower
            base_body_line = gmsh.model.geo.addLine(
                body_base_upper_pt, body_base_lower_pt,
            )

            # Farfield outer loop (CCW: fluid on left):
            # upper_spline -> nose_upper -> nose_base_conn -> nose_lower
            # -> lower_spline -> down_lower -> base_body -> down_upper
            body_nose_upper_pt = nose_upper_pts[-1]  # (0, R_nose)
            body_nose_lower_pt = nose_lower_pts[0]   # (0, -R_nose)

            # Connection between body nose upper and body nose lower
            nose_base_conn = gmsh.model.geo.addLine(
                body_nose_upper_pt, body_nose_lower_pt,
            )

            # Farfield loop: upper -> nose_upper -> nose_base_conn -> nose_lower -> lower -> down_lower -> base_body -> down_upper
            farfield_loop = gmsh.model.geo.addCurveLoop([
                upper_spline,           # outflow upper -> inflow
                nose_upper_spline,      # inflow -> body nose upper
                nose_base_conn,         # body nose upper -> body nose lower
                nose_lower_spline,      # body nose lower -> inflow
                lower_spline,           # inflow -> outflow lower
                down_lower_line,        # outflow lower -> body base lower
                base_body_line,         # body base lower -> body base upper
                down_upper_line,        # body base upper -> outflow upper
            ])

            # BL offset splines for upper and lower halves
            upper_offset_pts = [bl_nodes[i][n_bl] for i in range(n_body)]
            lower_offset_pts = [lower_bl_nodes[i][n_bl] for i in range(n_body)]

            upper_offset_spline = gmsh.model.geo.addSpline(upper_offset_pts)
            lower_offset_spline = gmsh.model.geo.addSpline(
                lower_offset_pts[::-1],
            )

            # Connect BL offsets at nose and base
            nose_conn = gmsh.model.geo.addLine(
                lower_offset_pts[0], upper_offset_pts[0],
            )
            base_conn = gmsh.model.geo.addLine(
                upper_offset_pts[-1], lower_offset_pts[-1],
            )

            # Inner BL loop (CW hole in the outer surface)
            bl_loop = gmsh.model.geo.addCurveLoop([
                upper_offset_spline,
                base_conn,
                lower_offset_spline,
                nose_conn,
            ])

            # Outer surface: farfield loop with BL hole
            outer_surface = gmsh.model.geo.addPlaneSurface(
                [farfield_loop, bl_loop],
            )

            # --- Physical groups for full 2D C-grid ---
            # Body curves: upper + base + lower
            upper_body_curves: list[int] = []
            for i in range(n_body - 1):
                upper_body_curves.append(
                    gmsh.model.geo.addLine(bl_nodes[i][0], bl_nodes[i + 1][0])
                )
            base_body_curve = gmsh.model.geo.addLine(
                bl_nodes[n_body - 1][0], lower_bl_nodes[n_body - 1][0],
            )
            lower_body_curves: list[int] = []
            for i in range(n_body - 1):
                lower_body_curves.append(
                    gmsh.model.geo.addLine(
                        lower_bl_nodes[n_body - 1 - i][0],
                        lower_bl_nodes[n_body - 2 - i][0],
                    )
                )
            gmsh.model.geo.addPhysicalGroup(
                1,
                upper_body_curves + [base_body_curve] + lower_body_curves,
                name="body",
            )

            # Farfield: upper boundary + nose closures + lower boundary
            gmsh.model.geo.addPhysicalGroup(
                1,
                [upper_spline, nose_upper_spline, nose_lower_spline,
                 lower_spline],
                name="farfield",
            )

            # Fluid: BL surfaces + outer surface
            gmsh.model.geo.addPhysicalGroup(
                2, bl_surfaces + [outer_surface], name="fluid",
            )
            # NO sym marker for full 2d

            offset_curves = [upper_offset_spline, lower_offset_spline]

        else:
            # ---- Axisymmetric C-grid: upper half only ----
            # Follows the SAME annular surface pattern as the O-grid:
            # - Inner boundary: BL offset spline (nose to base)
            # - Outer boundary: C-shaped farfield (upper arc + connector lines)
            # - Single annular surface between them
            #
            # Upper boundary: elliptical arc from outflow (right) to inflow (left)
            n_upper = 41
            upper_x, upper_r = _generate_cgrid_upper_boundary_points(
                domain.upper_center_x,
                domain.upper_center_r,
                domain.upper_semi_major,
                domain.upper_semi_minor,
                n=n_upper,
            )

            upper_pts: list[int] = []
            for j in range(n_upper):
                pt = gmsh.model.geo.addPoint(
                    float(upper_x[j]), float(upper_r[j]), 0,
                )
                upper_pts.append(pt)
            upper_spline = gmsh.model.geo.addSpline(upper_pts)

            # BL offset spline (top of BL, inner boundary of outer surface)
            outer_inner_pts = [bl_nodes[i][n_bl] for i in range(n_body)]
            offset_spline = gmsh.model.geo.addSpline(outer_inner_pts)

            # Connector lines: connect farfield boundary to BL offset
            # nose_conn: from inflow (x_inflow, 0) to BL offset start
            inflow_pt = upper_pts[-1]  # last point of upper spline (theta=pi)
            nose_conn = gmsh.model.geo.addLine(inflow_pt, outer_inner_pts[0])

            # downstream_conn: from BL offset end to outflow (x_outflow, 0)
            outflow_pt = upper_pts[0]  # first point of upper spline (theta=0)
            downstream_conn = gmsh.model.geo.addLine(
                outer_inner_pts[-1], outflow_pt,
            )

            # Outer surface loop (CCW: fluid on left)
            # Pattern: offset_spline, downstream_conn, upper_spline, nose_conn
            # This matches the O-grid pattern exactly.
            outer_loop = gmsh.model.geo.addCurveLoop([
                offset_spline,
                downstream_conn,
                upper_spline,
                nose_conn,
            ])
            outer_surface = gmsh.model.geo.addPlaneSurface([outer_loop])

            # --- Physical groups (SU2 markers) for axisymmetric ---
            # Body curves: all body contour line segments
            body_curves: list[int] = []
            for i in range(n_body - 1):
                body_curves.append(
                    gmsh.model.geo.addLine(bl_nodes[i][0], bl_nodes[i + 1][0])
                )
            gmsh.model.geo.addPhysicalGroup(1, body_curves, name="body")

            # Farfield: the C-grid upper boundary arc
            gmsh.model.geo.addPhysicalGroup(
                1, [upper_spline], name="farfield",
            )

            # Symmetry: lines on the axis r=0
            # Upstream: from inflow to body nose on axis
            sym_up_line = gmsh.model.geo.addLine(
                inflow_pt, bl_nodes[0][0],
            )
            # Downstream: from body base on axis to outflow
            body_base_axis_pt = gmsh.model.geo.addPoint(
                float(domain.x_base), 0.0, 0,
            )
            sym_down_line = gmsh.model.geo.addLine(
                body_base_axis_pt, outflow_pt,
            )
            gmsh.model.geo.addPhysicalGroup(
                1, [sym_up_line, sym_down_line], name="sym",
            )

            # Fluid: BL surfaces + outer surface
            gmsh.model.geo.addPhysicalGroup(
                2, bl_surfaces + [outer_surface], name="fluid",
            )

            offset_curves = [offset_spline]

        # --- Synchronize geometry ---
        gmsh.model.geo.synchronize()

        # ============================================================
        #  SIZE FIELDS (identical to O-grid)
        # ============================================================
        tier_mult = _TIER_SIZE_MULTIPLIERS[mesh_config.mesh_tier]
        body_length = config.computed_body_length

        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMin", 0.1 * R_nose * tier_mult,
        )
        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMax", 2.0 * body_length * tier_mult,
        )

        # Background mesh
        bg_tag = 100
        gmsh.model.mesh.field.add("Constant", bg_tag)
        bg_vin = 0.2 * body_length * tier_mult
        gmsh.model.mesh.field.setNumber(bg_tag, "VIn", bg_vin)
        gmsh.model.mesh.field.setNumber(bg_tag, "VOut", bg_vin)

        # Shock refinement
        shock_tag = 300
        if mesh_config.shock_refinement:
            _add_shock_refinement(
                config, mach, mesh_config, shock_tag, tier_mult,
            )

        # Junction refinement
        junc_tag = 400
        _add_junction_refinement(config, mesh_config, junc_tag, tier_mult)

        # Wake refinement
        wake_tag = 600
        _add_wake_refinement(config, mesh_config, wake_tag, tier_mult)

        # Distance-based size field for smooth BL-to-farfield transition
        distance_tag = 700
        gmsh.model.mesh.field.add("Distance", distance_tag)
        gmsh.model.mesh.field.setNumbers(distance_tag, "CurvesList", offset_curves)

        bl_edge_spacing = 0.5 * body_length / max(n_body - 1, 1)
        min_size = max(bl_edge_spacing, 0.01 * tier_mult)
        max_size = 0.3 * R_nose * tier_mult

        # Use upper_semi_minor as the characteristic farfield distance
        ramp_dist = domain.upper_semi_minor if hasattr(domain, "upper_semi_minor") else domain.r_inflow_upper
        ramp_coeff = (max_size - min_size) / ramp_dist

        math_tag = 701
        gmsh.model.mesh.field.add("MathEval", math_tag)
        gmsh.model.mesh.field.setString(
            math_tag,
            "F",
            f"Max({min_size}, Min({max_size}, "
            f"{min_size} + F{distance_tag} * {ramp_coeff}))",
        )

        # Combine fields with Min
        min_tag = 999
        field_ids: list[int] = [bg_tag]
        if mesh_config.shock_refinement:
            field_ids.append(shock_tag)
        field_ids.append(junc_tag)
        field_ids.append(wake_tag)
        field_ids.append(math_tag)

        gmsh.model.mesh.field.add("Min", min_tag)
        gmsh.model.mesh.field.setNumbers(min_tag, "FieldsList", field_ids)
        gmsh.model.mesh.field.setAsBackgroundMesh(min_tag)

        # ============================================================
        #  MESH GENERATION
        # ============================================================
        gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.Smoothing", 50)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Netgen")

        # --- Export ---
        gmsh.write(str(output_path))

    except Exception as exc:
        raise RuntimeError(f"Mesh generation failed: {exc}") from exc
    finally:
        try:
            gmsh.finalize()
        except OSError:
            pass

    return output_path


def generate_body_mesh(
    config: BluntBodyConfig,
    mesh_config: MeshConfig,
    mach: float,
    output_path: Path,
    aoa: float = 0.0,
) -> Path:
    """Generate a 2D Gmsh mesh for a spherically-blunted cone.

    Creates the computational domain with:
        - Body contour (sphere + cone) from Phase 1 geometry
        - Body-conforming elliptical (O-grid) farfield boundary
        - Symmetry axis along r=0 (axisymmetric) or full domain (full2d)
        - Structured boundary-layer cells at the body wall
        - Shock-region refinement from Billig correlation
        - Sphere-cone junction refinement
        - Wake refinement downstream of body base

    The elliptical farfield replaces the rectangular domain, reducing the
    total domain area by approximately 87% while maintaining adequate
    distance for all boundaries.

    When *aoa* != 0 the domain is forced to full2d regardless of the
    ``mesh_config.domain_type`` setting.  The angle of attack itself is
    applied by SU2 (freestream rotation), not by rotating the body mesh.

    The boundary layer is built by placing BL nodes at explicit positions
    along the outward normal from each body contour point.  Each axial
    segment gets a column of triangular elements with geometric growth
    from the wall.  This guarantees the requested first cell height.

    Args:
        config: Blunt body geometry configuration.
        mesh_config: Mesh refinement configuration.
        mach: Freestream Mach number (for shock refinement).
        output_path: Output .su2 mesh file path.
        aoa: Angle of attack in degrees (default 0.0).

    Returns:
        Path to the generated .su2 mesh file.

    Raises:
        RuntimeError: If mesh generation fails.
    """
    import gmsh

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Angle of attack handling ---
    if aoa != 0.0:
        if mesh_config.domain_type == "axisymmetric":
            print(
                f"  WARNING: axisymmetric domain requested but aoa={aoa} deg; "
                "forcing full2d"
            )
        # Force full2d when aoa is nonzero
        mesh_config = MeshConfig.for_tier(
            mesh_config.mesh_tier,
            domain_type="full2d",
        )

    # Dispatch to C-grid mesh if requested
    if mesh_config.domain_type == "cgrid":
        return generate_cgrid_mesh(
            config, mesh_config, mach, output_path, aoa=aoa,
        )

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("blunt_body")

        # --- Body contour ---
        x_body, r_body = generate_contour(config)
        R_nose = config.R_nose
        n_body = len(x_body)

        # --- O-grid domain computation ---
        domain = _build_ogrid_domain(config, mesh_config)

        # --- Boundary layer geometry ---
        first_h = mesh_config.resolve_first_cell_height(R_nose)
        n_bl = mesh_config.n_bl
        ratio = mesh_config.bl_growth_ratio

        # Compute outward normals at each body point.
        # For axisymmetric half-domain (r >= 0), the outward normal must point
        # away from the body surface.  We use the left-hand normal of the
        # tangent vector and verify it points away from the symmetry axis.
        normals = np.empty((n_body, 2))
        for i in range(n_body):
            if i == 0:
                dx = x_body[1] - x_body[0]
                dr = r_body[1] - r_body[0]
            elif i == n_body - 1:
                dx = x_body[-1] - x_body[-2]
                dr = r_body[-1] - r_body[-2]
            else:
                dx = x_body[i + 1] - x_body[i - 1]
                dr = r_body[i + 1] - r_body[i - 1]
            mag = np.hypot(dx, dr)
            if mag < 1e-15:
                normals[i] = [0.0, 1.0]
            else:
                # Left-hand normal of the tangent vector
                n = np.array([-dr / mag, dx / mag])
                # Ensure normal points away from symmetry axis (r >= 0 direction)
                # For r > 0, outward means n_r > 0; for r = 0, n_r > 0
                if n[1] < 0:
                    n = -n
                normals[i] = n

        # --- Create BL layer points ---
        # bl_nodes[i][k] = gmsh point tag for body point i, BL layer k
        # k=0 is the body surface, k=n_bl is the offset (BL outer edge)
        bl_nodes: list[list[int]] = []
        bl_node_coords: list[list[tuple[float, float]]] = []
        cumulative_h = np.zeros(n_bl + 1)
        for k in range(1, n_bl + 1):
            cumulative_h[k] = cumulative_h[k - 1] + first_h * ratio ** (k - 1)

        for i in range(n_body):
            layer_pts: list[int] = []
            layer_coords: list[tuple[float, float]] = []
            for k in range(n_bl + 1):
                x = x_body[i] + normals[i, 0] * cumulative_h[k]
                r = r_body[i] + normals[i, 1] * cumulative_h[k]
                pt = gmsh.model.geo.addPoint(float(x), float(r), 0)
                layer_pts.append(pt)
                layer_coords.append((float(x), float(r)))
            bl_nodes.append(layer_pts)
            bl_node_coords.append(layer_coords)

        # --- Create BL triangular surfaces ---
        # Using triangles instead of quads to avoid high aspect ratio elements
        # in the thin boundary layer near the wall.
        bl_surfaces: list[int] = []
        for i in range(n_body - 1):
            for k in range(n_bl):
                bl = bl_nodes[i][k]
                br = bl_nodes[i + 1][k]
                tr = bl_nodes[i + 1][k + 1]
                tl = bl_nodes[i][k + 1]
                # Split quad into 2 triangles
                loop1 = gmsh.model.geo.addCurveLoop([
                    gmsh.model.geo.addLine(bl, br),
                    gmsh.model.geo.addLine(br, tr),
                    gmsh.model.geo.addLine(tr, bl),
                ])
                surf1 = gmsh.model.geo.addPlaneSurface([loop1])
                bl_surfaces.append(surf1)

                loop2 = gmsh.model.geo.addCurveLoop([
                    gmsh.model.geo.addLine(bl, tr),
                    gmsh.model.geo.addLine(tr, tl),
                    gmsh.model.geo.addLine(tl, bl),
                ])
                surf2 = gmsh.model.geo.addPlaneSurface([loop2])
                bl_surfaces.append(surf2)

        # --- Full 2D: mirror body and BL to lower half ---
        is_full2d = mesh_config.domain_type == "full2d"

        if is_full2d:
            # Mirror body contour: negate r, reverse order for proper winding
            x_lower = x_body[::-1]
            r_lower = -r_body[::-1]
            n_lower = len(x_lower)

            # Create lower BL nodes by mirroring upper BL (negate r)
            lower_bl_nodes: list[list[int]] = []
            for i in range(n_body):
                layer_pts: list[int] = []
                for k in range(n_bl + 1):
                    x = float(bl_node_coords[i][k][0])
                    r = float(-bl_node_coords[i][k][1])
                    pt = gmsh.model.geo.addPoint(x, r, 0)
                    layer_pts.append(pt)
                lower_bl_nodes.append(layer_pts)

            # Create lower BL triangular surfaces (reversed winding for CCW)
            for i in range(n_body - 1):
                for k in range(n_bl):
                    bl = lower_bl_nodes[i][k]
                    br = lower_bl_nodes[i + 1][k]
                    tr = lower_bl_nodes[i + 1][k + 1]
                    tl = lower_bl_nodes[i][k + 1]
                    loop1 = gmsh.model.geo.addCurveLoop([
                        gmsh.model.geo.addLine(bl, tr),
                        gmsh.model.geo.addLine(tr, br),
                        gmsh.model.geo.addLine(br, bl),
                    ])
                    surf1 = gmsh.model.geo.addPlaneSurface([loop1])
                    bl_surfaces.append(surf1)

                    loop2 = gmsh.model.geo.addCurveLoop([
                        gmsh.model.geo.addLine(bl, tl),
                        gmsh.model.geo.addLine(tl, tr),
                        gmsh.model.geo.addLine(tr, bl),
                    ])
                    surf2 = gmsh.model.geo.addPlaneSurface([loop2])
                    bl_surfaces.append(surf2)

        # ============================================================
        #  O-GRID ELLIPTICAL BOUNDARY + PHYSICAL GROUPS
        # ============================================================
        # Curves used for the distance-based size field (defined in
        # both branches so the field section below can reference them).
        offset_curves: list[int]

        if is_full2d:
            # ---- Full 2D: complete ellipse (0 to 2 pi) ----
            n_full = 42
            theta_full = np.linspace(0, 2 * np.pi, n_full, endpoint=False)
            ell_full_x = domain.center_x + domain.semi_major * np.cos(theta_full)
            ell_full_r = domain.center_r + domain.semi_minor * np.sin(theta_full)

            ell_full_pts: list[int] = []
            for j in range(n_full):
                pt = gmsh.model.geo.addPoint(
                    float(ell_full_x[j]), float(ell_full_r[j]), 0,
                )
                ell_full_pts.append(pt)

            # Closed spline around the full ellipse (last -> first to close)
            ellipse_spline = gmsh.model.geo.addSpline(
                ell_full_pts + [ell_full_pts[0]],
            )

            # BL offset splines for upper and lower halves
            upper_offset_pts = [bl_nodes[i][n_bl] for i in range(n_body)]
            lower_offset_pts = [lower_bl_nodes[i][n_bl] for i in range(n_body)]

            upper_offset_spline = gmsh.model.geo.addSpline(upper_offset_pts)
            # Lower offset: reverse so the spline goes base -> nose
            lower_offset_spline = gmsh.model.geo.addSpline(
                lower_offset_pts[::-1],
            )

            # Connect BL offsets at nose and base to close the inner loop
            nose_conn = gmsh.model.geo.addLine(
                lower_offset_pts[0], upper_offset_pts[0],
            )
            base_conn = gmsh.model.geo.addLine(
                upper_offset_pts[-1], lower_offset_pts[-1],
            )

            # Outer surface: annular region between BL offset and ellipse
            # Inner loop (CW = hole): upper_offset, base_conn,
            #   lower_offset, nose_conn
            bl_loop = gmsh.model.geo.addCurveLoop([
                upper_offset_spline,
                base_conn,
                lower_offset_spline,
                nose_conn,
            ])
            # Outer loop (CCW): full ellipse
            ell_loop = gmsh.model.geo.addCurveLoop([ellipse_spline])
            outer_surface = gmsh.model.geo.addPlaneSurface([ell_loop, bl_loop])

            # --- Physical groups for full 2D ---
            # Body curves: upper + base + lower
            upper_body_curves: list[int] = []
            for i in range(n_body - 1):
                upper_body_curves.append(
                    gmsh.model.geo.addLine(bl_nodes[i][0], bl_nodes[i + 1][0])
                )
            # Base edge: upper base -> lower base
            base_body_curve = gmsh.model.geo.addLine(
                bl_nodes[n_body - 1][0], lower_bl_nodes[n_body - 1][0],
            )
            # Lower body curves: base -> nose (reversed index order)
            lower_body_curves: list[int] = []
            for i in range(n_body - 1):
                lower_body_curves.append(
                    gmsh.model.geo.addLine(
                        lower_bl_nodes[n_body - 1 - i][0],
                        lower_bl_nodes[n_body - 2 - i][0],
                    )
                )
            gmsh.model.geo.addPhysicalGroup(
                1,
                upper_body_curves + [base_body_curve] + lower_body_curves,
                name="body",
            )
            # Farfield: the full ellipse
            gmsh.model.geo.addPhysicalGroup(
                1, [ellipse_spline], name="farfield",
            )
            # Fluid: BL surfaces + outer surface
            gmsh.model.geo.addPhysicalGroup(
                2, bl_surfaces + [outer_surface], name="fluid",
            )
            # NO sym marker for full 2d

            offset_curves = [upper_offset_spline, lower_offset_spline]

        else:
            # ---- Axisymmetric: upper-half ellipse only ----
            # Generate upper-half ellipse points for the outer boundary.
            # Theta from 0 (base/right) to pi (nose/left) through the top.
            n_upper = 21  # odd so we get a center point at the top
            theta_upper = np.linspace(0.0, np.pi, n_upper)
            ell_upper_x = (
                domain.center_x + domain.semi_major * np.cos(theta_upper)
            )
            ell_upper_r = (
                domain.center_r + domain.semi_minor * np.sin(theta_upper)
            )

            upper_ell_pts: list[int] = []
            for j in range(n_upper):
                pt = gmsh.model.geo.addPoint(
                    float(ell_upper_x[j]), float(ell_upper_r[j]), 0,
                )
                upper_ell_pts.append(pt)

            # Open spline along the upper half of the ellipse (base -> nose)
            upper_ellipse_spline = gmsh.model.geo.addSpline(upper_ell_pts)

            # BL offset spline (top of BL, inner boundary of outer surface)
            outer_inner_pts = [bl_nodes[i][n_bl] for i in range(n_body)]
            offset_spline = gmsh.model.geo.addSpline(outer_inner_pts)

            # --- Symmetry lines (along the axis r=0) ---
            # Upstream: from ellipse nose (x_min, 0) to body nose (0, 0)
            # Ellipse nose is the last point in the upper half (theta=pi)
            ell_nose_pt = upper_ell_pts[-1]
            sym_up_line = gmsh.model.geo.addLine(ell_nose_pt, bl_nodes[0][0])

            # Downstream: from body base axis point to ellipse base (x_max, 0)
            # Place a point on the axis at the body base x-coordinate
            body_base_axis_pt = gmsh.model.geo.addPoint(
                float(config.computed_body_length), 0.0, 0,
            )
            # Ellipse base is the first point in the upper half (theta=0)
            ell_base_pt = upper_ell_pts[0]
            sym_down_line = gmsh.model.geo.addLine(
                body_base_axis_pt, ell_base_pt,
            )

            # --- Outer surface (offset BL to ellipse) ---
            # Connect BL offset to ellipse at upstream and downstream
            # Upstream: ellipse_nose -> BL offset start (near body nose)
            upstream_conn = gmsh.model.geo.addLine(
                ell_nose_pt, outer_inner_pts[0],
            )
            # Downstream: BL offset end (near body base) -> ellipse base
            downstream_conn = gmsh.model.geo.addLine(
                outer_inner_pts[-1], ell_base_pt,
            )

            # Outer surface loop (CCW in upper half-plane):
            # 1. offset_spline: BL offset from nose-end to base-end
            # 2. downstream_conn: BL offset end -> ellipse base
            # 3. upper_ellipse_spline: ellipse base -> nose via top
            # 4. upstream_conn: ellipse nose -> BL offset start
            outer_loop = gmsh.model.geo.addCurveLoop([
                offset_spline,
                downstream_conn,
                upper_ellipse_spline,
                upstream_conn,
            ])
            outer_surface = gmsh.model.geo.addPlaneSurface([outer_loop])

            # --- Physical groups (SU2 markers) for axisymmetric ---
            # Body curves: all body contour line segments
            body_curves = []
            for i in range(n_body - 1):
                body_curves.append(
                    gmsh.model.geo.addLine(bl_nodes[i][0], bl_nodes[i + 1][0])
                )
            gmsh.model.geo.addPhysicalGroup(1, body_curves, name="body")

            # Farfield: the upper ellipse spline
            gmsh.model.geo.addPhysicalGroup(
                1, [upper_ellipse_spline], name="farfield",
            )

            # Symmetry: both upstream and downstream lines on the axis
            gmsh.model.geo.addPhysicalGroup(
                1, [sym_up_line, sym_down_line], name="sym",
            )

            # Fluid: BL surfaces + outer surface
            gmsh.model.geo.addPhysicalGroup(
                2, bl_surfaces + [outer_surface], name="fluid",
            )

            offset_curves = [offset_spline]

        # --- Synchronize geometry ---
        gmsh.model.geo.synchronize()

        # ============================================================
        #  SIZE FIELDS FOR THE OUTER SURFACE
        # ============================================================
        tier_mult = _TIER_SIZE_MULTIPLIERS[mesh_config.mesh_tier]
        body_length = config.computed_body_length

        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMin", 0.1 * R_nose * tier_mult,
        )
        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMax", 2.0 * body_length * tier_mult,
        )

        # Background mesh (base cell size, scaled by tier)
        bg_tag = 100
        gmsh.model.mesh.field.add("Constant", bg_tag)
        bg_vin = 0.2 * body_length * tier_mult
        gmsh.model.mesh.field.setNumber(bg_tag, "VIn", bg_vin)
        gmsh.model.mesh.field.setNumber(bg_tag, "VOut", bg_vin)

        # Shock refinement
        shock_tag = 300
        if mesh_config.shock_refinement:
            _add_shock_refinement(
                config, mach, mesh_config, shock_tag, tier_mult,
            )

        # Junction refinement
        junc_tag = 400
        _add_junction_refinement(config, mesh_config, junc_tag, tier_mult)

        # Wake refinement
        wake_tag = 600
        _add_wake_refinement(config, mesh_config, wake_tag, tier_mult)

        # Distance-based size field for smooth BL-to-farfield transition.
        #
        # Measures distance from the offset spline (outer boundary of the
        # BL, inner boundary of the outer surface).  This gives distance=0
        # exactly at the inner boundary where node spacing is set by the
        # explicit BL placement (~0.016 m mean body contour spacing).
        #
        # The MathEval field ramps cell size linearly from min_size at the
        # inner boundary to max_size at the farfield, preventing the
        # catastrophic size ratio (1.048 m field vs 0.016 m nodes) that
        # causes inverted elements.
        distance_tag = 700
        gmsh.model.mesh.field.add("Distance", distance_tag)
        gmsh.model.mesh.field.setNumbers(distance_tag, "CurvesList", offset_curves)

        # Compute ramp parameters
        bl_edge_spacing = 0.5 * body_length / max(n_body - 1, 1)
        min_size = max(bl_edge_spacing, 0.01 * tier_mult)
        max_size = 0.3 * R_nose * tier_mult
        # Ramp coefficient: size = min_size + ramp_coeff * distance
        ramp_coeff = (max_size - min_size) / domain.semi_minor

        math_tag = 701
        gmsh.model.mesh.field.add("MathEval", math_tag)
        gmsh.model.mesh.field.setString(
            math_tag,
            "F",
            f"Max({min_size}, Min({max_size}, "
            f"{min_size} + F{distance_tag} * {ramp_coeff}))",
        )

        # Combine fields with Min
        min_tag = 999
        field_ids: list[int] = [bg_tag]
        if mesh_config.shock_refinement:
            field_ids.append(shock_tag)
        field_ids.append(junc_tag)
        field_ids.append(wake_tag)
        field_ids.append(math_tag)

        gmsh.model.mesh.field.add("Min", min_tag)
        gmsh.model.mesh.field.setNumbers(min_tag, "FieldsList", field_ids)
        gmsh.model.mesh.field.setAsBackgroundMesh(min_tag)

        # ============================================================
        #  MESH GENERATION
        # ============================================================
        gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.Smoothing", 50)
        gmsh.model.mesh.generate(2)

        # Post-generation optimization to fix inverted/sliver elements
        # caused by extreme BL-to-farfield size ratios.
        gmsh.model.mesh.optimize("Netgen")

        # --- Export ---
        gmsh.write(str(output_path))

    except Exception as exc:
        raise RuntimeError(f"Mesh generation failed: {exc}") from exc
    finally:
        try:
            gmsh.finalize()
        except OSError:
            pass

    return output_path


def generate_mesh_from_dxf(
    dxf_path: Path,
    output_path: Path,
    mesh_config: MeshConfig | None = None,
) -> Path:
    """Generate a 2D axisymmetric mesh from DXF contour points.

    Loads the body contour from a saved ``.npz`` file (keys ``x`` and ``r``)
    and builds a simple triangular mesh in the (x, r) half-plane.  The
    domain is bounded by:
      - the body contour (physical group ``body``)
      - upstream and downstream lines connecting the body to the axis
        (physical group ``farfield``)
      - the symmetry axis along r = 0 (physical group ``sym``)

    When *mesh_config* is ``None`` a default ``MeshConfig`` is used.

    Args:
        dxf_path: Path to the ``.npz`` file containing contour arrays
            with keys ``x`` and ``r``.
        output_path: Output ``.su2`` mesh file path.
        mesh_config: Mesh refinement configuration.  If ``None``, a
            default ``MeshConfig()`` is created.

    Returns:
        Path to the generated ``.su2`` mesh file.

    Raises:
        RuntimeError: If mesh generation fails.
    """
    import gmsh

    if mesh_config is None:
        mesh_config = MeshConfig()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("dxf_mesh")

        # --- Load contour ---
        data = np.load(dxf_path)
        x_contour: np.ndarray = data["x"]
        r_contour: np.ndarray = data["r"]

        # --- Body contour points ---
        body_pts: list[int] = []
        for i in range(len(x_contour)):
            pt = gmsh.model.geo.addPoint(float(x_contour[i]), float(r_contour[i]), 0)
            body_pts.append(pt)

        # --- Axis points ---
        axis_left = gmsh.model.geo.addPoint(0.0, 0.0, 0)
        axis_right = gmsh.model.geo.addPoint(float(x_contour[-1]), 0.0, 0)

        # --- Body lines ---
        body_lines: list[int] = []
        for i in range(len(body_pts) - 1):
            line = gmsh.model.geo.addLine(body_pts[i], body_pts[i + 1])
            body_lines.append(line)

        # --- Closing lines ---
        # Upstream: from axis left to body nose
        upstream = gmsh.model.geo.addLine(axis_left, body_pts[0])
        # Downstream: from body base to axis right
        downstream = gmsh.model.geo.addLine(body_pts[-1], axis_right)
        # Axis: from axis right back to axis left (symmetry axis)
        axis_line = gmsh.model.geo.addLine(axis_right, axis_left)

        # --- Loop and surface (CCW winding) ---
        # For CCW winding in the (x, r) plane, the enclosed region must
        # be on the LEFT of each curve when traversing the loop.  The
        # natural CCW order is: axis right, downstream up, body left,
        # upstream down.  We negate curves to reverse their direction.
        all_lines = (
            [-axis_line, -downstream]
            + [-l for l in reversed(body_lines)]
            + [-upstream]
        )
        loop = gmsh.model.geo.addCurveLoop(all_lines)
        surface = gmsh.model.geo.addPlaneSurface([loop])

        gmsh.model.geo.synchronize()

        # --- Physical groups (SU2 markers) ---
        gmsh.model.setPhysicalName(
            1, gmsh.model.geo.addPhysicalGroup(1, body_lines), "body",
        )
        gmsh.model.setPhysicalName(
            1, gmsh.model.geo.addPhysicalGroup(1, [axis_line]), "sym",
        )
        gmsh.model.setPhysicalName(
            1, gmsh.model.geo.addPhysicalGroup(1, [upstream, downstream]),
            "farfield",
        )
        gmsh.model.setPhysicalName(
            2, gmsh.model.geo.addPhysicalGroup(2, [surface]), "fluid",
        )

        # --- Mesh generation ---
        tier_mult = _TIER_SIZE_MULTIPLIERS[mesh_config.mesh_tier]
        R_nose = mesh_config.farfield_distance  # used as scale reference
        body_length = float(x_contour[-1] - x_contour[0])

        gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay
        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMin", 0.05 * body_length * tier_mult,
        )
        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMax", 0.5 * body_length * tier_mult,
        )
        gmsh.option.setNumber("Mesh.Smoothing", 50)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Netgen")

        # --- Export ---
        gmsh.write(str(output_path))

    except Exception as exc:
        raise RuntimeError(f"Mesh generation failed: {exc}") from exc
    finally:
        try:
            gmsh.finalize()
        except OSError:
            pass

    return output_path


