"""Gmsh O-grid mesh generation for spherically-blunted cones.

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

from .mesh_config import MeshConfig, OGridDomain

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
        #  O-GRID ELLIPTICAL BOUNDARY
        # ============================================================
        # Generate upper-half ellipse points for the outer boundary.
        # Theta from 0 (base/right) to pi (nose/left) through the top.
        n_upper = 21  # odd so we get a center point at the top
        theta_upper = np.linspace(0.0, np.pi, n_upper)
        ell_upper_x = domain.center_x + domain.semi_major * np.cos(theta_upper)
        ell_upper_r = domain.center_r + domain.semi_minor * np.sin(theta_upper)

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
        sym_down_line = gmsh.model.geo.addLine(body_base_axis_pt, ell_base_pt)

        # --- Outer surface (offset BL to ellipse) ---
        # Connect BL offset to ellipse at upstream and downstream
        # Upstream: ellipse_nose -> BL offset start (near body nose)
        upstream_conn = gmsh.model.geo.addLine(ell_nose_pt, outer_inner_pts[0])
        # Downstream: BL offset end (near body base) -> ellipse base
        downstream_conn = gmsh.model.geo.addLine(
            outer_inner_pts[-1], ell_base_pt,
        )

        # Outer surface loop (CCW in upper half-plane):
        # 1. offset_spline: BL offset from nose-end to base-end (left to right)
        # 2. downstream_conn: BL offset end -> ellipse base (right, outward)
        # 3. upper_ellipse_spline: ellipse base -> nose via top (right to left)
        # 4. upstream_conn: ellipse nose -> BL offset start (left, inward)
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
        gmsh.model.mesh.field.setNumbers(distance_tag, "CurvesList", [offset_spline])

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


def generate_fullview_mesh(
    dxf_path: Path,
    output_path: Path,
) -> Path:
    """Generate a full-view 2D mesh showing the complete Apollo CM body.

    Creates a rectangular farfield domain with the full body (upper and
    lower halves) centered inside.  The DXF contour provides the upper
    half only; the lower half is created by mirroring (negate r, reverse
    order).  The body is positioned closer to the entrance so the wake
    region downstream is well-resolved.

    Domain sizing:
        Upstream:   5 * R_nose = 23.47 m (from nose to left boundary)
        Downstream: 15 * body_length = 50.88 m (from base to right boundary)
        Lateral:    5 * max_radius = 9.78 m (above and below body)

    Boundary conditions:
        - body:  entire closed body surface (upper + lower halves)
        - farfield: all four edges of the rectangular domain
        - fluid: the 2D surface between body and farfield
        - NO sym marker (full 2D, not axisymmetric)

    Args:
        dxf_path: Path to the ``.npz`` file containing contour arrays
            with keys ``x`` and ``r`` (upper half only).
        output_path: Output ``.su2`` mesh file path.

    Returns:
        Path to the generated ``.su2`` mesh file.

    Raises:
        RuntimeError: If mesh generation fails.
    """
    import gmsh

    # --- Apollo CM geometry parameters ---
    R_nose = 4.694  # m (heat shield sphere radius)
    body_length = 3.3918  # m
    max_radius = 1.956  # m (half-height)

    # --- Domain sizing (5x upstream, 15x downstream, 5x lateral) ---
    upstream = 5.0 * R_nose  # 23.47 m from nose to left boundary
    downstream = 15.0 * body_length  # 50.88 m from base to right boundary
    lateral = 5.0 * max_radius  # 9.78 m above and below body

    # --- Body centered in domain ---
    x_center = body_length / 2.0  # 1.696 m
    y_center = 0.0

    x_min = x_center - upstream  # -21.77 m
    x_max = x_center + body_length + downstream  # 52.58 m
    y_min = y_center - lateral  # -9.78 m
    y_max = y_center + lateral  # 9.78 m

    # --- Load body contour from DXF (upper half only) ---
    data = np.load(dxf_path)
    x_upper: np.ndarray = data["x"]
    r_upper: np.ndarray = data["r"]

    # Mirror to create lower half: negate r, reverse order
    x_lower = x_upper[::-1]
    r_lower = -r_upper[::-1]

    # Concatenate upper + lower to form full closed body contour
    x_full = np.concatenate([x_upper, x_lower[1:]])
    r_full = np.concatenate([r_upper, r_lower[1:]])
    n_body = len(x_full)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("fullview_mesh")

        # ============================================================
        #  BODY CONTOUR (closed loop: upper half -> base close -> lower half -> nose)
        # ============================================================
        body_pts: list[int] = []
        for i in range(n_body):
            pt = gmsh.model.geo.addPoint(
                float(x_full[i]), float(r_full[i]), 0,
            )
            body_pts.append(pt)

        # Upper body curves (nose -> base, left to right)
        upper_body_lines: list[int] = []
        for i in range(len(x_upper) - 1):
            line = gmsh.model.geo.addLine(body_pts[i], body_pts[i + 1])
            upper_body_lines.append(line)

        # Base closing line: upper base point -> lower base point
        base_upper_idx = len(x_upper) - 1  # index 199
        base_lower_idx = len(x_upper)      # index 200
        base_close = gmsh.model.geo.addLine(
            body_pts[base_upper_idx], body_pts[base_lower_idx],
        )

        # Lower body curves (base -> nose, right to left)
        lower_body_lines: list[int] = []
        for i in range(base_lower_idx, n_body - 1):
            line = gmsh.model.geo.addLine(body_pts[i], body_pts[i + 1])
            lower_body_lines.append(line)

        # Nose closing line: last lower point back to nose
        nose_close = gmsh.model.geo.addLine(body_pts[-1], body_pts[0])

        # ============================================================
        #  RECTANGULAR FARFIELD (4 corners)
        # ============================================================
        far_bl = gmsh.model.geo.addPoint(x_min, y_min, 0)  # bottom-left
        far_br = gmsh.model.geo.addPoint(x_max, y_min, 0)  # bottom-right
        far_tr = gmsh.model.geo.addPoint(x_max, y_max, 0)  # top-right
        far_tl = gmsh.model.geo.addPoint(x_min, y_max, 0)  # top-left

        # Rectangle edges (CCW order)
        rect_left = gmsh.model.geo.addLine(far_tl, far_bl)    # top-left -> bottom-left
        rect_bottom = gmsh.model.geo.addLine(far_bl, far_br)  # left -> right
        rect_right = gmsh.model.geo.addLine(far_br, far_tr)   # bottom-right -> top-right
        rect_top = gmsh.model.geo.addLine(far_tr, far_tl)     # right -> left

        # ============================================================
        #  CURVE LOOPS AND SURFACE
        # ============================================================
        # Outer loop: farfield rectangle (CCW)
        outer_loop = gmsh.model.geo.addCurveLoop([
            rect_bottom,   # far_bl -> far_br (right)
            rect_right,    # far_br -> far_tr (up)
            rect_top,      # far_tr -> far_tl (left)
            rect_left,     # far_tl -> far_bl (down)
        ])

        # Inner loop: body contour (CW = upper -> base_close -> lower -> nose_close)
        # CW direction: right along top, down at base, left along bottom, up at nose
        body_loop = gmsh.model.geo.addCurveLoop(
            upper_body_lines + [base_close] + lower_body_lines + [nose_close],
        )

        # Fluid surface: region between farfield (outer) and body (inner hole)
        surface = gmsh.model.geo.addPlaneSurface([outer_loop, body_loop])

        gmsh.model.geo.synchronize()

        # ============================================================
        #  PHYSICAL GROUPS (SU2 markers)
        #  NO sym marker -- full 2D, not axisymmetric.
        # ============================================================
        # body: entire closed body surface
        gmsh.model.geo.addPhysicalGroup(
            1,
            upper_body_lines + [base_close] + lower_body_lines + [nose_close],
            name="body",
        )

        # farfield: all four rectangle edges
        gmsh.model.geo.addPhysicalGroup(
            1, [rect_left, rect_bottom, rect_right, rect_top], name="farfield",
        )

        # fluid: the 2D surface
        gmsh.model.geo.addPhysicalGroup(2, [surface], name="fluid")

        gmsh.model.geo.synchronize()

        # ============================================================
        #  MESH GENERATION
        # ============================================================
        gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 5.0)
        gmsh.option.setNumber("Mesh.Smoothing", 50)
        gmsh.model.mesh.generate(2)

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


def generate_rectangular_mesh(
    dxf_path: Path,
    output_path: Path,
) -> Path:
    """Generate a simple rectangular domain mesh with Apollo CM body.

    Creates a straightforward rectangular farfield boundary with the body
    contour integrated into the outer boundary. No O-grid, no boundary
    layer layers -- just triangles filling the domain between body and
    farfield.

    The body sits on the symmetry axis (r=0) at the nose. The fluid
    domain boundary traces: farfield rectangle -> downstream axis ->
    body base closing -> body surface (reversed) -> upstream axis.
    This avoids a hole whose closing line would cut through the body.

    Domain sizing (10x body dimensions):
        Upstream:   10 * R_nose = 46.94 m from nose
        Downstream: 10 * body_length = 33.92 m from base
        Lateral:    10 * max_radius = 19.56 m from axis

    Args:
        dxf_path: Path to the ``.npz`` file containing contour arrays
            with keys ``x`` and ``r``.
        output_path: Output ``.su2`` mesh file path.

    Returns:
        Path to the generated ``.su2`` mesh file.

    Raises:
        RuntimeError: If mesh generation fails.
    """
    import gmsh

    # --- Apollo CM geometry parameters ---
    R_nose = 4.694  # m (heat shield sphere radius)
    body_length = 3.3918  # m
    max_radius = 1.956  # m

    # --- Domain sizing (10x body dimensions) ---
    upstream = 10.0 * R_nose  # 46.94 m from nose
    downstream = 10.0 * body_length  # 33.92 m from base
    lateral = 10.0 * max_radius  # 19.56 m from axis

    x_min = -upstream  # -46.94 m (nose at x=0)
    x_max = body_length + downstream  # 37.31 m
    r_max = lateral  # 19.56 m

    # --- Load body contour from DXF ---
    data = np.load(dxf_path)
    x_body: np.ndarray = data["x"]
    r_body: np.ndarray = data["r"]
    n_body = len(x_body)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("rectangular_mesh")

        # ============================================================
        #  RECTANGULAR FARFIELD (4 corners)
        # ============================================================
        far_bl = gmsh.model.geo.addPoint(x_min, 0.0, 0)   # bottom-left
        far_br = gmsh.model.geo.addPoint(x_max, 0.0, 0)   # bottom-right
        far_tr = gmsh.model.geo.addPoint(x_max, r_max, 0)  # top-right
        far_tl = gmsh.model.geo.addPoint(x_min, r_max, 0)  # top-left

        # Rectangle edges (defined in natural direction, will be
        # referenced in the loop with appropriate signs)
        rect_left = gmsh.model.geo.addLine(far_tl, far_bl)   # top-left -> bottom-left
        rect_bottom = gmsh.model.geo.addLine(far_bl, far_br)  # left -> right
        rect_right = gmsh.model.geo.addLine(far_br, far_tr)   # bottom-right -> top-right
        rect_top = gmsh.model.geo.addLine(far_tr, far_tl)     # right -> left

        # ============================================================
        #  BODY CONTOUR (individual lines, not spline)
        # ============================================================
        body_pts: list[int] = []
        for i in range(n_body):
            pt = gmsh.model.geo.addPoint(
                float(x_body[i]), float(r_body[i]), 0,
            )
            body_pts.append(pt)

        body_lines: list[int] = []
        for i in range(n_body - 1):
            line = gmsh.model.geo.addLine(body_pts[i], body_pts[i + 1])
            body_lines.append(line)

        # ============================================================
        #  AXIS AND BASE POINTS
        # ============================================================
        # Point on the symmetry axis directly below the body base
        axis_base = gmsh.model.geo.addPoint(
            float(x_body[-1]), 0.0, 0,
        )

        # Upstream axis: from farfield bottom-left to body nose (east)
        axis_upstream = gmsh.model.geo.addLine(far_bl, body_pts[0])

        # Downstream axis: from body base axis point to farfield right (east)
        axis_downstream = gmsh.model.geo.addLine(axis_base, far_br)

        # Base closing: vertical line from body base down to axis (south)
        base_close = gmsh.model.geo.addLine(body_pts[-1], axis_base)

        # ============================================================
        #  CURVE LOOP AND SURFACE (single CCW loop, no holes)
        # ============================================================
        # The fluid domain boundary, traversed CCW (interior on LEFT):
        #   far_bl -> body[0]  (east along upstream axis, fluid above)
        #   body[0] -> body[-1] (east along body surface, fluid above)
        #   body[-1] -> axis_base (south along base closing, fluid right)
        #   axis_base -> far_br (east along downstream axis, fluid above)
        #   far_br -> far_tr (north on right edge, fluid left)
        #   far_tr -> far_tl (west on top edge, fluid below)
        #   far_tl -> far_bl (south on left edge, fluid right)
        outer_loop = gmsh.model.geo.addCurveLoop([
            axis_upstream,              # far_bl -> body[0]
            *body_lines,                # body[0] -> body[-1]
            base_close,                 # body[-1] -> axis_base
            axis_downstream,            # axis_base -> far_br
            rect_right,                 # far_br -> far_tr
            rect_top,                   # far_tr -> far_tl
            rect_left,                  # far_tl -> far_bl
        ])

        surface = gmsh.model.geo.addPlaneSurface([outer_loop])

        # ============================================================
        #  PHYSICAL GROUPS (SU2 markers)
        #  Must be added BEFORE synchronize() using geo.addPhysicalGroup
        #  with the name= keyword for proper SU2 export.
        # ============================================================
        # body: body contour lines + base closing (entire body surface)
        gmsh.model.geo.addPhysicalGroup(
            1, body_lines + [base_close], name="body",
        )

        # farfield: left, top, right edges (inflow/outflow/lateral)
        gmsh.model.geo.addPhysicalGroup(
            1, [rect_left, rect_top, rect_right], name="farfield",
        )

        # sym: upstream and downstream axis segments
        gmsh.model.geo.addPhysicalGroup(
            1, [axis_upstream, axis_downstream], name="sym",
        )

        # fluid: the surface
        gmsh.model.geo.addPhysicalGroup(2, [surface], name="fluid")

        gmsh.model.geo.synchronize()

        # ============================================================
        #  MESH GENERATION
        # ============================================================
        gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 5.0)
        gmsh.option.setNumber("Mesh.Smoothing", 50)
        gmsh.model.mesh.generate(2)

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
