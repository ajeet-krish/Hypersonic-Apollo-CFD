"""Gmsh shock-aligned mesh generation for spherically-blunted cones.

Produces a 2D axisymmetric mesh in the (x, r) plane with:
  - Structured boundary-layer cells at the body wall (manual quad layers)
  - Shock-region refinement based on Billig standoff correlation
  - Sphere-cone junction refinement
  - Farfield boundary at configurable distance
  - SU2-compatible physical group markers

The boundary layer is built by explicit node placement: for each body
contour point, additional nodes are placed along the outward normal at
geometrically-spaced distances.  Quads connect consecutive layers.
This bypasses the gmsh BoundaryLayer field and transfinite surface,
both of which are non-functional in gmsh 4.15.2.

References:
    Billig, F. S. (1967), "Shock-Wave Shapes Around Unswept- and
    Swept-Nose Bodies," J. Spacecraft & Rockets, 4(6), 822-823.
"""
from pathlib import Path

import numpy as np

from geometry.blunt_body import generate_contour
from geometry.config import BluntBodyConfig
from validation.billig import billig_blunted_cone

from .mesh_config import MeshConfig

# Mesh size multipliers per tier (inverse of cell-count multipliers).
# Draft uses 2x larger cells (fewer total), high uses 0.5x (more total).
_TIER_SIZE_MULTIPLIERS: dict[str, float] = {
    "draft": 2.0,
    "standard": 1.0,
    "high": 0.5,
}


def _build_domain_points(
    config: BluntBodyConfig,
    mesh_config: MeshConfig,
) -> tuple[float, float, float]:
    """Compute farfield boundary box coordinates.

    Args:
        config: Blunt body geometry parameters.
        mesh_config: Mesh configuration.

    Returns:
        (x_min, x_max, r_max) bounding the farfield domain.
        r_min is always 0 (axis of symmetry).
    """
    R_nose = config.R_nose
    body_length = config.computed_body_length
    ff = mesh_config.farfield_distance

    x_min = -ff * R_nose
    x_max = body_length + ff * R_nose
    r_max = ff * R_nose

    return x_min, x_max, r_max


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


def generate_body_mesh(
    config: BluntBodyConfig,
    mesh_config: MeshConfig,
    mach: float,
    output_path: Path,
) -> Path:
    """Generate a 2D axisymmetric Gmsh mesh for a spherically-blunted cone.

    Creates the computational domain with:
        - Body contour (sphere + cone) from Phase 1 geometry
        - Farfield boundary at configurable distance
        - Symmetry axis along r=0
        - Structured boundary-layer cells at the body wall
        - Shock-region refinement from Billig correlation
        - Sphere-cone junction refinement

    The boundary layer is built by placing BL nodes at explicit positions
    along the outward normal from each body contour point.  Each axial
    segment gets a column of quad elements with geometric growth from
    the wall.  This guarantees the requested first cell height.

    Args:
        config: Blunt body geometry configuration.
        mesh_config: Mesh refinement configuration.
        mach: Freestream Mach number (for shock refinement).
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
        gmsh.model.add("blunt_body")

        # --- Body contour ---
        x_body, r_body = generate_contour(config)
        R_nose = config.R_nose
        n_body = len(x_body)

        # --- Farfield bounds ---
        x_min, x_max, r_max = _build_domain_points(config, mesh_config)

        # --- Boundary layer geometry ---
        first_h = mesh_config.resolve_first_cell_height(R_nose)
        n_bl = mesh_config.n_bl
        ratio = mesh_config.bl_growth_ratio

        # Compute outward normals at each body point
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
        # bl_nodes[i][k] = gmsh point tag for body point i, BL layer k
        # k=0 is the body surface, k=n_bl is the offset (BL outer edge)
        bl_nodes: list[list[int]] = []
        cumulative_h = np.zeros(n_bl + 1)
        for k in range(1, n_bl + 1):
            cumulative_h[k] = cumulative_h[k - 1] + first_h * ratio ** (k - 1)

        for i in range(n_body):
            layer_pts: list[int] = []
            for k in range(n_bl + 1):
                x = x_body[i] + normals[i, 0] * cumulative_h[k]
                r = r_body[i] + normals[i, 1] * cumulative_h[k]
                pt = gmsh.model.geo.addPoint(float(x), float(r), 0)
                layer_pts.append(pt)
            bl_nodes.append(layer_pts)

        # --- Create BL quad surfaces ---
        bl_surfaces: list[int] = []
        for i in range(n_body - 1):
            for k in range(n_bl):
                bl = bl_nodes[i][k]
                br = bl_nodes[i + 1][k]
                tr = bl_nodes[i + 1][k + 1]
                tl = bl_nodes[i][k + 1]
                loop = gmsh.model.geo.addCurveLoop([
                    gmsh.model.geo.addLine(bl, br),
                    gmsh.model.geo.addLine(br, tr),
                    gmsh.model.geo.addLine(tr, tl),
                    gmsh.model.geo.addLine(tl, bl),
                ])
                surf = gmsh.model.geo.addPlaneSurface([loop])
                bl_surfaces.append(surf)

        # --- Farfield boundary points ---
        bl_ff = gmsh.model.geo.addPoint(x_min, 0.0, 0)
        tl = gmsh.model.geo.addPoint(x_min, r_max, 0)
        tr = gmsh.model.geo.addPoint(x_max, r_max, 0)
        br = gmsh.model.geo.addPoint(x_max, 0.0, 0)

        # --- Outer domain: offset contour to farfield ---
        # Use the top row of BL nodes (layer n_bl) as the inner boundary
        outer_inner_pts = [bl_nodes[i][n_bl] for i in range(n_body)]

        # Upstream line: from farfield left to body nose (on the axis)
        upstream_line = gmsh.model.geo.addLine(bl_ff, bl_nodes[0][0])

        # Downstream line: from body tail (on the axis) to farfield right
        downstream_line = gmsh.model.geo.addLine(bl_nodes[-1][0], br)

        # Farfield boundary curves
        far_top = gmsh.model.geo.addLine(tr, tl)
        far_left = gmsh.model.geo.addLine(bl_ff, tl)
        far_right = gmsh.model.geo.addLine(br, tr)

        # Outer domain surface: complex boundary
        # For the outer domain, create a single surface with the offset
        # contour as the inner boundary and farfield as the outer boundary.
        # We'll create the outer surface as a single large surface.
        # The offset contour is the inner boundary (top of BL).
        # The farfield is the outer boundary.

        # Offset spline from BL top nodes
        offset_spline = gmsh.model.geo.addSpline(outer_inner_pts)

        # Outer surface loop
        outer_loop = gmsh.model.geo.addCurveLoop([
            upstream_line,       # farfield left -> body nose (axis)
            gmsh.model.geo.addLine(bl_nodes[0][0], outer_inner_pts[0]),
            offset_spline,       # along the offset contour
            gmsh.model.geo.addLine(outer_inner_pts[-1], bl_nodes[-1][0]),
            downstream_line,     # body tail (axis) -> farfield right
            far_right,           # farfield right side
            far_top,             # farfield top
            -far_left,           # farfield left side (reversed)
        ])
        outer_surface = gmsh.model.geo.addPlaneSurface([outer_loop])

        # --- Physical groups (SU2 markers) ---
        # Body marker: all bottom curves of BL (body surface)
        body_curves = []
        for i in range(n_body - 1):
            body_curves.append(
                gmsh.model.geo.addLine(bl_nodes[i][0], bl_nodes[i + 1][0])
            )
        gmsh.model.geo.addPhysicalGroup(1, body_curves, name="body")

        # Farfield marker
        gmsh.model.geo.addPhysicalGroup(
            1, [far_left, far_top, far_right, downstream_line],
            name="farfield",
        )

        # Symmetry axis: upstream line
        gmsh.model.geo.addPhysicalGroup(1, [upstream_line], name="sym")

        # Fluid domain: BL surfaces + outer surface
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

        # Combine fields with Min
        min_tag = 999
        field_ids: list[int] = [bg_tag]
        if mesh_config.shock_refinement:
            field_ids.append(shock_tag)
        field_ids.append(junc_tag)

        gmsh.model.mesh.field.add("Min", min_tag)
        gmsh.model.mesh.field.setNumbers(min_tag, "FieldsList", field_ids)
        gmsh.model.mesh.field.setAsBackgroundMesh(min_tag)

        # ============================================================
        #  MESH GENERATION
        # ============================================================
        gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.Smoothing", 10)
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
