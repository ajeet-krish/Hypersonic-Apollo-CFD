"""Gmsh shock-aligned mesh generation for spherically-blunted cones.

Produces a 2D axisymmetric mesh in the (x, r) plane with:
  - Boundary-layer refinement at the body wall (Gmsh BoundaryLayer field)
  - Shock-region refinement based on Billig standoff correlation
  - Sphere-cone junction refinement
  - Farfield boundary at configurable distance
  - SU2-compatible physical group markers

References:
    Billig, F. S. (1967), "Shock-Wave Shapes Around Unswept- and
    Swept-Nose Bodies," J. Spacecraft & Rockets, 4(6), 822-823.
"""
from pathlib import Path

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


def _add_boundary_layer(
    body_curve_tags: list[int],
    surface_tag: int,
    mesh_config: MeshConfig,
    R_nose: float,
    bl_field_tag: int,
) -> None:
    """Add Gmsh BoundaryLayer field for wall-normal refinement.

    Uses the BoundaryLayer field with geometric spacing to produce
    first_cell_height at the wall with bl_growth_ratio progression.

    The total boundary layer thickness is computed from the desired
    first cell height, number of layers, and growth ratio:
        Thickness = h1 * (1 - ratio^N) / (1 - ratio)

    Args:
        body_curve_tags: Tags of the body wall curves.
        surface_tag: Gmsh surface tag (for recombine to quads).
        mesh_config: Mesh configuration.
        R_nose: Nose sphere radius (m).
        bl_field_tag: Field ID for the boundary layer.
    """
    import gmsh

    first_h = mesh_config.resolve_first_cell_height(R_nose)
    n_layers = mesh_config.n_bl
    ratio = mesh_config.bl_growth_ratio

    # Compute total BL thickness from geometric series:
    # Thickness = h1 * (1 - ratio^N) / (1 - ratio)
    if abs(ratio - 1.0) < 1e-10:
        thickness = first_h * n_layers
    else:
        thickness = first_h * (1.0 - ratio**n_layers) / (1.0 - ratio)

    gmsh.model.mesh.field.add("BoundaryLayer", bl_field_tag)
    gmsh.model.mesh.field.setNumbers(
        bl_field_tag, "CurvesList", body_curve_tags,
    )
    gmsh.model.mesh.field.setNumber(bl_field_tag, "Quads", 1)
    gmsh.model.mesh.field.setNumber(bl_field_tag, "NbLayers", n_layers)
    gmsh.model.mesh.field.setNumber(bl_field_tag, "Thickness", thickness)
    gmsh.model.mesh.field.setNumber(bl_field_tag, "ratio", ratio)

    # Enable quad elements in the BL region
    gmsh.model.geo.mesh.setRecombine(2, surface_tag)

    # Allow high anisotropy in the BL (no fan element limit)
    gmsh.option.setNumber("Mesh.BoundaryLayerFanElements", 0)
    gmsh.option.setNumber("Mesh.AnisoMax", 1e6)


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

    # Ball field: refined zone around the shock standoff region.
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

    The junction where the spherical nose meets the conical frustum
    requires smaller cells to capture the geometric curvature change.

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
        - Boundary-layer refinement at the body wall
        - Shock-region refinement from Billig correlation
        - Sphere-cone junction refinement

    The mesh is exported as an SU2-compatible .su2 file.

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

        # --- Farfield bounds ---
        x_min, x_max, r_max = _build_domain_points(config, mesh_config)

        # --- Create body wall points and spline ---
        body_pts: list[int] = []
        for i in range(len(x_body)):
            pt = gmsh.model.geo.addPoint(float(x_body[i]), float(r_body[i]), 0)
            body_pts.append(pt)
        body_spline = gmsh.model.geo.addSpline(body_pts)

        # --- Farfield boundary points ---
        bl = gmsh.model.geo.addPoint(x_min, 0.0, 0)
        tl = gmsh.model.geo.addPoint(x_min, r_max, 0)
        tr = gmsh.model.geo.addPoint(x_max, r_max, 0)
        br = gmsh.model.geo.addPoint(x_max, 0.0, 0)

        # --- Boundary curves ---
        upstream_line = gmsh.model.geo.addLine(bl, body_pts[0])
        downstream_line = gmsh.model.geo.addLine(body_pts[-1], br)
        far_top = gmsh.model.geo.addLine(tr, tl)
        far_left = gmsh.model.geo.addLine(bl, tl)
        far_right = gmsh.model.geo.addLine(br, tr)

        # --- Surface loop and surface ---
        surface_loop = gmsh.model.geo.addCurveLoop([
            upstream_line,      # bl -> body_pts[0]
            body_spline,        # body_pts[0] -> body_pts[-1]
            downstream_line,    # body_pts[-1] -> br
            far_right,          # br -> tr
            far_top,            # tr -> tl
            -far_left,          # tl -> bl  (reverse of bl -> tl)
        ])
        surface = gmsh.model.geo.addPlaneSurface([surface_loop])

        # --- Physical groups (SU2 markers) ---
        gmsh.model.geo.addPhysicalGroup(1, [body_spline], name="body")
        gmsh.model.geo.addPhysicalGroup(
            1, [far_left, far_top, far_right, downstream_line],
            name="farfield",
        )
        # Symmetry axis at r=0: upstream_line goes from farfield left
        # (r=0) to body nose tip (r=0) and IS in the surface loop.
        # The old axis_line (br->bl) was NOT in the surface loop and
        # produced invalid node IDs (-1) in the SU2 mesh export.
        gmsh.model.geo.addPhysicalGroup(1, [upstream_line], name="sym")
        gmsh.model.geo.addPhysicalGroup(2, [surface], name="fluid")

        # --- Synchronize geometry before setting up mesh fields ---
        gmsh.model.geo.synchronize()

        # --- Tier-based mesh size multiplier ---
        # draft=2.0 (coarser, fewer cells), standard=1.0, high=0.5 (finer, more)
        tier_mult = _TIER_SIZE_MULTIPLIERS[mesh_config.mesh_tier]
        first_h = mesh_config.resolve_first_cell_height(R_nose)

        # --- Global mesh size constraints ---
        # CharacteristicLengthMin must be smaller than the BL first cell
        # height so gmsh does not clamp BL element sizes to the global min.
        body_length = config.computed_body_length
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5 * first_h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.0 * body_length * tier_mult)

        # --- Background mesh (base cell size, scaled by tier) ---
        bg_tag = 100
        gmsh.model.mesh.field.add("Constant", bg_tag)
        bg_vin = 0.2 * body_length * tier_mult
        gmsh.model.mesh.field.setNumber(bg_tag, "VIn", bg_vin)
        gmsh.model.mesh.field.setNumber(bg_tag, "VOut", bg_vin)

        # --- Boundary layer refinement ---
        bl_tag = 200
        _add_boundary_layer([body_spline], surface, mesh_config, R_nose, bl_tag)

        # --- Shock refinement ---
        shock_tag = 300
        if mesh_config.shock_refinement:
            _add_shock_refinement(
                config, mach, mesh_config, shock_tag, tier_mult,
            )

        # --- Junction refinement ---
        junc_tag = 400
        _add_junction_refinement(config, mesh_config, junc_tag, tier_mult)

        # --- Combine fields with Min ---
        min_tag = 999
        field_ids = [bg_tag, bl_tag]
        if mesh_config.shock_refinement:
            field_ids.append(shock_tag)
        field_ids.append(junc_tag)

        gmsh.model.mesh.field.add("Min", min_tag)
        gmsh.model.mesh.field.setNumbers(min_tag, "FieldsList", field_ids)
        gmsh.model.mesh.field.setAsBackgroundMesh(min_tag)

        # --- Generate mesh ---
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
