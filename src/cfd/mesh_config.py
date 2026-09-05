"""Gmsh mesh generation configuration for blunt body CFD."""
from dataclasses import dataclass

# Tier multipliers relative to standard cell counts
_TIER_MULTIPLIERS: dict[str, float] = {
    "draft": 0.5,
    "standard": 1.0,
    "high": 2.0,
}


@dataclass(frozen=True)
class OGridDomain:
    """Parameters for an elliptical (O-grid) farfield boundary.

    The domain is an ellipse centered at (center_x, center_r) with
    semi-axes semi_major (axial) and semi_minor (radial).  The body
    sits inside this ellipse, and the mesh fills the annular region
    between the body contour (plus BL) and the ellipse.

    Attributes:
        center_x: Axial center of the ellipse (m).
        center_r: Radial center of the ellipse (m), typically 0 (symmetry axis).
        semi_major: Axial semi-axis (half-width) of the ellipse (m).
        semi_minor: Radial semi-axis (half-height) of the ellipse (m).
        x_nose: Axial coordinate of the body nose tip (m).
        x_base: Axial coordinate of the body base (m).
        r_base: Radial coordinate of the body base (m).
        x_min: Leftmost axial extent of the ellipse (m).
        x_max: Rightmost axial extent of the ellipse (m).
        r_max: Maximum radial extent of the ellipse (m).
    """

    center_x: float
    center_r: float
    semi_major: float
    semi_minor: float
    x_nose: float
    x_base: float
    r_base: float
    x_min: float
    x_max: float
    r_max: float


@dataclass(frozen=True)
class MeshConfig:
    """Configuration for Gmsh boundary-layer mesh on a spherically-blunted cone.

    Attributes:
        n_bl: Boundary layer cells normal to body wall.
        first_cell_height: Height of first cell off the wall (m). None = auto.
        bl_growth_ratio: Geometric growth ratio for boundary layer cells.
        n_axial_nose: Axial cells in the spherical nose region.
        n_axial_cone: Axial cells in the conical frustum region.
        n_radial: Radial cells from body wall to farfield.
        shock_refinement: Enable shock-region refinement using Billig standoff.
        shock_standoff_factor: Refinement zone = factor * delta (shock standoff).
        farfield_distance: Farfield boundary distance in x * R_nose units.
        mesh_tier: Refinement tier (draft, standard, high).
        domain_type: Mesh domain type ('axisymmetric' or 'full2d').
        upstream_factor: Upstream distance as multiple of R_nose.
        downstream_factor: Downstream distance as multiple of body diameter.
        lateral_factor: Lateral distance as multiple of R_nose.
    """

    n_bl: int = 50
    first_cell_height: float | None = None
    bl_growth_ratio: float = 1.10
    n_axial_nose: int = 60
    n_axial_cone: int = 80
    n_radial: int = 80
    shock_refinement: bool = True
    shock_standoff_factor: float = 1.5
    farfield_distance: float = 25.0
    mesh_tier: str = "standard"
    domain_type: str = "axisymmetric"
    upstream_factor: float = 8.0
    downstream_factor: float = 12.0
    lateral_factor: float = 8.0

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        if self.n_bl < 10:
            raise ValueError(
                f"n_bl must be >= 10, got {self.n_bl}"
            )
        if not (1.01 <= self.bl_growth_ratio <= 1.5):
            raise ValueError(
                f"bl_growth_ratio must be in [1.01, 1.5], got {self.bl_growth_ratio}"
            )
        if self.mesh_tier not in ("draft", "standard", "high"):
            raise ValueError(
                f"mesh_tier must be 'draft', 'standard', or 'high', got '{self.mesh_tier}'"
            )
        if self.domain_type not in ("axisymmetric", "full2d"):
            raise ValueError(
                f"domain_type must be 'axisymmetric' or 'full2d', got '{self.domain_type}'"
            )
        if self.n_axial_nose < 10:
            raise ValueError(
                f"n_axial_nose must be >= 10, got {self.n_axial_nose}"
            )
        if self.n_axial_cone < 10:
            raise ValueError(
                f"n_axial_cone must be >= 10, got {self.n_axial_cone}"
            )
        if self.n_radial < 10:
            raise ValueError(
                f"n_radial must be >= 10, got {self.n_radial}"
            )

    def resolve_first_cell_height(self, R_nose: float) -> float:
        """Return effective first cell height, targeting y+ < 1 for hypersonic RANS.

        For y+ < 1 with typical hypersonic Re numbers, the first cell height
        must be ~1e-6 * R_nose (not 1e-4 which gives y+ ~100-1000).

        Args:
            R_nose: Nose sphere radius (m).

        Returns:
            First cell height in meters.
        """
        if self.first_cell_height is not None and self.first_cell_height > 0:
            return self.first_cell_height
        return 1e-6 * R_nose

    @property
    def total_axial_cells(self) -> int:
        """Total number of axial cells along the body surface."""
        return self.n_axial_nose + self.n_axial_cone

    @property
    def estimated_cell_count(self) -> int:
        """Rough proxy estimate of total 2D cells (axial * radial counts).

        Note: Actual gmsh cell counts differ because the mesh is unstructured
        and controlled by size fields (BL, shock, junction refinement), not
        by structured grid counts. This estimate scales with tier to give a
        relative sense of mesh density between draft/standard/high.
        """
        return self.total_axial_cells * self.n_radial

    @property
    def effective_farfield_distance(self) -> float:
        """Farfield distance adjusted for domain type.

        For full2d domains the farfield is slightly larger to accommodate
        the mirrored body geometry without clipping the shock layer.
        """
        if self.domain_type == "full2d":
            return self.farfield_distance * 1.2
        return self.farfield_distance

    @classmethod
    def for_tier(cls, tier: str, **overrides: object) -> "MeshConfig":
        """Create MeshConfig scaled for the given refinement tier.

        Tier targets (approximate 2D cell counts):
            draft:    ~50-80K cells
            standard: ~150-250K cells
            high:     ~400-500K cells

        Args:
            tier: Refinement tier (draft, standard, high).
            **overrides: Any MeshConfig field to override after scaling.

        Returns:
            MeshConfig with cell counts scaled for the tier.
        """
        if tier not in _TIER_MULTIPLIERS:
            raise ValueError(
                f"tier must be 'draft', 'standard', or 'high', got '{tier}'"
            )
        mult = _TIER_MULTIPLIERS[tier]

        config = cls(
            n_bl=max(10, int(50 * mult)),
            first_cell_height=overrides.pop("first_cell_height", None),
            bl_growth_ratio=float(overrides.pop("bl_growth_ratio", 1.10)),
            n_axial_nose=max(10, int(60 * mult)),
            n_axial_cone=max(10, int(80 * mult)),
            n_radial=max(10, int(80 * mult)),
            shock_refinement=bool(overrides.pop("shock_refinement", True)),
            shock_standoff_factor=float(overrides.pop("shock_standoff_factor", 1.5)),
            farfield_distance=float(overrides.pop("farfield_distance", 25.0)),
            mesh_tier=tier,
            domain_type=str(overrides.pop("domain_type", "axisymmetric")),
            upstream_factor=float(overrides.pop("upstream_factor", 8.0)),
            downstream_factor=float(overrides.pop("downstream_factor", 12.0)),
            lateral_factor=float(overrides.pop("lateral_factor", 8.0)),
        )

        # Apply any remaining overrides via replace pattern
        # (frozen dataclass, so we rebuild)
        if overrides:
            field_dict = {
                "n_bl": config.n_bl,
                "first_cell_height": config.first_cell_height,
                "bl_growth_ratio": config.bl_growth_ratio,
                "n_axial_nose": config.n_axial_nose,
                "n_axial_cone": config.n_axial_cone,
                "n_radial": config.n_radial,
                "shock_refinement": config.shock_refinement,
                "shock_standoff_factor": config.shock_standoff_factor,
                "farfield_distance": config.farfield_distance,
                "mesh_tier": config.mesh_tier,
                "domain_type": config.domain_type,
                "upstream_factor": config.upstream_factor,
                "downstream_factor": config.downstream_factor,
                "lateral_factor": config.lateral_factor,
            }
            field_dict.update(overrides)  # type: ignore[arg-type]
            config = cls(**field_dict)

        return config
