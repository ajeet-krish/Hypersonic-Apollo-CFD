"""Gmsh mesh generation configuration for blunt body CFD."""
from dataclasses import dataclass

# Tier multipliers relative to standard cell counts
_TIER_MULTIPLIERS: dict[str, float] = {
    "draft": 0.5,
    "standard": 1.0,
    "high": 2.0,
}


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
        """Return effective first cell height, falling back to 1e-6 * R_nose.

        For y+ < 1 with typical hypersonic Re numbers, the first cell height
        must be ~1e-6 * R_nose (not 1e-3 which was the old default).

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
            }
            field_dict.update(overrides)  # type: ignore[arg-type]
            config = cls(**field_dict)

        return config
