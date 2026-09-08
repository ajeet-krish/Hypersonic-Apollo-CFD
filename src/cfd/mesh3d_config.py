"""Configuration for 3D mesh generation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Mesh3DConfig:
    """Configuration for 3D Gmsh mesh generation.

    Attributes:
        mesh_tier: Refinement tier (draft, standard, high).
        farfield_radius_factor: Farfield radius as multiple of body length.
        boundary_layers: Number of prism layers in boundary layer.
        first_cell_height: Height of first cell off the wall (m). None = auto.
        bl_growth_ratio: Geometric growth ratio for boundary layer cells.
        max_element_size: Maximum element size in the farfield (m).
        min_element_size: Minimum element size near the body (m).
    """

    mesh_tier: str = "draft"
    farfield_radius_factor: float = 20.0
    boundary_layers: int = 20
    first_cell_height: float | None = None
    bl_growth_ratio: float = 1.2
    max_element_size: float = 2.0
    min_element_size: float = 0.01

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        if self.mesh_tier not in ("draft", "standard", "high"):
            raise ValueError(
                f"mesh_tier must be 'draft', 'standard', or 'high', got '{self.mesh_tier}'"
            )
        if self.farfield_radius_factor < 5.0:
            raise ValueError(
                f"farfield_radius_factor must be >= 5.0, got {self.farfield_radius_factor}"
            )
        if self.boundary_layers < 5:
            raise ValueError(
                f"boundary_layers must be >= 5, got {self.boundary_layers}"
            )
        if self.bl_growth_ratio < 1.01 or self.bl_growth_ratio > 1.5:
            raise ValueError(
                f"bl_growth_ratio must be in [1.01, 1.5], got {self.bl_growth_ratio}"
            )

    @property
    def effective_first_cell_height(self) -> float:
        """Return effective first cell height, targeting y+ < 1 for hypersonic RANS.

        For y+ < 1 with typical hypersonic Re numbers, the first cell height
        must be ~1e-6 * R_nose.
        """
        if self.first_cell_height is not None and self.first_cell_height > 0:
            return self.first_cell_height
        return 1e-6  # Default for hypersonic RANS

    @classmethod
    def for_tier(cls, tier: str, **overrides: object) -> "Mesh3DConfig":
        """Create Mesh3DConfig scaled for the given refinement tier.

        Tier targets (approximate 3D cell counts):
            draft:    ~2-3M cells
            standard: ~10-15M cells
            high:     ~40-50M cells

        Args:
            tier: Refinement tier (draft, standard, high).
            **overrides: Any Mesh3DConfig field to override after scaling.

        Returns:
            Mesh3DConfig with parameters scaled for the tier.
        """
        tier_params: dict[str, dict[str, float]] = {
            "draft": {
                "max_element_size": 2.0,
                "min_element_size": 0.05,
                "boundary_layers": 15,
            },
            "standard": {
                "max_element_size": 1.0,
                "min_element_size": 0.01,
                "boundary_layers": 20,
            },
            "high": {
                "max_element_size": 0.5,
                "min_element_size": 0.005,
                "boundary_layers": 30,
            },
        }

        params = tier_params.get(tier, tier_params["draft"])
        params.update(overrides)  # type: ignore[arg-type]

        return cls(mesh_tier=tier, **params)  # type: ignore[arg-type]
