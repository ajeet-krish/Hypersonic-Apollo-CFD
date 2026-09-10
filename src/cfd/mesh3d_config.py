"""Configuration for 3D cylindrical wind tunnel mesh generation.

Provides configuration for generating 3D tetrahedral meshes with
prismatic boundary layers in a cylindrical domain around hypersonic
blunt body geometries.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Mesh3DConfig:
    """Configuration for 3D cylindrical wind tunnel mesh.

    Attributes:
        mesh_tier: Refinement tier (draft, standard, high).
        upstream_factor: Upstream distance as multiple of R_nose.
        downstream_factor: Downstream distance as multiple of body diameter.
        lateral_factor: Cylinder radius as multiple of R_nose.
        boundary_layers: Number of prismatic BL layers.
        first_cell_height: First cell height off wall (m). None = auto.
        bl_growth_ratio: Geometric growth ratio for BL cells.
        min_element_size: Minimum element size near body (m).
        max_element_size: Maximum element size in farfield (m).
        use_symmetry: Use y=0 symmetry plane for 0 deg AoA.
        mach: Freestream Mach number for shock standoff sizing.
    """

    mesh_tier: str = "draft"
    upstream_factor: float = 10.0
    downstream_factor: float = 20.0
    lateral_factor: float = 8.0
    boundary_layers: int = 40
    first_cell_height: float | None = None
    bl_growth_ratio: float = 1.10
    min_element_size: float = 0.005
    max_element_size: float = 2.0
    use_symmetry: bool = True
    mach: float = 15.6

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        if self.mesh_tier not in ("draft", "standard", "high"):
            raise ValueError(
                f"mesh_tier must be 'draft', 'standard', or 'high', got '{self.mesh_tier}'"
            )
        if self.upstream_factor < 5.0:
            raise ValueError(
                f"upstream_factor must be >= 5.0, got {self.upstream_factor}"
            )
        if self.downstream_factor < 10.0:
            raise ValueError(
                f"downstream_factor must be >= 10.0, got {self.downstream_factor}"
            )
        if self.lateral_factor < 3.0:
            raise ValueError(
                f"lateral_factor must be >= 3.0, got {self.lateral_factor}"
            )
        if self.boundary_layers < 10:
            raise ValueError(
                f"boundary_layers must be >= 10, got {self.boundary_layers}"
            )
        if self.bl_growth_ratio < 1.01 or self.bl_growth_ratio > 1.5:
            raise ValueError(
                f"bl_growth_ratio must be in [1.01, 1.5], got {self.bl_growth_ratio}"
            )
        if self.min_element_size <= 0:
            raise ValueError(
                f"min_element_size must be > 0, got {self.min_element_size}"
            )
        if self.max_element_size <= self.min_element_size:
            raise ValueError(
                f"max_element_size must be > min_element_size ({self.min_element_size}), "
                f"got {self.max_element_size}"
            )

    def effective_first_cell_height(self, R_nose: float) -> float:
        """Return effective first cell height for y+ < 1.

        For hypersonic RANS, y+ < 1 requires first cell height ~1e-6 * R_nose.

        Args:
            R_nose: Nose sphere radius (m).

        Returns:
            First cell height in meters.
        """
        if self.first_cell_height is not None and self.first_cell_height > 0:
            return self.first_cell_height
        return 1e-6 * R_nose

    @classmethod
    def for_tier(cls, tier: str, **overrides: object) -> "Mesh3DConfig":
        """Create Mesh3DConfig scaled for the given refinement tier.

        Tier targets (approximate 3D cell counts, sizes in mm):
            draft:    ~2-5M cells   (min=50mm, max=2000mm)
            standard: ~10-15M cells (min=20mm, max=1000mm)
            high:     ~30-50M cells (min=5mm, max=500mm)

        Args:
            tier: Refinement tier (draft, standard, high).
            **overrides: Any Mesh3DConfig field to override.

        Returns:
            Mesh3DConfig with parameters scaled for the tier.
        """
        tier_params: dict[str, dict[str, object]] = {
            "draft": {
                "min_element_size": 50.0,    # mm - coarse, ~2-5M elements
                "max_element_size": 2000.0,  # mm
                "boundary_layers": 20,
            },
            "standard": {
                "min_element_size": 20.0,    # mm - moderate, ~10-15M elements
                "max_element_size": 1000.0,  # mm
                "boundary_layers": 40,
            },
            "high": {
                "min_element_size": 5.0,     # mm - fine, ~30-50M elements
                "max_element_size": 500.0,   # mm
                "boundary_layers": 60,
            },
        }

        params = tier_params.get(tier, tier_params["draft"])
        # Apply overrides (remove None values)
        clean_overrides = {k: v for k, v in overrides.items() if v is not None}
        params.update(clean_overrides)  # type: ignore[arg-type]

        return cls(mesh_tier=tier, **params)  # type: ignore[arg-type]
