"""Tests for MeshConfig frozen dataclass."""
import pytest

from cfd.mesh_config import MeshConfig


class TestMeshConfigDefaults:
    """Tests for MeshConfig default values."""

    def test_default_values(self):
        """Verify default configuration values."""
        config = MeshConfig()
        assert config.n_bl == 30
        assert config.first_cell_height is None
        assert config.bl_growth_ratio == 1.10
        assert config.n_axial_nose == 60
        assert config.n_axial_cone == 80
        assert config.n_radial == 80
        assert config.shock_refinement is True
        assert config.shock_standoff_factor == 1.5
        assert config.farfield_distance == 25.0
        assert config.mesh_tier == "standard"

    def test_frozen_dataclass(self):
        """MeshConfig should be immutable."""
        config = MeshConfig()
        with pytest.raises(AttributeError):
            config.n_bl = 50  # type: ignore[misc]


class TestMeshConfigForTier:
    """Tests for MeshConfig.for_tier() factory method."""

    def test_draft_tier(self):
        """Draft tier should have roughly half the cells of standard."""
        config = MeshConfig.for_tier("draft")
        assert config.mesh_tier == "draft"
        assert config.n_bl == 15
        assert config.n_axial_nose == 30
        assert config.n_axial_cone == 40
        assert config.n_radial == 40

    def test_standard_tier(self):
        """Standard tier should use default cell counts."""
        config = MeshConfig.for_tier("standard")
        assert config.mesh_tier == "standard"
        assert config.n_bl == 30
        assert config.n_axial_nose == 60
        assert config.n_axial_cone == 80
        assert config.n_radial == 80

    def test_high_tier(self):
        """High tier should have roughly double the cells of standard."""
        config = MeshConfig.for_tier("high")
        assert config.mesh_tier == "high"
        assert config.n_bl == 60
        assert config.n_axial_nose == 120
        assert config.n_axial_cone == 160
        assert config.n_radial == 160

    def test_invalid_tier(self):
        """Invalid tier should raise ValueError."""
        with pytest.raises(ValueError, match="tier must be"):
            MeshConfig.for_tier("ultra")

    def test_for_tier_with_overrides(self):
        """for_tier should accept field overrides."""
        config = MeshConfig.for_tier("draft", n_bl=20, first_cell_height=1e-6)
        assert config.n_bl == 20
        assert config.first_cell_height == 1e-6
        assert config.mesh_tier == "draft"

    def test_cell_targets_draft(self):
        """Draft tier should target roughly 50-80K cells."""
        config = MeshConfig.for_tier("draft")
        est = config.estimated_cell_count
        assert 1000 <= est <= 200000, f"Draft estimate {est} out of range"

    def test_cell_targets_standard(self):
        """Standard tier should target roughly 150-250K cells."""
        config = MeshConfig.for_tier("standard")
        est = config.estimated_cell_count
        assert 5000 <= est <= 500000, f"Standard estimate {est} out of range"

    def test_cell_targets_high(self):
        """High tier should target roughly 400-500K cells."""
        config = MeshConfig.for_tier("high")
        est = config.estimated_cell_count
        assert 10000 <= est <= 1000000, f"High estimate {est} out of range"


class TestMeshConfigResolveFirstCellHeight:
    """Tests for resolve_first_cell_height method."""

    def test_auto_height(self):
        """When first_cell_height is None, should return 1e-5 * R_nose."""
        config = MeshConfig()
        result = config.resolve_first_cell_height(0.1)
        assert result == pytest.approx(1e-6)

    def test_explicit_height(self):
        """When first_cell_height is set, should return it directly."""
        config = MeshConfig(first_cell_height=5e-7)
        result = config.resolve_first_cell_height(0.1)
        assert result == pytest.approx(5e-7)

    def test_auto_height_with_apollo(self):
        """Auto height for Apollo CM R_nose = 0.196 m."""
        config = MeshConfig()
        result = config.resolve_first_cell_height(0.196)
        assert result == pytest.approx(1.96e-6)


class TestMeshConfigValidation:
    """Tests for MeshConfig validation."""

    def test_n_bl_too_small(self):
        """n_bl < 10 should raise ValueError."""
        with pytest.raises(ValueError, match="n_bl must be >= 10"):
            MeshConfig(n_bl=5)

    def test_bl_growth_ratio_too_low(self):
        """bl_growth_ratio < 1.01 should raise ValueError."""
        with pytest.raises(ValueError, match="bl_growth_ratio must be"):
            MeshConfig(bl_growth_ratio=1.0)

    def test_bl_growth_ratio_too_high(self):
        """bl_growth_ratio > 1.5 should raise ValueError."""
        with pytest.raises(ValueError, match="bl_growth_ratio must be"):
            MeshConfig(bl_growth_ratio=1.6)

    def test_invalid_mesh_tier(self):
        """Invalid mesh_tier should raise ValueError."""
        with pytest.raises(ValueError, match="mesh_tier must be"):
            MeshConfig(mesh_tier="ultra")

    def test_n_axial_nose_too_small(self):
        """n_axial_nose < 10 should raise ValueError."""
        with pytest.raises(ValueError, match="n_axial_nose must be >= 10"):
            MeshConfig(n_axial_nose=5)

    def test_n_axial_cone_too_small(self):
        """n_axial_cone < 10 should raise ValueError."""
        with pytest.raises(ValueError, match="n_axial_cone must be >= 10"):
            MeshConfig(n_axial_cone=5)

    def test_n_radial_too_small(self):
        """n_radial < 10 should raise ValueError."""
        with pytest.raises(ValueError, match="n_radial must be >= 10"):
            MeshConfig(n_radial=5)


class TestMeshConfigProperties:
    """Tests for MeshConfig computed properties."""

    def test_total_axial_cells(self):
        """total_axial_cells = n_axial_nose + n_axial_cone."""
        config = MeshConfig(n_axial_nose=60, n_axial_cone=80)
        assert config.total_axial_cells == 140

    def test_estimated_cell_count(self):
        """estimated_cell_count = total_axial * n_radial."""
        config = MeshConfig(n_axial_nose=60, n_axial_cone=80, n_radial=80)
        assert config.estimated_cell_count == 140 * 80
