"""Tests for CaseConfig and PipelineStage."""
import pytest

from geometry.presets import apollo_cm, generic
from pipeline.case_config import CaseConfig, PipelineStage


class TestCaseConfig:
    """Tests for CaseConfig dataclass."""

    def test_default_values(self):
        """Verify default configuration values."""
        config = CaseConfig(
            name="test", label="Test", preset_fn=generic,
        )
        assert config.mach == 8.0
        assert config.altitude == 30000.0
        assert config.gamma == 1.4
        assert config.mesh_tier == "standard"
        assert config.su2_iterations == 5000
        assert config.su2_cfl == 1.0

    def test_output_dir(self):
        """output_dir should be output/{name}."""
        config = CaseConfig(
            name="apollo-cm", label="Apollo CM", preset_fn=apollo_cm,
        )
        assert config.output_dir == "output/apollo-cm"

    def test_images_dir(self):
        """images_dir should be docs/assets/images/{name}."""
        config = CaseConfig(
            name="apollo-cm", label="Apollo CM", preset_fn=apollo_cm,
        )
        assert config.images_dir == "docs/assets/images/apollo-cm"

    def test_frozen_dataclass(self):
        """CaseConfig should be immutable (frozen)."""
        config = CaseConfig(
            name="test", label="Test", preset_fn=generic,
        )
        with pytest.raises(AttributeError):
            config.name = "changed"  # type: ignore[misc]

    def test_custom_values(self):
        """Should accept custom values."""
        config = CaseConfig(
            name="custom",
            label="Custom",
            preset_fn=generic,
            mach=12.0,
            altitude=50000.0,
            gamma=1.3,
        )
        assert config.mach == 12.0
        assert config.altitude == 50000.0
        assert config.gamma == 1.3

    def test_preset_fn_callable(self):
        """preset_fn should return BluntBodyConfig."""
        config = CaseConfig(
            name="apollo", label="Apollo", preset_fn=apollo_cm,
        )
        body = config.preset_fn()
        assert body.R_nose == 0.196

    def test_equal_configs(self):
        """Two configs with same params should be equal."""
        c1 = CaseConfig(name="a", label="A", preset_fn=generic)
        c2 = CaseConfig(name="a", label="A", preset_fn=generic)
        assert c1 == c2


class TestPipelineStage:
    """Tests for PipelineStage enum."""

    def test_geometry_value(self):
        """GEOMETRY should have value 'geometry'."""
        assert PipelineStage.GEOMETRY.value == "geometry"

    def test_mesh_value(self):
        """MESH should have value 'mesh'."""
        assert PipelineStage.MESH.value == "mesh"

    def test_su2_value(self):
        """SU2 should have value 'su2'."""
        assert PipelineStage.SU2.value == "su2"

    def test_postprocess_value(self):
        """POSTPROCESS should have value 'postprocess'."""
        assert PipelineStage.POSTPROCESS.value == "postprocess"

    def test_validation_value(self):
        """VALIDATION should have value 'validation'."""
        assert PipelineStage.VALIDATION.value == "validation"

    def test_gci_value(self):
        """GCI should have value 'gci'."""
        assert PipelineStage.GCI.value == "gci"

    def test_sweep_value(self):
        """SWEEP should have value 'sweep'."""
        assert PipelineStage.SWEEP.value == "sweep"

    def test_site_value(self):
        """SITE should have value 'site'."""
        assert PipelineStage.SITE.value == "site"

    def test_from_string(self):
        """Should construct from string value."""
        assert PipelineStage("geometry") == PipelineStage.GEOMETRY
        assert PipelineStage("mesh") == PipelineStage.MESH
