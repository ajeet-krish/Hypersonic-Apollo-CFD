"""Tests for preset blunt body configurations."""
from geometry.presets import apollo_cm, generic


class TestPresets:
    """Tests for preset blunt body configurations."""

    def test_apollo_cm_R_nose(self):
        """Apollo CM should have R_nose=0.196."""
        config = apollo_cm()
        assert config.R_nose == 0.196

    def test_apollo_cm_half_angle(self):
        """Apollo CM should have half_angle=50.0."""
        config = apollo_cm()
        assert config.half_angle == 50.0

    def test_apollo_cm_base_radius(self):
        """Apollo CM should have base_radius=1.955."""
        config = apollo_cm()
        assert config.base_radius == 1.955

    def test_apollo_cm_num_points(self):
        """Apollo CM should have num_points=400."""
        config = apollo_cm()
        assert config.num_points == 400

    def test_apollo_cm_passes_validation(self):
        """Apollo CM preset should pass BluntBodyConfig.validate()."""
        from geometry.config import BluntBodyConfig
        config = apollo_cm()
        validated = BluntBodyConfig.validate(**config.__dict__)
        assert validated.R_nose == 0.196

    def test_generic_defaults(self):
        """Generic preset should have correct defaults."""
        config = generic()
        assert config.R_nose == 0.1
        assert config.half_angle == 45.0
        assert config.base_radius == 0.5

    def test_generic_custom(self):
        """Generic preset should accept custom parameters."""
        config = generic(R_shield=0.2, cone_half_angle=60.0, base_radius=1.0)
        assert config.R_nose == 0.2
        assert config.half_angle == 60.0
        assert config.base_radius == 1.0

    def test_generic_num_points(self):
        """Generic preset should default to 300 points."""
        config = generic()
        assert config.num_points == 300

    def test_generic_passes_validation(self):
        """Generic preset should pass BluntBodyConfig.validate()."""
        from geometry.config import BluntBodyConfig
        config = generic()
        validated = BluntBodyConfig.validate(**config.__dict__)
        assert validated.R_nose == 0.1

    def test_apollo_cm_docstring_source(self):
        """Apollo CM docstring should reference Graves & Witte 1963."""
        assert "Graves" in apollo_cm.__doc__
        assert "Witte" in apollo_cm.__doc__
