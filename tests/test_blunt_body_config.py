"""Tests for BluntBodyConfig dataclass."""
import math

import pytest

from geometry.config import BluntBodyConfig


class TestBluntBodyConfig:
    """Tests for BluntBodyConfig properties and defaults."""

    def test_default_values(self):
        """Verify default configuration values."""
        config = BluntBodyConfig()
        assert config.R_nose == 0.196
        assert config.half_angle == 50.0
        assert config.base_radius == 1.955
        assert config.body_length == 0.0
        assert config.num_points == 300

    def test_half_angle_rad(self):
        """half_angle_rad should be radians conversion."""
        config = BluntBodyConfig(half_angle=50.0)
        assert config.half_angle_rad == pytest.approx(math.radians(50.0), rel=1e-10)

    def test_junction_x(self):
        """junction_x = R_nose * sin(half_angle_rad)."""
        config = BluntBodyConfig(R_nose=0.196, half_angle=50.0)
        theta = math.radians(50.0)
        expected = 0.196 * math.sin(theta)
        assert config.junction_x == pytest.approx(expected, rel=1e-10)

    def test_junction_r(self):
        """junction_r = R_nose * (1 - cos(half_angle_rad))."""
        config = BluntBodyConfig(R_nose=0.196, half_angle=50.0)
        theta = math.radians(50.0)
        expected = 0.196 * (1.0 - math.cos(theta))
        assert config.junction_r == pytest.approx(expected, rel=1e-10)

    def test_computed_body_length(self):
        """computed_body_length = (base_r - junction_r)/tan(theta) + junction_x."""
        config = BluntBodyConfig(R_nose=0.196, half_angle=50.0, base_radius=1.955)
        theta = math.radians(50.0)
        junction_r = 0.196 * (1.0 - math.cos(theta))
        junction_x = 0.196 * math.sin(theta)
        expected = (1.955 - junction_r) / math.tan(theta) + junction_x
        assert config.computed_body_length == pytest.approx(expected, rel=1e-10)

    def test_computed_base_radius_default(self):
        """Without body_length, computed_base_radius = base_radius."""
        config = BluntBodyConfig(base_radius=1.955)
        assert config.computed_base_radius == pytest.approx(1.955, rel=1e-10)

    def test_computed_base_radius_with_body_length(self):
        """With body_length set, computed_base_radius uses it."""
        config = BluntBodyConfig(
            R_nose=0.196, half_angle=50.0,
            base_radius=1.955, body_length=2.0,
        )
        theta = math.radians(50.0)
        junction_r = 0.196 * (1.0 - math.cos(theta))
        junction_x = 0.196 * math.sin(theta)
        expected = junction_r + (2.0 - junction_x) * math.tan(theta)
        assert config.computed_base_radius == pytest.approx(expected, rel=1e-10)

    def test_frozen_dataclass(self):
        """BluntBodyConfig should be immutable (frozen)."""
        config = BluntBodyConfig()
        with pytest.raises(AttributeError):
            config.R_nose = 0.5  # type: ignore[misc]

    def test_equal_configs(self):
        """Two configs with same params should be equal."""
        c1 = BluntBodyConfig(R_nose=0.196, half_angle=50.0)
        c2 = BluntBodyConfig(R_nose=0.196, half_angle=50.0)
        assert c1 == c2

    def test_different_configs_not_equal(self):
        """Configs with different params should not be equal."""
        c1 = BluntBodyConfig(R_nose=0.196)
        c2 = BluntBodyConfig(R_nose=0.1)
        assert c1 != c2


class TestBluntBodyConfigValidation:
    """Tests for BluntBodyConfig.validate() classmethod."""

    def test_validate_default(self):
        """Default config should pass validation."""
        config = BluntBodyConfig.validate()
        assert config.R_nose == 0.196

    def test_validate_negative_R_nose(self):
        """Negative R_nose should raise ValueError."""
        with pytest.raises(ValueError, match="R_nose must be > 0"):
            BluntBodyConfig.validate(R_nose=-0.1)

    def test_validate_zero_R_nose(self):
        """Zero R_nose should raise ValueError."""
        with pytest.raises(ValueError, match="R_nose must be > 0"):
            BluntBodyConfig.validate(R_nose=0.0)

    def test_validate_half_angle_zero(self):
        """Zero half_angle should raise ValueError."""
        with pytest.raises(ValueError, match="half_angle must be > 0 and < 85"):
            BluntBodyConfig.validate(half_angle=0.0)

    def test_validate_half_angle_too_large(self):
        """half_angle >= 85 should raise ValueError."""
        with pytest.raises(ValueError, match="half_angle must be > 0 and < 85"):
            BluntBodyConfig.validate(half_angle=85.0)

    def test_validate_half_angle_84_ok(self):
        """half_angle of 84 should pass (just under limit)."""
        config = BluntBodyConfig.validate(half_angle=84.0)
        assert config.half_angle == 84.0

    def test_validate_base_radius_too_small(self):
        """base_radius <= junction_r should raise ValueError."""
        # For R_nose=1.0, half_angle=50: junction_r ~ 0.234
        with pytest.raises(ValueError, match="base_radius must be > junction_r"):
            BluntBodyConfig.validate(R_nose=1.0, half_angle=50.0, base_radius=0.1)

    def test_validate_too_few_points(self):
        """num_points < 10 should raise ValueError."""
        with pytest.raises(ValueError, match="num_points must be >= 10"):
            BluntBodyConfig.validate(num_points=5)

    def test_validate_returns_frozen(self):
        """Validated config should still be frozen."""
        config = BluntBodyConfig.validate()
        with pytest.raises(AttributeError):
            config.R_nose = 0.5  # type: ignore[misc]
