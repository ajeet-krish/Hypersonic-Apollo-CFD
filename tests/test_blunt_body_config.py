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
        assert config.num_points == 400

    def test_half_angle_rad(self):
        """half_angle_rad should be radians conversion."""
        config = BluntBodyConfig(cone_half_angle=50.0)
        assert config.half_angle_rad == pytest.approx(math.radians(50.0), rel=1e-10)

    def test_junction_x(self):
        """junction_x = R_shield * sin(phi_j) where phi_j depends on max_radius."""
        theta = math.radians(50.0)
        # Set max_radius so that phi_j = theta (matches old geometry)
        max_r = 0.196 * (1.0 - math.cos(theta))
        config = BluntBodyConfig(
            R_shield=0.196, cone_half_angle=50.0,
            max_radius=max_r, base_radius=0.05,
        )
        expected = 0.196 * math.sin(theta)
        assert config.junction_x == pytest.approx(expected, rel=1e-10)

    def test_junction_r(self):
        """junction_r = max_radius (when no fillet)."""
        theta = math.radians(50.0)
        max_r = 0.196 * (1.0 - math.cos(theta))
        config = BluntBodyConfig(
            R_shield=0.196, cone_half_angle=50.0,
            max_radius=max_r, base_radius=0.05,
        )
        assert config.junction_r == pytest.approx(max_r, rel=1e-10)

    def test_computed_body_length(self):
        """computed_body_length = x_j + |max_radius - base_radius|/tan(theta)."""
        theta = math.radians(50.0)
        max_r = 0.196 * (1.0 - math.cos(theta))
        x_j = 0.196 * math.sin(theta)
        base_r = 0.05
        config = BluntBodyConfig(
            R_shield=0.196, cone_half_angle=50.0,
            max_radius=max_r, base_radius=base_r,
        )
        expected = x_j + abs(max_r - base_r) / math.tan(theta)
        assert config.computed_body_length == pytest.approx(expected, rel=1e-10)

    def test_computed_base_radius_default(self):
        """Without body_length, computed_base_radius should be base_radius.

        Note: computed_base_radius was removed; base_radius is used directly.
        """
        config = BluntBodyConfig(base_radius=1.955)
        assert config.base_radius == pytest.approx(1.955, rel=1e-10)

    def test_computed_base_radius_with_body_length(self):
        """With body_length set, geometry should still be consistent."""
        config = BluntBodyConfig(
            R_shield=0.196, cone_half_angle=50.0,
            base_radius=0.05, body_length=2.0,
        )
        assert config.computed_body_length == pytest.approx(2.0, rel=1e-10)

    def test_frozen_dataclass(self):
        """BluntBodyConfig should be immutable (frozen)."""
        config = BluntBodyConfig()
        with pytest.raises(AttributeError):
            config.R_nose = 0.5  # type: ignore[misc]

    def test_equal_configs(self):
        """Two configs with same params should be equal."""
        c1 = BluntBodyConfig(R_shield=0.196, cone_half_angle=50.0)
        c2 = BluntBodyConfig(R_shield=0.196, cone_half_angle=50.0)
        assert c1 == c2

    def test_different_configs_not_equal(self):
        """Configs with different params should not be equal."""
        c1 = BluntBodyConfig(R_shield=0.196)
        c2 = BluntBodyConfig(R_shield=0.1)
        assert c1 != c2


class TestBluntBodyConfigValidation:
    """Tests for BluntBodyConfig.validate() classmethod."""

    def test_validate_default(self):
        """Default config should pass validation."""
        config = BluntBodyConfig.validate()
        assert config.R_nose == 0.196

    def test_validate_negative_R_shield(self):
        """Negative R_shield should raise ValueError."""
        with pytest.raises(ValueError, match="R_shield must be > 0"):
            BluntBodyConfig.validate(R_shield=-0.1)

    def test_validate_zero_R_shield(self):
        """Zero R_shield should raise ValueError."""
        with pytest.raises(ValueError, match="R_shield must be > 0"):
            BluntBodyConfig.validate(R_shield=0.0)

    def test_validate_cone_half_angle_zero(self):
        """Zero cone_half_angle should raise ValueError."""
        with pytest.raises(ValueError, match="cone_half_angle must be in \\(0, 90\\)"):
            BluntBodyConfig.validate(cone_half_angle=0.0)

    def test_validate_cone_half_angle_too_large(self):
        """cone_half_angle >= 90 should raise ValueError."""
        with pytest.raises(ValueError, match="cone_half_angle must be in \\(0, 90\\)"):
            BluntBodyConfig.validate(cone_half_angle=90.0)

    def test_validate_cone_half_angle_84_ok(self):
        """cone_half_angle of 84 should pass (under limit of 90)."""
        config = BluntBodyConfig.validate(cone_half_angle=84.0)
        assert config.cone_half_angle == 84.0

    def test_validate_too_few_points(self):
        """num_points < 10 should raise ValueError."""
        with pytest.raises(ValueError, match="num_points must be >= 10"):
            BluntBodyConfig.validate(num_points=5)

    def test_validate_returns_frozen(self):
        """Validated config should still be frozen."""
        config = BluntBodyConfig.validate()
        with pytest.raises(AttributeError):
            config.R_nose = 0.5  # type: ignore[misc]
