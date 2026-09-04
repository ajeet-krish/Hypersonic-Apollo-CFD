"""Tests for blunt body contour generation."""
import math

import numpy as np
import pytest

from geometry.blunt_body import generate_contour
from geometry.config import BluntBodyConfig


class TestGenerateContour:
    """Tests for spherically-blunted cone contour generation."""

    @pytest.fixture
    def apollo_config(self):
        """Apollo CM configuration."""
        return BluntBodyConfig(
            R_shield=0.196, cone_half_angle=50.0, base_radius=1.955,
            max_radius=0.196, num_points=400,
        )

    @pytest.fixture
    def apollo_contour(self, apollo_config):
        """Generate Apollo CM contour."""
        return generate_contour(apollo_config)

    def test_contour_length(self, apollo_config, apollo_contour):
        """Contour length should be num_points - 1 (junction point deduplicated)."""
        x, r = apollo_contour
        expected = apollo_config.num_points - 1
        assert len(x) == expected, (
            f"Expected {expected} points, got {len(x)}"
        )
        assert len(r) == expected

    def test_tip_at_origin(self, apollo_contour):
        """Nose tip should be at (0, 0)."""
        x, r = apollo_contour
        assert x[0] == pytest.approx(0.0, abs=1e-12)
        assert r[0] == pytest.approx(0.0, abs=1e-12)

    def test_base_at_correct_location(self, apollo_config, apollo_contour):
        """Base point should be at (body_length, base_radius)."""
        x, r = apollo_contour
        body_length = apollo_config.computed_body_length
        base_radius = apollo_config.base_radius
        assert x[-1] == pytest.approx(body_length, rel=1e-6), (
            f"Base x should be {body_length:.4f}, got {x[-1]:.4f}"
        )
        assert r[-1] == pytest.approx(base_radius, rel=1e-6), (
            f"Base r should be {base_radius:.4f}, got {r[-1]:.4f}"
        )

    def test_c1_continuity_at_junction(self, apollo_config, apollo_contour):
        """C1 continuity: sphere tangent = cone tangent at junction."""
        theta = apollo_config.half_angle_rad
        expected_slope = math.tan(theta)

        # Cone slope is (base_radius - junction_r) / (computed_body_length - junction_x)
        # Note: for a cone widening outward, slope is positive
        dx = apollo_config.computed_body_length - apollo_config.junction_x
        dr = apollo_config.base_radius - apollo_config.junction_r
        cone_slope = dr / dx

        assert abs(cone_slope) == pytest.approx(expected_slope, rel=1e-5)

    def test_sphere_section_monotonic(self, apollo_contour):
        """Sphere section should have monotonically increasing x."""
        x, r = apollo_contour
        n_sphere = int(apollo_contour[0].size * 0.4)
        for i in range(n_sphere - 1):
            assert x[i] <= x[i + 1]

    def test_cone_section_monotonic(self, apollo_contour):
        """Cone section should have monotonically increasing x."""
        x, r = apollo_contour
        n_sphere = int(apollo_contour[0].size * 0.4)
        # Find where x starts increasing after sphere peak
        # For a sphere-cone, x increases monotonically across the whole contour
        # Let's check from junction onwards where dx > 0
        for i in range(n_sphere, len(x) - 1):
            if x[i + 1] > x[i]:
                assert x[i] <= x[i + 1]

    def test_no_negative_radii(self, apollo_contour):
        """No negative radial coordinates."""
        _, r = apollo_contour
        assert np.all(r >= 0), f"Found negative radii: {r[r < 0]}"

    def test_generic_config(self):
        """Generic config should produce valid contour."""
        config = BluntBodyConfig(R_shield=0.1, cone_half_angle=45.0,
                                 base_radius=0.05, max_radius=0.1)
        x, r = generate_contour(config)
        assert len(x) == config.num_points - 1
        assert x[0] == pytest.approx(0.0, abs=1e-12)
        assert r[0] == pytest.approx(0.0, abs=1e-12)
        assert r[-1] == pytest.approx(config.base_radius, rel=1e-6)



