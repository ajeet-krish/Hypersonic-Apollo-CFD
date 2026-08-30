"""Tests for blunt body contour generation."""
import math

import numpy as np
import pytest

from geometry.blunt_body import _cone_frustum, _sphere_nose, generate_contour
from geometry.config import BluntBodyConfig


class TestGenerateContour:
    """Tests for spherically-blunted cone contour generation."""

    @pytest.fixture
    def apollo_config(self):
        """Apollo CM configuration."""
        return BluntBodyConfig(
            R_nose=0.196, half_angle=50.0, base_radius=1.955, num_points=400,
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
        """C1 continuity: sphere tangent = cone tangent at junction.

        The sphere slope at the junction is:
            dr/dx = d(R*(1-cos(phi)))/d(R*sin(phi)) = sin(phi)/cos(phi) = tan(phi) at phi=half_angle

        The cone slope is:
            dr/dx = (base_r - junction_r)/(body_length - junction_x) = tan(half_angle)

        Both are tan(half_angle) by construction. Verify this analytically.
        """
        theta = apollo_config.half_angle_rad
        expected_slope = math.tan(theta)

        # Sphere derivative at junction (parametric: x=R*sin(phi), r=R*(1-cos(phi)))
        # dr/dx = (dr/dphi)/(dx/dphi) = R*sin(phi)/(R*cos(phi)) = tan(phi)
        # At phi = half_angle: dr/dx = tan(half_angle)
        sphere_slope = math.tan(theta)

        # Cone derivative (linear from junction to base)
        cone_slope = (apollo_config.base_radius - apollo_config.junction_r) / (
            apollo_config.computed_body_length - apollo_config.junction_x
        )

        assert sphere_slope == pytest.approx(cone_slope, rel=1e-10), (
            f"C1 discontinuity: sphere slope {sphere_slope:.6f} != cone slope {cone_slope:.6f}"
        )
        assert sphere_slope == pytest.approx(expected_slope, rel=1e-10)

    def test_sphere_section_monotonic(self, apollo_contour):
        """Sphere section should have monotonically increasing x and r."""
        x, r = apollo_contour
        # Sphere is first half of contour
        n_half = len(x) // 2
        for i in range(n_half - 1):
            assert x[i] <= x[i + 1]
            assert r[i] <= r[i + 1]

    def test_cone_section_monotonic(self, apollo_contour):
        """Cone section should have monotonically increasing x and r."""
        x, r = apollo_contour
        n_half = len(x) // 2
        for i in range(n_half, len(x) - 1):
            assert x[i] <= x[i + 1]
            assert r[i] <= r[i + 1]

    def test_no_negative_radii(self, apollo_contour):
        """No negative radial coordinates."""
        _, r = apollo_contour
        assert np.all(r >= 0), f"Found negative radii: {r[r < 0]}"

    def test_generic_config(self):
        """Generic config should produce valid contour."""
        config = BluntBodyConfig(R_nose=0.1, half_angle=45.0, base_radius=0.5)
        x, r = generate_contour(config)
        assert len(x) == config.num_points - 1
        assert x[0] == pytest.approx(0.0, abs=1e-12)
        assert r[0] == pytest.approx(0.0, abs=1e-12)
        assert r[-1] == pytest.approx(config.base_radius, rel=1e-6)


class TestSphereNose:
    """Tests for spherical nose cap helper."""

    def test_tip_at_origin(self):
        """First point should be at (0, 0)."""
        x, r = _sphere_nose(0.1, math.radians(50.0), 100)
        assert x[0] == pytest.approx(0.0, abs=1e-12)
        assert r[0] == pytest.approx(0.0, abs=1e-12)

    def test_junction_matches_formula(self):
        """Last point should match junction formulas."""
        R = 0.1
        theta = math.radians(50.0)
        x, r = _sphere_nose(R, theta, 100)
        assert x[-1] == pytest.approx(R * math.sin(theta), rel=1e-10)
        assert r[-1] == pytest.approx(R * (1.0 - math.cos(theta)), rel=1e-10)

    def test_monotonic(self):
        """x and r should be monotonically increasing."""
        x, r = _sphere_nose(0.2, math.radians(60.0), 50)
        for i in range(len(x) - 1):
            assert x[i] < x[i + 1]
            assert r[i] < r[i + 1]


class TestConeFrustum:
    """Tests for conical frustum helper."""

    def test_endpoints(self):
        """Should connect start to end linearly."""
        x, r = _cone_frustum(0.1, 0.05, 1.0, 0.5, 100)
        assert x[0] == pytest.approx(0.1, rel=1e-10)
        assert r[0] == pytest.approx(0.05, rel=1e-10)
        assert x[-1] == pytest.approx(1.0, rel=1e-10)
        assert r[-1] == pytest.approx(0.5, rel=1e-10)

    def test_linear(self):
        """r should vary linearly with x."""
        x, r = _cone_frustum(0.0, 0.0, 1.0, 0.5, 50)
        for i in range(len(x)):
            expected_r = 0.5 * x[i]
            assert r[i] == pytest.approx(expected_r, rel=1e-10)
