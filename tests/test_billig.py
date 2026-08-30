"""Tests for Billig shock standoff correlations."""
import math

import pytest

from validation.billig import (
    BilligResult,
    billig_blunted_cone,
    billig_sphere,
)


class TestBilligBluntedCone:
    """Tests for blunted cone shock standoff."""

    def test_high_mach_limit(self):
        """As M -> inf, delta/R -> 0.143."""
        result = billig_blunted_cone(R_nose=0.1, M=100.0)
        assert result.delta_over_R == pytest.approx(0.143, abs=0.001)

    def test_mach_8(self):
        """Verify at M=8."""
        result = billig_blunted_cone(R_nose=0.1, M=8.0)
        expected = 0.143 * math.exp(3.24 / 64.0)
        assert result.delta_over_R == pytest.approx(expected, rel=1e-6)

    def test_delta_calculation(self):
        """delta = delta_over_R * R_nose."""
        result = billig_blunted_cone(R_nose=0.196, M=8.0)
        assert result.delta == pytest.approx(result.delta_over_R * 0.196, rel=1e-10)

    def test_stores_inputs(self):
        """Result should store R_nose and M."""
        result = billig_blunted_cone(R_nose=0.5, M=10.0)
        assert result.R_nose == 0.5
        assert result.M == 10.0

    def test_formula_string(self):
        """Formula should describe the correlation."""
        result = billig_blunted_cone(R_nose=0.1, M=8.0)
        assert "0.143" in result.formula
        assert "exp" in result.formula

    def test_result_type(self):
        """Should return BilligResult."""
        result = billig_blunted_cone(R_nose=0.1, M=8.0)
        assert isinstance(result, BilligResult)

    def test_value_error_mach_1(self):
        """M <= 1.5 should raise ValueError."""
        with pytest.raises(ValueError, match="M must be > 1.5"):
            billig_blunted_cone(R_nose=0.1, M=1.0)

    def test_value_error_mach_1_5(self):
        """M = 1.5 should raise ValueError."""
        with pytest.raises(ValueError, match="M must be > 1.5"):
            billig_blunted_cone(R_nose=0.1, M=1.5)

    def test_mach_2_reasonable(self):
        """M=2 should give delta/R > 0.143."""
        result = billig_blunted_cone(R_nose=0.1, M=2.0)
        assert result.delta_over_R > 0.143


class TestBilligSphere:
    """Tests for sphere shock standoff."""

    def test_high_mach_limit(self):
        """As M -> inf, delta/R -> 0.78."""
        result = billig_sphere(R_nose=0.1, M=100.0)
        assert result.delta_over_R == pytest.approx(0.78, abs=0.01)

    def test_mach_8(self):
        """Verify at M=8."""
        result = billig_sphere(R_nose=0.1, M=8.0)
        expected = 0.78 * math.exp(3.24 / 64.0)
        assert result.delta_over_R == pytest.approx(expected, rel=1e-6)

    def test_different_from_cone(self):
        """Sphere correlation should differ from cone correlation."""
        R = 0.1
        M = 8.0
        cone = billig_blunted_cone(R, M)
        sphere = billig_sphere(R, M)
        assert sphere.delta_over_R > cone.delta_over_R, (
            f"Sphere delta/R ({sphere.delta_over_R}) should be > cone ({cone.delta_over_R})"
        )

    def test_formula_string(self):
        """Formula should describe the correlation."""
        result = billig_sphere(R_nose=0.1, M=8.0)
        assert "0.78" in result.formula

    def test_value_error_mach_1(self):
        """M <= 1.5 should raise ValueError."""
        with pytest.raises(ValueError, match="M must be > 1.5"):
            billig_sphere(R_nose=0.1, M=1.0)
