"""Edge case and error handling tests.

Focused on the specific edge cases requested:
- BluntBodyConfig.validate() with various invalid inputs
- normal_shock with M1=1.0
- billig_blunted_cone with M=1.0
- standard_atmosphere with altitude=-1 and altitude=90000
- gamma_curve_fit with T=100 (clamped) and T=10000 (clamped)

Plus additional edge cases for robustness.
"""
import math

import numpy as np
import pytest

from geometry.blunt_body import generate_contour
from geometry.config import BluntBodyConfig
from physics.atmosphere import standard_atmosphere
from physics.real_gas import gamma_curve_fit
from validation.billig import billig_blunted_cone, billig_sphere
from validation.fay_riddell import sutton_graves
from validation.newtonian import modified_newtonian_cp, stagnation_cp
from validation.shock_relations import normal_shock

# ============================================================================
# BluntBodyConfig.validate() edge cases
# ============================================================================


class TestBluntBodyConfigEdgeCases:
    """Edge cases for BluntBodyConfig.validate()."""

    def test_negative_R_shield(self):
        """Negative R_shield must raise ValueError."""
        with pytest.raises(ValueError, match="R_shield must be > 0"):
            BluntBodyConfig.validate(R_shield=-0.5)

    def test_zero_R_shield(self):
        """Zero R_shield must raise ValueError."""
        with pytest.raises(ValueError, match="R_shield must be > 0"):
            BluntBodyConfig.validate(R_shield=0.0)

    def test_cone_half_angle_zero(self):
        """cone_half_angle=0 must raise ValueError (degenerate: no cone)."""
        with pytest.raises(ValueError, match="cone_half_angle must be in \\(0, 90\\)"):
            BluntBodyConfig.validate(cone_half_angle=0.0)

    def test_cone_half_angle_90(self):
        """cone_half_angle=90 must raise ValueError (>=90 rejected).

        cone_half_angle=90 would mean the cone opens perpendicular to the axis,
        which is geometrically invalid for this model.
        """
        with pytest.raises(ValueError, match="cone_half_angle must be in \\(0, 90\\)"):
            BluntBodyConfig.validate(cone_half_angle=90.0)

    def test_cone_half_angle_negative(self):
        """Negative cone_half_angle must raise ValueError."""
        with pytest.raises(ValueError, match="cone_half_angle must be in \\(0, 90\\)"):
            BluntBodyConfig.validate(cone_half_angle=-10.0)

    def test_cone_half_angle_exactly_85(self):
        """cone_half_angle=85 should pass (within 0, 90)."""
        config = BluntBodyConfig.validate(cone_half_angle=85.0)
        assert config.cone_half_angle == 85.0

    def test_num_points_too_few(self):
        """num_points < 10 must raise ValueError."""
        with pytest.raises(ValueError, match="num_points must be >= 10"):
            BluntBodyConfig.validate(num_points=3)

    def test_num_points_exactly_10(self):
        """num_points=10 should pass (boundary)."""
        config = BluntBodyConfig.validate(num_points=10)
        assert config.num_points == 10


# ============================================================================
# Normal shock edge cases
# ============================================================================


class TestNormalShockEdgeCases:
    """Edge cases for normal shock relations."""

    def test_M1_exactly_1(self):
        """M1=1.0 must raise ValueError (no shock at sonic speed)."""
        with pytest.raises(ValueError, match="M1 must be > 1.0"):
            normal_shock(1.0)

    def test_M1_just_above_1(self):
        """M1=1.001 should work (weak shock)."""
        result = normal_shock(1.001)
        # Very weak shock: M2 ~ 1, p_ratio ~ 1
        assert result.M2 < 1.0
        assert result.p_ratio > 1.0
        assert result.p_ratio < 1.01  # Very small pressure jump

    def test_M1_subsonic(self):
        """M1 < 1 must raise ValueError."""
        with pytest.raises(ValueError, match="M1 must be > 1.0"):
            normal_shock(0.5)

    def test_M1_negative(self):
        """M1 < 0 must raise ValueError."""
        with pytest.raises(ValueError, match="M1 must be > 1.0"):
            normal_shock(-1.0)

    def test_M1_very_large(self):
        """M1=1000 should work without overflow."""
        result = normal_shock(1000.0)
        assert result.M2 > 0
        assert result.M2 < 1.0
        assert result.p_ratio > 0
        assert result.p0_ratio > 0

    def test_custom_gamma(self):
        """Should accept non-standard gamma."""
        result = normal_shock(5.0, gamma=1.2)
        assert result.M2 < 1.0
        assert result.p_ratio > 1.0


# ============================================================================
# Billig edge cases
# ============================================================================


class TestBilligEdgeCases:
    """Edge cases for Billig shock standoff correlations."""

    def test_M_exactly_1(self):
        """M=1.0 must raise ValueError."""
        with pytest.raises(ValueError, match="M must be > 1.5"):
            billig_blunted_cone(R_nose=0.1, M=1.0)

    def test_M_exactly_1_sphere(self):
        """M=1.0 for sphere must raise ValueError."""
        with pytest.raises(ValueError, match="M must be > 1.5"):
            billig_sphere(R_nose=0.1, M=1.0)

    def test_M_just_above_15(self):
        """M=1.501 should work (just above threshold)."""
        result = billig_blunted_cone(R_nose=0.1, M=1.501)
        assert result.delta_over_R > 0.143

    def test_M_exactly_15(self):
        """M=1.5 must raise ValueError."""
        with pytest.raises(ValueError, match="M must be > 1.5"):
            billig_blunted_cone(R_nose=0.1, M=1.5)

    def test_M_subsonic(self):
        """M < 1 must raise ValueError."""
        with pytest.raises(ValueError, match="M must be > 1.5"):
            billig_blunted_cone(R_nose=0.1, M=0.5)

    def test_negative_M(self):
        """M < 0 must raise ValueError."""
        with pytest.raises(ValueError, match="M must be > 1.5"):
            billig_blunted_cone(R_nose=0.1, M=-5.0)

    def test_M_very_large(self):
        """M=1000 should work and approach limit."""
        result = billig_blunted_cone(R_nose=0.1, M=1000.0)
        assert result.delta_over_R == pytest.approx(0.143, abs=1e-6)

    def test_negative_R_nose(self):
        """Negative R_nose should still compute (no validation in Billig)."""
        result = billig_blunted_cone(R_nose=-0.1, M=8.0)
        # The formula doesn't validate R_nose sign
        assert result.delta < 0  # Negative radius gives negative distance


# ============================================================================
# Standard atmosphere edge cases
# ============================================================================


class TestAtmosphereEdgeCases:
    """Edge cases for standard atmosphere model."""

    def test_negative_altitude(self):
        """Altitude < 0 must raise ValueError."""
        with pytest.raises(ValueError, match="Altitude must be between"):
            standard_atmosphere(-1.0)

    def test_altitude_90km(self):
        """Altitude = 90000 m (above 86 km limit) must raise ValueError."""
        with pytest.raises(ValueError, match="Altitude must be between"):
            standard_atmosphere(90000.0)

    def test_altitude_exactly_0(self):
        """Altitude = 0 should work (sea level)."""
        atm = standard_atmosphere(0.0)
        assert atm.temperature == pytest.approx(288.15, abs=0.01)

    def test_altitude_exactly_86km(self):
        """Altitude = 86000 m should work (upper boundary)."""
        atm = standard_atmosphere(86000.0)
        assert atm.pressure > 0
        assert atm.temperature > 0

    def test_altitude_just_below_86km(self):
        """Altitude = 85999 m should work."""
        atm = standard_atmosphere(85999.0)
        assert atm.pressure > 0

    def test_very_large_negative(self):
        """Large negative altitude must raise ValueError."""
        with pytest.raises(ValueError, match="Altitude must be between"):
            standard_atmosphere(-100000.0)

    def test_very_large_positive(self):
        """Large positive altitude must raise ValueError."""
        with pytest.raises(ValueError, match="Altitude must be between"):
            standard_atmosphere(1000000.0)


# ============================================================================
# gamma_curve_fit edge cases
# ============================================================================


class TestRealGasEdgeCases:
    """Edge cases for real-gas gamma(T)."""

    def test_T_100_clamped(self):
        """T=100K should clamp to 200K behavior."""
        props = gamma_curve_fit(100.0)
        props_200 = gamma_curve_fit(200.0)
        assert props.gamma == pytest.approx(props_200.gamma, rel=1e-10)
        # Stored temperature should be the input
        assert props.temperature == 100.0

    def test_T_10000_clamped(self):
        """T=10000K should clamp to 6000K behavior."""
        props = gamma_curve_fit(10000.0)
        props_6000 = gamma_curve_fit(6000.0)
        assert props.gamma == pytest.approx(props_6000.gamma, rel=1e-10)
        # Stored temperature should be the input
        assert props.temperature == 10000.0

    def test_T_exactly_200(self):
        """T=200K should be the lower boundary (no clamping needed)."""
        props = gamma_curve_fit(200.0)
        assert props.temperature == 200.0
        assert props.gamma > 0

    def test_T_exactly_6000(self):
        """T=6000K should be the upper boundary (no clamping needed)."""
        props = gamma_curve_fit(6000.0)
        assert props.temperature == 6000.0
        assert props.gamma > 0

    def test_T_exactly_1000(self):
        """T=1000K is the transition point between low and high coefficients."""
        props_low = gamma_curve_fit(999.0)
        props_1000 = gamma_curve_fit(1000.0)
        props_high = gamma_curve_fit(1001.0)
        # Should be continuous at the transition
        assert props_1000.gamma == pytest.approx(props_low.gamma, abs=0.01)
        assert props_1000.gamma == pytest.approx(props_high.gamma, abs=0.01)

    def test_extreme_low_temperature(self):
        """T=1K should clamp to 200K without error."""
        props = gamma_curve_fit(1.0)
        assert props.gamma > 0

    def test_extreme_high_temperature(self):
        """T=100000K should clamp to 6000K without error."""
        props = gamma_curve_fit(100000.0)
        assert props.gamma > 0

    def test_T_zero(self):
        """T=0K should clamp to 200K."""
        props = gamma_curve_fit(0.0)
        props_200 = gamma_curve_fit(200.0)
        assert props.gamma == pytest.approx(props_200.gamma, rel=1e-10)

    def test_negative_temperature(self):
        """Negative T should clamp to 200K."""
        props = gamma_curve_fit(-100.0)
        props_200 = gamma_curve_fit(200.0)
        assert props.gamma == pytest.approx(props_200.gamma, rel=1e-10)


# ============================================================================
# Sutton-Graves edge cases
# ============================================================================


class TestSuttonGravesEdgeCases:
    """Edge cases for Sutton-Graves heating correlation."""

    def test_zero_density(self):
        """Zero density should give zero heating."""
        result = sutton_graves(rho_inf=0.0, V_inf=1000.0, R_nose=0.1)
        assert result.q_stag == 0.0

    def test_zero_velocity(self):
        """Zero velocity should give zero heating."""
        result = sutton_graves(rho_inf=1.0, V_inf=0.0, R_nose=0.1)
        assert result.q_stag == 0.0

    def test_both_zero(self):
        """Both zero should give zero heating."""
        result = sutton_graves(rho_inf=0.0, V_inf=0.0, R_nose=0.1)
        assert result.q_stag == 0.0

    def test_very_small_density(self):
        """Very small density should give very small heating.

        q = 1.83e-4 * sqrt(1e-10/0.1) * 1e9 = 5.79 W/m^2
        This is tiny compared to typical hypersonic values (100+ kW/m^2).
        """
        result = sutton_graves(rho_inf=1e-10, V_inf=1000.0, R_nose=0.1)
        assert result.q_stag > 0
        assert result.q_stag < 10.0  # Much smaller than typical values

    def test_very_large_velocity(self):
        """Very large velocity should not overflow."""
        result = sutton_graves(rho_inf=0.0184, V_inf=10000.0, R_nose=0.1)
        assert result.q_stag > 0
        assert math.isfinite(result.q_stag)


# ============================================================================
# Newtonian Cp edge cases
# ============================================================================


class TestNewtonianEdgeCases:
    """Edge cases for modified Newtonian Cp."""

    def test_theta_negative(self):
        """Negative theta should work (sin^2 is even)."""
        cp_pos = modified_newtonian_cp(np.pi / 4, M=8.0)
        cp_neg = modified_newtonian_cp(-np.pi / 4, M=8.0)
        assert cp_pos == pytest.approx(cp_neg, rel=1e-10)

    def test_theta_beyond_pi_2(self):
        """theta > pi/2 should still compute sin^2(theta) <= 1."""
        cp = modified_newtonian_cp(np.pi, M=8.0)
        # sin(pi) = 0, so Cp = 0
        assert cp == pytest.approx(0.0, abs=1e-12)

    def test_theta_pi(self):
        """At theta=pi, sin(pi)=0, so Cp=0."""
        cp = modified_newtonian_cp(np.pi, M=8.0)
        assert cp == pytest.approx(0.0, abs=1e-12)

    def test_array_empty(self):
        """Empty array should return empty result."""
        thetas = np.array([])
        result = modified_newtonian_cp(thetas, M=8.0)
        assert len(result) == 0

    def test_M_very_large(self):
        """At very large M, Cp_max should approach ~2.0."""
        cp = stagnation_cp(100.0)
        assert cp > 1.8
        assert cp < 2.1


# ============================================================================
# Geometry edge cases
# ============================================================================


class TestGeometryEdgeCases:
    """Edge cases for blunt body geometry."""

    def test_very_small_R_nose(self):
        """Very small R_shield should still produce valid contour."""
        config = BluntBodyConfig(
            R_shield=0.001, cone_half_angle=45.0,
            max_radius=0.001, base_radius=0.0005,
        )
        x, r = generate_contour(config)
        assert len(x) > 0
        assert x[0] == pytest.approx(0.0, abs=1e-12)
        assert r[-1] == pytest.approx(config.base_radius, rel=1e-4)

    def test_very_large_R_nose(self):
        """Very large R_shield should still produce valid contour."""
        config = BluntBodyConfig(R_shield=100.0, cone_half_angle=45.0, base_radius=500.0)
        x, r = generate_contour(config)
        assert len(x) > 0
        assert r[-1] == pytest.approx(config.base_radius, rel=1e-6)

    def test_very_small_half_angle(self):
        """Very small half_angle (1 degree) should work."""
        config = BluntBodyConfig(R_shield=0.1, cone_half_angle=1.0,
                                 max_radius=0.05, base_radius=0.5)
        x, r = generate_contour(config)
        assert len(x) > 0
        assert r[-1] == pytest.approx(config.base_radius, rel=1e-4)

    def test_large_half_angle(self):
        """Large half_angle (84 degrees) should work."""
        config = BluntBodyConfig(R_shield=0.1, cone_half_angle=84.0,
                                 max_radius=0.05, base_radius=5.0)
        x, r = generate_contour(config)
        assert len(x) > 0
        assert r[-1] == pytest.approx(config.base_radius, rel=1e-4)
