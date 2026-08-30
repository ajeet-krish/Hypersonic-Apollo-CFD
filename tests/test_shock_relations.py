"""Tests for normal shock relations (Rankine-Hugoniot)."""
import math

import pytest

from validation.shock_relations import ShockResult, normal_shock


class TestNormalShock:
    """Tests for normal shock relations."""

    def test_conservation_mass(self):
        """rho1 * u1 = rho2 * u2 (mass conservation)."""
        M1 = 8.0
        gamma = 1.4
        result = normal_shock(M1, gamma)
        # rho2/rho1 = result.rho_ratio
        # u2/u1 = M2*a2 / (M1*a1) = (M2/M1) * sqrt(T2/T1)
        # rho1*u1 = rho2*u2 => rho_ratio * (M2/M1)*sqrt(T_ratio) = 1
        check = result.rho_ratio * (result.M2 / M1) * math.sqrt(result.T_ratio)
        assert check == pytest.approx(1.0, rel=1e-6), (
            f"Mass conservation violated: {check}"
        )

    def test_conservation_momentum(self):
        """p1 + rho1*u1^2 = p2 + rho2*u2^2 (momentum conservation).

        Normalized by p1: 1 + gamma*M1^2 = (rho_ratio*T_ratio) * (1 + gamma*M2^2)
        since p_ratio = rho_ratio * T_ratio.
        """
        M1 = 8.0
        gamma = 1.4
        result = normal_shock(M1, gamma)
        lhs = 1.0 + gamma * M1**2
        # p_ratio = rho_ratio * T_ratio, and rho2*u2^2/p1 = rho_ratio * M2^2 * T_ratio * gamma
        rhs = result.rho_ratio * result.T_ratio * (1.0 + gamma * result.M2**2)
        assert lhs == pytest.approx(rhs, rel=1e-6), (
            f"Momentum conservation violated: LHS={lhs}, RHS={rhs}"
        )

    def test_m2_less_than_1(self):
        """Downstream Mach should be subsonic for supersonic upstream."""
        result = normal_shock(8.0)
        assert result.M2 < 1.0

    def test_m2_approaches_0(self):
        """Very high M1 should give M2 approaching sqrt((gamma-1)/(2*gamma))."""
        result = normal_shock(50.0)
        gamma = 1.4
        m2_limit = math.sqrt((gamma - 1.0) / (2.0 * gamma))
        assert result.M2 == pytest.approx(m2_limit, abs=0.01)

    def test_pressure_ratio_increases_with_mach(self):
        """Pressure ratio should increase with Mach number."""
        r5 = normal_shock(5.0)
        r8 = normal_shock(8.0)
        r12 = normal_shock(12.0)
        assert r5.p_ratio < r8.p_ratio < r12.p_ratio

    def test_total_pressure_decreases_with_mach(self):
        """Total pressure ratio should decrease (more loss) with Mach."""
        r5 = normal_shock(5.0)
        r8 = normal_shock(8.0)
        r12 = normal_shock(12.0)
        assert r5.p0_ratio > r8.p0_ratio > r12.p0_ratio

    def test_density_ratio_limit(self):
        """At M->inf, rho_ratio -> (gamma+1)/(gamma-1)."""
        result = normal_shock(100.0)
        gamma = 1.4
        limit = (gamma + 1.0) / (gamma - 1.0)
        assert result.rho_ratio == pytest.approx(limit, abs=0.01)

    def test_result_type(self):
        """Should return ShockResult."""
        result = normal_shock(8.0)
        assert isinstance(result, ShockResult)

    def test_mach_1_downstream(self):
        """M1=1 should raise ValueError (per spec: M1 <= 1.0 is invalid)."""
        with pytest.raises(ValueError, match="M1 must be > 1.0"):
            normal_shock(1.0)

    def test_value_error_subsonic(self):
        """M1 < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="M1 must be > 1.0"):
            normal_shock(0.5)

    def test_value_error_exactly_1(self):
        """M1 = 1.0 should raise ValueError (trivial case not supported)."""
        with pytest.raises(ValueError, match="M1 must be > 1.0"):
            normal_shock(1.0)
