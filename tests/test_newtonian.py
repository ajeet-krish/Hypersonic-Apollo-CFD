"""Tests for modified Newtonian pressure coefficient."""
import numpy as np
import pytest

from validation.newtonian import modified_newtonian_cp, stagnation_cp


class TestModifiedNewtonianCp:
    """Tests for modified Newtonian Cp distribution."""

    def test_cp_max_at_stagnation(self):
        """Cp at theta=pi/2 should equal Cp_max from modified Newtonian theory.

        Modified Newtonian Cp_max depends on Mach number and approaches 2.0
        only at very high Mach. At M=8, Cp_max ~ 1.83.
        """
        M = 8.0
        cp_stag = stagnation_cp(M)
        # Modified Newtonian Cp_max at M=8 is about 1.83 (< 2.0)
        assert cp_stag > 1.0, f"Cp_max should be > 1.0, got {cp_stag}"
        assert cp_stag < 2.0, f"Modified Newtonian Cp_max should be < 2.0 at M=8, got {cp_stag}"

    def test_cp_zero_at_theta_zero(self):
        """Cp should be 0 at theta=0 (tangent to flow)."""
        cp = modified_newtonian_cp(0.0, M=8.0)
        assert cp == pytest.approx(0.0, abs=1e-12)

    def test_cp_at_45_degrees(self):
        """Cp at 45 degrees should be Cp_max * 0.5."""
        M = 8.0
        cp_max = stagnation_cp(M)
        cp_45 = modified_newtonian_cp(np.pi / 4.0, M)
        assert cp_45 == pytest.approx(cp_max * 0.5, rel=1e-6)

    def test_array_input(self):
        """Should accept numpy array input."""
        thetas = np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])
        result = modified_newtonian_cp(thetas, M=8.0)
        assert isinstance(result, np.ndarray)
        assert len(result) == 5
        assert result[0] == pytest.approx(0.0, abs=1e-12)
        assert result[-1] == pytest.approx(stagnation_cp(8.0), rel=1e-6)

    def test_monotonic_increase(self):
        """Cp should increase monotonically from 0 to pi/2."""
        thetas = np.linspace(0, np.pi / 2, 50)
        cps = modified_newtonian_cp(thetas, M=8.0)
        for i in range(len(cps) - 1):
            assert cps[i] <= cps[i + 1], (
                f"Cp not monotonic: Cp[{i}]={cps[i]} > Cp[{i+1}]={cps[i+1]}"
            )

    def test_cp_max_increases_with_mach(self):
        """Cp_max should increase with Mach number."""
        cp5 = stagnation_cp(5.0)
        cp8 = stagnation_cp(8.0)
        cp12 = stagnation_cp(12.0)
        assert cp5 < cp8 < cp12

    def test_gamma_parameter(self):
        """Should accept custom gamma."""
        cp_14 = stagnation_cp(8.0, gamma=1.4)
        cp_12 = stagnation_cp(8.0, gamma=1.2)
        # Different gamma should give different Cp
        assert cp_14 != pytest.approx(cp_12, rel=0.01)


class TestStagnationCp:
    """Tests for stagnation point Cp."""

    def test_stagnation_cp_matches_theta_pi2(self):
        """stagnation_cp should equal modified_newtonian_cp(pi/2)."""
        M = 8.0
        cp_stag = stagnation_cp(M)
        cp_pi2 = modified_newtonian_cp(np.pi / 2.0, M)
        assert cp_stag == pytest.approx(cp_pi2, rel=1e-10)

    def test_returns_float(self):
        """Should return a float."""
        result = stagnation_cp(8.0)
        assert isinstance(result, float)
