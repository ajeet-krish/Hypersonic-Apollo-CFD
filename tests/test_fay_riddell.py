"""Tests for Sutton-Graves stagnation heating correlation."""
import math

import pytest

from validation.fay_riddell import FayRiddellResult, sutton_graves


class TestSuttonGraves:
    """Tests for Sutton-Graves correlation."""

    def test_hand_computed_case(self):
        """Verify against hand-computed value.

        q = C * sqrt(rho/R) * V^3
        C = 1.83e-4, rho = 1.0, R = 0.1, V = 1000
        q = 1.83e-4 * sqrt(1.0/0.1) * 1e9
          = 1.83e-4 * 3.16227766 * 1e9
          = 578,896.8
        """
        result = sutton_graves(rho_inf=1.0, V_inf=1000.0, R_nose=0.1)
        expected = 1.83e-4 * math.sqrt(1.0 / 0.1) * 1000.0**3
        assert result.q_stag == pytest.approx(expected, rel=1e-6)

    def test_units_kw_conversion(self):
        """q_stag_kw should be q_stag / 1000."""
        result = sutton_graves(rho_inf=0.5, V_inf=2000.0, R_nose=0.2)
        assert result.q_stag_kw == pytest.approx(result.q_stag / 1000.0, rel=1e-10)

    def test_formula_string(self):
        """Formula should describe Sutton-Graves form."""
        result = sutton_graves(rho_inf=1.0, V_inf=1000.0, R_nose=0.1)
        assert "sqrt(rho_inf / R_nose)" in result.formula
        assert "V_inf^3" in result.formula

    def test_correct_form_not_chapman(self):
        """MUST use sqrt(rho/R), NOT sqrt(rho^3/R)."""
        # Test with rho != 1 where the two forms diverge
        result_sg = sutton_graves(rho_inf=0.01, V_inf=2500.0, R_nose=0.196)
        q_chapman = 1.83e-4 * math.sqrt(0.01**3 / 0.196) * 2500.0**3
        # Sutton-Graves and Chapman should differ for rho != 1
        assert result_sg.q_stag != pytest.approx(q_chapman, rel=0.01), (
            "Sutton-Graves and Chapman forms must differ for rho != 1"
        )
        # Verify the form used is sqrt(rho/R), not sqrt(rho^3/R)
        expected_sg = 1.83e-4 * math.sqrt(0.01 / 0.196) * 2500.0**3
        assert result_sg.q_stag == pytest.approx(expected_sg, rel=1e-6)

    def test_result_type(self):
        """Should return FayRiddellResult."""
        result = sutton_graves(rho_inf=1.0, V_inf=1000.0, R_nose=0.1)
        assert isinstance(result, FayRiddellResult)

    def test_stores_inputs(self):
        """Result should store input parameters."""
        result = sutton_graves(rho_inf=0.0184, V_inf=2400.0, R_nose=0.196)
        assert result.R_nose == 0.196
        assert result.rho_inf == 0.0184
        assert result.V_inf == 2400.0

    def test_zero_density(self):
        """Zero density should give zero heating."""
        result = sutton_graves(rho_inf=0.0, V_inf=1000.0, R_nose=0.1)
        assert result.q_stag == 0.0

    def test_zero_velocity(self):
        """Zero velocity should give zero heating."""
        result = sutton_graves(rho_inf=1.0, V_inf=0.0, R_nose=0.1)
        assert result.q_stag == 0.0

    def test_apollo_cm_case(self):
        """Apollo CM at 30km, M=8 should give reasonable heating."""
        # Approximate: rho ~ 0.0184 kg/m^3, a ~ 322 m/s, V ~ 2576 m/s
        rho = 0.0184
        V = 322.0 * 8.0
        R = 0.196
        result = sutton_graves(rho, V, R)
        # Should be in the range of hundreds of kW/m^2
        assert result.q_stag_kw > 100.0
        assert result.q_stag_kw < 10000.0
