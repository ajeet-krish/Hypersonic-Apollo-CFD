"""Tests for real-gas properties using NASA polynomial fits."""
import pytest

from physics.real_gas import GasProperties, gamma_correction_factor, gamma_curve_fit


class TestGammaCurveFit:
    """Tests for temperature-dependent gamma."""

    def test_gamma_at_300k(self):
        """gamma at 300K should be ~1.4 (within 2%)."""
        props = gamma_curve_fit(300.0)
        assert props.gamma == pytest.approx(1.4, abs=0.03)

    def test_gamma_at_3000k_below_13(self):
        """gamma at 3000K should be < 1.3 (real-gas drop)."""
        props = gamma_curve_fit(3000.0)
        assert props.gamma < 1.3, (
            f"gamma at 3000K should be < 1.3, got {props.gamma}"
        )

    def test_gamma_at_6000k(self):
        """gamma at 6000K should be significantly below 1.4."""
        props = gamma_curve_fit(6000.0)
        assert props.gamma < 1.3

    def test_gamma_decreases_with_temperature(self):
        """gamma should decrease as temperature increases."""
        g300 = gamma_curve_fit(300.0).gamma
        g1000 = gamma_curve_fit(1000.0).gamma
        g3000 = gamma_curve_fit(3000.0).gamma
        assert g300 > g1000 > g3000

    def test_temperature_stored(self):
        """Result should store the input temperature."""
        props = gamma_curve_fit(500.0)
        assert props.temperature == 500.0

    def test_result_type(self):
        """Should return GasProperties."""
        props = gamma_curve_fit(300.0)
        assert isinstance(props, GasProperties)

    def test_cv_positive(self):
        """cv should be positive."""
        props = gamma_curve_fit(1000.0)
        assert props.cv > 0

    def test_cp_equals_cv_plus_R(self):
        """cp = cv + R_specific."""
        props = gamma_curve_fit(500.0)
        assert props.cp == pytest.approx(props.cv + props.R_specific, rel=1e-6)

    def test_gamma_equals_cp_over_cv(self):
        """gamma = cp / cv."""
        props = gamma_curve_fit(500.0)
        assert props.gamma == pytest.approx(props.cp / props.cv, rel=1e-10)

    def test_clamp_below_200k(self):
        """Temperatures below 200K should clamp to 200K values."""
        props_low = gamma_curve_fit(100.0)
        props_200 = gamma_curve_fit(200.0)
        assert props_low.gamma == pytest.approx(props_200.gamma, rel=1e-10)

    def test_clamp_above_6000k(self):
        """Temperatures above 6000K should clamp to 6000K values."""
        props_high = gamma_curve_fit(10000.0)
        props_6000 = gamma_curve_fit(6000.0)
        assert props_high.gamma == pytest.approx(props_6000.gamma, rel=1e-10)


class TestGammaCorrectionFactor:
    """Tests for gamma correction factor."""

    def test_correction_at_300k(self):
        """At 300K, correction should be ~1.0 (ideal gas, within 3%)."""
        cf = gamma_correction_factor(300.0, gamma_ref=1.4)
        assert cf == pytest.approx(1.0, abs=0.03)

    def test_correction_below_1_for_high_t(self):
        """At high T, correction should be < 1 (gamma_real < gamma_ref)."""
        cf = gamma_correction_factor(3000.0, gamma_ref=1.4)
        assert cf < 1.0, (
            f"Correction factor at 3000K should be < 1.0, got {cf}"
        )

    def test_correction_formula(self):
        """Should compute sqrt(gamma_real / gamma_ref)."""
        T = 2000.0
        gamma_ref = 1.4
        props = gamma_curve_fit(T)
        expected = (props.gamma / gamma_ref) ** 0.5
        cf = gamma_correction_factor(T, gamma_ref)
        assert cf == pytest.approx(expected, rel=1e-10)

    def test_correction_decreases_with_temperature(self):
        """Correction should decrease as T increases."""
        cf300 = gamma_correction_factor(300.0)
        cf1000 = gamma_correction_factor(1000.0)
        cf3000 = gamma_correction_factor(3000.0)
        assert cf300 >= cf1000 >= cf3000
