"""Tests for US Standard Atmosphere 1976."""
import math

import pytest

from physics.atmosphere import AtmosphereResult, standard_atmosphere


class TestStandardAtmosphere:
    """Tests for standard atmosphere model."""

    def test_sea_level_temperature(self):
        """Sea level temperature should be 288.15 K."""
        atm = standard_atmosphere(0.0)
        assert atm.temperature == pytest.approx(288.15, abs=0.01)

    def test_sea_level_pressure(self):
        """Sea level pressure should be 101325 Pa."""
        atm = standard_atmosphere(0.0)
        assert atm.pressure == pytest.approx(101325.0, abs=1.0)

    def test_sea_level_density(self):
        """Sea level density should be ~1.225 kg/m^3."""
        atm = standard_atmosphere(0.0)
        assert atm.density == pytest.approx(1.225, rel=0.01)

    def test_30km_temperature(self):
        """At 30 km, T ~ 226.5 K."""
        atm = standard_atmosphere(30000.0)
        assert atm.temperature == pytest.approx(226.5, abs=1.0)

    def test_30km_pressure(self):
        """At 30 km, P ~ 1197 Pa (US Standard Atmosphere 1976, within 3%)."""
        atm = standard_atmosphere(30000.0)
        assert atm.pressure == pytest.approx(1197.0, rel=0.03)

    def test_30km_density(self):
        """At 30 km, rho ~ 0.0184 kg/m^3 (US Standard Atmosphere 1976, within 3%)."""
        atm = standard_atmosphere(30000.0)
        assert atm.density == pytest.approx(0.0184, rel=0.03)

    def test_11km_temperature(self):
        """At 11 km (tropopause), T should be ~216.65 K."""
        atm = standard_atmosphere(11000.0)
        assert atm.temperature == pytest.approx(216.65, abs=0.5)

    def test_temperature_decreases_in_troposphere(self):
        """Temperature should decrease from 0 to 11 km."""
        atm0 = standard_atmosphere(0.0)
        atm11 = standard_atmosphere(11000.0)
        assert atm0.temperature > atm11.temperature

    def test_pressure_always_decreases(self):
        """Pressure should decrease monotonically with altitude."""
        altitudes = [0, 11000, 20000, 32000, 47000, 60000, 80000]
        pressures = [standard_atmosphere(h).pressure for h in altitudes]
        for i in range(len(pressures) - 1):
            assert pressures[i] > pressures[i + 1], (
                f"Pressure not decreasing: {pressures[i]} at {altitudes[i]}m "
                f">= {pressures[i+1]} at {altitudes[i+1]}m"
            )

    def test_speed_of_sound(self):
        """Speed of sound = sqrt(gamma * R * T)."""
        atm = standard_atmosphere(0.0)
        expected = math.sqrt(1.4 * 287.0528 * 288.15)
        assert atm.speed_of_sound == pytest.approx(expected, rel=1e-4)

    def test_viscosity_positive(self):
        """Dynamic viscosity should be positive."""
        atm = standard_atmosphere(30000.0)
        assert atm.dynamic_viscosity > 0

    def test_result_type(self):
        """Should return AtmosphereResult."""
        atm = standard_atmosphere(30000.0)
        assert isinstance(atm, AtmosphereResult)

    def test_altitude_stored(self):
        """Result should store the input altitude."""
        atm = standard_atmosphere(50000.0)
        assert atm.altitude == 50000.0

    def test_value_error_negative(self):
        """Negative altitude should raise ValueError."""
        with pytest.raises(ValueError, match="Altitude must be between"):
            standard_atmosphere(-100.0)

    def test_value_error_too_high(self):
        """Altitude > 86 km should raise ValueError."""
        with pytest.raises(ValueError, match="Altitude must be between"):
            standard_atmosphere(90000.0)

    def test_47km_temperature(self):
        """At 47 km (stratopause), T should be ~270.65 K."""
        atm = standard_atmosphere(47000.0)
        assert atm.temperature == pytest.approx(270.65, abs=1.0)

    def test_86km_pressure_very_low(self):
        """At 86 km, pressure should be very low (~3 Pa)."""
        atm = standard_atmosphere(86000.0)
        assert atm.pressure < 10.0
        assert atm.pressure > 0.0
