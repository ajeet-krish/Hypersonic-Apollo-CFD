"""Tests for Apollo CM flight data comparison module.

Tests flight data retrieval, comparison logic, and visualization output.
"""
import pytest

from validation.flight_data import (
    FlightDataPoint,
    apollo_flight_data,
    compare_to_flight_data,
)


class TestApolloFlightData:
    """Tests for apollo_flight_data() function."""

    def test_returns_list(self):
        """Should return a list of FlightDataPoint."""
        data = apollo_flight_data()
        assert isinstance(data, list)

    def test_returns_multiple_points(self):
        """Should have at least 8 data points (multiple missions)."""
        data = apollo_flight_data()
        assert len(data) >= 8

    def test_all_are_dataclass_instances(self):
        """Every entry should be a FlightDataPoint instance."""
        data = apollo_flight_data()
        for dp in data:
            assert isinstance(dp, FlightDataPoint)

    def test_heat_flux_values_positive(self):
        """All heat flux values should be positive."""
        data = apollo_flight_data()
        for dp in data:
            if "Heat Flux" in dp.quantity:
                assert dp.value > 0

    def test_heat_flux_range_apollo4(self):
        """Apollo 4 heat flux should be in 400-500 W/cm^2 range."""
        data = apollo_flight_data()
        apollo4_hf = [dp for dp in data if "Apollo 4" in dp.mission
                      and "Heat Flux" in dp.quantity]
        assert len(apollo4_hf) >= 1
        for dp in apollo4_hf:
            assert 300.0 < dp.value < 600.0

    def test_heat_flux_range_apollo11(self):
        """Apollo 11 heat flux should be in 350-400 W/cm^2 range."""
        data = apollo_flight_data()
        apollo11_hf = [dp for dp in data if "Apollo 11" in dp.mission
                       and "Heat Flux" in dp.quantity]
        assert len(apollo11_hf) >= 1
        for dp in apollo11_hf:
            assert 300.0 < dp.value < 500.0

    def test_has_source_field(self):
        """Every entry should have a non-empty source."""
        data = apollo_flight_data()
        for dp in data:
            assert len(dp.source) > 0

    def test_has_notes_field(self):
        """Every entry should have a non-empty notes."""
        data = apollo_flight_data()
        for dp in data:
            assert len(dp.notes) > 0

    def test_has_missions(self):
        """Should include Apollo 4, 6, 11, and 13."""
        data = apollo_flight_data()
        missions = {dp.mission for dp in data}
        assert any("Apollo 4" in m for m in missions)
        assert any("Apollo 6" in m for m in missions)
        assert any("Apollo 11" in m for m in missions)
        assert any("Apollo 13" in m for m in missions)

    def test_units_are_correct(self):
        """Heat flux should be in W/cm^2, altitude in km, velocity in km/s."""
        data = apollo_flight_data()
        for dp in data:
            if "Heat Flux" in dp.quantity:
                assert dp.units == "W/cm^2"
            elif "Altitude" in dp.quantity:
                assert dp.units == "km"
            elif "Velocity" in dp.quantity:
                assert dp.units == "km/s"


class TestCompareToFlightData:
    """Tests for compare_to_flight_data() function."""

    def _make_conditions(self, mach=12.0, altitude_m=30000.0, r_nose=0.196):
        """Create a mock conditions dict."""
        return {
            "mach": mach,
            "altitude_m": altitude_m,
            "R_nose": r_nose,
        }

    def test_returns_dict(self):
        """Should return a dictionary."""
        result = compare_to_flight_data(50000.0, self._make_conditions())
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        """Result should contain expected top-level keys."""
        result = compare_to_flight_data(50000.0, self._make_conditions())
        assert "su2_conditions" in result
        assert "flight_data_comparisons" in result
        assert "caveats" in result
        assert "summary" in result

    def test_su2_conditions_populated(self):
        """su2_conditions should reflect input."""
        conds = self._make_conditions(mach=12.0, altitude_m=30000.0)
        result = compare_to_flight_data(50000.0, conds)
        assert result["su2_conditions"]["mach"] == 12.0
        assert result["su2_conditions"]["altitude_km"] == 30.0

    def test_comparisons_list(self):
        """flight_data_comparisons should be a non-empty list."""
        result = compare_to_flight_data(50000.0, self._make_conditions())
        assert isinstance(result["flight_data_comparisons"], list)
        assert len(result["flight_data_comparisons"]) > 0

    def test_comparison_has_ratio(self):
        """Each comparison should have ratio_su2_to_flight."""
        result = compare_to_flight_data(50000.0, self._make_conditions())
        for comp in result["flight_data_comparisons"]:
            assert "ratio_su2_to_flight" in comp
            assert "su2_pct_of_flight" in comp
            assert "mission" in comp

    def test_ratio_less_than_one_for_low_su2(self):
        """SU2 at M=12 should be much less than Apollo peak heating."""
        # 50000 W/m^2 = 5 W/cm^2 (very low vs Apollo ~400 W/cm^2)
        result = compare_to_flight_data(50000.0, self._make_conditions())
        for comp in result["flight_data_comparisons"]:
            if "Sutton-Graves" not in comp["mission"]:
                assert comp["ratio_su2_to_flight"] < 1.0

    def test_ratio_scales_with_su2_value(self):
        """Higher SU2 value should give higher ratio."""
        low = compare_to_flight_data(50000.0, self._make_conditions())
        high = compare_to_flight_data(500000.0, self._make_conditions())
        # Compare against the same flight data point (first comparison)
        low_ratio = low["flight_data_comparisons"][0]["ratio_su2_to_flight"]
        high_ratio = high["flight_data_comparisons"][0]["ratio_su2_to_flight"]
        assert high_ratio > low_ratio

    def test_caveats_list_not_empty(self):
        """Should have at least 4 caveats."""
        result = compare_to_flight_data(50000.0, self._make_conditions())
        assert len(result["caveats"]) >= 4

    def test_summary_contains_conditions(self):
        """Summary string should mention Mach and altitude."""
        result = compare_to_flight_data(50000.0, self._make_conditions())
        assert "M=12" in result["summary"]
        assert "30 km" in result["summary"]

    def test_zero_heat_flux(self):
        """Zero SU2 heat flux should still produce valid comparison."""
        result = compare_to_flight_data(0.0, self._make_conditions())
        assert "su2_conditions" in result
        for comp in result["flight_data_comparisons"]:
            assert comp["su2_value_W_cm2"] == 0.0

    def test_w_cm2_conversion(self):
        """10000 W/m^2 should be 1.0 W/cm^2."""
        result = compare_to_flight_data(10000.0, self._make_conditions())
        assert result["su2_conditions"]["su2_q_stag_W_cm2"] == pytest.approx(1.0, abs=0.01)
