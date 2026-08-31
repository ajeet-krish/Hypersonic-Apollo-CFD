"""Tests for validation comparison module (triple validation).

Tests Sutton-Graves, Billig, Newtonian comparisons and report assembly.
"""
import math

import numpy as np
import pytest

from validation.compare import (
    ValidationResult,
    build_validation_report,
    compare_shock_standoff,
    compare_stagnation_heat_flux,
    compare_surface_cp,
    save_validation_report,
)


class TestCompareStagnationHeatFlux:
    """Tests for compare_stagnation_heat_flux function."""

    def test_exact_match_passes(self):
        """SU2 value matching Sutton-Graves should PASS."""
        rho = 0.0184
        V = 2400.0
        R = 0.1
        expected = 1.83e-4 * math.sqrt(rho / R) * V ** 3
        result = compare_stagnation_heat_flux(expected, rho, V, R)
        assert result.status == "PASS"
        assert result.error_pct == pytest.approx(0.0, abs=0.01)

    def test_ten_percent_error_passes(self):
        """10% error should PASS with 20% target."""
        rho = 0.0184
        V = 2400.0
        R = 0.1
        sg = 1.83e-4 * math.sqrt(rho / R) * V ** 3
        su2_val = sg * 1.10  # 10% high
        result = compare_stagnation_heat_flux(su2_val, rho, V, R, target_pct=20.0)
        assert result.status == "PASS"
        assert result.error_pct == pytest.approx(10.0, abs=0.5)

    def test_thirty_percent_error_fails(self):
        """30% error should FAIL with 20% target."""
        rho = 0.0184
        V = 2400.0
        R = 0.1
        sg = 1.83e-4 * math.sqrt(rho / R) * V ** 3
        su2_val = sg * 0.70  # 30% low
        result = compare_stagnation_heat_flux(su2_val, rho, V, R, target_pct=20.0)
        assert result.status == "FAIL"
        assert result.error_pct == pytest.approx(30.0, abs=0.5)

    def test_error_pct_correct(self):
        """Error percentage should be computed correctly."""
        rho = 0.0184
        V = 2400.0
        R = 0.1
        sg = 1.83e-4 * math.sqrt(rho / R) * V ** 3
        su2_val = sg * 1.15  # 15% high
        result = compare_stagnation_heat_flux(su2_val, rho, V, R)
        assert result.error_pct == pytest.approx(15.0, abs=0.5)

    def test_result_is_dataclass(self):
        """Should return a ValidationResult dataclass."""
        result = compare_stagnation_heat_flux(1000.0, 0.018, 2400.0, 0.1)
        assert isinstance(result, ValidationResult)

    def test_analytical_value_matches_sg(self):
        """Analytical value should match Sutton-Graves formula."""
        rho = 0.0184
        V = 2400.0
        R = 0.1
        result = compare_stagnation_heat_flux(1000.0, rho, V, R)
        expected = 1.83e-4 * math.sqrt(rho / R) * V ** 3
        assert result.analytical_value == pytest.approx(expected, rel=1e-10)


class TestCompareShockStandoff:
    """Tests for compare_shock_standoff function."""

    def test_exact_match_passes(self):
        """SU2 value matching Billig should PASS."""
        R = 0.1
        M = 8.0
        expected = 0.143 * math.exp(3.24 / M ** 2)
        result = compare_shock_standoff(expected, R, M)
        assert result.status == "PASS"
        assert result.error_pct == pytest.approx(0.0, abs=0.01)

    def test_five_percent_error_passes(self):
        """5% error should PASS with 10% target."""
        R = 0.1
        M = 8.0
        expected = 0.143 * math.exp(3.24 / M ** 2)
        su2_val = expected * 1.05
        result = compare_shock_standoff(su2_val, R, M, target_pct=10.0)
        assert result.status == "PASS"
        assert result.error_pct == pytest.approx(5.0, abs=0.5)

    def test_fifteen_percent_error_fails(self):
        """15% error should FAIL with 10% target."""
        R = 0.1
        M = 8.0
        expected = 0.143 * math.exp(3.24 / M ** 2)
        su2_val = expected * 1.15
        result = compare_shock_standoff(su2_val, R, M, target_pct=10.0)
        assert result.status == "FAIL"
        assert result.error_pct == pytest.approx(15.0, abs=0.5)


class TestCompareSurfaceCp:
    """Tests for compare_surface_cp function."""

    def test_perfect_newtonian_passes(self):
        """SU2 matching Newtonian should PASS."""
        M = 8.0
        theta = np.linspace(31, 89, 50)
        theta_rad = np.deg2rad(theta)
        # Newtonian Cp
        from validation.newtonian import modified_newtonian_cp
        cp = modified_newtonian_cp(theta_rad, M)
        result = compare_surface_cp(cp, theta, M)
        assert result.status == "PASS"
        assert result.error_pct == pytest.approx(0.0, abs=0.01)

    def test_ten_percent_error_passes(self):
        """10% mean error should PASS with 15% target."""
        M = 8.0
        theta = np.linspace(31, 89, 50)
        theta_rad = np.deg2rad(theta)
        from validation.newtonian import modified_newtonian_cp
        cp_newton = modified_newtonian_cp(theta_rad, M)
        cp_su2 = cp_newton * 1.10  # 10% high
        result = compare_surface_cp(cp_su2, theta, M, target_pct=15.0)
        assert result.status == "PASS"

    def test_excludes_sphere_nose(self):
        """Should exclude theta < 30 from comparison."""
        M = 8.0
        # Include both sphere and cone regions
        theta = np.linspace(5, 89, 100)
        theta_rad = np.deg2rad(theta)
        from validation.newtonian import modified_newtonian_cp
        cp_newton = modified_newtonian_cp(theta_rad, M)
        result = compare_surface_cp(cp_newton, theta, M)
        # Only cone section (theta > 30) is compared
        assert result.status == "PASS"

    def test_returns_result_dataclass(self):
        """Should return a ValidationResult."""
        M = 8.0
        theta = np.linspace(31, 89, 50)
        cp = np.ones(50) * 0.5
        result = compare_surface_cp(cp, theta, M)
        assert isinstance(result, ValidationResult)


class TestBuildValidationReport:
    """Tests for build_validation_report function."""

    def _make_mock_data(self):
        """Create mock data for report testing."""
        su2_results = {
            "case": "generic",
            "stagnation": {"heat_flux_W_m2": 34000.0},
            "shock_standoff": {"delta_over_R": 0.148},
            "field_extrema": {"cp_max": 1.94},
            "real_gas_correction": {"q_corrected_W_m2": 32776.0},
        }
        freestream = {
            "M": 8.0,
            "rho_inf": 0.0184,
            "V_inf": 2400.0,
            "altitude": 30000.0,
        }
        geometry = {
            "R_nose": 0.1,
            "half_angle": 45.0,
            "base_radius": 0.5,
        }
        return su2_results, freestream, geometry

    def test_has_expected_keys(self):
        """Report should contain expected keys."""
        su2, fs, geo = self._make_mock_data()
        report = build_validation_report(su2, fs, geo)
        assert "case" in report
        assert "mach" in report
        assert "all_pass" in report
        assert "results" in report
        assert "summary_table" in report

    def test_results_length(self):
        """Should have 3 validation results."""
        su2, fs, geo = self._make_mock_data()
        report = build_validation_report(su2, fs, geo)
        assert len(report["results"]) == 3

    def test_summary_table_length(self):
        """Summary table should have 3 entries."""
        su2, fs, geo = self._make_mock_data()
        report = build_validation_report(su2, fs, geo)
        assert len(report["summary_table"]) == 3

    def test_all_results_are_dicts(self):
        """Each result should be a dictionary."""
        su2, fs, geo = self._make_mock_data()
        report = build_validation_report(su2, fs, geo)
        for r in report["results"]:
            assert isinstance(r, dict)
            assert "quantity" in r
            assert "su2_value" in r
            assert "analytical_value" in r
            assert "error_pct" in r
            assert "status" in r

    def test_summary_table_entries(self):
        """Summary table entries should have expected keys."""
        su2, fs, geo = self._make_mock_data()
        report = build_validation_report(su2, fs, geo)
        for entry in report["summary_table"]:
            assert "quantity" in entry
            assert "su2" in entry
            assert "analytical" in entry
            assert "error_pct" in entry
            assert "target" in entry
            assert "status" in entry

    def test_json_serializable(self):
        """Report should be JSON-serializable."""
        import json
        su2, fs, geo = self._make_mock_data()
        report = build_validation_report(su2, fs, geo)
        json_str = json.dumps(report)
        assert len(json_str) > 0


class TestSaveValidationReport:
    """Tests for save_validation_report function."""

    def test_saves_json(self, tmp_path):
        """Should save a valid JSON file."""
        report = {
            "case": "test",
            "all_pass": True,
            "results": [],
            "summary_table": [],
        }
        output_path = tmp_path / "validation" / "validation.json"
        saved = save_validation_report(report, output_path)
        assert saved.exists()

        import json
        with open(saved) as f:
            loaded = json.load(f)
        assert loaded["case"] == "test"
        assert loaded["all_pass"] is True
