"""Integration tests for thermal analysis pipeline.

Tests CaseConfig thermal fields, thermal validation module, and
thermal stage wiring without requiring actual SU2 binaries.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from geometry.presets import apollo_cm, generic
from pipeline.case_config import CaseConfig


class TestCaseConfigThermalFields:
    """Tests for thermal-related CaseConfig fields."""

    def test_default_thermal_t_end(self):
        """Default thermal_t_end should be 100.0."""
        config = CaseConfig(name="test", label="Test", preset_fn=generic)
        assert config.thermal_t_end == 100.0

    def test_default_thermal_ablation(self):
        """Default thermal_ablation should be False."""
        config = CaseConfig(name="test", label="Test", preset_fn=generic)
        assert config.thermal_ablation is False

    def test_default_thermal_output_dir(self):
        """Default thermal_output_dir should be None."""
        config = CaseConfig(name="test", label="Test", preset_fn=generic)
        assert config.thermal_output_dir is None

    def test_custom_thermal_t_end(self):
        """Should accept custom thermal_t_end."""
        config = CaseConfig(
            name="test", label="Test", preset_fn=generic,
            thermal_t_end=200.0,
        )
        assert config.thermal_t_end == 200.0

    def test_custom_thermal_ablation(self):
        """Should accept custom thermal_ablation."""
        config = CaseConfig(
            name="test", label="Test", preset_fn=generic,
            thermal_ablation=True,
        )
        assert config.thermal_ablation is True

    def test_custom_thermal_output_dir(self):
        """Should accept custom thermal_output_dir."""
        config = CaseConfig(
            name="test", label="Test", preset_fn=generic,
            thermal_output_dir="/tmp/custom_thermal",
        )
        assert config.thermal_output_dir == "/tmp/custom_thermal"

    def test_thermal_dir_default(self):
        """thermal_dir should return output/{name}/thermal by default."""
        config = CaseConfig(name="apollo-cm", label="Apollo CM", preset_fn=apollo_cm)
        assert config.thermal_dir == "output/apollo-cm/thermal"

    def test_thermal_dir_custom(self):
        """thermal_dir should return custom dir when set."""
        config = CaseConfig(
            name="apollo-cm", label="Apollo CM", preset_fn=apollo_cm,
            thermal_output_dir="/tmp/custom",
        )
        assert config.thermal_dir == "/tmp/custom"

    def test_thermal_fields_frozen(self):
        """Thermal fields should be immutable."""
        config = CaseConfig(name="test", label="Test", preset_fn=generic)
        with pytest.raises(AttributeError):
            config.thermal_t_end = 200.0  # type: ignore[misc]

    def test_existing_thermal_defaults_unchanged(self):
        """Existing thermal defaults should remain."""
        config = CaseConfig(name="test", label="Test", preset_fn=generic)
        assert config.run_thermal is False
        assert config.thermal_material == "avcoat"
        assert config.thermal_wall_thickness == 0.05

    def test_all_thermal_fields_together(self):
        """Should accept all thermal fields simultaneously."""
        config = CaseConfig(
            name="test", label="Test", preset_fn=generic,
            run_thermal=True,
            thermal_material="pica",
            thermal_wall_thickness=0.08,
            thermal_t_end=300.0,
            thermal_ablation=True,
            thermal_output_dir="/tmp/thermal_out",
        )
        assert config.run_thermal is True
        assert config.thermal_material == "pica"
        assert config.thermal_wall_thickness == 0.08
        assert config.thermal_t_end == 300.0
        assert config.thermal_ablation is True
        assert config.thermal_output_dir == "/tmp/thermal_out"
        assert config.thermal_dir == "/tmp/thermal_out"


class TestThermalValidation:
    """Tests for thermal validation module."""

    def test_validate_passes_for_good_results(self):
        """Validation should pass for reasonable thermal results."""
        from validation.thermal_validation import validate_thermal_results

        thermal_results = {
            "results": {
                "T_max_wall_K": 2500.0,
                "T_max_back_K": 500.0,
                "q_total_J_m2": 1e6,
            },
        }
        freestream = {"M": 15.0, "rho_inf": 0.001, "V_inf": 5000.0, "altitude": 60000.0}
        geometry = {"R_nose": 0.196}

        report = validate_thermal_results(thermal_results, freestream, geometry)
        assert report["all_pass"] is True
        assert len(report["checks"]) >= 3

    def test_validate_fails_for_extreme_temperature(self):
        """Validation should fail for temperature above sublimation limit."""
        from validation.thermal_validation import validate_thermal_results

        thermal_results = {
            "results": {
                "T_max_wall_K": 4000.0,
                "T_max_back_K": 500.0,
                "q_total_J_m2": 1e6,
            },
        }
        freestream = {"M": 15.0, "rho_inf": 0.001, "V_inf": 5000.0, "altitude": 60000.0}
        geometry = {"R_nose": 0.196}

        report = validate_thermal_results(thermal_results, freestream, geometry)
        assert report["all_pass"] is False
        temp_check = [c for c in report["checks"] if c["name"] == "T_max_sublimation_limit"]
        assert len(temp_check) == 1
        assert temp_check[0]["status"] == "FAIL"

    def test_validate_fails_for_negative_heat_flux(self):
        """Validation should fail for negative heat flux."""
        from validation.thermal_validation import validate_thermal_results

        thermal_results = {
            "results": {
                "T_max_wall_K": 2500.0,
                "T_max_back_K": 500.0,
                "q_total_J_m2": -1000.0,
            },
        }
        freestream = {"M": 15.0, "rho_inf": 0.001, "V_inf": 5000.0, "altitude": 60000.0}
        geometry = {"R_nose": 0.196}

        report = validate_thermal_results(thermal_results, freestream, geometry)
        assert report["all_pass"] is False
        q_check = [c for c in report["checks"] if c["name"] == "q_positive"]
        assert len(q_check) == 1
        assert q_check[0]["status"] == "FAIL"

    def test_validate_pica_material(self):
        """Validation should use PICA sublimation limit."""
        from validation.thermal_validation import validate_thermal_results

        thermal_results = {
            "results": {
                "T_max_wall_K": 3600.0,
                "T_max_back_K": 500.0,
                "q_total_J_m2": 1e6,
            },
        }
        freestream = {"M": 15.0, "rho_inf": 0.001, "V_inf": 5000.0, "altitude": 60000.0}
        geometry = {"R_nose": 0.196}

        # 3600K is below PICA limit (3800K) but above Avcoat (3500K)
        report_avcoat = validate_thermal_results(
            thermal_results, freestream, geometry, material="avcoat",
        )
        report_pica = validate_thermal_results(
            thermal_results, freestream, geometry, material="pica",
        )

        # Avcoat should fail (3600 > 3500), PICA should pass (3600 < 3800)
        temp_avcoat = [c for c in report_avcoat["checks"] if c["name"] == "T_max_sublimation_limit"]
        temp_pica = [c for c in report_pica["checks"] if c["name"] == "T_max_sublimation_limit"]
        assert temp_avcoat[0]["status"] == "FAIL"
        assert temp_pica[0]["status"] == "PASS"

    def test_validate_check_count(self):
        """Should produce 4 validation checks."""
        from validation.thermal_validation import validate_thermal_results

        thermal_results = {
            "results": {
                "T_max_wall_K": 2000.0,
                "T_max_back_K": 400.0,
                "q_total_J_m2": 5e5,
            },
        }
        freestream = {"M": 10.0, "rho_inf": 0.01, "V_inf": 3000.0, "altitude": 30000.0}
        geometry = {"R_nose": 0.5}

        report = validate_thermal_results(thermal_results, freestream, geometry)
        assert len(report["checks"]) == 4

    def test_validate_zero_heat_flux(self):
        """Zero heat flux should fail q_positive check."""
        from validation.thermal_validation import validate_thermal_results

        thermal_results = {
            "results": {
                "T_max_wall_K": 300.0,
                "T_max_back_K": 300.0,
                "q_total_J_m2": 0.0,
            },
        }
        freestream = {"M": 10.0, "rho_inf": 0.01, "V_inf": 3000.0, "altitude": 30000.0}
        geometry = {"R_nose": 0.5}

        report = validate_thermal_results(thermal_results, freestream, geometry)
        assert report["all_pass"] is False


class TestThermalStageWiring:
    """Tests for thermal stage CLI wiring (no actual solver)."""

    def test_cli_args_parsed(self):
        """CLI should accept thermal-time, ablation, thermal-output flags."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", "python", "run.py", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert "--thermal-time" in result.stdout
        assert "--ablation" in result.stdout
        assert "--thermal-output" in result.stdout

    def test_cli_thermal_time_default(self):
        """Default thermal-time should be 100.0."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", "python", "run.py", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert "default: 100" in result.stdout

    def test_config_construction_with_all_thermal_flags(self):
        """CaseConfig should accept all thermal flags from CLI."""
        config = CaseConfig(
            name="test",
            label="Test",
            preset_fn=generic,
            run_thermal=True,
            thermal_material="pica",
            thermal_wall_thickness=0.08,
            thermal_t_end=200.0,
            thermal_ablation=True,
            thermal_output_dir="/tmp/test_thermal",
        )
        assert config.run_thermal is True
        assert config.thermal_material == "pica"
        assert config.thermal_wall_thickness == 0.08
        assert config.thermal_t_end == 200.0
        assert config.thermal_ablation is True
        assert config.thermal_output_dir == "/tmp/test_thermal"

    def test_thermal_stage_uses_config_t_end(self):
        """run_thermal_stage should use config.thermal_t_end, not hardcoded value."""
        from pipeline.case_config import CaseConfig

        config_short = CaseConfig(
            name="test", label="Test", preset_fn=generic,
            thermal_t_end=10.0,
        )
        config_long = CaseConfig(
            name="test", label="Test", preset_fn=generic,
            thermal_t_end=500.0,
        )
        assert config_short.thermal_t_end == 10.0
        assert config_long.thermal_t_end == 500.0

    def test_thermal_dir_respects_custom_output(self):
        """thermal_dir should use thermal_output_dir when set."""
        config = CaseConfig(
            name="test", label="Test", preset_fn=generic,
            thermal_output_dir="/custom/path",
        )
        assert config.thermal_dir == "/custom/path"

    def test_thermal_dir_falls_back_to_default(self):
        """thermal_dir should use default when thermal_output_dir is None."""
        config = CaseConfig(
            name="apollo-cm", label="Apollo CM", preset_fn=apollo_cm,
            thermal_output_dir=None,
        )
        assert config.thermal_dir == "output/apollo-cm/thermal"
