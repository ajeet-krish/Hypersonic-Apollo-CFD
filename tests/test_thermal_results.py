"""Tests for thermal results dataclass and serialization.

Covers ThermalResult1D construction, save_thermal_results, and
build_thermal_summary functions.
"""
import json

import numpy as np
import pytest

from thermal.results import ThermalResult1D, build_thermal_summary, save_thermal_results


def _make_result(**kwargs) -> ThermalResult1D:
    """Create a ThermalResult1D with sensible defaults for testing."""
    defaults = dict(
        z=np.linspace(0, 0.05, 11),
        T_initial=np.full(11, 300.0),
        T_final=np.linspace(500.0, 300.0, 11),
        T_history=np.vstack([
            np.full(11, 300.0),
            np.linspace(400.0, 300.0, 11),
            np.linspace(500.0, 300.0, 11),
        ]),
        t_history=np.array([0.0, 0.5, 1.0]),
        t_end=1.0,
        T_max_wall=500.0,
        T_max_back=300.0,
        q_total=10000.0,
        material="AVCOAT-5026",
        wall_thickness=0.05,
    )
    defaults.update(kwargs)
    return ThermalResult1D(**defaults)


class TestThermalResult1D:
    """Tests for ThermalResult1D dataclass."""

    def test_construction(self):
        """Should construct with all required fields."""
        result = _make_result()
        assert result.material == "AVCOAT-5026"
        assert result.wall_thickness == pytest.approx(0.05)

    def test_numpy_arrays(self):
        """Should store numpy arrays."""
        result = _make_result()
        assert isinstance(result.z, np.ndarray)
        assert isinstance(result.T_initial, np.ndarray)
        assert isinstance(result.T_final, np.ndarray)

    def test_mutable(self):
        """ThermalResult1D is mutable (unlike frozen config)."""
        result = _make_result()
        result.T_max_wall = 600.0
        assert result.T_max_wall == 600.0


class TestBuildThermalSummary:
    """Tests for build_thermal_summary function."""

    def test_has_all_keys(self):
        """Summary should contain expected top-level keys."""
        result = _make_result()
        summary = build_thermal_summary(result)

        assert "config" in summary
        assert "results" in summary
        assert "profiles" in summary
        assert "time" in summary

    def test_config_keys(self):
        """Config sub-dict should have expected keys."""
        result = _make_result()
        summary = build_thermal_summary(result)

        assert "material" in summary["config"]
        assert "wall_thickness_m" in summary["config"]
        assert "t_end_s" in summary["config"]

    def test_results_keys(self):
        """Results sub-dict should have expected keys."""
        result = _make_result()
        summary = build_thermal_summary(result)

        assert "T_max_wall_K" in summary["results"]
        assert "T_max_back_K" in summary["results"]
        assert "q_total_J_m2" in summary["results"]

    def test_profiles_keys(self):
        """Profiles sub-dict should have expected keys."""
        result = _make_result()
        summary = build_thermal_summary(result)

        assert "z_m" in summary["profiles"]
        assert "T_initial_K" in summary["profiles"]
        assert "T_final_K" in summary["profiles"]

    def test_json_serializable(self):
        """Summary should be JSON-serializable."""
        result = _make_result()
        summary = build_thermal_summary(result)

        json_str = json.dumps(summary)
        assert len(json_str) > 0

    def test_rounding(self):
        """Numerical values should be rounded for readability."""
        result = _make_result(T_max_wall=500.123456)
        summary = build_thermal_summary(result)
        assert summary["results"]["T_max_wall_K"] == 500.12

    def test_profile_length(self):
        """Profile arrays should have same length as z."""
        result = _make_result()
        summary = build_thermal_summary(result)

        assert len(summary["profiles"]["z_m"]) == len(result.z)
        assert len(summary["profiles"]["T_initial_K"]) == len(result.T_initial)
        assert len(summary["profiles"]["T_final_K"]) == len(result.T_final)

    def test_time_info(self):
        """Time sub-dict should have correct values."""
        result = _make_result(
            t_history=np.array([0.0, 0.1, 0.2, 0.3]),
            t_end=0.3,
        )
        summary = build_thermal_summary(result)
        assert summary["time"]["t_end_s"] == pytest.approx(0.3)
        assert summary["time"]["n_steps"] == 4
        assert summary["time"]["dt_effective_s"] == pytest.approx(0.1)


class TestSaveThermalResults:
    """Tests for save_thermal_results function."""

    def test_saves_json(self, tmp_path):
        """Should save a valid JSON file."""
        result = _make_result()
        output_path = tmp_path / "thermal" / "thermal.json"
        saved = save_thermal_results(result, output_path)

        assert saved.exists()
        with open(saved) as f:
            loaded = json.load(f)
        assert loaded["results"]["T_max_wall_K"] == 500.0

    def test_creates_parent_directories(self, tmp_path):
        """Should create parent directories if they don't exist."""
        result = _make_result()
        output_path = tmp_path / "a" / "b" / "c" / "thermal.json"
        saved = save_thermal_results(result, output_path)

        assert saved.exists()

    def test_roundtrip_data(self, tmp_path):
        """Saved data should match original result."""
        result = _make_result(T_max_wall=750.25, q_total=42000.0)
        output_path = tmp_path / "thermal.json"
        save_thermal_results(result, output_path)

        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["results"]["T_max_wall_K"] == 750.25
        assert loaded["results"]["q_total_J_m2"] == 42000.0

    def test_overwrites_existing(self, tmp_path):
        """Should overwrite existing file without error."""
        result = _make_result()
        output_path = tmp_path / "thermal.json"

        save_thermal_results(result, output_path)
        save_thermal_results(result, output_path)

        with open(output_path) as f:
            loaded = json.load(f)
        assert "results" in loaded

    def test_single_step_time(self, tmp_path):
        """Should handle single time step (dt_effective = 0)."""
        result = _make_result(
            t_history=np.array([0.0]),
            t_end=0.0,
        )
        output_path = tmp_path / "thermal.json"
        save_thermal_results(result, output_path)

        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded["time"]["dt_effective_s"] == 0.0
