"""Tests for SU2Solver (unit tests only; no actual SU2 runs)."""
from pathlib import Path

from cfd.config import get_su2_binary
from cfd.solver import SU2Results, SU2Solver


class TestGetSU2Binary:
    """Tests for SU2 binary discovery."""

    def test_finds_binary(self):
        """Should find SU2_CFD binary at known path."""
        binary = get_su2_binary()
        assert binary.exists()
        assert "SU2_CFD" in binary.name

    def test_returns_path(self):
        """Should return a Path object."""
        binary = get_su2_binary()
        assert isinstance(binary, Path)


class TestSU2Results:
    """Tests for SU2Results dataclass defaults."""

    def test_default_converged(self):
        """Default should not be converged."""
        results = SU2Results()
        assert results.converged is False

    def test_default_iterations(self):
        """Default iterations should be 0."""
        results = SU2Results()
        assert results.iterations == 0

    def test_default_residual_drop(self):
        """Default residual drop should be 0."""
        results = SU2Results()
        assert results.residual_drop == 0.0

    def test_default_history(self):
        """Default history should be empty."""
        results = SU2Results()
        assert results.history == []

    def test_default_stagnation_values(self):
        """Default stagnation values should be None."""
        results = SU2Results()
        assert results.stagnation_heat_flux is None
        assert results.stagnation_pressure is None
        assert results.max_mach is None

    def test_custom_values(self):
        """Should accept custom values."""
        results = SU2Results(
            converged=True,
            iterations=5000,
            residual_drop=4.5,
            stagnation_pressure=150000.0,
            max_mach=8.2,
        )
        assert results.converged is True
        assert results.iterations == 5000
        assert results.residual_drop == 4.5
        assert results.stagnation_pressure == 150000.0
        assert results.max_mach == 8.2


class TestSU2SolverInit:
    """Tests for SU2Solver initialization."""

    def test_default_binary(self):
        """Should auto-discover SU2 binary."""
        solver = SU2Solver()
        assert solver.su2_cfd.exists()

    def test_custom_binary(self, tmp_path: Path):
        """Should accept custom binary path."""
        fake_binary = tmp_path / "SU2_CFD"
        fake_binary.write_text("#!/bin/sh\nexit 0\n")
        solver = SU2Solver(su2_cfd=fake_binary)
        assert solver.su2_cfd == fake_binary


class TestParseHistory:
    """Tests for _parse_history method."""

    def test_parse_valid_history(self, tmp_path: Path):
        """Should parse a valid SU2 history.csv."""
        history_content = (
            '"Inner_Iter","rms[Rho]","rms[RhoU]","rms[RhoV]","rms[RhoE]"\n'
            '1,1.000e+00,2.500e-01,3.000e-01,5.000e-01\n'
            '2,8.000e-01,2.000e-01,2.500e-01,4.000e-01\n'
            '3,5.000e-01,1.500e-01,2.000e-01,3.000e-01\n'
        )
        history_path = tmp_path / "history.csv"
        history_path.write_text(history_content)

        solver = SU2Solver.__new__(SU2Solver)
        history = solver._parse_history(history_path)

        assert len(history) == 3
        assert history[0]["Inner_Iter"] == "1"
        assert history[0]["rms[Rho]"] == "1.000e+00"
        assert history[2]["rms[Rho]"] == "5.000e-01"

    def test_parse_empty_history(self, tmp_path: Path):
        """Should handle empty history file."""
        history_path = tmp_path / "history.csv"
        history_path.write_text("")

        solver = SU2Solver.__new__(SU2Solver)
        history = solver._parse_history(history_path)

        assert history == []

    def test_parse_history_with_blank_lines(self, tmp_path: Path):
        """Should skip blank lines."""
        history_content = (
            '"ITER","rms[Rho]"\n'
            '1,1.0e+00\n'
            '\n'
            '2,5.0e-01\n'
            '\n'
        )
        history_path = tmp_path / "history.csv"
        history_path.write_text(history_content)

        solver = SU2Solver.__new__(SU2Solver)
        history = solver._parse_history(history_path)

        assert len(history) == 2

    def test_parse_history_missing_file(self, tmp_path: Path):
        """Should handle missing file gracefully."""
        solver = SU2Solver.__new__(SU2Solver)
        history = solver._parse_history(tmp_path / "nonexistent.csv")
        assert history == []


class TestComputeResidualDrop:
    """Tests for _compute_residual_drop static method."""

    def test_residual_drop_positive(self):
        """Should compute positive drop for decreasing residuals (log10 scale)."""
        history = [
            {"rms[Rho]": "-1.0"},   # 10^-1 = 0.1
            {"rms[Rho]": "-2.0"},   # 10^-2 = 0.01
            {"rms[Rho]": "-3.0"},   # 10^-3 = 0.001
            {"rms[Rho]": "-4.0"},   # 10^-4 = 0.0001
        ]
        drop = SU2Solver._compute_residual_drop(history)
        assert abs(drop - 3.0) < 0.01

    def test_residual_drop_zero(self):
        """Should return 0 for identical residuals."""
        history = [
            {"rms[Rho]": "-1.0"},
            {"rms[Rho]": "-1.0"},
        ]
        drop = SU2Solver._compute_residual_drop(history)
        assert abs(drop) < 0.01

    def test_residual_drop_empty(self):
        """Should return 0 for empty history."""
        drop = SU2Solver._compute_residual_drop([])
        assert drop == 0.0

    def test_residual_drop_single_entry(self):
        """Should return 0 for single entry."""
        history = [{"rms[Rho]": "-1.0"}]
        drop = SU2Solver._compute_residual_drop(history)
        assert drop == 0.0

    def test_residual_drop_rms_density_key(self):
        """Should handle RMS_DENSITY key format."""
        history = [
            {"RMS_DENSITY": "-1.0"},
            {"RMS_DENSITY": "-5.0"},
        ]
        drop = SU2Solver._compute_residual_drop(history)
        assert abs(drop - 4.0) < 0.01

    def test_residual_drop_diverging(self):
        """Should return negative for diverging residuals."""
        history = [
            {"rms[Rho]": "-4.0"},
            {"rms[Rho]": "-1.0"},
        ]
        drop = SU2Solver._compute_residual_drop(history)
        assert drop < 0


class TestFindRestartFile:
    """Tests for _find_restart_file static method."""

    def test_finds_restart_dat(self, tmp_path):
        """Should find restart.dat (SU2 v8.x naming)."""
        (tmp_path / "restart.dat").touch()
        result = SU2Solver._find_restart_file(tmp_path)
        assert result == tmp_path / "restart.dat"

    def test_no_restart_returns_none(self, tmp_path):
        """Should return None when no restart files exist."""
        result = SU2Solver._find_restart_file(tmp_path)
        assert result is None

    def test_prefers_restart_dat_over_flow_restart(self, tmp_path):
        """Should prefer restart.dat over flow_restart_*.dat."""
        (tmp_path / "restart.dat").touch()
        (tmp_path / "flow_restart_000100.dat").touch()
        result = SU2Solver._find_restart_file(tmp_path)
        assert result.name == "restart.dat"

    def test_finds_flow_restart_latest(self, tmp_path):
        """Should find the latest flow_restart file when no restart.dat."""
        (tmp_path / "flow_restart_000050.dat").touch()
        (tmp_path / "flow_restart_000100.dat").touch()
        (tmp_path / "flow_restart_000025.dat").touch()
        result = SU2Solver._find_restart_file(tmp_path)
        assert result.name == "flow_restart_000100.dat"

    def test_empty_dir_returns_none(self, tmp_path):
        """Empty directory returns None."""
        result = SU2Solver._find_restart_file(tmp_path)
        assert result is None

    def test_only_flow_restart_files(self, tmp_path):
        """Should fall back to flow_restart when restart.dat absent."""
        (tmp_path / "flow_restart_000200.dat").touch()
        result = SU2Solver._find_restart_file(tmp_path)
        assert result.name == "flow_restart_000200.dat"


class TestSU2SolverRunNotFound:
    """Tests for SU2Solver.run with missing binary."""

    def test_missing_binary_returns_not_converged(self, tmp_path: Path):
        """Run with non-existent binary should return unconverged results."""
        solver = SU2Solver(su2_cfd=tmp_path / "nonexistent" / "SU2_CFD")
        results = solver.run(
            config_path=tmp_path / "config.cfg",
            workdir=tmp_path / "work",
        )
        assert results.converged is False
        assert results.iterations == 0
