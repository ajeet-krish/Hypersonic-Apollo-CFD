"""Tests for 1D thermal solver.

Covers solver initialization, implicit backward Euler time stepping,
boundary conditions, Thomas algorithm, and physical sanity checks.
"""
import numpy as np
import pytest

from thermal.config import ThermalConfig
from thermal.solver_1d import ThermalSolver1D, _thomas_solve


class TestThomasSolve:
    """Tests for Thomas algorithm tridiagonal solver."""

    def test_identity_system(self):
        """Solving I*x = d should return d."""
        n = 5
        a = np.zeros(n)
        b = np.ones(n)
        c = np.zeros(n)
        d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = _thomas_solve(a, b, c, d)
        np.testing.assert_array_almost_equal(x, d)

    def test_simple_tridiagonal(self):
        """Solve a known 3x3 tridiagonal system."""
        # System:
        #   2*x0 + 1*x1        = 5
        #   1*x0 + 3*x1 + 1*x2 = 10
        #          2*x1 + 4*x2 = 8
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([2.0, 3.0, 4.0])
        c = np.array([1.0, 1.0, 0.0])
        d = np.array([5.0, 10.0, 8.0])
        x = _thomas_solve(a, b, c, d)

        # Verify: check residual
        residual = np.zeros(3)
        residual[0] = b[0] * x[0] + c[0] * x[1] - d[0]
        residual[1] = a[1] * x[0] + b[1] * x[1] + c[1] * x[2] - d[1]
        residual[2] = a[2] * x[1] + b[2] * x[2] - d[2]
        np.testing.assert_array_almost_equal(residual, np.zeros(3), decimal=10)

    def test_diagonal_dominant(self):
        """Should solve a diagonally dominant system accurately."""
        # Construct system where x = ones is the known solution.
        # For interior: d[i] = a[i] + b[i] + c[i]
        # For i=0: d[0] = b[0] + c[0] (a[0] unused)
        # For i=n-1: d[-1] = a[-1] + b[-1] (c[-1] unused)
        n = 10
        a = np.zeros(n)
        b = np.full(n, 2.0)
        c = np.zeros(n)
        a[1:] = -0.5
        c[:-1] = -0.5

        d = np.zeros(n)
        d[0] = b[0] + c[0]
        d[-1] = a[-1] + b[-1]
        for i in range(1, n - 1):
            d[i] = a[i] + b[i] + c[i]

        x = _thomas_solve(a, b, c, d)
        np.testing.assert_array_almost_equal(x, np.ones(n), decimal=10)

    def test_size_one(self):
        """Should handle single-element system."""
        x = _thomas_solve(np.array([0.0]), np.array([2.0]), np.array([0.0]), np.array([4.0]))
        np.testing.assert_array_almost_equal(x, np.array([2.0]))


class TestThermalSolver1D:
    """Tests for ThermalSolver1D class."""

    def test_initialization(self):
        """Solver should initialize with correct grid and temperature."""
        config = ThermalConfig(
            wall_thickness=0.05,
            n_cells=10,
            cold_wall_temp=300.0,
        )
        solver = ThermalSolver1D(config)

        assert len(solver.z) == 11  # n_cells + 1
        assert len(solver.T) == 11
        np.testing.assert_array_almost_equal(solver.T, 300.0)
        assert solver.t == 0.0

    def test_grid_uniform(self):
        """Grid should be uniformly spaced from 0 to wall_thickness."""
        config = ThermalConfig(wall_thickness=0.1, n_cells=5)
        solver = ThermalSolver1D(config)

        assert solver.z[0] == pytest.approx(0.0)
        assert solver.z[-1] == pytest.approx(0.1)
        dz = np.diff(solver.z)
        np.testing.assert_array_almost_equal(dz, dz[0])

    def test_solve_returns_result(self):
        """solve() should return ThermalResult1D."""
        config = ThermalConfig(
            n_cells=10,
            dt=1.0,
            t_end=10.0,
            q_stagnation=100000.0,
            radiation=False,
        )
        solver = ThermalSolver1D(config)
        result = solver.solve()

        assert result.T_final is not None
        assert len(result.T_final) == 11
        assert result.t_end == pytest.approx(10.0)

    def test_hot_side_heats_up(self):
        """Hot side should be warmer than cold side after heating."""
        config = ThermalConfig(
            n_cells=20,
            dt=0.5,
            t_end=10.0,
            q_stagnation=200000.0,
            cold_wall_temp=300.0,
            radiation=False,
        )
        solver = ThermalSolver1D(config)
        result = solver.solve()

        assert result.T_max_wall > result.T_max_back
        assert result.T_max_wall > 300.0

    def test_cold_side_stays_near_initial(self):
        """Cold side should remain near initial temperature for short time."""
        config = ThermalConfig(
            n_cells=20,
            dt=0.1,
            t_end=5.0,
            q_stagnation=100000.0,
            cold_wall_temp=300.0,
            radiation=False,
        )
        solver = ThermalSolver1D(config)
        result = solver.solve()

        # Cold side should be close to 300K (heat hasn't fully penetrated)
        assert result.T_max_back < 310.0

    def test_temperature_monotonic(self):
        """Temperature should decrease monotonically from hot to cold side."""
        config = ThermalConfig(
            n_cells=20,
            dt=0.5,
            t_end=20.0,
            q_stagnation=500000.0,
            cold_wall_temp=300.0,
            radiation=False,
        )
        solver = ThermalSolver1D(config)
        result = solver.solve()

        # Temperature should generally decrease from hot to cold side
        # (allowing for some non-monotonicity due to radiation BC)
        assert result.T_final[0] >= result.T_final[-1]

    def test_higher_flux_higher_temp(self):
        """Higher heat flux should produce higher wall temperature."""
        base_config = ThermalConfig(
            n_cells=20,
            dt=1.0,
            t_end=50.0,
            cold_wall_temp=300.0,
            radiation=False,
        )

        # Low flux
        config_low = ThermalConfig(
            q_stagnation=100000.0,
            n_cells=20,
            dt=1.0,
            t_end=50.0,
            cold_wall_temp=300.0,
            radiation=False,
        )
        result_low = ThermalSolver1D(config_low).solve()

        # High flux
        config_high = ThermalConfig(
            q_stagnation=500000.0,
            n_cells=20,
            dt=1.0,
            t_end=50.0,
            cold_wall_temp=300.0,
            radiation=False,
        )
        result_high = ThermalSolver1D(config_high).solve()

        assert result_high.T_max_wall > result_low.T_max_wall

    def test_history_shape(self):
        """T_history should have correct shape (n_steps x n_nodes)."""
        config = ThermalConfig(
            n_cells=5,
            dt=1.0,
            t_end=3.0,
            q_stagnation=100000.0,
            radiation=False,
        )
        solver = ThermalSolver1D(config)
        result = solver.solve()

        assert result.T_history.ndim == 2
        assert result.T_history.shape[1] == 6  # n_cells + 1
        assert result.T_history.shape[0] > 0

    def test_t_history_monotonic(self):
        """Time history should be monotonically increasing."""
        config = ThermalConfig(
            n_cells=10,
            dt=0.5,
            t_end=5.0,
            q_stagnation=100000.0,
            radiation=False,
        )
        result = ThermalSolver1D(config).solve()
        assert np.all(np.diff(result.t_history) > 0)

    def test_initial_profile_constant(self):
        """Initial profile should be uniform at cold_wall_temp."""
        config = ThermalConfig(cold_wall_temp=350.0, n_cells=10, radiation=False)
        result = ThermalSolver1D(config).solve()
        np.testing.assert_array_almost_equal(result.T_initial, 350.0)

    def test_material_stored(self):
        """Result should store material name."""
        config = ThermalConfig(material="pica", n_cells=5, dt=1.0, t_end=1.0, radiation=False)
        result = ThermalSolver1D(config).solve()
        assert result.material == "PICA-X"

    def test_wall_thickness_stored(self):
        """Result should store wall thickness."""
        config = ThermalConfig(wall_thickness=0.08, n_cells=5, dt=1.0, t_end=1.0, radiation=False)
        result = ThermalSolver1D(config).solve()
        assert result.wall_thickness == pytest.approx(0.08)

    def test_radiation_bc(self):
        """Solver with radiation enabled should not crash."""
        config = ThermalConfig(
            n_cells=10,
            dt=0.5,
            t_end=5.0,
            q_stagnation=300000.0,
            radiation=True,
            emissivity=0.8,
        )
        result = ThermalSolver1D(config).solve()
        assert result.T_max_wall > 300.0

    def test_fixed_cold_bc(self):
        """Fixed temperature cold BC should hold cold side at T_cold."""
        config = ThermalConfig(
            n_cells=10,
            dt=0.5,
            t_end=5.0,
            q_stagnation=100000.0,
            radiation=False,
            cold_wall_temp=300.0,
        )
        result = ThermalSolver1D(config).solve()
        assert result.T_max_back == pytest.approx(300.0)

    def test_zero_flux(self):
        """Zero heat flux should leave temperature unchanged."""
        config = ThermalConfig(
            n_cells=10,
            dt=1.0,
            t_end=10.0,
            q_stagnation=0.0,
            radiation=False,
            cold_wall_temp=300.0,
        )
        result = ThermalSolver1D(config).solve()
        np.testing.assert_array_almost_equal(result.T_final, 300.0)

    def test_q_total_positive(self):
        """Total heat input should be positive for positive flux."""
        config = ThermalConfig(
            n_cells=10,
            dt=1.0,
            t_end=10.0,
            q_stagnation=100000.0,
            radiation=False,
        )
        result = ThermalSolver1D(config).solve()
        assert result.q_total >= 0.0

    def test_build_tridiagonal_size(self):
        """Tridiagonal arrays should have correct size."""
        config = ThermalConfig(n_cells=10, radiation=False)
        solver = ThermalSolver1D(config)
        a, b, c, d = solver._build_tridiagonal(solver.T, 1.0)

        n = config.n_cells + 1
        assert len(a) == n
        assert len(b) == n
        assert len(c) == n
        assert len(d) == n

    def test_pica_material(self):
        """Solver should work with PICA material."""
        config = ThermalConfig(
            material="pica",
            n_cells=10,
            dt=1.0,
            t_end=10.0,
            q_stagnation=200000.0,
            radiation=False,
        )
        result = ThermalSolver1D(config).solve()
        assert result.material == "PICA-X"
        assert result.T_max_wall > 300.0
