"""Tests for 2D axisymmetric thermal solver.

Covers solver initialization, implicit backward Euler time stepping,
boundary conditions, sparse system assembly, and physical sanity checks.
"""
import numpy as np
import pytest

from thermal.config import ThermalConfig2D
from thermal.solver_2d import ThermalSolver2D


def _make_config(**kwargs) -> ThermalConfig2D:
    """Create a ThermalConfig2D with sensible defaults for testing."""
    n_s = kwargs.get("n_s", 10)
    defaults = dict(
        material="avcoat",
        wall_thickness=0.05,
        n_s=n_s,
        n_z=10,
        dt=1.0,
        t_end=10.0,
        q_surface=(500000.0,),
        s_surface=tuple(np.linspace(0.0, 1.0, n_s).tolist()),
        cold_wall_temp=300.0,
        radiation=False,
        emissivity=0.8,
    )
    defaults.update(kwargs)
    return ThermalConfig2D(**defaults)


class TestThermalConfig2D:
    """Tests for ThermalConfig2D dataclass."""

    def test_default_construction(self):
        """Should construct with all defaults."""
        config = ThermalConfig2D()
        assert config.material == "avcoat"
        assert config.wall_thickness == pytest.approx(0.05)
        assert config.n_s == 50
        assert config.n_z == 50

    def test_frozen(self):
        """Config should be immutable."""
        config = ThermalConfig2D()
        with pytest.raises(AttributeError):
            config.material = "pica"  # type: ignore[misc]

    def test_custom_values(self):
        """Should accept custom values."""
        config = ThermalConfig2D(
            material="pica",
            wall_thickness=0.1,
            n_s=20,
            n_z=30,
            dt=0.5,
            t_end=50.0,
        )
        assert config.material == "pica"
        assert config.wall_thickness == pytest.approx(0.1)
        assert config.n_s == 20
        assert config.n_z == 30


class TestThermalSolver2D:
    """Tests for ThermalSolver2D class."""

    def test_initialization(self):
        """Solver should initialize with correct grid and temperature."""
        config = _make_config(n_s=10, n_z=10, wall_thickness=0.05)
        solver = ThermalSolver2D(config)

        assert solver.n_s == 10
        assert solver.n_z == 10
        assert solver.T.shape == (10, 11)
        np.testing.assert_array_almost_equal(solver.T, 300.0)
        assert solver.t == 0.0

    def test_grid_uniform_z(self):
        """z-grid should be uniformly spaced from 0 to wall_thickness."""
        config = _make_config(wall_thickness=0.1, n_z=5, n_s=3)
        solver = ThermalSolver2D(config)

        assert solver.z[0] == pytest.approx(0.0)
        assert solver.z[-1] == pytest.approx(0.1)
        dz = np.diff(solver.z)
        np.testing.assert_array_almost_equal(dz, dz[0])

    def test_solve_returns_result(self):
        """solve() should return ThermalResult2D."""
        config = _make_config(n_s=5, n_z=5, dt=1.0, t_end=5.0, radiation=False)
        solver = ThermalSolver2D(config)
        result = solver.solve()

        assert result.T_final is not None
        assert result.T_final.shape == (5, 6)
        assert result.s.shape == (5,)
        assert result.z.shape == (6,)

    def test_hot_side_heats_up(self):
        """Hot side should be warmer than cold side after heating."""
        config = _make_config(
            n_s=5, n_z=10, dt=0.5, t_end=10.0,
            q_surface=(200000.0,),
            cold_wall_temp=300.0, radiation=False,
        )
        solver = ThermalSolver2D(config)
        result = solver.solve()

        assert result.T_max_wall > result.T_max_back
        assert result.T_max_wall > 300.0

    def test_cold_side_stays_near_initial(self):
        """Cold side should remain near initial temperature for short time."""
        config = _make_config(
            n_s=5, n_z=10, dt=0.1, t_end=2.0,
            q_surface=(100000.0,),
            cold_wall_temp=300.0, radiation=False,
        )
        solver = ThermalSolver2D(config)
        result = solver.solve()

        # Cold side should be close to 300K
        assert result.T_max_back < 320.0

    def test_uniform_flux_uniform_wall(self):
        """Uniform heat flux should produce uniform wall temperature along s."""
        s_surface = tuple(np.linspace(0.0, 1.0, 5).tolist())
        config = _make_config(
            n_s=5, n_z=10, dt=0.5, t_end=5.0,
            q_surface=(300000.0,),
            s_surface=s_surface, radiation=False,
        )
        solver = ThermalSolver2D(config)
        result = solver.solve()

        # Wall temperatures at z=0 should be nearly equal for uniform flux
        T_wall = result.T_final[:, 0]
        assert np.std(T_wall) < 50.0  # Allow some variation due to s-direction conduction

    def test_history_shape(self):
        """T_wall_history should have correct shape."""
        config = _make_config(n_s=5, n_z=5, dt=1.0, t_end=3.0, radiation=False)
        solver = ThermalSolver2D(config)
        result = solver.solve()

        assert result.T_wall_history.ndim == 2
        assert result.T_wall_history.shape[1] == 5  # n_s
        assert result.T_wall_history.shape[0] > 0

    def test_t_history_monotonic(self):
        """Time history should be monotonically increasing."""
        config = _make_config(n_s=5, n_z=5, dt=0.5, t_end=5.0, radiation=False)
        result = ThermalSolver2D(config).solve()
        assert np.all(np.diff(result.t_history) > 0)

    def test_initial_profile_constant(self):
        """Initial profile should be uniform at cold_wall_temp."""
        config = _make_config(cold_wall_temp=350.0, n_s=5, n_z=5, radiation=False)
        result = ThermalSolver2D(config).solve()
        np.testing.assert_array_almost_equal(result.T_initial, 350.0)

    def test_material_stored(self):
        """Result should store material name."""
        config = _make_config(
            material="pica", n_s=3, n_z=3, dt=1.0, t_end=1.0, radiation=False,
        )
        result = ThermalSolver2D(config).solve()
        assert result.material == "PICA-X"

    def test_wall_thickness_stored(self):
        """Result should store wall thickness."""
        config = _make_config(
            wall_thickness=0.08, n_s=3, n_z=3, dt=1.0, t_end=1.0, radiation=False,
        )
        result = ThermalSolver2D(config).solve()
        assert result.wall_thickness == pytest.approx(0.08)

    def test_radiation_bc(self):
        """Solver with radiation enabled should not crash."""
        config = _make_config(
            n_s=5, n_z=5, dt=0.5, t_end=5.0,
            q_surface=(300000.0,),
            radiation=True, emissivity=0.8,
        )
        result = ThermalSolver2D(config).solve()
        assert result.T_max_wall > 300.0

    def test_fixed_cold_bc(self):
        """Fixed temperature cold BC should hold cold side at T_cold."""
        config = _make_config(
            n_s=5, n_z=5, dt=0.5, t_end=5.0,
            q_surface=(100000.0,),
            radiation=False, cold_wall_temp=300.0,
        )
        result = ThermalSolver2D(config).solve()
        np.testing.assert_array_almost_equal(
            result.T_final[:, -1], 300.0, decimal=10,
        )

    def test_zero_flux(self):
        """Zero heat flux should leave temperature unchanged."""
        config = _make_config(
            n_s=5, n_z=5, dt=1.0, t_end=10.0,
            q_surface=(0.0,),
            radiation=False, cold_wall_temp=300.0,
        )
        result = ThermalSolver2D(config).solve()
        np.testing.assert_array_almost_equal(result.T_final, 300.0)

    def test_q_total_positive(self):
        """Total heat input should be positive for positive flux."""
        config = _make_config(
            n_s=5, n_z=5, dt=1.0, t_end=10.0,
            q_surface=(100000.0,), radiation=False,
        )
        result = ThermalSolver2D(config).solve()
        assert result.q_total >= 0.0

    def test_pica_material(self):
        """Solver should work with PICA material."""
        config = _make_config(
            material="pica", n_s=5, n_z=5, dt=1.0, t_end=5.0,
            q_surface=(200000.0,), radiation=False,
        )
        result = ThermalSolver2D(config).solve()
        assert result.material == "PICA-X"
        assert result.T_max_wall > 300.0

    def test_temperature_2d_field_shape(self):
        """T_final should be 2D with shape (n_s, n_z+1)."""
        config = _make_config(n_s=8, n_z=12)
        result = ThermalSolver2D(config).solve()
        assert result.T_final.shape == (8, 13)

    def test_surface_coordinates_match_config(self):
        """Surface coordinates should match input s_surface."""
        s_vals = (0.0, 0.25, 0.5, 0.75, 1.0)
        config = _make_config(
            n_s=5, n_z=5, s_surface=s_vals, radiation=False,
        )
        result = ThermalSolver2D(config).solve()
        np.testing.assert_array_almost_equal(result.s, np.array(s_vals))

    def test_higher_flux_higher_temp(self):
        """Higher heat flux should produce higher wall temperature."""
        config_low = _make_config(
            n_s=5, n_z=5, dt=1.0, t_end=10.0,
            q_surface=(100000.0,), radiation=False,
        )
        result_low = ThermalSolver2D(config_low).solve()

        config_high = _make_config(
            n_s=5, n_z=5, dt=1.0, t_end=10.0,
            q_surface=(500000.0,), radiation=False,
        )
        result_high = ThermalSolver2D(config_high).solve()

        assert result_high.T_max_wall > result_low.T_max_wall
