"""Tests for charring ablation model.

Covers AblationConfig, AblationModel physics, material char properties,
1D and 2D solver coupling, AblationResult dataclasses, serialization,
and visualization smoke tests.
"""
import numpy as np
import pytest

from thermal.ablation import AblationModel
from thermal.config import AblationConfig, ThermalConfig, ThermalConfig2D
from thermal.materials import AVCOAT, PICA, ThermalMaterial
from thermal.results import (
    AblationResult1D,
    AblationResult2D,
    build_ablation_summary_1d,
    build_ablation_summary_2d,
    save_ablation_results_1d,
    save_ablation_results_2d,
)
from thermal.solver_1d import ThermalSolver1D
from thermal.solver_2d import ThermalSolver2D


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def avcoat_config() -> AblationConfig:
    """Default AblationConfig for AVCOAT."""
    return AblationConfig()


@pytest.fixture
def pica_config() -> AblationConfig:
    """AblationConfig with higher pyrolysis rate for PICA."""
    return AblationConfig(
        pyrolysis_A=5.0e6,
        pyrolysis_Ea=8.0e4,
        blow_coeff=0.3,
    )


@pytest.fixture
def avcoat_model(avcoat_config: AblationConfig) -> AblationModel:
    """AblationModel for AVCOAT."""
    return AblationModel(AVCOAT, avcoat_config)


@pytest.fixture
def pica_model(pica_config: AblationConfig) -> AblationModel:
    """AblationModel for PICA."""
    return AblationModel(PICA, pica_config)


# ---------------------------------------------------------------------------
# AblationConfig tests
# ---------------------------------------------------------------------------

class TestAblationConfig:
    """Tests for AblationConfig dataclass."""

    def test_default_values(self):
        """Should have sensible default values."""
        cfg = AblationConfig()
        assert cfg.pyrolysis_A == pytest.approx(1.0e6)
        assert cfg.pyrolysis_Ea == pytest.approx(1.0e5)
        assert cfg.pyrolysis_R == pytest.approx(8.314)
        assert cfg.blow_coeff == pytest.approx(0.5)
        assert cfg.surface_emissivity == pytest.approx(0.9)

    def test_frozen(self):
        """Should be immutable."""
        cfg = AblationConfig()
        with pytest.raises(AttributeError):
            cfg.pyrolysis_A = 999.0  # type: ignore[misc]

    def test_custom_values(self):
        """Should accept custom values."""
        cfg = AblationConfig(
            pyrolysis_A=2.0e6,
            pyrolysis_Ea=1.2e5,
            pyrolysis_R=8.314,
            blow_coeff=0.7,
            surface_emissivity=0.85,
        )
        assert cfg.pyrolysis_A == 2.0e6
        assert cfg.blow_coeff == 0.7


# ---------------------------------------------------------------------------
# Material char properties tests
# ---------------------------------------------------------------------------

class TestMaterialCharProperties:
    """Tests for ThermalMaterial char properties."""

    def test_avcoat_has_char_density(self):
        """AVCOAT should have char_density defined."""
        assert AVCOAT.char_density == pytest.approx(180.0)

    def test_avcoat_has_char_conductivity(self):
        """AVCOAT should have char_conductivity defined."""
        assert AVCOAT.char_conductivity == pytest.approx(1.5)

    def test_avcoat_has_char_specific_heat(self):
        """AVCOAT should have char_specific_heat defined."""
        assert AVCOAT.char_specific_heat == pytest.approx(800.0)

    def test_pica_has_char_density(self):
        """PICA should have char_density defined."""
        assert PICA.char_density == pytest.approx(120.0)

    def test_pica_has_char_conductivity(self):
        """PICA should have char_conductivity defined."""
        assert PICA.char_conductivity == pytest.approx(1.0)

    def test_pica_has_char_specific_heat(self):
        """PICA should have char_specific_heat defined."""
        assert PICA.char_specific_heat == pytest.approx(700.0)

    def test_material_without_char_fallback(self):
        """Material without char properties should fallback to k_at/cp_at."""
        mat = ThermalMaterial(
            name="test",
            density=100.0,
            thermal_conductivity=2.5,
            specific_heat=800.0,
            decomposition_temperature=500.0,
            char_temperature=2000.0,
            heat_of_pyrolysis=1e6,
            emissivity=0.9,
        )
        assert mat.k_at_with_ablation(300.0, 100.0) == mat.k_at(300.0)
        assert mat.cp_at_with_ablation(300.0, 100.0) == mat.cp_at(300.0)

    def test_k_at_with_ablation_virgin(self):
        """At virgin density, k should equal k_at(T)."""
        T = 600.0
        k_virgin = AVCOAT.k_at_with_ablation(T, AVCOAT.density)
        k_ref = AVCOAT.k_at(T)
        assert k_virgin == pytest.approx(k_ref)

    def test_k_at_with_ablation_char(self):
        """At char density, k should equal char_conductivity."""
        T = 600.0
        k_char = AVCOAT.k_at_with_ablation(T, AVCOAT.char_density)
        assert k_char == pytest.approx(AVCOAT.char_conductivity)

    def test_k_at_with_ablation_intermediate(self):
        """At intermediate density, k should interpolate."""
        T = 600.0
        rho_mid = (AVCOAT.density + AVCOAT.char_density) / 2.0
        k_mid = AVCOAT.k_at_with_ablation(T, rho_mid)
        k_virgin = AVCOAT.k_at(T)
        k_char = AVCOAT.char_conductivity
        expected = 0.5 * k_virgin + 0.5 * k_char
        assert k_mid == pytest.approx(expected, rel=0.01)

    def test_cp_at_with_ablation_virgin(self):
        """At virgin density, cp should equal cp_at(T)."""
        T = 600.0
        cp_virgin = AVCOAT.cp_at_with_ablation(T, AVCOAT.density)
        cp_ref = AVCOAT.cp_at(T)
        assert cp_virgin == pytest.approx(cp_ref)

    def test_cp_at_with_ablation_char(self):
        """At char density, cp should equal char_specific_heat."""
        T = 600.0
        cp_char = AVCOAT.cp_at_with_ablation(T, AVCOAT.char_density)
        assert cp_char == pytest.approx(AVCOAT.char_specific_heat)


# ---------------------------------------------------------------------------
# AblationModel tests
# ---------------------------------------------------------------------------

class TestAblationModel:
    """Tests for AblationModel class."""

    def test_initialization(self, avcoat_model: AblationModel):
        """Should store material and config."""
        assert avcoat_model.material is AVCOAT
        assert avcoat_model.config.pyrolysis_A == 1.0e6

    def test_heat_of_pyrolysis(self, avcoat_model: AblationModel):
        """Should have heat of pyrolysis from material."""
        assert avcoat_model.heat_of_pyrolysis == AVCOAT.heat_of_pyrolysis

    def test_pyrolysis_rate_below_decomposition(self, avcoat_model: AblationModel):
        """Rate should be zero below decomposition temperature."""
        rho = np.array([AVCOAT.density, AVCOAT.density])
        T = np.array([300.0, 400.0])
        drho_dt = avcoat_model.compute_pyrolysis_rate(rho, T)
        np.testing.assert_array_almost_equal(drho_dt, 0.0)

    def test_pyrolysis_rate_above_decomposition(self, avcoat_model: AblationModel):
        """Rate should be negative above decomposition temperature."""
        rho = np.array([AVCOAT.density, AVCOAT.density])
        T = np.array([1000.0, 1500.0])
        drho_dt = avcoat_model.compute_pyrolysis_rate(rho, T)
        assert np.all(drho_dt < 0.0)

    def test_pyrolysis_rate_magnitude_increases_with_temp(self, avcoat_model: AblationModel):
        """Rate magnitude should increase with temperature."""
        rho = np.array([AVCOAT.density])
        T1 = np.array([800.0])
        T2 = np.array([1500.0])
        rate1 = avcoat_model.compute_pyrolysis_rate(rho, T1)
        rate2 = avcoat_model.compute_pyrolysis_rate(rho, T2)
        assert abs(rate2[0]) > abs(rate1[0])

    def test_update_density_clamps_to_char(self, avcoat_model: AblationModel):
        """Density should not go below char density."""
        rho = np.full(5, AVCOAT.density)
        T = np.full(5, 3000.0)  # Very high temperature
        dt = 100.0  # Large time step
        rho_new, _ = avcoat_model.update_density(rho, T, dt)
        assert np.all(rho_new >= AVCOAT.char_density)

    def test_update_density_returns_rate(self, avcoat_model: AblationModel):
        """Should return both new density and rate."""
        rho = np.array([AVCOAT.density])
        T = np.array([1000.0])
        dt = 0.1
        rho_new, drho_dt = avcoat_model.update_density(rho, T, dt)
        assert rho_new.shape == rho.shape
        assert drho_dt.shape == rho.shape

    def test_blowing_factor_virgin(self, avcoat_model: AblationModel):
        """Blowing factor should be 1.0 for virgin material."""
        rho_surface = np.array([AVCOAT.density])
        B = avcoat_model.compute_blowing_factor(rho_surface)
        assert B[0] == pytest.approx(1.0)

    def test_blowing_factor_reduced_with_ablation(self, avcoat_model: AblationModel):
        """Blowing factor should decrease with mass loss."""
        rho_virgin = np.array([AVCOAT.density])
        rho_ablated = np.array([AVCOAT.char_density])
        B_virgin = avcoat_model.compute_blowing_factor(rho_virgin)
        B_ablated = avcoat_model.compute_blowing_factor(rho_ablated)
        assert B_ablated[0] < B_virgin[0]

    def test_blowing_factor_minimum(self, avcoat_model: AblationModel):
        """Blowing factor should not go below 0.1."""
        rho_surface = np.array([AVCOAT.char_density])
        B = avcoat_model.compute_blowing_factor(rho_surface)
        assert B[0] >= 0.1

    def test_pyrolysis_heat_sink_positive(self, avcoat_model: AblationModel):
        """Heat sink should be positive for non-zero drho_dt."""
        drho_dt = np.array([-100.0, -50.0, 0.0])
        q_sink = avcoat_model.compute_pyrolysis_heat_sink(drho_dt)
        assert q_sink[0] > 0.0
        assert q_sink[1] > 0.0
        assert q_sink[2] == pytest.approx(0.0)

    def test_recession_rate_zero_for_cold(self, avcoat_model: AblationModel):
        """Recession rate should be zero when net flux is zero."""
        T_surface = np.array([300.0])
        q_net = np.array([0.0])
        rho_surface = np.array([AVCOAT.density])
        v = avcoat_model.compute_recession_rate(T_surface, q_net, rho_surface)
        assert v[0] == pytest.approx(0.0)

    def test_recession_rate_positive_for_heating(self, avcoat_model: AblationModel):
        """Recession rate should be positive for significant heating."""
        T_surface = np.array([2000.0])
        q_net = np.array([1e6])  # 1 MW/m^2
        rho_surface = np.array([AVCOAT.density])
        v = avcoat_model.compute_recession_rate(T_surface, q_net, rho_surface)
        assert v[0] > 0.0

    def test_pica_model(self, pica_model: AblationModel):
        """PICA model should work correctly."""
        rho = np.array([PICA.density])
        T = np.array([1200.0])
        drho_dt = pica_model.compute_pyrolysis_rate(rho, T)
        assert drho_dt[0] < 0.0


# ---------------------------------------------------------------------------
# Solver 1D with ablation tests
# ---------------------------------------------------------------------------

class TestThermalSolver1DAblation:
    """Tests for ThermalSolver1D with ablation coupling."""

    def test_initialization_without_ablation(self):
        """Should work without ablation config (backward compatible)."""
        config = ThermalConfig(n_cells=10, dt=1.0, t_end=1.0, radiation=False)
        solver = ThermalSolver1D(config)
        assert solver.ablation is None
        assert solver.rho is None

    def test_initialization_with_ablation(self):
        """Should initialize density field when ablation is enabled."""
        config = ThermalConfig(n_cells=10, dt=1.0, t_end=1.0, radiation=False)
        ablation_cfg = AblationConfig()
        solver = ThermalSolver1D(config, ablation_config=ablation_cfg)
        assert solver.ablation is not None
        assert solver.rho is not None
        assert len(solver.rho) == 11
        np.testing.assert_array_almost_equal(solver.rho, AVCOAT.density)

    def test_solve_without_ablation_returns_thermal_result(self):
        """Should return ThermalResult1D when no ablation."""
        from thermal.results import ThermalResult1D
        config = ThermalConfig(
            n_cells=10, dt=1.0, t_end=5.0,
            q_stagnation=100000.0, radiation=False,
        )
        result = ThermalSolver1D(config).solve()
        # When ablation is disabled (ablation_config=None), solver returns ThermalResult1D
        assert isinstance(result, ThermalResult1D)

    def test_solve_with_ablation_returns_ablation_result(self):
        """Should return AblationResult1D when ablation is enabled."""
        config = ThermalConfig(
            n_cells=10, dt=1.0, t_end=5.0,
            q_stagnation=100000.0, radiation=False,
        )
        ablation_cfg = AblationConfig()
        result = ThermalSolver1D(config, ablation_config=ablation_cfg).solve()
        assert isinstance(result, AblationResult1D)

    def test_ablation_result_has_density_profiles(self):
        """AblationResult1D should have initial and final density."""
        config = ThermalConfig(
            n_cells=10, dt=0.5, t_end=10.0,
            q_stagnation=200000.0, radiation=False,
        )
        ablation_cfg = AblationConfig()
        result = ThermalSolver1D(config, ablation_config=ablation_cfg).solve()
        assert result.rho_initial is not None
        assert result.rho_final is not None
        assert len(result.rho_initial) == 11
        assert len(result.rho_final) == 11

    def test_density_decreases_with_ablation(self):
        """Density should decrease from initial value during ablation."""
        config = ThermalConfig(
            n_cells=10, dt=0.5, t_end=20.0,
            q_stagnation=500000.0, radiation=False,
        )
        ablation_cfg = AblationConfig(pyrolysis_A=1.0e8)
        result = ThermalSolver1D(config, ablation_config=ablation_cfg).solve()
        # At least some nodes should have lower density
        assert np.any(result.rho_final < result.rho_initial)

    def test_ablation_result_metrics(self):
        """AblationResult1D should have recession, char depth, mass loss."""
        config = ThermalConfig(
            n_cells=10, dt=0.5, t_end=10.0,
            q_stagnation=300000.0, radiation=False,
        )
        ablation_cfg = AblationConfig()
        result = ThermalSolver1D(config, ablation_config=ablation_cfg).solve()
        assert hasattr(result, "recession_m")
        assert hasattr(result, "char_depth_m")
        assert hasattr(result, "mass_loss_kg_m2")
        assert hasattr(result, "ablation_rate_mm_s")
        assert result.recession_m >= 0.0
        assert result.mass_loss_kg_m2 >= 0.0

    def test_hot_side_heats_up_with_ablation(self):
        """Hot side should still heat up with ablation enabled."""
        config = ThermalConfig(
            n_cells=20, dt=0.5, t_end=10.0,
            q_stagnation=200000.0, radiation=False,
        )
        ablation_cfg = AblationConfig()
        result = ThermalSolver1D(config, ablation_config=ablation_cfg).solve()
        assert result.T_max_wall > 300.0

    def test_temperature_history_shape(self):
        """Temperature history should have correct shape."""
        config = ThermalConfig(
            n_cells=5, dt=1.0, t_end=3.0,
            q_stagnation=100000.0, radiation=False,
        )
        ablation_cfg = AblationConfig()
        result = ThermalSolver1D(config, ablation_config=ablation_cfg).solve()
        assert result.T_history.ndim == 2
        assert result.T_history.shape[1] == 6  # n_cells + 1

    def test_density_history_shape(self):
        """Density history should have correct shape."""
        config = ThermalConfig(
            n_cells=5, dt=1.0, t_end=3.0,
            q_stagnation=100000.0, radiation=False,
        )
        ablation_cfg = AblationConfig()
        result = ThermalSolver1D(config, ablation_config=ablation_cfg).solve()
        assert result.rho_history.ndim == 2
        assert result.rho_history.shape[1] == 6  # n_cells + 1


# ---------------------------------------------------------------------------
# Solver 2D with ablation tests
# ---------------------------------------------------------------------------

class TestThermalSolver2DAblation:
    """Tests for ThermalSolver2D with ablation coupling."""

    def test_initialization_with_ablation(self):
        """Should initialize density field when ablation is enabled."""
        n_s = 5
        s = np.linspace(0, 0.1, n_s)
        config = ThermalConfig2D(
            n_s=n_s, n_z=5, dt=1.0, t_end=1.0,
            q_surface=(500000.0,),
            s_surface=tuple(s),
            radiation=False,
        )
        ablation_cfg = AblationConfig()
        solver = ThermalSolver2D(config, ablation_config=ablation_cfg)
        assert solver.ablation is not None
        assert solver.rho is not None
        assert solver.rho.shape == (5, 6)

    def test_solve_returns_ablation_result(self):
        """Should return AblationResult2D when ablation is enabled."""
        n_s = 3
        s = np.linspace(0, 0.1, n_s)
        config = ThermalConfig2D(
            n_s=n_s, n_z=5, dt=1.0, t_end=5.0,
            q_surface=(500000.0,),
            s_surface=tuple(s),
            radiation=False,
        )
        ablation_cfg = AblationConfig()
        result = ThermalSolver2D(config, ablation_config=ablation_cfg).solve()
        assert isinstance(result, AblationResult2D)

    def test_ablation_result_metrics(self):
        """AblationResult2D should have per-point metrics."""
        n_s = 3
        s = np.linspace(0, 0.1, n_s)
        config = ThermalConfig2D(
            n_s=n_s, n_z=5, dt=0.5, t_end=5.0,
            q_surface=(500000.0,),
            s_surface=tuple(s),
            radiation=False,
        )
        ablation_cfg = AblationConfig()
        result = ThermalSolver2D(config, ablation_config=ablation_cfg).solve()
        assert hasattr(result, "recession_m")
        assert hasattr(result, "char_depth_m")
        assert hasattr(result, "mass_loss_kg_m2")
        assert hasattr(result, "ablation_rate_mm_s")
        assert len(result.recession_m) == n_s


# ---------------------------------------------------------------------------
# AblationResult serialization tests
# ---------------------------------------------------------------------------

class TestAblationResultSerialization:
    """Tests for ablation result serialization."""

    def _make_ablation_result_1d(self) -> AblationResult1D:
        """Create a minimal AblationResult1D for testing."""
        n = 11
        return AblationResult1D(
            z=np.linspace(0, 0.05, n),
            T_initial=np.full(n, 300.0),
            T_final=np.linspace(500.0, 300.0, n),
            T_history=np.vstack([np.full(n, 300.0), np.linspace(500.0, 300.0, n)]),
            t_history=np.array([0.0, 1.0]),
            t_end=1.0,
            T_max_wall=500.0,
            T_max_back=300.0,
            q_total=10000.0,
            material="AVCOAT-5026",
            wall_thickness=0.05,
            rho_initial=np.full(n, 512.0),
            rho_final=np.linspace(512.0, 300.0, n),
            rho_history=np.vstack([np.full(n, 512.0), np.linspace(512.0, 300.0, n)]),
            recession_m=0.001,
            char_depth_m=0.02,
            mass_loss_kg_m2=0.5,
            ablation_rate_mm_s=0.001,
        )

    def _make_ablation_result_2d(self) -> AblationResult2D:
        """Create a minimal AblationResult2D for testing."""
        n_s = 3
        n_z = 5
        n_nodes_z = n_z + 1
        T_final = np.zeros((n_s, n_nodes_z))
        for i in range(n_s):
            T_final[i, :] = np.linspace(500.0, 300.0, n_nodes_z)
        rho_final = np.zeros((n_s, n_nodes_z))
        for i in range(n_s):
            rho_final[i, :] = np.linspace(512.0, 400.0, n_nodes_z)
        return AblationResult2D(
            s=np.linspace(0, 0.1, n_s),
            z=np.linspace(0, 0.05, n_nodes_z),
            T_initial=np.full((n_s, n_nodes_z), 300.0),
            T_final=T_final,
            T_wall_history=np.full((2, n_s), 300.0),
            t_history=np.array([0.0, 1.0]),
            q_surface=np.full(n_s, 500000.0),
            T_max_wall=500.0,
            T_max_back=300.0,
            q_total=10000.0,
            material="AVCOAT-5026",
            wall_thickness=0.05,
            rho_initial=np.full((n_s, n_nodes_z), 512.0),
            rho_final=rho_final,
            rho_history=np.full((2, n_s), 512.0),
            recession_m=np.array([0.001, 0.002, 0.001]),
            char_depth_m=np.array([0.02, 0.03, 0.02]),
            mass_loss_kg_m2=np.array([0.5, 1.0, 0.5]),
            ablation_rate_mm_s=np.array([0.001, 0.002, 0.001]),
        )

    def test_build_ablation_summary_1d(self):
        """Should build a valid summary dict for 1D."""
        result = self._make_ablation_result_1d()
        summary = build_ablation_summary_1d(result)
        assert "config" in summary
        assert "results" in summary
        assert "profiles" in summary
        assert "recession_m" in summary["results"]
        assert "char_depth_m" in summary["results"]

    def test_build_ablation_summary_2d(self):
        """Should build a valid summary dict for 2D."""
        result = self._make_ablation_result_2d()
        summary = build_ablation_summary_2d(result)
        assert "config" in summary
        assert "results" in summary
        assert "surface" in summary

    def test_save_ablation_results_1d(self, tmp_path):
        """Should save 1D ablation results to JSON."""
        result = self._make_ablation_result_1d()
        output_path = tmp_path / "ablation" / "ablation_1d.json"
        saved = save_ablation_results_1d(result, output_path)
        assert saved.exists()

    def test_save_ablation_results_2d(self, tmp_path):
        """Should save 2D ablation results to JSON."""
        result = self._make_ablation_result_2d()
        output_path = tmp_path / "ablation" / "ablation_2d.json"
        saved = save_ablation_results_2d(result, output_path)
        assert saved.exists()

    def test_ablation_summary_1d_json_serializable(self):
        """1D ablation summary should be JSON-serializable."""
        import json
        result = self._make_ablation_result_1d()
        summary = build_ablation_summary_1d(result)
        json_str = json.dumps(summary)
        assert len(json_str) > 0

    def test_ablation_summary_2d_json_serializable(self):
        """2D ablation summary should be JSON-serializable."""
        import json
        result = self._make_ablation_result_2d()
        summary = build_ablation_summary_2d(result)
        json_str = json.dumps(summary)
        assert len(json_str) > 0
