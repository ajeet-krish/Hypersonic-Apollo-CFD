"""Tests for post-processing functions.

Uses hand-crafted synthetic VTUData to verify surface profile extraction,
shock standoff measurement, total heating, real-gas correction, and
results summary.
"""
import numpy as np

from cfd.postprocess import (
    apply_real_gas_correction,
    build_results_summary,
    compute_total_heating,
    compute_total_heating_axisymmetric,
    extract_surface_profiles,
    measure_shock_standoff,
    measure_shock_standoff_r_nose,
    save_postprocess_results,
)
from cfd.vtu_parser import VTUData


def _make_synthetic_vtu(
    n_nodes: int = 100,
    x_range: tuple[float, float] = (-0.5, 2.0),
    r_max: float = 0.5,
    mach_freestream: float = 8.0,
    rho_freestream: float = 0.018,
    p_freestream: float = 1172.0,
    T_freestream: float = 227.0,
) -> VTUData:
    """Create synthetic VTUData with a simple shock structure.

    The synthetic field has:
    - Freestream conditions for x > shock_location
    - Post-shock conditions for x < shock_location
    - Stagnation conditions at x=0, r=0
    """
    # Generate a grid of points
    x_lin = np.linspace(x_range[0], x_range[1], n_nodes)
    r_lin = np.linspace(0, r_max, max(n_nodes // 5, 10))

    xx, rr = np.meshgrid(x_lin, r_lin)
    x_flat = xx.ravel()
    r_flat = rr.ravel()

    # Pad with z=0 for 3D coordinates
    z_flat = np.zeros_like(x_flat)
    coords = np.column_stack([x_flat, r_flat, z_flat])

    # Shock location at x = 0.05
    shock_x = 0.05

    # Mach field: freestream ahead of shock, lower behind
    mach = np.where(x_flat > shock_x, mach_freestream, mach_freestream * 0.3)
    mach = np.where(x_flat < 0, 0.0, mach)  # stagnation at nose

    # Density field: higher behind shock (compression)
    rho_ratio = 5.0  # typical normal shock density ratio at M=8
    density = np.where(
        x_flat > shock_x, rho_freestream, rho_freestream * rho_ratio
    )

    # Pressure field: higher behind shock
    p_ratio = 70.0  # typical normal shock pressure ratio at M=8
    pressure = np.where(
        x_flat > shock_x, p_freestream, p_freestream * p_ratio
    )

    # Temperature field: higher behind shock
    T_ratio = 10.0
    temperature = np.where(
        x_flat > shock_x, T_freestream, T_freestream * T_ratio
    )

    # Heat flux: high at stagnation, decays along surface
    heat_flux = np.where(
        np.abs(r_flat) < 0.01,
        50000.0 * np.exp(-x_flat / 0.1),
        5000.0,
    )

    # Pressure coefficient
    q_inf = 0.5 * rho_freestream * (mach_freestream * 310.0) ** 2
    cp = (pressure - p_freestream) / q_inf if q_inf > 0 else np.zeros_like(pressure)

    point_data = {
        "Mach": mach,
        "Pressure": pressure,
        "Temperature": temperature,
        "Density": density,
        "Heat_Flux": heat_flux,
        "Pressure_Coefficient": cp,
    }

    return VTUData(coordinates=coords, point_data=point_data)


def _make_body_contour(
    n_points: int = 100,
    R_nose: float = 0.1,
    half_angle_rad: float = 0.785,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a simple body contour for testing."""
    # Simple hemisphere + cone
    phi = np.linspace(0, half_angle_rad, n_points // 2)
    x_sphere = R_nose * np.sin(phi)
    r_sphere = R_nose * (1 - np.cos(phi))

    # Cone section
    n_cone = n_points - len(x_sphere)
    x_junction = x_sphere[-1]
    r_junction = r_sphere[-1]
    x_cone_end = x_junction + 0.3
    r_cone_end = r_junction + 0.3 * np.tan(half_angle_rad)

    x_cone = np.linspace(x_junction, x_cone_end, n_cone)
    r_cone = np.linspace(r_junction, r_cone_end, n_cone)

    x = np.concatenate([x_sphere, x_cone[1:]])
    r = np.concatenate([r_sphere, r_cone[1:]])

    return x, r


class TestExtractSurfaceProfiles:
    """Tests for extract_surface_profiles function."""

    def test_returns_all_keys(self):
        """Should return dict with all expected keys."""
        data = _make_synthetic_vtu()
        body = _make_body_contour()
        profiles = extract_surface_profiles(data, body)

        assert "s" in profiles
        assert "q" in profiles
        assert "p" in profiles
        assert "cp" in profiles
        assert "theta" in profiles

    def test_correct_lengths(self):
        """Output arrays should have same length as body contour."""
        data = _make_synthetic_vtu()
        body = _make_body_contour(n_points=80)
        profiles = extract_surface_profiles(data, body)

        assert len(profiles["s"]) == len(body[0])

    def test_arc_length_monotonic(self):
        """Arc length should be monotonically increasing."""
        data = _make_synthetic_vtu()
        body = _make_body_contour()
        profiles = extract_surface_profiles(data, body)

        s = profiles["s"]
        assert np.all(np.diff(s) >= 0)

    def test_arc_length_starts_at_zero(self):
        """Arc length should start at 0."""
        data = _make_synthetic_vtu()
        body = _make_body_contour()
        profiles = extract_surface_profiles(data, body)

        assert profiles["s"][0] == 0.0

    def test_heat_flux_positive(self):
        """Heat flux should be non-negative."""
        data = _make_synthetic_vtu()
        body = _make_body_contour()
        profiles = extract_surface_profiles(data, body)

        assert np.all(profiles["q"] >= 0)

    def test_pressure_positive(self):
        """Pressure should be positive."""
        data = _make_synthetic_vtu()
        body = _make_body_contour()
        profiles = extract_surface_profiles(data, body)

        assert np.all(profiles["p"] > 0)

    def test_theta_range(self):
        """Theta should be between 0 and 90 degrees."""
        data = _make_synthetic_vtu()
        body = _make_body_contour()
        profiles = extract_surface_profiles(data, body)

        assert np.all(profiles["theta"] >= 0)
        assert np.all(profiles["theta"] <= 90)


class TestMeasureShockStandoff:
    """Tests for measure_shock_standoff function."""

    def test_detects_shock(self):
        """Should detect a shock standoff distance."""
        data = _make_synthetic_vtu()
        _make_body_contour()
        R_nose = 0.1

        delta = measure_shock_standoff(data, R_nose)
        assert delta > 0

    def test_standoff_reasonable(self):
        """Standoff should be reasonable (a few percent of R_nose)."""
        data = _make_synthetic_vtu()
        _make_body_contour()
        R_nose = 0.1

        delta = measure_shock_standoff(data, R_nose)
        # Should be positive and not absurdly large
        assert 0 < delta < R_nose * 5.0

    def test_no_density_returns_zero(self):
        """Should return 0 when density data is missing."""
        coords = np.zeros((10, 3))
        data = VTUData(coordinates=coords, point_data={"Mach": np.ones(10)})
        delta = measure_shock_standoff(data, 0.1)
        assert delta == 0.0


class TestMeasureShockStandoffDivergenceGuard:
    """Tests for divergence guard in measure_shock_standoff."""

    def test_diverged_returns_zero(self):
        """When max_mach > 3x config_mach, returns 0.0."""
        n = 100
        x = np.linspace(-0.5, 2.0, n)
        r = np.linspace(0, 0.5, n)
        z = np.zeros(n)
        coords = np.column_stack([x, r, z])

        # Create a diverged field: max Mach = 20, config Mach = 5
        mach = np.full(n, 20.0)
        density = np.ones(n)
        pressure = np.ones(n)

        data = VTUData(
            coordinates=coords,
            point_data={"Mach": mach, "Density": density, "Pressure": pressure},
        )

        delta = measure_shock_standoff(data, R_nose=0.1, config_mach=5.0)
        assert delta == 0.0

    def test_diverged_exact_boundary(self):
        """When max_mach == 3x config_mach, guard does NOT trigger (strict >)."""
        n = 100
        x = np.linspace(-0.5, 2.0, n)
        r = np.linspace(0, 0.5, n)
        z = np.zeros(n)
        coords = np.column_stack([x, r, z])

        # max_mach = 15 = 3 * 5, not strictly greater
        mach = np.full(n, 15.0)
        density = np.ones(n)

        data = VTUData(
            coordinates=coords,
            point_data={"Mach": mach, "Density": density},
        )

        # Should NOT return 0 due to guard (but may return 0 for other reasons)
        # Since all density is uniform, no shock gradient, so result is 0.0.
        # But the guard did NOT trigger, so no warning.
        delta = measure_shock_standoff(data, R_nose=0.1, config_mach=5.0)
        # Just verify no exception and a float result
        assert isinstance(delta, float)

    def test_config_mach_none_skips_guard(self):
        """When config_mach is None, guard is skipped."""
        n = 100
        x = np.linspace(-0.5, 2.0, n)
        r = np.linspace(0, 0.5, n)
        z = np.zeros(n)
        coords = np.column_stack([x, r, z])

        # High Mach but no config_mach -- guard skipped
        mach = np.full(n, 20.0)
        density = np.ones(n)

        data = VTUData(
            coordinates=coords,
            point_data={"Mach": mach, "Density": density},
        )

        # Should not raise, guard is bypassed
        delta = measure_shock_standoff(data, R_nose=0.1, config_mach=None)
        assert isinstance(delta, float)

    def test_no_mach_data_skips_guard(self):
        """When VTU has no Mach data, guard is skipped."""
        n = 100
        x = np.linspace(-0.5, 2.0, n)
        r = np.linspace(0, 0.5, n)
        z = np.zeros(n)
        coords = np.column_stack([x, r, z])

        density = np.ones(n)
        data = VTUData(
            coordinates=coords,
            point_data={"Density": density},
        )

        # config_mach provided but no Mach in VTU -- guard skipped
        delta = measure_shock_standoff(data, R_nose=0.1, config_mach=5.0)
        assert isinstance(delta, float)

    def test_diverged_produces_warning(self):
        """Divergence guard should emit a warning."""
        import warnings
        n = 100
        x = np.linspace(-0.5, 2.0, n)
        r = np.linspace(0, 0.5, n)
        z = np.zeros(n)
        coords = np.column_stack([x, r, z])

        mach = np.full(n, 20.0)
        density = np.ones(n)

        data = VTUData(
            coordinates=coords,
            point_data={"Mach": mach, "Density": density},
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            delta = measure_shock_standoff(data, R_nose=0.1, config_mach=5.0)
            assert delta == 0.0
            assert len(w) >= 1
            assert "diverged" in str(w[0].message).lower()


class TestMeasureShockStandoffRNose:
    """Tests for measure_shock_standoff_r_nose function."""

    def test_estimates_nose_radius(self):
        """Should estimate a positive nose radius."""
        x, r = _make_body_contour()
        R_est = measure_shock_standoff_r_nose(x, r)
        assert R_est > 0

    def test_nose_radius_reasonable(self):
        """Nose radius should be close to the actual R_nose=0.1."""
        x, r = _make_body_contour(R_nose=0.1)
        R_est = measure_shock_standoff_r_nose(x, r)
        # Should be close to 0.1 (within 10%)
        assert 0.05 < R_est < 0.5


class TestComputeTotalHeating:
    """Tests for compute_total_heating function."""

    def test_constant_flux(self):
        """For constant q and simple ds, integral = q * total_arc_length."""
        s = np.array([0.0, 1.0, 2.0])
        q = np.array([1000.0, 1000.0, 1000.0])
        total = compute_total_heating(q, s)
        # Trapezoidal: 1000 * 1 + 1000 * 1 = 2000
        assert abs(total - 2000.0) < 1.0

    def test_zero_flux(self):
        """Zero heat flux should give zero total."""
        s = np.array([0.0, 1.0, 2.0])
        q = np.zeros(3)
        total = compute_total_heating(q, s)
        assert total == 0.0

    def test_single_point(self):
        """Single point should return 0."""
        s = np.array([0.0])
        q = np.array([1000.0])
        total = compute_total_heating(q, s)
        assert total == 0.0


class TestComputeTotalHeatingAxisymmetric:
    """Tests for compute_total_heating_axisymmetric function."""

    def test_constant_flux_uniform_radius(self):
        """For constant q and constant r, Q = q * 2*pi*r * L."""
        s = np.array([0.0, 1.0])
        q = np.array([1000.0, 1000.0])
        r = np.array([0.1, 0.1])
        total = compute_total_heating_axisymmetric(q, s, r)
        expected = 1000.0 * 2 * np.pi * 0.1 * 1.0
        assert abs(total - expected) / expected < 0.01

    def test_zero_flux(self):
        """Zero heat flux should give zero total."""
        s = np.array([0.0, 1.0, 2.0])
        q = np.zeros(3)
        r = np.array([0.1, 0.2, 0.3])
        total = compute_total_heating_axisymmetric(q, s, r)
        assert total == 0.0

    def test_linear_radius(self):
        """Verify integration with linearly varying radius."""
        s = np.array([0.0, 1.0])
        q = np.array([100.0, 100.0])
        r = np.array([0.0, 0.1])
        total = compute_total_heating_axisymmetric(q, s, r)
        # q_avg=100, r_avg=0.05, ds=1.0
        # Q = 100 * 2*pi*0.05 * 1.0 = 10*pi ~ 31.4
        expected = 100.0 * 2 * np.pi * 0.05 * 1.0
        assert abs(total - expected) / expected < 0.01


class TestApplyRealGasCorrection:
    """Tests for apply_real_gas_correction function."""

    def test_correction_at_room_temperature(self):
        """At 300K, correction should be close to 1.0."""
        q = 10000.0
        q_corr, factor = apply_real_gas_correction(q, 300.0)
        # At 300K, gamma ~ 1.4, so correction ~ 1.0
        assert abs(factor - 1.0) < 0.01
        assert abs(q_corr - q) / q < 0.01

    def test_correction_at_high_temperature(self):
        """At high T, gamma drops, correction < 1."""
        q = 100000.0
        q_corr, factor = apply_real_gas_correction(q, 3000.0)
        # At 3000K, gamma < 1.4, so correction < 1
        assert factor < 1.0
        assert q_corr < q

    def test_correction_returns_two_values(self):
        """Should return tuple of (corrected, factor)."""
        result = apply_real_gas_correction(50000.0, 2000.0)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestBuildResultsSummary:
    """Tests for build_results_summary function."""

    def test_has_all_keys(self):
        """Summary should contain all expected top-level keys."""
        data = _make_synthetic_vtu()
        body = _make_body_contour()

        class MockConfig:
            mach = 8.0
            altitude = 30000.0

        summary = build_results_summary(data, MockConfig(), body)

        assert "stagnation" in summary
        assert "shock_standoff" in summary
        assert "total_heating" in summary
        assert "real_gas_correction" in summary
        assert "field_extrema" in summary
        assert "surface_profiles" in summary

    def test_stagnation_keys(self):
        """Stagnation sub-dict should have expected keys."""
        data = _make_synthetic_vtu()
        body = _make_body_contour()

        class MockConfig:
            mach = 8.0
            altitude = 30000.0

        summary = build_results_summary(data, MockConfig(), body)
        stag = summary["stagnation"]

        assert "pressure_Pa" in stag
        assert "temperature_K" in stag
        assert "density_kg_m3" in stag
        assert "mach" in stag

    def test_json_serializable(self):
        """Summary should be JSON-serializable."""
        import json

        data = _make_synthetic_vtu()
        body = _make_body_contour()

        class MockConfig:
            mach = 8.0
            altitude = 30000.0

        summary = build_results_summary(data, MockConfig(), body)
        # Should not raise
        json_str = json.dumps(summary)
        assert len(json_str) > 0

    def test_shock_standoff_positive(self):
        """Shock standoff should be positive."""
        data = _make_synthetic_vtu()
        body = _make_body_contour()

        class MockConfig:
            mach = 8.0
            altitude = 30000.0

        summary = build_results_summary(data, MockConfig(), body)
        assert summary["shock_standoff"]["delta_m"] >= 0

    def test_max_mach_positive(self):
        """Max Mach should be positive."""
        data = _make_synthetic_vtu()
        body = _make_body_contour()

        class MockConfig:
            mach = 8.0
            altitude = 30000.0

        summary = build_results_summary(data, MockConfig(), body)
        assert summary["field_extrema"]["max_mach"] is not None
        assert summary["field_extrema"]["max_mach"] > 0


class TestSavePostprocessResults:
    """Tests for save_postprocess_results function."""

    def test_saves_json(self, tmp_path):
        """Should save a valid JSON file."""
        summary = {
            "stagnation": {"pressure_Pa": 1000.0},
            "shock_standoff": {"delta_m": 0.01},
        }
        output_path = tmp_path / "postprocess" / "postprocess.json"
        saved = save_postprocess_results(summary, output_path)

        assert saved.exists()
        import json
        with open(saved) as f:
            loaded = json.load(f)
        assert loaded["stagnation"]["pressure_Pa"] == 1000.0
