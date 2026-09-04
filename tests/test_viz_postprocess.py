"""Tests for post-processing visualization functions.

Verifies that plot functions produce valid PNG files using synthetic VTUData.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from cfd.vtu_parser import VTUData


def _make_vtu_for_viz(
    n_nodes: int = 200,
) -> VTUData:
    """Create synthetic VTUData for visualization tests."""
    # Create a simple grid
    x = np.linspace(-0.5, 2.0, 50)
    r = np.linspace(0, 0.5, 4)
    xx, rr = np.meshgrid(x, r)
    x_flat = xx.ravel()
    r_flat = rr.ravel()
    z_flat = np.zeros_like(x_flat)
    coords = np.column_stack([x_flat, r_flat, z_flat])

    # Simple fields
    shock_x = 0.05
    mach = np.where(x_flat > shock_x, 8.0, 2.4)
    mach = np.where(x_flat < 0, 0.0, mach)

    density = np.where(x_flat > shock_x, 0.018, 0.09)
    pressure = np.where(x_flat > shock_x, 1172.0, 82000.0)
    temperature = np.where(x_flat > shock_x, 227.0, 2270.0)

    point_data = {
        "Mach": mach,
        "Pressure": pressure,
        "Temperature": temperature,
        "Density": density,
        "Heat_Flux": np.where(x_flat < 0.1, 30000.0, 5000.0),
    }

    return VTUData(coordinates=coords, point_data=point_data)


def _make_body_contour() -> tuple[np.ndarray, np.ndarray]:
    """Simple body contour for testing."""
    phi = np.linspace(0, 0.785, 50)
    x = 0.1 * np.sin(phi)
    r = 0.1 * (1 - np.cos(phi))
    # Add cone
    x_cone = np.linspace(x[-1], x[-1] + 0.3, 30)
    r_cone = np.linspace(r[-1], r[-1] + 0.3, 30)
    x = np.concatenate([x, x_cone[1:]])
    r = np.concatenate([r, r_cone[1:]])
    return x, r


class TestPlotMachContour:
    """Tests for plot_mach_contour."""

    def test_creates_png(self, tmp_path):
        """Should create a PNG file."""
        from viz.contour import plot_mach_contour
        data = _make_vtu_for_viz()
        output = tmp_path / "mach_contour.png"
        result = plot_mach_contour(data, output)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_with_body_contour(self, tmp_path):
        """Should work with body contour overlay."""
        from viz.contour import plot_mach_contour
        data = _make_vtu_for_viz()
        body = _make_body_contour()
        output = tmp_path / "mach_contour_body.png"
        result = plot_mach_contour(data, output, body_contour=body)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_no_mach_data(self, tmp_path):
        """Should handle missing Mach data gracefully."""
        from viz.contour import plot_mach_contour
        data = VTUData(coordinates=np.zeros((10, 3)))
        output = tmp_path / "mach_empty.png"
        result = plot_mach_contour(data, output)
        assert result.exists()


class TestPlotPressureContour:
    """Tests for plot_pressure_contour."""

    def test_creates_png(self, tmp_path):
        """Should create a PNG file."""
        from viz.contour import plot_pressure_contour
        data = _make_vtu_for_viz()
        output = tmp_path / "pressure_contour.png"
        result = plot_pressure_contour(data, output)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_no_pressure_data(self, tmp_path):
        """Should handle missing pressure data gracefully."""
        from viz.contour import plot_pressure_contour
        data = VTUData(coordinates=np.zeros((10, 3)))
        output = tmp_path / "pressure_empty.png"
        result = plot_pressure_contour(data, output)
        assert result.exists()


class TestPlotTemperatureContour:
    """Tests for plot_temperature_contour."""

    def test_creates_png(self, tmp_path):
        """Should create a PNG file."""
        from viz.contour import plot_temperature_contour
        data = _make_vtu_for_viz()
        output = tmp_path / "temperature_contour.png"
        result = plot_temperature_contour(data, output)
        assert result.exists()
        assert result.stat().st_size > 0


class TestPlotSurfaceHeatFlux:
    """Tests for plot_surface_heat_flux."""

    def test_creates_png(self, tmp_path):
        """Should create a PNG file."""
        from viz.heat_flux import plot_surface_heat_flux
        s = np.linspace(0, 0.5, 50)
        q = 50000.0 * np.exp(-s / 0.1)
        output = tmp_path / "heat_flux.png"
        result = plot_surface_heat_flux(s, q, output)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_empty_data(self, tmp_path):
        """Should handle empty arrays gracefully."""
        from viz.heat_flux import plot_surface_heat_flux
        s = np.array([])
        q = np.array([])
        output = tmp_path / "heat_flux_empty.png"
        result = plot_surface_heat_flux(s, q, output)
        assert result.exists()


class TestPlotShockStructure:
    """Tests for plot_shock_structure."""

    def test_creates_png(self, tmp_path):
        """Should create a PNG file."""
        from viz.shock import plot_shock_structure
        data = _make_vtu_for_viz()
        output = tmp_path / "shock_structure.png"
        result = plot_shock_structure(data, output)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_with_body_contour(self, tmp_path):
        """Should work with body contour overlay."""
        from viz.shock import plot_shock_structure
        data = _make_vtu_for_viz()
        body = _make_body_contour()
        output = tmp_path / "shock_structure_body.png"
        result = plot_shock_structure(data, output, body_contour=body)
        assert result.exists()


class TestPlotShockStandoffMeasurement:
    """Tests for plot_shock_standoff_measurement."""

    def test_creates_png(self, tmp_path):
        """Should create a PNG file."""
        from viz.shock import plot_shock_standoff_measurement
        data = _make_vtu_for_viz()
        output = tmp_path / "shock_standoff.png"
        result = plot_shock_standoff_measurement(data, output, R_nose=0.1)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_no_density(self, tmp_path):
        """Should handle missing density gracefully."""
        from viz.shock import plot_shock_standoff_measurement
        data = VTUData(coordinates=np.zeros((10, 3)))
        output = tmp_path / "shock_empty.png"
        result = plot_shock_standoff_measurement(data, output, R_nose=0.1)
        assert result.exists()


class TestPlotBody3d:
    """Tests for plot_body_3d."""

    def test_creates_png(self, tmp_path):
        """Should create a PNG file."""
        from geometry.config import BluntBodyConfig
        from viz.geometry_3d import plot_body_3d

        config = BluntBodyConfig(
            R_shield=0.1,
            cone_half_angle=45.0,
            base_radius=0.5,
            num_points=100,
        )
        output = tmp_path / "body_3d.png"
        result = plot_body_3d(config, output)
        assert result.exists()
        assert result.stat().st_size > 0
