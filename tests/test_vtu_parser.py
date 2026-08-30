"""Tests for VTU parser (ASCII format)."""
from pathlib import Path

import numpy as np
import pytest

from cfd.vtu_parser import (
    VTUData,
    extract_stagnation_values,
    extract_surface_heat_flux,
    parse_vtu,
)

# Minimal valid ASCII VTU for testing (5 nodes, 2 triangles)
_MINIMAL_VTU = """\
<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="5" NumberOfCells="2">
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">
          0.0 0.0 0.0  1.0 0.0 0.0  0.5 0.5 0.0  1.5 0.5 0.0  0.0 1.0 0.0
        </DataArray>
      </Points>
      <CellData>
      </CellData>
      <PointData>
        <DataArray type="Float32" Name="Mach" format="ascii">
          0.0 0.5 0.8 2.1 0.3
        </DataArray>
        <DataArray type="Float32" Name="Pressure" format="ascii">
          50000.0 48000.0 45000.0 20000.0 49000.0
        </DataArray>
        <DataArray type="Float32" Name="Temperature" format="ascii">
          300.0 295.0 290.0 250.0 298.0
        </DataArray>
        <DataArray type="Float32" Name="Density" format="ascii">
          1.225 1.200 1.150 0.900 1.210
        </DataArray>
      </PointData>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">
          0 1 2  1 3 2
        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
          3 6
        </DataArray>
        <DataArray type="Int32" Name="types" format="ascii">
          5 5
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

# VTU with no point data (empty fields)
_EMPTY_FIELDS_VTU = """\
<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="3" NumberOfCells="1">
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">
          0.0 0.0 0.0  1.0 0.0 0.0  0.5 0.5 0.0
        </DataArray>
      </Points>
      <PointData>
      </PointData>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">
          0 1 2
        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
          3
        </DataArray>
        <DataArray type="Int32" Name="types" format="ascii">
          5
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

# VTU with multi-component velocity field
_VELOCITY_VTU = """\
<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="3" NumberOfCells="1">
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">
          0.0 0.0 0.0  1.0 0.0 0.0  0.5 0.5 0.0
        </DataArray>
      </Points>
      <PointData>
        <DataArray type="Float32" NumberOfComponents="3" Name="Velocity" format="ascii">
          100.0 0.0 0.0  150.0 10.0 0.0  120.0 5.0 0.0
        </DataArray>
      </PointData>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">
          0 1 2
        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
          3
        </DataArray>
        <DataArray type="Int32" Name="types" format="ascii">
          5
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""


@pytest.fixture
def minimal_vtu_path(tmp_path: Path) -> Path:
    """Write minimal VTU to temp file and return path."""
    path = tmp_path / "test.vtu"
    path.write_text(_MINIMAL_VTU)
    return path


@pytest.fixture
def empty_fields_vtu_path(tmp_path: Path) -> Path:
    """Write empty-fields VTU to temp file and return path."""
    path = tmp_path / "empty.vtu"
    path.write_text(_EMPTY_FIELDS_VTU)
    return path


@pytest.fixture
def velocity_vtu_path(tmp_path: Path) -> Path:
    """Write velocity-field VTU to temp file and return path."""
    path = tmp_path / "velocity.vtu"
    path.write_text(_VELOCITY_VTU)
    return path


class TestParseVTU:
    """Tests for parse_vtu function."""

    def test_parse_coordinates(self, minimal_vtu_path: Path):
        """Should extract correct coordinates."""
        data = parse_vtu(minimal_vtu_path)
        assert data.coordinates.shape == (5, 3)
        np.testing.assert_allclose(data.coordinates[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(data.coordinates[4], [0.0, 1.0, 0.0])

    def test_parse_mach(self, minimal_vtu_path: Path):
        """Should extract Mach number field."""
        data = parse_vtu(minimal_vtu_path)
        assert data.mach is not None
        np.testing.assert_allclose(data.mach, [0.0, 0.5, 0.8, 2.1, 0.3])

    def test_parse_pressure(self, minimal_vtu_path: Path):
        """Should extract pressure field."""
        data = parse_vtu(minimal_vtu_path)
        assert data.pressure is not None
        np.testing.assert_allclose(
            data.pressure, [50000.0, 48000.0, 45000.0, 20000.0, 49000.0],
        )

    def test_parse_temperature(self, minimal_vtu_path: Path):
        """Should extract temperature field."""
        data = parse_vtu(minimal_vtu_path)
        assert data.temperature is not None
        np.testing.assert_allclose(
            data.temperature, [300.0, 295.0, 290.0, 250.0, 298.0],
        )

    def test_parse_density(self, minimal_vtu_path: Path):
        """Should extract density field."""
        data = parse_vtu(minimal_vtu_path)
        assert data.density is not None
        np.testing.assert_allclose(
            data.density, [1.225, 1.200, 1.150, 0.900, 1.210],
        )

    def test_empty_fields(self, empty_fields_vtu_path: Path):
        """VTU with no point data should parse without error."""
        data = parse_vtu(empty_fields_vtu_path)
        assert data.coordinates.shape == (3, 3)
        assert data.mach is None
        assert data.pressure is None

    def test_velocity_component_split(self, velocity_vtu_path: Path):
        """Multi-component Velocity field should be available in point_data."""
        data = parse_vtu(velocity_vtu_path)
        assert "Velocity" in data.point_data
        vel = data.point_data["Velocity"]
        assert vel.shape == (3, 3)
        np.testing.assert_allclose(vel[0], [100.0, 0.0, 0.0])


class TestExtractStagnationValues:
    """Tests for extract_stagnation_values function."""

    def test_finds_max_pressure(self, minimal_vtu_path: Path):
        """Should find the node with maximum pressure."""
        data = parse_vtu(minimal_vtu_path)
        stag = extract_stagnation_values(data)
        # Node 0 has highest pressure (50000.0)
        assert stag["Pressure"] == 50000.0

    def test_stagnation_temperature(self, minimal_vtu_path: Path):
        """Should return temperature at max-pressure node."""
        data = parse_vtu(minimal_vtu_path)
        stag = extract_stagnation_values(data)
        assert stag["Temperature"] == 300.0

    def test_stagnation_mach(self, minimal_vtu_path: Path):
        """Should return Mach at max-pressure node."""
        data = parse_vtu(minimal_vtu_path)
        stag = extract_stagnation_values(data)
        assert stag["Mach"] == 0.0  # stagnation point: Mach = 0

    def test_stagnation_density(self, minimal_vtu_path: Path):
        """Should return density at max-pressure node."""
        data = parse_vtu(minimal_vtu_path)
        stag = extract_stagnation_values(data)
        assert stag["Density"] == 1.225

    def test_stagnation_coordinates(self, minimal_vtu_path: Path):
        """Should return coordinates at max-pressure node."""
        data = parse_vtu(minimal_vtu_path)
        stag = extract_stagnation_values(data)
        assert stag["x"] == 0.0
        assert stag["r"] == 0.0

    def test_empty_data(self):
        """Empty VTU data should return empty dict."""
        data = VTUData(coordinates=np.zeros((3, 3)))
        stag = extract_stagnation_values(data)
        assert stag == {}


class TestExtractSurfaceHeatFlux:
    """Tests for extract_surface_heat_flux function."""

    def test_no_heat_flux_field(self, minimal_vtu_path: Path):
        """Should return None when heat flux field is absent."""
        data = parse_vtu(minimal_vtu_path)
        hf = extract_surface_heat_flux(data)
        assert hf is None

    def test_with_heat_flux_field(self, tmp_path: Path):
        """Should extract Heat_Flux field if present."""
        vtu_content = """\
<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="3" NumberOfCells="1">
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">
          0.0 0.0 0.0  1.0 0.0 0.0  0.5 0.5 0.0
        </DataArray>
      </Points>
      <PointData>
        <DataArray type="Float32" Name="Heat_Flux" format="ascii">
          100000.0 50000.0 75000.0
        </DataArray>
      </PointData>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">
          0 1 2
        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
          3
        </DataArray>
        <DataArray type="Int32" Name="types" format="ascii">
          5
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""
        path = tmp_path / "heat_flux.vtu"
        path.write_text(vtu_content)
        data = parse_vtu(path)
        hf = extract_surface_heat_flux(data)
        assert hf is not None
        np.testing.assert_allclose(hf, [100000.0, 50000.0, 75000.0])


class TestVTUDataProperties:
    """Tests for VTUData convenience properties."""

    def test_mach_property(self, minimal_vtu_path: Path):
        """mach property should return Mach array."""
        data = parse_vtu(minimal_vtu_path)
        assert data.mach is not None
        assert len(data.mach) == 5

    def test_pressure_property(self, minimal_vtu_path: Path):
        """pressure property should return Pressure array."""
        data = parse_vtu(minimal_vtu_path)
        assert data.pressure is not None
        assert len(data.pressure) == 5

    def test_temperature_property(self, minimal_vtu_path: Path):
        """temperature property should return Temperature array."""
        data = parse_vtu(minimal_vtu_path)
        assert data.temperature is not None
        assert len(data.temperature) == 5

    def test_density_property(self, minimal_vtu_path: Path):
        """density property should return Density array."""
        data = parse_vtu(minimal_vtu_path)
        assert data.density is not None
        assert len(data.density) == 5
