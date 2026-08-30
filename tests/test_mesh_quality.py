"""Tests for mesh quality metrics and SU2 mesh validation."""
import math
from pathlib import Path

import pytest

from cfd.mesh_quality import (
    _element_area,
    _element_quality,
    _parse_su2_mesh,
    check_mesh_quality,
    estimate_y_plus,
    validate_su2_mesh,
)

# Minimal valid 2D SU2 mesh with a single quad element
_MINIMAL_SU2 = """\
NDIME= 2
NELEM= 1
9 0 1 2 3
NPOIN= 4
0.0 0.0 0.0
1.0 0.0 0.0
1.0 1.0 0.0
0.0 1.0 0.0
NMARK= 3
MARKER_TAG= body
MARKER_ELEMS= 1
3 0 1
MARKER_TAG= farfield
MARKER_ELEMS= 1
3 1 2
MARKER_TAG= sym
MARKER_ELEMS= 1
3 2 3
"""

# Invalid SU2 mesh (no markers)
_MINIMAL_SU2_NO_MARKERS = """\
NDIME= 2
NELEM= 1
9 0 1 2 3
NPOIN= 4
0.0 0.0 0.0
1.0 0.0 0.0
1.0 1.0 0.0
0.0 1.0 0.0
NMARK= 0
"""

# SU2 mesh with negative-volume (inverted) element
_MINIMAL_SU2_NEGATIVE = """\
NDIME= 2
NELEM= 1
9 0 3 2 1
NPOIN= 4
0.0 0.0 0.0
1.0 0.0 0.0
1.0 1.0 0.0
0.0 1.0 0.0
NMARK= 3
MARKER_TAG= body
MARKER_ELEMS= 1
3 0 1
MARKER_TAG= farfield
MARKER_ELEMS= 1
3 1 2
MARKER_TAG= sym
MARKER_ELEMS= 1
3 2 3
"""


@pytest.fixture
def minimal_su2(tmp_path: Path) -> Path:
    """Write a minimal valid SU2 mesh to a temp file and return its path."""
    mesh_file = tmp_path / "test.su2"
    mesh_file.write_text(_MINIMAL_SU2)
    return mesh_file


@pytest.fixture
def no_marker_su2(tmp_path: Path) -> Path:
    """Write a SU2 mesh with no markers to a temp file."""
    mesh_file = tmp_path / "no_marker.su2"
    mesh_file.write_text(_MINIMAL_SU2_NO_MARKERS)
    return mesh_file


@pytest.fixture
def negative_su2(tmp_path: Path) -> Path:
    """Write a SU2 mesh with a negative-volume element."""
    mesh_file = tmp_path / "negative.su2"
    mesh_file.write_text(_MINIMAL_SU2_NEGATIVE)
    return mesh_file


class TestParseSu2Mesh:
    """Tests for _parse_su2_mesh."""

    def test_parse_nodes(self, minimal_su2: Path):
        """Should parse 4 nodes from the minimal mesh."""
        nodes, _elements, _markers = _parse_su2_mesh(minimal_su2)
        assert len(nodes) == 4
        assert nodes.shape[1] == 3

    def test_parse_elements(self, minimal_su2: Path):
        """Should parse 1 element from the minimal mesh."""
        _nodes, elements, _markers = _parse_su2_mesh(minimal_su2)
        assert len(elements) == 1
        assert elements[0][0] == 9  # quad element type

    def test_parse_markers(self, minimal_su2: Path):
        """Should parse 3 markers from the minimal mesh."""
        _nodes, _elements, markers = _parse_su2_mesh(minimal_su2)
        assert "body" in markers
        assert "farfield" in markers
        assert "sym" in markers
        assert len(markers["body"]) == 1

    def test_parse_empty_file(self, tmp_path: Path):
        """Empty file should raise ValueError."""
        empty_file = tmp_path / "empty.su2"
        empty_file.write_text("")
        with pytest.raises(ValueError, match="No nodes"):
            _parse_su2_mesh(empty_file)


class TestElementArea:
    """Tests for _element_area."""

    def test_unit_square(self, minimal_su2: Path):
        """Unit square element should have area 1.0."""
        nodes, elements, _ = _parse_su2_mesh(minimal_su2)
        area = _element_area(nodes, elements[0])
        assert abs(area) == pytest.approx(1.0, abs=0.01)

    def test_triangle_area(self):
        """Right triangle with legs 1 and 1 should have area 0.5."""
        import numpy as np
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        elem = np.array([3, 0, 1, 2])  # triangle
        area = _element_area(nodes, elem)
        assert abs(area) == pytest.approx(0.5, abs=0.01)


class TestElementQuality:
    """Tests for _element_quality."""

    def test_unit_square_quality(self, minimal_su2: Path):
        """Unit square should have high quality."""
        nodes, elements, _ = _parse_su2_mesh(minimal_su2)
        area = _element_area(nodes, elements[0])
        q = _element_quality(area, elements[0], nodes)
        assert 0.8 <= q <= 1.0, f"Unit square quality {q} out of range"


class TestValidateSu2Mesh:
    """Tests for validate_su2_mesh."""

    def test_valid_mesh(self, minimal_su2: Path):
        """Valid mesh should return True."""
        assert validate_su2_mesh(minimal_su2) is True

    def test_no_markers(self, no_marker_su2: Path):
        """Mesh without required markers should return False."""
        assert validate_su2_mesh(no_marker_su2) is False

    def test_negative_volume(self, negative_su2: Path):
        """Mesh with negative-volume cells should return False."""
        assert validate_su2_mesh(negative_su2) is False

    def test_nonexistent_file(self, tmp_path: Path):
        """Non-existent file should return False."""
        assert validate_su2_mesh(tmp_path / "nope.su2") is False


class TestCheckMeshQuality:
    """Tests for check_mesh_quality."""

    def test_returns_expected_keys(self, minimal_su2: Path):
        """Should return all expected quality keys."""
        result = check_mesh_quality(minimal_su2)
        expected_keys = {
            "n_cells", "min_quality", "mean_quality", "pct_bad_cells",
            "has_body_marker", "has_farfield_marker", "has_sym_marker",
        }
        assert set(result.keys()) == expected_keys

    def test_cell_count(self, minimal_su2: Path):
        """Should report 1 cell for the minimal mesh."""
        result = check_mesh_quality(minimal_su2)
        assert result["n_cells"] == 1

    def test_markers_present(self, minimal_su2: Path):
        """Should detect all three required markers."""
        result = check_mesh_quality(minimal_su2)
        assert result["has_body_marker"] is True
        assert result["has_farfield_marker"] is True
        assert result["has_sym_marker"] is True

    def test_nonexistent_file(self, tmp_path: Path):
        """Non-existent file should return zero metrics."""
        result = check_mesh_quality(tmp_path / "nope.su2")
        assert result["n_cells"] == 0
        assert result["has_body_marker"] is False


class TestEstimateYPlus:
    """Tests for estimate_y_plus."""

    def test_sanity(self):
        """Y+ should be a positive finite number for reasonable inputs."""
        y_plus = estimate_y_plus(
            first_cell_height=1e-5,
            freestream_velocity=2500.0,
            freestream_density=0.01,
            freestream_viscosity=1e-5,
            wall_shear_estimate=1.0,
        )
        assert math.isfinite(y_plus)
        assert y_plus > 0

    def test_nan_on_invalid_inputs(self):
        """NaN should be returned for non-positive density or shear."""
        assert math.isnan(estimate_y_plus(
            first_cell_height=1e-5,
            freestream_velocity=2500.0,
            freestream_density=0.0,
            freestream_viscosity=1e-5,
            wall_shear_estimate=1.0,
        ))
        assert math.isnan(estimate_y_plus(
            first_cell_height=1e-5,
            freestream_velocity=2500.0,
            freestream_density=0.01,
            freestream_viscosity=1e-5,
            wall_shear_estimate=0.0,
        ))
        assert math.isnan(estimate_y_plus(
            first_cell_height=0.0,
            freestream_velocity=2500.0,
            freestream_density=0.01,
            freestream_viscosity=1e-5,
            wall_shear_estimate=1.0,
        ))

    def test_scales_with_height(self):
        """Y+ should scale linearly with first cell height."""
        y1 = estimate_y_plus(
            first_cell_height=1e-5,
            freestream_velocity=2500.0,
            freestream_density=0.01,
            freestream_viscosity=1e-5,
            wall_shear_estimate=1.0,
        )
        y2 = estimate_y_plus(
            first_cell_height=2e-5,
            freestream_velocity=2500.0,
            freestream_density=0.01,
            freestream_viscosity=1e-5,
            wall_shear_estimate=1.0,
        )
        assert y2 == pytest.approx(2.0 * y1)
