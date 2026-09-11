"""Tests for thermal material database.

Covers ThermalMaterial dataclass, piecewise interpolation, pre-defined
materials, and the get_material lookup function.
"""
import pytest

from thermal.materials import (
    AVCOAT,
    PICA,
    MATERIALS,
    ThermalMaterial,
    _piecewise_interpolate,
    get_material,
)


class TestThermalMaterial:
    """Tests for ThermalMaterial dataclass."""

    def test_frozen_dataclass(self):
        """ThermalMaterial should be immutable."""
        mat = AVCOAT
        with pytest.raises(AttributeError):
            mat.density = 999.0  # type: ignore[misc]

    def test_k_at_ref_temperature(self):
        """k_at should return reference k at first table temperature."""
        # AVCOAT k_table starts at (300, 0.5)
        assert AVCOAT.k_at(300.0) == pytest.approx(0.5)

    def test_cp_at_ref_temperature(self):
        """cp_at should return reference cp at first table temperature."""
        # AVCOAT cp_table starts at (300, 1000)
        assert AVCOAT.cp_at(300.0) == pytest.approx(1000.0)

    def test_k_interpolation(self):
        """k_at should interpolate linearly between table entries."""
        # AVCOAT: k=0.8 at T=600, k=1.2 at T=1000
        # At T=800 (midpoint), k should be 1.0
        k_800 = AVCOAT.k_at(800.0)
        assert k_800 == pytest.approx(1.0, abs=0.01)

    def test_cp_interpolation(self):
        """cp_at should interpolate linearly between table entries."""
        # AVCOAT: cp=1200 at T=600, cp=1500 at T=1000
        # At T=800 (midpoint), cp should be 1350
        cp_800 = AVCOAT.cp_at(800.0)
        assert cp_800 == pytest.approx(1350.0, abs=1.0)

    def test_k_clamp_low(self):
        """k_at should clamp to table minimum for T below table range."""
        # AVCOAT k_table starts at T=300 with k=0.5
        assert AVCOAT.k_at(100.0) == pytest.approx(0.5)

    def test_k_clamp_high(self):
        """k_at should clamp to table maximum for T above table range."""
        # AVCOAT k_table ends at T=2000 with k=2.5
        assert AVCOAT.k_at(5000.0) == pytest.approx(2.5)

    def test_cp_clamp_low(self):
        """cp_at should clamp to table minimum for T below table range."""
        assert AVCOAT.cp_at(100.0) == pytest.approx(1000.0)

    def test_cp_clamp_high(self):
        """cp_at should clamp to table maximum for T above table range."""
        assert AVCOAT.cp_at(5000.0) == pytest.approx(2000.0)

    def test_constant_material_k(self):
        """Material without k_table should return constant k."""
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
        assert mat.k_at(300.0) == 2.5
        assert mat.k_at(3000.0) == 2.5

    def test_constant_material_cp(self):
        """Material without cp_table should return constant cp."""
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
        assert mat.cp_at(300.0) == 800.0
        assert mat.cp_at(3000.0) == 800.0


class TestPiecewiseInterpolate:
    """Tests for _piecewise_interpolate helper."""

    def test_exact_first_point(self):
        """Should return exact value at first table point."""
        table = ((100, 1.0), (200, 2.0), (300, 3.0))
        assert _piecewise_interpolate(100, table) == 1.0

    def test_exact_last_point(self):
        """Should return exact value at last table point."""
        table = ((100, 1.0), (200, 2.0), (300, 3.0))
        assert _piecewise_interpolate(300, table) == 3.0

    def test_midpoint(self):
        """Should return midpoint value at midpoint temperature."""
        table = ((100, 1.0), (300, 3.0))
        assert _piecewise_interpolate(200, table) == 2.0

    def test_below_range(self):
        """Should clamp to first value below table range."""
        table = ((100, 1.0), (200, 2.0))
        assert _piecewise_interpolate(50, table) == 1.0

    def test_above_range(self):
        """Should clamp to last value above table range."""
        table = ((100, 1.0), (200, 2.0))
        assert _piecewise_interpolate(500, table) == 2.0

    def test_nonlinear_table(self):
        """Should interpolate correctly in non-uniform table."""
        table = ((0, 0), (10, 100), (100, 200))
        # At T=5 (midpoint of first segment): 0 + 0.5 * 100 = 50
        assert _piecewise_interpolate(5, table) == 50.0
        # At T=55 (midpoint of second segment): 100 + 0.5 * 100 = 150
        assert _piecewise_interpolate(55, table) == 150.0


class TestPreDefinedMaterials:
    """Tests for pre-defined AVCOAT and PICA materials."""

    def test_avcoat_properties(self):
        """AVCOAT should have correct base properties."""
        assert AVCOAT.name == "AVCOAT-5026"
        assert AVCOAT.density == 512.0
        assert AVCOAT.thermal_conductivity == 0.5
        assert AVCOAT.specific_heat == 1000.0
        assert AVCOAT.decomposition_temperature == 600.0
        assert AVCOAT.char_temperature == 3000.0
        assert AVCOAT.heat_of_pyrolysis == 1.5e6
        assert AVCOAT.emissivity == 0.8

    def test_pica_properties(self):
        """PICA should have correct base properties."""
        assert PICA.name == "PICA-X"
        assert PICA.density == 240.0
        assert PICA.thermal_conductivity == 0.5
        assert PICA.specific_heat == 1000.0
        assert PICA.decomposition_temperature == 500.0
        assert PICA.char_temperature == 3200.0
        assert PICA.heat_of_pyrolysis == 2.0e6
        assert PICA.emissivity == 0.85

    def test_avcoat_has_tables(self):
        """AVCOAT should have k and cp tables."""
        assert AVCOAT.k_table is not None
        assert AVCOAT.cp_table is not None

    def test_pica_has_tables(self):
        """PICA should have k and cp tables."""
        assert PICA.k_table is not None
        assert PICA.cp_table is not None

    def test_materials_dict(self):
        """MATERIALS dict should contain both materials."""
        assert "avcoat" in MATERIALS
        assert "pica" in MATERIALS

    def test_avcoat_k_increases_with_temp(self):
        """AVCOAT conductivity should increase with temperature."""
        k_300 = AVCOAT.k_at(300.0)
        k_1000 = AVCOAT.k_at(1000.0)
        k_2000 = AVCOAT.k_at(2000.0)
        assert k_300 < k_1000 < k_2000

    def test_avcoat_cp_increases_with_temp(self):
        """AVCOAT specific heat should increase with temperature."""
        cp_300 = AVCOAT.cp_at(300.0)
        cp_1000 = AVCOAT.cp_at(1000.0)
        cp_2000 = AVCOAT.cp_at(2000.0)
        assert cp_300 < cp_1000 < cp_2000


class TestGetMaterial:
    """Tests for get_material lookup function."""

    def test_avcoat_lowercase(self):
        """Should find AVCOAT by lowercase name."""
        mat = get_material("avcoat")
        assert mat is AVCOAT

    def test_avcoat_uppercase(self):
        """Should find AVCOAT by uppercase name."""
        mat = get_material("AVCOAT")
        assert mat is AVCOAT

    def test_avcoat_mixed_case(self):
        """Should find AVCOAT by mixed case name."""
        mat = get_material("AvCoAt")
        assert mat is AVCOAT

    def test_pica_lowercase(self):
        """Should find PICA by lowercase name."""
        mat = get_material("pica")
        assert mat is PICA

    def test_pica_with_whitespace(self):
        """Should find PICA with surrounding whitespace."""
        mat = get_material("  pica  ")
        assert mat is PICA

    def test_unknown_material_raises(self):
        """Should raise ValueError for unknown material."""
        with pytest.raises(ValueError, match="Unknown material"):
            get_material("titanium")

    def test_unknown_material_message(self):
        """Error message should list available materials."""
        with pytest.raises(ValueError, match="avcoat") as exc_info:
            get_material("carbon_carbon")
        assert "pica" in str(exc_info.value)

    def test_return_type(self):
        """Should return ThermalMaterial instance."""
        mat = get_material("avcoat")
        assert isinstance(mat, ThermalMaterial)
