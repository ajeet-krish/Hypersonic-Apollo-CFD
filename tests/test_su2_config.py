"""Tests for SU2HypersonicConfig and config generation."""
from pathlib import Path

import pytest

from cfd.config import SU2HypersonicConfig, get_su2_binary


class TestSU2HypersonicConfigDefaults:
    """Tests for SU2HypersonicConfig default values."""

    def test_default_solver(self):
        """Default solver should be RANS."""
        config = SU2HypersonicConfig()
        assert config.solver == "RANS"

    def test_default_turb_model(self):
        """Default turbulence model should be SA."""
        config = SU2HypersonicConfig()
        assert config.turb_model == "SA"

    def test_default_axisymmetric(self):
        """Axisymmetric should be enabled by default."""
        config = SU2HypersonicConfig()
        assert config.axisymmetric is True

    def test_default_mach(self):
        """Default Mach number should be 8.0."""
        config = SU2HypersonicConfig()
        assert config.mach == 8.0

    def test_default_aoa(self):
        """Default angle of attack should be 0.0."""
        config = SU2HypersonicConfig()
        assert config.aoa == 0.0

    def test_default_freestream_pressure(self):
        """Default pressure should be ~1172 Pa (30 km atmosphere)."""
        config = SU2HypersonicConfig()
        assert config.freestream_pressure == 1172.0

    def test_default_freestream_temperature(self):
        """Default temperature should be ~226.65 K (30 km atmosphere)."""
        config = SU2HypersonicConfig()
        assert config.freestream_temperature == 226.65

    def test_default_wall_temperature(self):
        """Default wall temperature should be 2500 K."""
        config = SU2HypersonicConfig()
        assert config.wall_temperature == 2500.0

    def test_default_cfl(self):
        """Default CFL should be 0.1."""
        config = SU2HypersonicConfig()
        assert config.cfl_number == 0.1

    def test_default_iterations(self):
        """Default iterations should be 10000."""
        config = SU2HypersonicConfig()
        assert config.iterations == 10000

    def test_default_markers(self):
        """Default boundary markers should be body, farfield, sym."""
        config = SU2HypersonicConfig()
        assert config.wall_marker == "body"
        assert config.farfield_marker == "farfield"
        assert config.sym_marker == "sym"

    def test_default_gamma(self):
        """Default gamma should be 1.4."""
        config = SU2HypersonicConfig()
        assert config.gamma == 1.4

    def test_default_gas_constant(self):
        """Default gas constant should be 287.058."""
        config = SU2HypersonicConfig()
        assert config.gas_constant == 287.058

    def test_default_output_files(self):
        """Default output files should include RESTART, PARAVIEW, and SURFACE_CSV."""
        config = SU2HypersonicConfig()
        assert "RESTART" in config.output_files
        assert "PARAVIEW" in config.output_files
        assert "SURFACE_CSV" in config.output_files

    def test_default_turbulence_intensity(self):
        """Default turbulence intensity should be 0.05."""
        config = SU2HypersonicConfig()
        assert config.freestream_turbulence_intensity == 0.05

    def test_default_turbulence_viscosity_ratio(self):
        """Default turbulence viscosity ratio should be 10.0."""
        config = SU2HypersonicConfig()
        assert config.freestream_turbulence_viscosity_ratio == 10.0

    def test_default_cfl_adapt_min(self):
        """Default CFL adapt min should be 0.1."""
        config = SU2HypersonicConfig()
        assert config.cfl_adapt_min == 0.1

    def test_default_cfl_adapt_max(self):
        """Default CFL adapt max should be 2.0."""
        config = SU2HypersonicConfig()
        assert config.cfl_adapt_max == 2.0


class TestSU2HypersonicConfigWrite:
    """Tests for config file generation."""

    def test_write_produces_cfg(self, tmp_path: Path):
        """write() should produce a .cfg file."""
        config = SU2HypersonicConfig()
        result = config.write(tmp_path)
        assert result.exists()
        assert result.suffix == ".cfg"

    def test_cfg_contains_required_keys(self, tmp_path: Path):
        """Generated .cfg should contain all required SU2 keys."""
        config = SU2HypersonicConfig()
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()

        required_keys = [
            "SOLVER= RANS",
            "KIND_TURB_MODEL= SA",
            "MATH_PROBLEM= DIRECT",
            "AXISYMMETRIC= YES",
            "MACH_NUMBER= 8.0",
            "AOA= 0.0",
            "FREESTREAM_PRESSURE= 1172.0",
            "FREESTREAM_TEMPERATURE= 226.65",
            "FREESTREAM_DENSITY= 0.0184",
            "FREESTREAM_VISCOSITY= 1.477e-05",
            "MARKER_ISOTHERMAL= ( body, 2500.0 )",
            "MARKER_FAR= ( farfield )",
            "MARKER_SYM= ( sym )",
            "CONV_NUM_METHOD_FLOW= AUSM",
            "MUSCL_FLOW= YES",
            "SLOPE_LIMITER_FLOW= VENKATAKRISHNAN",
            "ENTROPY_FIX_COEFF= 0.1",
            "CONV_NUM_METHOD_TURB= SCALAR_UPWIND",
            "MUSCL_TURB= NO",
            "TIME_DISCRE_FLOW= EULER_IMPLICIT",
            "TIME_DISCRE_TURB= EULER_IMPLICIT",
            "LINEAR_SOLVER= BCGSTAB",
            "LINEAR_SOLVER_PREC= ILU",
            "CFL_NUMBER= 0.1",
            "CFL_ADAPT= YES",
            "ITER= 10000",
            "CONV_FIELD= RMS_DENSITY",
            "CONV_RESIDUAL_MINVAL= -6.0",
            "REF_AREA= 1.0",
            "REF_LENGTH= 1.0",
            "RESTART_SOL= NO",
            "MESH_FILENAME= mesh.su2",
            "MESH_FORMAT= SU2",
            "FREESTREAM_TURBULENCEINTENSITY= 0.05",
            "FREESTREAM_TURB2LAMVISCRATIO= 10.0",
        ]

        for key in required_keys:
            assert key in content, f"Missing key: {key}"

    def test_cfg_custom_mesh_filename(self, tmp_path: Path):
        """write() should use the provided mesh filename."""
        config = SU2HypersonicConfig()
        cfg_path = config.write(tmp_path, mesh_filename="custom.su2")
        content = cfg_path.read_text()
        assert "MESH_FILENAME= custom.su2" in content

    def test_cfg_no_restart_by_default(self, tmp_path: Path):
        """Default config should have RESTART_SOL= NO."""
        config = SU2HypersonicConfig()
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "RESTART_SOL= NO" in content

    def test_cfg_axisymmetric_no(self, tmp_path: Path):
        """Disabling axisymmetric should write NO."""
        config = SU2HypersonicConfig(axisymmetric=False)
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "AXISYMMETRIC= NO" in content

    def test_cfg_custom_mach(self, tmp_path: Path):
        """Custom Mach should appear in config."""
        config = SU2HypersonicConfig(mach=12.0)
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "MACH_NUMBER= 12.0" in content

    def test_cfg_custom_wall_temperature(self, tmp_path: Path):
        """Custom wall temperature should appear in isothermal BC."""
        config = SU2HypersonicConfig(wall_temperature=500.0)
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "MARKER_ISOTHERMAL= ( body, 500.0 )" in content

    def test_cfg_cfl_adapt_params(self, tmp_path: Path):
        """CFL adapt parameters should be present."""
        config = SU2HypersonicConfig()
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "CFL_ADAPT_PARAM= ( 0.1, 2.0, 0.5, 1.5 )" in content

    def test_cfg_cfl_adapt_params_custom(self, tmp_path: Path):
        """Custom CFL adapt parameters should appear in config."""
        config = SU2HypersonicConfig(
            cfl_adapt_min=0.005,
            cfl_adapt_max=0.5,
            cfl_adapt_decrease=0.5,
            cfl_adapt_increase=1.5,
        )
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "CFL_ADAPT_PARAM= ( 0.005, 0.5, 0.5, 1.5 )" in content

    def test_cfg_gas_properties(self, tmp_path: Path):
        """Gas properties should be in the config."""
        config = SU2HypersonicConfig()
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "GAMMA_VALUE= 1.4" in content
        assert "GAS_CONSTANT= 287.058" in content

    def test_cfg_freestream_rans(self, tmp_path: Path):
        """RANS config should include density, viscosity, and turbulence init."""
        config = SU2HypersonicConfig()
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "FREESTREAM_DENSITY=" in content
        assert "FREESTREAM_VISCOSITY=" in content
        assert "FREESTREAM_TURBULENCEINTENSITY=" in content
        assert "FREESTREAM_TURB2LAMVISCRATIO=" in content

    def test_cfg_output_files(self, tmp_path: Path):
        """Output files section should be correct."""
        config = SU2HypersonicConfig()
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "OUTPUT_FILES= ( RESTART, PARAVIEW, SURFACE_CSV )" in content

    def test_cfg_history_output(self, tmp_path: Path):
        """History output fields should be present."""
        config = SU2HypersonicConfig()
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "HISTORY_OUTPUT=" in content

    def test_write_creates_directory(self, tmp_path: Path):
        """write() should create the output directory if needed."""
        config = SU2HypersonicConfig()
        out_dir = tmp_path / "nested" / "dir"
        cfg_path = config.write(out_dir)
        assert cfg_path.exists()


class TestAsEuler:
    """Tests for as_euler() method."""

    def test_as_euler_returns_copy(self):
        """as_euler() should return a new config, not modify original."""
        original = SU2HypersonicConfig()
        euler = original.as_euler()
        assert original.solver == "RANS"
        assert euler.solver == "EULER"

    def test_as_euler_preserves_mach(self):
        """as_euler() should preserve Mach number."""
        config = SU2HypersonicConfig(mach=12.0)
        euler = config.as_euler()
        assert euler.mach == 12.0

    def test_as_euler_preserves_wall_temp(self):
        """as_euler() should preserve wall temperature."""
        config = SU2HypersonicConfig(wall_temperature=500.0)
        euler = config.as_euler()
        assert euler.wall_temperature == 500.0

    def test_as_euler_cfg_content(self, tmp_path: Path):
        """Euler config should have SOLVER= EULER and no turbulence."""
        config = SU2HypersonicConfig().as_euler()
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "SOLVER= EULER" in content
        assert "KIND_TURB_MODEL" not in content
        assert "TIME_DISCRE_TURB" not in content
        assert "MARKER_EULER= ( body )" in content
        assert "MARKER_ISOTHERMAL" not in content

    def test_as_euler_no_turbulence_section(self, tmp_path: Path):
        """Euler config should not contain turbulence numerics."""
        config = SU2HypersonicConfig().as_euler()
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "CONV_NUM_METHOD_TURB" not in content
        assert "MUSCL_TURB" not in content
        assert "FREESTREAM_TURBULENCEINTENSITY" not in content
        assert "FREESTREAM_TURB2LAMVISCRATIO" not in content


class TestWithFirstOrder:
    """Tests for with_first_order() method."""

    def test_with_first_order_returns_copy(self):
        """with_first_order() should return a new config."""
        original = SU2HypersonicConfig()
        first_order = original.with_first_order()
        assert original.muscl is True
        assert first_order.muscl is False

    def test_with_first_order_cfg_content(self, tmp_path: Path):
        """First-order config should have MUSCL_FLOW= NO."""
        config = SU2HypersonicConfig().with_first_order()
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "MUSCL_FLOW= NO" in content

    def test_with_first_order_preserves_solver(self):
        """with_first_order() should preserve solver type."""
        config = SU2HypersonicConfig()
        first_order = config.with_first_order()
        assert first_order.solver == "RANS"


class TestWithCfl:
    """Tests for with_cfl() method."""

    def test_with_cfl_returns_copy(self):
        """with_cfl() should return a new config."""
        original = SU2HypersonicConfig()
        modified = original.with_cfl(0.05)
        assert original.cfl_number == 0.1
        assert modified.cfl_number == 0.05

    def test_with_cfl_preserves_other_fields(self):
        """with_cfl() should preserve other fields."""
        config = SU2HypersonicConfig(mach=12.0)
        modified = config.with_cfl(0.01)
        assert modified.mach == 12.0

    def test_with_cfl_in_cfg(self, tmp_path: Path):
        """CFL change should appear in config."""
        config = SU2HypersonicConfig().with_cfl(0.05)
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "CFL_NUMBER= 0.05" in content


class TestWithTurbulenceInit:
    """Tests for with_turbulence_init() method."""

    def test_with_turbulence_init_returns_copy(self):
        """with_turbulence_init() should return a new config."""
        original = SU2HypersonicConfig()
        modified = original.with_turbulence_init(
            intensity=0.01, viscosity_ratio=100.0,
        )
        assert original.freestream_turbulence_intensity == 0.05
        assert original.freestream_turbulence_viscosity_ratio == 10.0
        assert modified.freestream_turbulence_intensity == 0.01
        assert modified.freestream_turbulence_viscosity_ratio == 100.0

    def test_with_turbulence_init_in_cfg(self, tmp_path: Path):
        """Turbulence init should appear in config."""
        config = SU2HypersonicConfig().with_turbulence_init(
            intensity=0.01, viscosity_ratio=50.0,
        )
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "FREESTREAM_TURBULENCEINTENSITY= 0.01" in content
        assert "FREESTREAM_TURB2LAMVISCRATIO= 50.0" in content


class TestWithCflAdapt:
    """Tests for with_cfl_adapt() method."""

    def test_with_cfl_adapt_returns_copy(self):
        """with_cfl_adapt() should return a new config."""
        original = SU2HypersonicConfig()
        modified = original.with_cfl_adapt(
            cfl_min=0.05, cfl_max=3.0, decrease=0.5, increase=200.0,
        )
        assert original.cfl_adapt_min == 0.1
        assert modified.cfl_adapt_min == 0.05
        assert modified.cfl_adapt_max == 3.0

    def test_with_cfl_adapt_in_cfg(self, tmp_path: Path):
        """CFL adapt params should appear in config."""
        config = SU2HypersonicConfig().with_cfl_adapt(
            cfl_min=0.001, cfl_max=0.3, decrease=0.3, increase=2.0,
        )
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "CFL_ADAPT_PARAM= ( 0.001, 0.3, 0.3, 2.0 )" in content


class TestWithRestart:
    """Tests for with_restart() method."""

    def test_with_restart_returns_copy(self, tmp_path: Path):
        """with_restart() should return a new config, not modify original."""
        original = SU2HypersonicConfig()
        restarted = original.with_restart(tmp_path / "restart.dat")
        assert original._restart_sol is False
        assert restarted._restart_sol is True

    def test_with_restart_sets_restart_filename(self, tmp_path: Path):
        """with_restart() should set the restart filename."""
        config = SU2HypersonicConfig()
        restart_path = tmp_path / "solution_restart.dat"
        restarted = config.with_restart(restart_path)
        assert restarted._restart_filename == str(restart_path)

    def test_with_restart_cfg_content(self, tmp_path: Path):
        """Config written from restart should have RESTART_SOL= YES."""
        config = SU2HypersonicConfig()
        restart_path = tmp_path / "restart_00050.dat"
        restarted = config.with_restart(restart_path)
        cfg_path = restarted.write(tmp_path / "restart_cfg")
        content = cfg_path.read_text()
        assert "RESTART_SOL= YES" in content
        assert "RESTART_FILENAME= " in content

    def test_with_restart_preserves_other_fields(self, tmp_path: Path):
        """with_restart() should preserve all other config fields."""
        config = SU2HypersonicConfig(mach=10.0, wall_temperature=400.0)
        restarted = config.with_restart(tmp_path / "restart.dat")
        assert restarted.mach == 10.0
        assert restarted.wall_temperature == 400.0


class TestWithMach:
    """Tests for with_mach() method."""

    def test_with_mach_returns_copy(self):
        """with_mach() should return a new config."""
        original = SU2HypersonicConfig()
        modified = original.with_mach(12.0)
        assert original.mach == 8.0
        assert modified.mach == 12.0

    def test_with_mach_preserves_other_fields(self):
        """with_mach() should preserve other fields."""
        config = SU2HypersonicConfig(wall_temperature=500.0)
        modified = config.with_mach(6.0)
        assert modified.wall_temperature == 500.0

    def test_with_mach_in_cfg(self, tmp_path: Path):
        """Mach change should appear in written config."""
        config = SU2HypersonicConfig().with_mach(15.0)
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "MACH_NUMBER= 15.0" in content


class TestWithWallTemperature:
    """Tests for with_wall_temperature() method."""

    def test_with_wall_temperature_returns_copy(self):
        """with_wall_temperature() should return a new config."""
        original = SU2HypersonicConfig()
        modified = original.with_wall_temperature(600.0)
        assert original.wall_temperature == 2500.0
        assert modified.wall_temperature == 600.0

    def test_with_wall_temperature_preserves_other_fields(self):
        """with_wall_temperature() should preserve other fields."""
        config = SU2HypersonicConfig(mach=12.0)
        modified = config.with_wall_temperature(800.0)
        assert modified.mach == 12.0

    def test_with_wall_temperature_in_cfg(self, tmp_path: Path):
        """Wall temperature change should appear in written config."""
        config = SU2HypersonicConfig().with_wall_temperature(1000.0)
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "MARKER_ISOTHERMAL= ( body, 1000.0 )" in content


class TestGetSU2Binary:
    """Tests for SU2 binary discovery."""

    def test_finds_binary_at_known_path(self):
        """Should find SU2_CFD at the known install path."""
        binary = get_su2_binary()
        assert binary.exists()
        assert binary.name == "SU2_CFD"


class TestSU2HypersonicConfigAsFull2d:
    """Tests for as_full2d() method."""

    def test_returns_deep_copy(self):
        original = SU2HypersonicConfig()
        copy = original.as_full2d()
        assert copy is not original

    def test_sets_axisymmetric_false(self):
        original = SU2HypersonicConfig()
        copy = original.as_full2d()
        assert copy.axisymmetric is False

    def test_original_unchanged(self):
        original = SU2HypersonicConfig()
        _ = original.as_full2d()
        assert original.axisymmetric is True

    def test_cfg_excludes_sym_for_full2d(self, tmp_path):
        config = SU2HypersonicConfig().as_full2d()
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "MARKER_SYM" not in content

    def test_cfg_includes_sym_for_axisymmetric(self, tmp_path):
        config = SU2HypersonicConfig()
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "MARKER_SYM" in content

    def test_full2d_with_aoa_combined(self, tmp_path):
        config = SU2HypersonicConfig().as_full2d().with_aoa(20.0)
        assert config.axisymmetric is False
        assert config.aoa == 20.0
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "AXISYMMETRIC= NO" in content
        assert "AOA= 20.0" in content
        assert "MARKER_SYM" not in content


class TestSU2HypersonicConfigWithAoa:
    """Tests for with_aoa() method."""

    def test_returns_deep_copy(self):
        original = SU2HypersonicConfig()
        copy = original.with_aoa(15.0)
        assert copy is not original

    def test_sets_aoa(self):
        original = SU2HypersonicConfig()
        copy = original.with_aoa(15.0)
        assert copy.aoa == 15.0

    def test_original_unchanged(self):
        original = SU2HypersonicConfig()
        _ = original.with_aoa(15.0)
        assert original.aoa == 0.0

    def test_cfg_contains_aoa(self, tmp_path):
        config = SU2HypersonicConfig().with_aoa(15.0)
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "AOA= 15.0" in content

    def test_preserves_other_fields(self):
        original = SU2HypersonicConfig(mach=10.0, wall_temperature=3000.0)
        copy = original.with_aoa(10.0)
        assert copy.mach == 10.0
        assert copy.wall_temperature == 3000.0


class TestCfgFileIntegrity:
    """Tests for overall config file integrity."""

    def test_no_em_dashes(self, tmp_path: Path):
        """Config file should not contain em dashes."""
        config = SU2HypersonicConfig()
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()
        assert "\u2014" not in content  # em dash
        assert "\u2013" not in content  # en dash

    def test_all_numeric_values_parseable(self, tmp_path: Path):
        """All numeric values in the config should be valid floats."""
        config = SU2HypersonicConfig()
        cfg_path = config.write(tmp_path)
        content = cfg_path.read_text()

        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                value = value.strip()
                # Skip non-numeric values
                if value in ("YES", "NO", "SI", "RANS", "SA", "ROE",
                             "AUSM", "AUSMPLUS",
                             "EULER_IMPLICIT", "FGMRES", "BCGSTAB", "ILU",
                             "MINMOD",  # legacy skip entry
                             "VENKATAKRISHNAN_WANG", "VENKATAKRISHNAN",
                             "SCALAR_UPWIND", "DIRECT", "RMS_DENSITY",
                             "SU2"):
                    continue
                if value.startswith(("(", "flow")):
                    continue
                if key.strip() in ("SCREEN_OUTPUT", "OUTPUT_FILES",
                                    "HISTORY_OUTPUT", "MESH_FILENAME",
                                    "VOLUME_FILENAME", "HISTORY_FILENAME",
                                    "SURFACE_FILENAME"):
                    continue
                # Should be parseable as float
                try:
                    float(value.split()[0])
                except (ValueError, IndexError):
                    pytest.fail(f"Non-numeric value for {key.strip()}: {value}")
