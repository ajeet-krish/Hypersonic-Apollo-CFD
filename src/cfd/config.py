"""SU2 hypersonic RANS configuration generator.

Generates valid SU2 .cfg files for axisymmetric RANS simulations of
spherically-blunted cones in hypersonic freestream conditions.
"""
import copy
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SU2HypersonicConfig:
    """SU2 configuration for axisymmetric RANS hypersonic blunt body simulation.

    Attributes:
        solver: CFD solver type (EULER, RANS, etc.)
        turb_model: Turbulence model (SA, SST, etc.)
        axisymmetric: Enable axisymmetric formulation
        mach: Freestream Mach number
        aoa: Angle of attack (degrees)
        freestream_pressure: Freestream static pressure (Pa)
        freestream_temperature: Freestream static temperature (K)
        freestream_density: Freestream density (kg/m^3). Required for RANS.
        freestream_viscosity: Freestream dynamic viscosity (Pa*s). Required for RANS.
        wall_temperature: Isothermal wall temperature (K)
        wall_marker: Boundary marker name for the body wall
        farfield_marker: Boundary marker name for the farfield
        sym_marker: Boundary marker name for the symmetry axis
        cfl_number: CFL number for time stepping
        cfl_adapt: Enable CFL adaptation
        iterations: Maximum number of solver iterations
        conv_residual_minval: Convergence residual minimum (log10)
        conv_num_method: Convective numerical method
        muscl: Enable MUSCL reconstruction
        limiter: Slope limiter for MUSCL
        gamma: Ratio of specific heats
        gas_constant: Specific gas constant (J/(kg*K))
        ref_area: Reference area for force coefficients (m^2)
        ref_length: Reference length for force coefficients (m)
        output_files: Tuple of output file formats
        history_output: Tuple of history output fields
    """

    # Solver
    solver: str = "RANS"
    turb_model: str = "SA"
    axisymmetric: bool = True

    # Freestream conditions
    mach: float = 8.0
    aoa: float = 0.0
    freestream_pressure: float = 1172.0   # Pa, 30 km US Standard Atmosphere
    freestream_temperature: float = 226.65  # K, 30 km US Standard Atmosphere
    freestream_density: float = 0.0184  # kg/m^3, 30 km US Standard Atmosphere
    freestream_viscosity: float = 1.477e-5  # Pa*s, 30 km (Sutherland)
    reynolds_number: float = 2.95e6  # Re_L, computed from atmosphere

    # Wall
    wall_temperature: float = 300.0  # K (isothermal)
    wall_marker: str = "body"

    # Boundaries
    farfield_marker: str = "farfield"
    sym_marker: str = "sym"

    # Numerics
    cfl_number: float = 0.1
    cfl_adapt: bool = True
    iterations: int = 10000
    conv_residual_minval: float = -6.0
    conv_num_method: str = "AUSM"
    muscl: bool = True
    limiter: str = "VENKATAKRISHNAN"

    # Gas properties
    gamma: float = 1.4
    gas_constant: float = 287.058  # J/(kg*K)

    # Reference values
    ref_area: float = 1.0  # m^2
    ref_length: float = 1.0  # m

    # Output
    output_files: tuple[str, ...] = field(
        default_factory=lambda: ("RESTART", "PARAVIEW"),
    )
    history_output: tuple[str, ...] = field(
        default_factory=lambda: ("ITER", "RMS_RES", "LIFT", "DRAG"),
    )

    # Internal state for restart (set via with_restart())
    _restart_sol: bool = False
    _restart_filename: str = ""

    def write(self, path: Path, mesh_filename: str = "mesh.su2") -> Path:
        """Generate SU2 .cfg configuration file.

        Args:
            path: Directory to write the config file into.
            mesh_filename: Name of the .su2 mesh file (relative to workdir).

        Returns:
            Path to the written .cfg file.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        output_files_str = ", ".join(self.output_files)
        history_output_str = ", ".join(self.history_output)

        # Build restart lines
        if self._restart_sol:
            restart_lines = (
                f"RESTART_SOL= YES\n"
                f"RESTART_FILENAME= {self._restart_filename}"
            )
        else:
            restart_lines = "RESTART_SOL= NO"

        config_content = f"""% ------- Hypersonic Blunt Body Aerothermodynamics - RANS Config --------
% Axisymmetric RANS, {self.solver}, M={self.mach}, Alt=30 km
% Turbulence: {self.turb_model}, Wall T={self.wall_temperature:.0f} K (isothermal)

% -------------------- SOLVER CONFIGURATION --------------------
SOLVER= {self.solver}
KIND_TURB_MODEL= {self.turb_model}
MATH_PROBLEM= DIRECT
{restart_lines}
AXISYMMETRIC= {'YES' if self.axisymmetric else 'NO'}

% -------------------- GAS PROPERTIES -------------------------
GAMMA_VALUE= {self.gamma}
GAS_CONSTANT= {self.gas_constant}
SYSTEM_MEASUREMENTS= SI

% -------------------- FREESTREAM CONDITIONS -------------------
MACH_NUMBER= {self.mach}
AOA= {self.aoa}
FREESTREAM_PRESSURE= {self.freestream_pressure}
FREESTREAM_TEMPERATURE= {self.freestream_temperature}
FREESTREAM_DENSITY= {self.freestream_density}
FREESTREAM_VISCOSITY= {self.freestream_viscosity}
REYNOLDS_NUMBER= {self.reynolds_number}
REYNOLDS_LENGTH= {self.ref_length}

% -------------------- BOUNDARY CONDITIONS ---------------------
MARKER_ISOTHERMAL= ( {self.wall_marker}, {self.wall_temperature:.1f} )
MARKER_FAR= ( {self.farfield_marker} )
MARKER_SYM= ( {self.sym_marker} )

% -------------------- NUMERICAL METHOD ------------------------
CONV_NUM_METHOD_FLOW= {self.conv_num_method}
MUSCL_FLOW= {'YES' if self.muscl else 'NO'}
SLOPE_LIMITER_FLOW= {self.limiter}

% Turbulence numerics
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO
SLOPE_LIMITER_TURB= VENKATAKRISHNAN

% Time discretization
TIME_DISCRE_FLOW= EULER_IMPLICIT
TIME_DISCRE_TURB= EULER_IMPLICIT

% -------------------- LINEAR SOLVER ---------------------------
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 10

% -------------------- MULTIGRID ------------------------------
MGLEVEL= 0

% -------------------- CONVERGENCE -----------------------------
ITER= {self.iterations}
CFL_NUMBER= {self.cfl_number}
CFL_ADAPT= {'YES' if self.cfl_adapt else 'NO'}
CFL_ADAPT_PARAM= ( 0.1, 2.0, 0.5, 100.0 )
CONV_FIELD= RMS_DENSITY
CONV_RESIDUAL_MINVAL= {self.conv_residual_minval}
CONV_STARTITER= 100
CONV_CAUCHY_EPS= 1E-6

% -------------------- REFERENCE VALUES ------------------------
REF_AREA= {self.ref_area}
REF_LENGTH= {self.ref_length}

% -------------------- OUTPUT ----------------------------------
SCREEN_OUTPUT= (INNER_ITER, RMS_DENSITY, LIFT, DRAG)
OUTPUT_FILES= ( {output_files_str} )
VOLUME_FILENAME= flow
HISTORY_OUTPUT= ( {history_output_str} )

% -------------------- MESH ------------------------------------
MESH_FILENAME= {mesh_filename}
MESH_FORMAT= SU2
"""
        config_path = path / "config.cfg"
        config_path.write_text(config_content)
        return config_path

    def with_restart(self, restart_file: Path) -> "SU2HypersonicConfig":
        """Return a copy configured for restart from a previous solution.

        Args:
            restart_file: Path to the restart file (relative to workdir).

        Returns:
            New SU2HypersonicConfig with RESTART_SOL=YES.
        """
        new = copy.deepcopy(self)
        object.__setattr__(new, "_restart_sol", True)
        object.__setattr__(new, "_restart_filename", str(restart_file))
        return new

    def with_mach(self, mach: float) -> "SU2HypersonicConfig":
        """Return a copy with a different Mach number (for Mach ramping).

        Args:
            mach: New freestream Mach number.

        Returns:
            New SU2HypersonicConfig with updated Mach.
        """
        new = copy.deepcopy(self)
        new.mach = mach
        return new

    def with_wall_temperature(self, t_wall: float) -> "SU2HypersonicConfig":
        """Return a copy with a different wall temperature.

        Args:
            t_wall: New isothermal wall temperature (K).

        Returns:
            New SU2HypersonicConfig with updated wall temperature.
        """
        new = copy.deepcopy(self)
        new.wall_temperature = t_wall
        return new


def get_su2_binary() -> Path:
    """Locate the SU2_CFD binary.

    Searches in order: SU2_CFD_BIN env var, known install paths, PATH.

    Returns:
        Path to the SU2_CFD binary.

    Raises:
        FileNotFoundError: If no SU2_CFD binary is found.
    """
    import os

    # Environment variable
    env_path = os.environ.get("SU2_CFD_BIN")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    # Known install paths
    common_paths = [
        Path("/Users/ajeet/SU2_CFD/bin/SU2_CFD"),
        Path.home() / "SU2_CFD/bin/SU2_CFD",
    ]

    for p in common_paths:
        if p.exists():
            return p

    # Fallback: check PATH
    import shutil
    on_path = shutil.which("SU2_CFD")
    if on_path:
        return Path(on_path)

    raise FileNotFoundError(
        "SU2_CFD binary not found. Set SU2_CFD_BIN env var or install SU2."
    )
