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
        cfl_adapt_min: Minimum CFL for CFL adaptation
        cfl_adapt_max: Maximum CFL for CFL adaptation
        cfl_adapt_decrease: CFL decrease factor for adaptation
        cfl_adapt_increase: CFL increase factor for adaptation
        iterations: Maximum number of solver iterations
        conv_residual_minval: Convergence residual minimum (log10)
        conv_num_method: Convective numerical method
        muscl: Enable MUSCL reconstruction
        limiter: Slope limiter for MUSCL
        gamma: Ratio of specific heats
        gas_constant: Specific gas constant (J/(kg*K))
        ref_area: Reference area for force coefficients (m^2)
        ref_length: Reference length for force coefficients (m)
        freestream_turbulence_intensity: Turbulence intensity for SA initialization
        freestream_turbulence_viscosity_ratio: Turbulence viscosity ratio for SA
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
    wall_temperature: float = 2500.0  # AVCOAT ablative heat shield approximate equilibrium temperature
    wall_marker: str = "body"

    # Boundaries
    farfield_marker: str = "farfield"
    sym_marker: str = "sym"

    # Numerics
    cfl_number: float = 0.1
    cfl_adapt: bool = True
    cfl_adapt_min: float = 0.1
    cfl_adapt_max: float = 2.0
    cfl_adapt_decrease: float = 0.5
    cfl_adapt_increase: float = 1.5  # Critical for hypersonic stability; large jumps cause divergence
    iterations: int = 10000
    conv_residual_minval: float = -6.0
    conv_num_method: str = "AUSM"
    muscl: bool = True
    limiter: str = "VENKATAKRISHNAN"
    entropy_fix_coeff: float = 0.1
    linear_solver: str = "BCGSTAB"
    linear_solver_prec: str = "ILU"
    linear_solver_error: float = 1e-6
    linear_solver_iter: int = 10

    # Gas properties
    gamma: float = 1.4
    gas_constant: float = 287.058  # J/(kg*K)

    # Reference values
    ref_area: float = 1.0  # m^2
    ref_length: float = 1.0  # m

    # Turbulence initialization (critical for SA convergence in hypersonic flows)
    freestream_turbulence_intensity: float = 0.05
    freestream_turbulence_viscosity_ratio: float = 10.0

    # Output
    output_files: tuple[str, ...] = field(
        default_factory=lambda: ("RESTART", "PARAVIEW", "SURFACE_CSV"),
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

        # Build turbulence model section (only for RANS)
        if self.solver == "RANS":
            turb_section = f"""KIND_TURB_MODEL= {self.turb_model}
FREESTREAM_TURBULENCEINTENSITY= {self.freestream_turbulence_intensity}
FREESTREAM_TURB2LAMVISCRATIO= {self.freestream_turbulence_viscosity_ratio}

% Turbulence numerics
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO
SLOPE_LIMITER_TURB= VENKATAKRISHNAN

TIME_DISCRE_TURB= EULER_IMPLICIT"""
        else:
            turb_section = ""

        # CFL adapt params
        cfl_adapt_params = (
            f"( {self.cfl_adapt_min}, {self.cfl_adapt_max}, "
            f"{self.cfl_adapt_decrease}, {self.cfl_adapt_increase} )"
        )

        # Build boundary conditions
        sym_line = f"MARKER_SYM= ( {self.sym_marker} )" if self.axisymmetric else ""
        bc_parts: list[str] = []
        if self.solver == "EULER":
            bc_parts.append(f"MARKER_EULER= ( {self.wall_marker} )")
        else:
            bc_parts.append(
                f"MARKER_ISOTHERMAL= ( {self.wall_marker}, {self.wall_temperature:.1f} )"
            )
        bc_parts.append(f"MARKER_FAR= ( {self.farfield_marker} )")
        if sym_line:
            bc_parts.append(sym_line)
        bc_section = "\n".join(bc_parts)

        config_content = f"""% ------- Hypersonic Blunt Body Aerothermodynamics - {self.solver} Config --------
% Axisymmetric {self.solver}, M={self.mach}, Alt=30 km

% -------------------- SOLVER CONFIGURATION --------------------
SOLVER= {self.solver}
{turb_section}
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
{bc_section}

% -------------------- NUMERICAL METHOD ------------------------
CONV_NUM_METHOD_FLOW= {self.conv_num_method}
MUSCL_FLOW= {'YES' if self.muscl else 'NO'}
SLOPE_LIMITER_FLOW= {self.limiter}
ENTROPY_FIX_COEFF= {self.entropy_fix_coeff}

% Time discretization
TIME_DISCRE_FLOW= EULER_IMPLICIT

% -------------------- LINEAR SOLVER ---------------------------
LINEAR_SOLVER= {self.linear_solver}
LINEAR_SOLVER_PREC= {self.linear_solver_prec}
LINEAR_SOLVER_ERROR= {self.linear_solver_error}
LINEAR_SOLVER_ITER= {self.linear_solver_iter}

% -------------------- MULTIGRID ------------------------------
MGLEVEL= 0

% -------------------- CONVERGENCE -----------------------------
ITER= {self.iterations}
CFL_NUMBER= {self.cfl_number}
CFL_ADAPT= {'YES' if self.cfl_adapt else 'NO'}
CFL_ADAPT_PARAM= {cfl_adapt_params}
CONV_FIELD= RMS_DENSITY
CONV_RESIDUAL_MINVAL= {self.conv_residual_minval}
CONV_STARTITER= 100

% -------------------- REFERENCE VALUES ------------------------
REF_AREA= {self.ref_area}
REF_LENGTH= {self.ref_length}

% -------------------- OUTPUT ----------------------------------
SCREEN_OUTPUT= (INNER_ITER, RMS_DENSITY, LIFT, DRAG)
OUTPUT_FILES= ( {output_files_str} )
VOLUME_FILENAME= flow
SURFACE_FILENAME= surface_flow
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

    def with_aoa(self, aoa: float) -> "SU2HypersonicConfig":
        """Return a copy with angle of attack (degrees).

        Args:
            aoa: Angle of attack in degrees (positive = nose up).

        Returns:
            New SU2HypersonicConfig with updated AoA.
        """
        new = copy.deepcopy(self)
        new.aoa = aoa
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

    def as_euler(self) -> "SU2HypersonicConfig":
        """Return a copy configured as an inviscid Euler solver.

        Used for the first stage of the Euler-then-RANS workflow:
        establish the bow shock cleanly before enabling turbulence.

        Returns:
            New SU2HypersonicConfig with SOLVER=EULER and no turbulence.
        """
        new = copy.deepcopy(self)
        new.solver = "EULER"
        return new

    def as_first_order_rans(self) -> "SU2HypersonicConfig":
        """Return a copy configured as first-order RANS (no MUSCL).

        First-order spatial accuracy is more stable for initial convergence
        in hypersonic flows. Use this for the initial RANS stage, then
        restart with second-order MUSCL for the final solution.

        Returns:
            New SU2HypersonicConfig with MUSCL_FLOW=NO.
        """
        new = copy.deepcopy(self)
        new.muscl = False
        return new

    def with_first_order(self) -> "SU2HypersonicConfig":
        """Return a copy with first-order spatial accuracy (MUSCL disabled).

        Used for initial stabilization before switching to second-order.

        Returns:
            New SU2HypersonicConfig with MUSCL_FLOW=NO.
        """
        new = copy.deepcopy(self)
        new.muscl = False
        return new

    def with_cfl(self, cfl: float) -> "SU2HypersonicConfig":
        """Return a copy with a different CFL number.

        Args:
            cfl: New CFL number.

        Returns:
            New SU2HypersonicConfig with updated CFL.
        """
        new = copy.deepcopy(self)
        new.cfl_number = cfl
        return new

    def with_turbulence_init(
        self,
        intensity: float = 0.05,
        viscosity_ratio: float = 10.0,
    ) -> "SU2HypersonicConfig":
        """Return a copy with different turbulence initialization for SA.

        Args:
            intensity: Freestream turbulence intensity (0-1).
            viscosity_ratio: Freestream turbulence-to-laminar viscosity ratio.

        Returns:
            New SU2HypersonicConfig with updated turbulence init.
        """
        new = copy.deepcopy(self)
        new.freestream_turbulence_intensity = intensity
        new.freestream_turbulence_viscosity_ratio = viscosity_ratio
        return new

    def with_cfl_adapt(
        self,
        cfl_min: float = 0.01,
        cfl_max: float = 1.0,
        decrease: float = 0.5,
        increase: float = 1.5,
    ) -> "SU2HypersonicConfig":
        """Return a copy with different CFL adaptation parameters.

        Args:
            cfl_min: Minimum CFL during adaptation.
            cfl_max: Maximum CFL during adaptation.
            decrease: CFL decrease factor when divergence detected.
            increase: CFL increase factor when converging.

        Returns:
            New SU2HypersonicConfig with updated CFL adapt params.
        """
        new = copy.deepcopy(self)
        new.cfl_adapt_min = cfl_min
        new.cfl_adapt_max = cfl_max
        new.cfl_adapt_decrease = decrease
        new.cfl_adapt_increase = increase
        return new

    def as_full2d(self) -> "SU2HypersonicConfig":
        """Return a copy configured for full 2D (not axisymmetric).

        Used for full external flow simulations where the body is not
        symmetric about the x-axis, such as angle of attack studies.

        Returns:
            New SU2HypersonicConfig with AXISYMMETRIC=NO.
        """
        new = copy.deepcopy(self)
        new.axisymmetric = False
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
