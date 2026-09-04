"""SU2 solver interface for hypersonic blunt body simulations.

Runs SU2_CFD as a subprocess, parses convergence history, and extracts
stagnation-point values from the solution VTU file.
"""
from __future__ import annotations

import copy
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .config import get_su2_binary
from .vtu_parser import extract_stagnation_values, parse_vtu

if TYPE_CHECKING:
    from .convergence import ConvergenceStrategy
    from .config import SU2HypersonicConfig

logger = logging.getLogger(__name__)


@dataclass
class SU2Results:
    """Parsed results from an SU2 simulation.

    Attributes:
        converged: Whether the simulation converged (residual drop > 3 orders
                   or final RMS density < 10^-4).
        iterations: Number of iterations actually run.
        residual_drop: Orders-of-magnitude drop in RMS density residual.
        final_residual: Final RMS density residual (log10 scale).
        history: List of parsed history.csv rows as dicts.
        stagnation_heat_flux: Stagnation heat flux (W/m^2), if available.
        stagnation_pressure: Stagnation-point static pressure (Pa).
        max_mach: Maximum Mach number in the domain.
    """

    converged: bool = False
    iterations: int = 0
    residual_drop: float = 0.0
    final_residual: float = 0.0
    history: list[dict] = field(default_factory=list)
    stagnation_heat_flux: float | None = None
    stagnation_pressure: float | None = None
    max_mach: float | None = None


class SU2Solver:
    """Run SU2_CFD and parse results."""

    def __init__(self, su2_cfd: Path | None = None) -> None:
        """Initialize solver with path to SU2_CFD binary.

        Args:
            su2_cfd: Path to SU2_CFD binary. If None, auto-discover.
        """
        self.su2_cfd = su2_cfd or get_su2_binary()

    def run(
        self,
        config_path: Path,
        workdir: Path,
        timeout: int = 7200,
    ) -> SU2Results:
        """Execute SU2_CFD and return parsed results.

        Args:
            config_path: Path to SU2 .cfg config file.
            workdir: Working directory for simulation.
            timeout: Maximum runtime in seconds.

        Returns:
            Parsed simulation results.
        """
        workdir.mkdir(parents=True, exist_ok=True)

        # SU2 expects config file relative to working directory
        config_name = Path(config_path).name
        cmd = [str(self.su2_cfd), config_name]

        logger.info("Running SU2: %s", " ".join(cmd))
        logger.info("Working directory: %s", workdir)

        try:
            result = subprocess.run(
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

            if result.returncode != 0:
                logger.warning(
                    "SU2 returned non-zero exit code %d. "
                    "Attempting to parse partial results.",
                    result.returncode,
                )
                if result.stderr:
                    # Log last 20 lines of stderr
                    stderr_lines = result.stderr.strip().splitlines()
                    for line in stderr_lines[-20:]:
                        logger.warning("  stderr: %s", line)

            return self.parse_results(workdir)

        except subprocess.TimeoutExpired:
            logger.error("SU2 timed out after %ds", timeout)
            return self.parse_results(workdir)
        except FileNotFoundError:
            logger.error("SU2 binary not found: %s", self.su2_cfd)
            return SU2Results(converged=False)

    def parse_results(self, workdir: Path) -> SU2Results:
        """Parse history.csv and flow.vtu for simulation results.

        Convergence is determined by either:
        1. Residual drop > 3.0 orders within a single run, OR
        2. Final RMS density residual < -4.0 (absolute level)

        The second criterion handles multi-stage convergence (e.g. first-order
        RANS then second-order restart) where the history only captures the
        last stage and the drop metric underestimates total convergence.

        Args:
            workdir: Directory containing SU2 output files.

        Returns:
            Parsed simulation results.
        """
        results = SU2Results()

        # Parse history.csv
        history_path = workdir / "history.csv"
        if history_path.exists():
            results.history = self._parse_history(history_path)
            if results.history:
                results.iterations = len(results.history)
                results.residual_drop = self._compute_residual_drop(results.history)
                results.final_residual = self._compute_final_residual(
                    results.history,
                )
                # Converged if drop > 3 orders OR final residual < -3
                results.converged = (
                    results.residual_drop > 3.0
                    or results.final_residual < -3.0
                )

        # Parse flow.vtu for stagnation values
        vtu_path = workdir / "flow.vtu"
        if vtu_path.exists():
            try:
                vtu_data = parse_vtu(vtu_path)
                stag = extract_stagnation_values(vtu_data)
                if "Pressure" in stag:
                    results.stagnation_pressure = stag["Pressure"]
                if "Temperature" in stag:
                    # Convert temperature to heat flux placeholder
                    # (actual heat flux requires surface integration)
                    results.stagnation_heat_flux = None
                if vtu_data.mach is not None:
                    results.max_mach = float(vtu_data.mach.max())
            except (ValueError, OSError) as exc:
                logger.warning("Failed to parse VTU: %s", exc)

        return results

    def _parse_history(self, history_path: Path) -> list[dict]:
        """Parse SU2 history.csv file.

        Handles SU2's quoted header format and various column layouts.

        Args:
            history_path: Path to history.csv.

        Returns:
            List of row dictionaries.
        """
        history: list[dict] = []
        try:
            with open(history_path, "r") as f:
                lines = f.readlines()

            if not lines:
                return history

            # SU2 history.csv format:
            # Line 1: Header line with quoted column names
            # Line 2+: Data rows
            header_line = lines[0].strip()
            header = [h.strip().strip('"') for h in header_line.split(",")]

            for line in lines[1:]:
                line_stripped = line.strip()
                if not line_stripped:
                    continue

                values = [v.strip() for v in line_stripped.split(",")]
                if len(values) == len(header):
                    row = dict(zip(header, values))
                    history.append(row)

        except (OSError, ValueError) as exc:
            logger.warning("Failed to parse history: %s", exc)
        return history

    @staticmethod
    def _compute_residual_drop(history: list[dict]) -> float:
        """Compute orders-of-magnitude residual drop from history.

        SU2 history.csv stores residuals in log10 scale (e.g., -1.5 = 10^-1.5).
        The drop is computed as: first_residual - last_residual (in log10).
        A drop of 3.0 means 3 orders of magnitude improvement.

        Looks for 'rms[Rho]' or 'RMS_DENSITY' columns.

        Args:
            history: Parsed history rows.

        Returns:
            Residual drop in orders of magnitude (positive = improving).
        """
        if len(history) < 2:
            return 0.0

        # Try various SU2 column name conventions
        rho_key = None
        for candidate in ("rms[Rho]", "RMS_DENSITY", "Rho", "Residual"):
            if candidate in history[0]:
                rho_key = candidate
                break

        if rho_key is None:
            return 0.0

        try:
            first_val = float(history[0][rho_key])
            last_val = float(history[-1][rho_key])
            # SU2 stores residuals in log10 scale
            # Positive drop = residual decreased (improved)
            return first_val - last_val
        except (ValueError, KeyError):
            return 0.0

    @staticmethod
    def _compute_final_residual(history: list[dict]) -> float:
        """Compute the final RMS density residual (log10 scale).

        Looks for 'rms[Rho]' or 'RMS_DENSITY' columns.

        Args:
            history: Parsed history rows.

        Returns:
            Final residual in log10 scale (e.g. -4.0 means 10^-4).
        """
        if not history:
            return 0.0

        rho_key = None
        for candidate in ("rms[Rho]", "RMS_DENSITY", "Rho", "Residual"):
            if candidate in history[0]:
                rho_key = candidate
                break

        if rho_key is None:
            return 0.0

        try:
            return float(history[-1][rho_key])
        except (ValueError, KeyError):
            return 0.0

    def run_stages(
        self,
        strategy: ConvergenceStrategy,
        base_config: SU2HypersonicConfig,
        workdir: Path,
        mesh_filename: str = "mesh.su2",
    ) -> SU2Results:
        """Execute a multi-stage convergence strategy.

        Runs each stage sequentially, using the previous stage's restart file
        as the initial condition for the next stage.

        Args:
            strategy: Multi-stage convergence strategy to execute.
            base_config: Base SU2 configuration (freestream, wall, gas, etc.).
            workdir: Working directory for simulation files.
            mesh_filename: Name of the .su2 mesh file.

        Returns:
            Results from the final stage.
        """
        workdir.mkdir(parents=True, exist_ok=True)
        final_results = SU2Results()

        for i, stage in enumerate(strategy.stages):
            is_first = i == 0
            stage_num = i + 1
            total = len(strategy.stages)

            logger.info(
                "Stage %d/%d: %s (M=%.1f, CFL=%.4f, iters=%d)",
                stage_num, total, stage.name, stage.mach,
                stage.cfl, stage.iterations,
            )

            # Build stage config from base
            cfg = copy.deepcopy(base_config)
            cfg.mach = stage.mach
            cfg.cfl_number = stage.cfl
            cfg.iterations = stage.iterations
            cfg.muscl = stage.muscl
            cfg.linear_solver = stage.linear_solver
            cfg.linear_solver_error = stage.linear_solver_error
            cfg.linear_solver_iter = stage.linear_solver_iter
            cfg.cfl_adapt_min = stage.cfl_adapt_min
            cfg.cfl_adapt_max = stage.cfl_adapt_max
            cfg.cfl_adapt_decrease = stage.cfl_adapt_decrease
            cfg.cfl_adapt_increase = stage.cfl_adapt_increase

            # Enable restart from previous stage (not the first stage)
            if not is_first:
                restart_file = self._find_restart_file(workdir)
                if restart_file is None:
                    logger.warning(
                        "No restart file found after stage %d. "
                        "Continuing without restart.",
                        stage_num - 1,
                    )
                else:
                    # Copy restart to solution.dat for SU2 to read
                    solution_path = workdir / "solution.dat"
                    shutil.copy2(restart_file, solution_path)
                    cfg = cfg.with_restart(Path("solution.dat"))
                    logger.info(
                        "Restart from %s (copied to solution.dat)",
                        restart_file.name,
                    )

            # Only output RESTART files (not PARAVIEW) for intermediate stages
            if not is_first:
                cfg.output_files = ("RESTART",)
            else:
                cfg.output_files = ("RESTART",)

            # Write config and run
            cfg_path = cfg.write(workdir, mesh_filename=mesh_filename)
            logger.info("Config: %s", cfg_path)

            results = self.run(cfg_path, workdir, timeout=7200)

            logger.info(
                "Stage %d complete: %d iters, drop=%.2f orders, "
                "final_res=%.2f, converged=%s",
                stage_num, results.iterations, results.residual_drop,
                results.final_residual, results.converged,
            )

            if not results.converged:
                logger.warning(
                    "Stage %d (%s) did not converge. "
                    "Continuing to next stage with partial solution.",
                    stage_num, stage.name,
                )

            final_results = results

        return final_results

    @staticmethod
    def _find_restart_file(workdir: Path) -> Path | None:
        """Find the latest SU2 restart file in the working directory.

        SU2 v8.x names restart files as 'restart.dat' (or flow_restart_XXX.dat
        in older versions).

        Args:
            workdir: Directory to search.

        Returns:
            Path to the restart file, or None if not found.
        """
        # Try SU2 v8.x naming first: restart.dat
        restart_v8 = workdir / "restart.dat"
        if restart_v8.exists():
            return restart_v8

        # Fallback: flow_restart_000XXX.dat (older SU2 versions)
        restart_files = list(workdir.glob("flow_restart_*.dat"))
        if not restart_files:
            return None

        # Sort by iteration number (the numeric part of the filename)
        def _iter_num(p: Path) -> int:
            match = re.search(r"flow_restart_(\d+)\.dat", p.name)
            return int(match.group(1)) if match else 0

        restart_files.sort(key=_iter_num)
        return restart_files[-1]
