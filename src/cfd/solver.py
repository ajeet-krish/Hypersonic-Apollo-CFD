"""SU2 solver interface for hypersonic blunt body simulations.

Runs SU2_CFD as a subprocess, parses convergence history, and extracts
stagnation-point values from the solution VTU file.
"""
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config import get_su2_binary
from .vtu_parser import extract_stagnation_values, parse_vtu

logger = logging.getLogger(__name__)


@dataclass
class SU2Results:
    """Parsed results from an SU2 simulation.

    Attributes:
        converged: Whether the simulation converged (residual drop > 3 orders).
        iterations: Number of iterations actually run.
        residual_drop: Orders-of-magnitude drop in RMS density residual.
        history: List of parsed history.csv rows as dicts.
        stagnation_heat_flux: Stagnation heat flux (W/m^2), if available.
        stagnation_pressure: Stagnation-point static pressure (Pa).
        max_mach: Maximum Mach number in the domain.
    """

    converged: bool = False
    iterations: int = 0
    residual_drop: float = 0.0
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
                results.converged = results.residual_drop > 3.0

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
