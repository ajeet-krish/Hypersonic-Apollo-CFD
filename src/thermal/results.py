"""Results dataclass and serialization for thermal analysis."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ThermalResult1D:
    """Results from 1D thermal simulation.

    Attributes:
        z: Wall coordinates (m), shape (n_cells+1,).
        T_initial: Initial temperature profile (K), shape (n_cells+1,).
        T_final: Final temperature profile (K), shape (n_cells+1,).
        T_history: Temperature at each time step (n_time x n_cells+1).
        t_history: Time array (s), shape (n_time,).
        t_end: Total simulation time (s).
        T_max_wall: Maximum hot-side wall temperature (K).
        T_max_back: Maximum cold-side temperature (K).
        q_total: Total heat input (J/m^2).
        material: Material name used.
        wall_thickness: Wall thickness (m).
    """

    z: np.ndarray
    T_initial: np.ndarray
    T_final: np.ndarray
    T_history: np.ndarray
    t_history: np.ndarray
    t_end: float
    T_max_wall: float
    T_max_back: float
    q_total: float
    material: str
    wall_thickness: float


def save_thermal_results(result: ThermalResult1D, output_path: Path) -> Path:
    """Save thermal results to JSON.

    Serializes the 1D thermal solution into a JSON file compatible with
    the project's existing output format.

    Args:
        result: ThermalResult1D from the thermal solver.
        output_path: Path to the output JSON file.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = build_thermal_summary(result)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    return output_path


def build_thermal_summary(result: ThermalResult1D) -> dict:
    """Build summary dict for JSON serialization.

    Args:
        result: ThermalResult1D from the thermal solver.

    Returns:
        Dictionary with all thermal results.
    """
    return {
        "config": {
            "material": result.material,
            "wall_thickness_m": result.wall_thickness,
            "t_end_s": result.t_end,
        },
        "results": {
            "T_max_wall_K": round(result.T_max_wall, 2),
            "T_max_back_K": round(result.T_max_back, 2),
            "q_total_J_m2": round(result.q_total, 2),
        },
        "profiles": {
            "z_m": [round(float(z), 6) for z in result.z],
            "T_initial_K": [round(float(T), 2) for T in result.T_initial],
            "T_final_K": [round(float(T), 2) for T in result.T_final],
        },
        "time": {
            "t_end_s": result.t_end,
            "n_steps": len(result.t_history),
            "dt_effective_s": (
                round(float(result.t_history[1] - result.t_history[0]), 6)
                if len(result.t_history) > 1
                else 0.0
            ),
        },
    }
