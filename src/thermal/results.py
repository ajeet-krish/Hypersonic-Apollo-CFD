"""Results dataclass and serialization for thermal analysis."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ThermalResult2D:
    """Results from 2D thermal simulation.

    Attributes:
        s: Surface coordinates (m), shape (n_s,).
        z: Wall thickness coordinates (m), shape (n_z+1,).
        T_initial: Initial temperature field (n_s x n_z+1).
        T_final: Final temperature field (n_s x n_z+1).
        T_wall_history: Wall temperature at each time step (n_time x n_s).
        t_history: Time array (s).
        q_surface: Surface heat flux distribution (W/m^2).
        T_max_wall: Maximum hot-side wall temperature (K).
        T_max_back: Maximum cold-side temperature (K).
        q_total: Total heat input (J/m^2).
        material: Material name used.
        wall_thickness: Wall thickness (m).
    """

    s: np.ndarray
    z: np.ndarray
    T_initial: np.ndarray
    T_final: np.ndarray
    T_wall_history: np.ndarray
    t_history: np.ndarray
    q_surface: np.ndarray
    T_max_wall: float
    T_max_back: float
    q_total: float
    material: str
    wall_thickness: float


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


@dataclass
class AblationResult1D:
    """Results from 1D thermal simulation with ablation.

    Extends ThermalResult1D with density evolution and ablation metrics.

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
        rho_initial: Initial density profile (kg/m^3).
        rho_final: Final density profile (kg/m^3).
        rho_history: Density at each time step (n_time x n_cells+1).
        recession_m: Total surface recession (m).
        char_depth_m: Char layer depth (m).
        mass_loss_kg_m2: Total mass loss per unit area (kg/m^2).
        ablation_rate_mm_s: Average ablation rate (mm/s).
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
    rho_initial: np.ndarray
    rho_final: np.ndarray
    rho_history: np.ndarray
    recession_m: float
    char_depth_m: float
    mass_loss_kg_m2: float
    ablation_rate_mm_s: float


@dataclass
class AblationResult2D:
    """Results from 2D thermal simulation with ablation.

    Extends ThermalResult2D with density evolution and ablation metrics.

    Attributes:
        s: Surface coordinates (m), shape (n_s,).
        z: Wall thickness coordinates (m), shape (n_z+1,).
        T_initial: Initial temperature field (n_s x n_z+1).
        T_final: Final temperature field (n_s x n_z+1).
        T_wall_history: Wall temperature at each time step (n_time x n_s).
        t_history: Time array (s).
        q_surface: Surface heat flux distribution (W/m^2).
        T_max_wall: Maximum hot-side wall temperature (K).
        T_max_back: Maximum cold-side temperature (K).
        q_total: Total heat input (J/m^2).
        material: Material name used.
        wall_thickness: Wall thickness (m).
        rho_initial: Initial density field (n_s x n_z+1).
        rho_final: Final density field (n_s x n_z+1).
        rho_history: Density at wall surface over time (n_time x n_s).
        recession_m: Surface recession at each s point (m).
        char_depth_m: Char depth at each s point (m).
        mass_loss_kg_m2: Mass loss per unit area at each s point (kg/m^2).
        ablation_rate_mm_s: Ablation rate at each s point (mm/s).
    """

    s: np.ndarray
    z: np.ndarray
    T_initial: np.ndarray
    T_final: np.ndarray
    T_wall_history: np.ndarray
    t_history: np.ndarray
    q_surface: np.ndarray
    T_max_wall: float
    T_max_back: float
    q_total: float
    material: str
    wall_thickness: float
    rho_initial: np.ndarray
    rho_final: np.ndarray
    rho_history: np.ndarray
    recession_m: np.ndarray
    char_depth_m: np.ndarray
    mass_loss_kg_m2: np.ndarray
    ablation_rate_mm_s: np.ndarray


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


def save_thermal_results_2d(result: ThermalResult2D, output_path: Path) -> Path:
    """Save 2D thermal results to JSON.

    Serializes the 2D thermal solution into a JSON file compatible with
    the project's existing output format.

    Args:
        result: ThermalResult2D from the 2D thermal solver.
        output_path: Path to the output JSON file.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = build_thermal_summary_2d(result)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    return output_path


def build_thermal_summary_2d(result: ThermalResult2D) -> dict:
    """Build summary dict for 2D JSON serialization.

    Args:
        result: ThermalResult2D from the 2D thermal solver.

    Returns:
        Dictionary with all 2D thermal results.
    """
    return {
        "config": {
            "material": result.material,
            "wall_thickness_m": result.wall_thickness,
            "n_s": len(result.s),
            "n_z": len(result.z) - 1,
        },
        "results": {
            "T_max_wall_K": round(result.T_max_wall, 2),
            "T_max_back_K": round(result.T_max_back, 2),
            "q_total_J_m2": round(result.q_total, 2),
        },
        "surface": {
            "s_m": [round(float(s), 6) for s in result.s],
            "q_surface_W_m2": [round(float(q), 2) for q in result.q_surface],
        },
        "profiles": {
            "z_m": [round(float(z), 6) for z in result.z],
            "T_initial_K": [
                [round(float(T), 2) for T in row]
                for row in result.T_initial
            ],
            "T_final_K": [
                [round(float(T), 2) for T in row]
                for row in result.T_final
            ],
        },
        "time": {
            "n_steps": len(result.t_history),
            "dt_effective_s": (
                round(float(result.t_history[1] - result.t_history[0]), 6)
                if len(result.t_history) > 1
                else 0.0
            ),
        },
    }


def save_ablation_results_1d(result: AblationResult1D, output_path: Path) -> Path:
    """Save 1D ablation results to JSON.

    Serializes the 1D ablation solution into a JSON file including
    density evolution and ablation metrics.

    Args:
        result: AblationResult1D from the 1D thermal solver with ablation.
        output_path: Path to the output JSON file.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = build_ablation_summary_1d(result)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    return output_path


def build_ablation_summary_1d(result: AblationResult1D) -> dict:
    """Build summary dict for 1D ablation JSON serialization.

    Args:
        result: AblationResult1D from the 1D thermal solver with ablation.

    Returns:
        Dictionary with all 1D ablation results.
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
            "recession_m": round(result.recession_m, 6),
            "char_depth_m": round(result.char_depth_m, 6),
            "mass_loss_kg_m2": round(result.mass_loss_kg_m2, 4),
            "ablation_rate_mm_s": round(result.ablation_rate_mm_s, 4),
        },
        "profiles": {
            "z_m": [round(float(z), 6) for z in result.z],
            "T_initial_K": [round(float(T), 2) for T in result.T_initial],
            "T_final_K": [round(float(T), 2) for T in result.T_final],
            "rho_initial_kg_m3": [round(float(r), 2) for r in result.rho_initial],
            "rho_final_kg_m3": [round(float(r), 2) for r in result.rho_final],
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


def save_ablation_results_2d(result: AblationResult2D, output_path: Path) -> Path:
    """Save 2D ablation results to JSON.

    Serializes the 2D ablation solution into a JSON file including
    density evolution and ablation metrics.

    Args:
        result: AblationResult2D from the 2D thermal solver with ablation.
        output_path: Path to the output JSON file.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = build_ablation_summary_2d(result)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    return output_path


def build_ablation_summary_2d(result: AblationResult2D) -> dict:
    """Build summary dict for 2D ablation JSON serialization.

    Args:
        result: AblationResult2D from the 2D thermal solver with ablation.

    Returns:
        Dictionary with all 2D ablation results.
    """
    return {
        "config": {
            "material": result.material,
            "wall_thickness_m": result.wall_thickness,
            "n_s": len(result.s),
            "n_z": len(result.z) - 1,
        },
        "results": {
            "T_max_wall_K": round(result.T_max_wall, 2),
            "T_max_back_K": round(result.T_max_back, 2),
            "q_total_J_m2": round(result.q_total, 2),
            "recession_m_max": round(float(np.max(result.recession_m)), 6),
            "char_depth_m_max": round(float(np.max(result.char_depth_m)), 6),
            "mass_loss_kg_m2_max": round(float(np.max(result.mass_loss_kg_m2)), 4),
            "ablation_rate_mm_s_max": round(float(np.max(result.ablation_rate_mm_s)), 4),
        },
        "surface": {
            "s_m": [round(float(s), 6) for s in result.s],
            "q_surface_W_m2": [round(float(q), 2) for q in result.q_surface],
        },
        "profiles": {
            "z_m": [round(float(z), 6) for z in result.z],
        },
        "time": {
            "n_steps": len(result.t_history),
            "dt_effective_s": (
                round(float(result.t_history[1] - result.t_history[0]), 6)
                if len(result.t_history) > 1
                else 0.0
            ),
        },
    }
