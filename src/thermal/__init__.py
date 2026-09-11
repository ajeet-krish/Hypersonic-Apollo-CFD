"""Thermal analysis for hypersonic heat shield design.

Provides material database, 1D and 2D thermal solvers, heat flux extraction,
ablation modeling, and results serialization for ablative heat shield analysis.
"""
from .ablation import AblationModel
from .config import AblationConfig, ThermalConfig, ThermalConfig2D
from .heat_flux import heat_flux_distribution
from .materials import AVCOAT, PICA, ThermalMaterial, get_material
from .results import (
    AblationResult1D,
    AblationResult2D,
    ThermalResult1D,
    ThermalResult2D,
    build_thermal_summary,
    save_thermal_results,
    save_thermal_results_2d,
)
from .solver_1d import ThermalSolver1D
from .solver_2d import ThermalSolver2D

__all__ = [
    "AVCOAT",
    "AblationConfig",
    "AblationModel",
    "AblationResult1D",
    "AblationResult2D",
    "PICA",
    "ThermalConfig",
    "ThermalConfig2D",
    "ThermalMaterial",
    "ThermalResult1D",
    "ThermalResult2D",
    "ThermalSolver1D",
    "ThermalSolver2D",
    "build_thermal_summary",
    "get_material",
    "heat_flux_distribution",
    "save_thermal_results",
    "save_thermal_results_2d",
]
