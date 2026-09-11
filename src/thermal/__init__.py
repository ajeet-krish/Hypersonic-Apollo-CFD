"""Thermal analysis for hypersonic heat shield design.

Provides material database, 1D thermal solver, heat flux extraction,
and results serialization for ablative heat shield analysis.
"""
from .config import ThermalConfig
from .heat_flux import heat_flux_distribution
from .materials import AVCOAT, PICA, ThermalMaterial, get_material
from .results import ThermalResult1D, build_thermal_summary, save_thermal_results
from .solver_1d import ThermalSolver1D

__all__ = [
    "AVCOAT",
    "PICA",
    "ThermalConfig",
    "ThermalMaterial",
    "ThermalResult1D",
    "ThermalSolver1D",
    "build_thermal_summary",
    "get_material",
    "heat_flux_distribution",
    "save_thermal_results",
]
