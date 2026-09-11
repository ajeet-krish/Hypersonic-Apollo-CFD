"""Configuration for thermal analysis."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ThermalConfig:
    """Configuration for 1D thermal analysis of a heat shield wall.

    Attributes:
        material: Heat shield material name (e.g., 'avcoat', 'pica').
        wall_thickness: Total wall thickness (m).
        n_cells: Number of cells through wall thickness.
        dt: Time step (s).
        t_end: Total simulation time (s).
        q_stagnation: Stagnation point heat flux (W/m^2).
        q_distribution: Heat flux distribution type ('uniform' or 'sinusoidal').
        cold_wall_temp: Cold side boundary temperature (K).
        radiation: Whether to include radiation on the cold side.
        emissivity: Surface emissivity for radiation.
    """

    material: str = "avcoat"
    wall_thickness: float = 0.05  # 50 mm
    n_cells: int = 50
    dt: float = 0.1  # seconds
    t_end: float = 100.0  # seconds
    q_stagnation: float = 500000.0  # 500 kW/m^2
    q_distribution: str = "uniform"
    cold_wall_temp: float = 300.0
    radiation: bool = True
    emissivity: float = 0.8
