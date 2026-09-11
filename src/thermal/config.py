"""Configuration for thermal analysis."""
from dataclasses import dataclass


@dataclass(frozen=True)
class AblationConfig:
    """Configuration for charring ablation model.

    Attributes:
        pyrolysis_A: Pre-exponential factor (1/s).
        pyrolysis_Ea: Activation energy (J/mol).
        pyrolysis_R: Gas constant (J/(mol*K)).
        blow_coeff: Blowing correction coefficient.
        surface_emissivity: Surface emissivity for re-radiation.
    """

    pyrolysis_A: float = 1.0e6  # 1/s
    pyrolysis_Ea: float = 1.0e5  # J/mol
    pyrolysis_R: float = 8.314  # J/(mol*K)
    blow_coeff: float = 0.5
    surface_emissivity: float = 0.9


@dataclass(frozen=True)
class ThermalConfig2D:
    """Configuration for 2D axisymmetric thermal analysis.

    Attributes:
        material: Heat shield material name.
        wall_thickness: Total wall thickness (m).
        n_s: Number of surface points along body.
        n_z: Number of cells through wall thickness.
        dt: Time step (s).
        t_end: Total simulation time (s).
        q_surface: Surface heat flux array (W/m^2) - from VTU extraction.
        s_surface: Surface coordinate array (m) - from VTU extraction.
        cold_wall_temp: Cold side boundary temperature (K).
        radiation: Whether to include radiation on cold side.
        emissivity: Surface emissivity for radiation.
    """

    material: str = "avcoat"
    wall_thickness: float = 0.05
    n_s: int = 50
    n_z: int = 50
    dt: float = 0.1
    t_end: float = 100.0
    q_surface: tuple[float, ...] = (500000.0,)
    s_surface: tuple[float, ...] = (0.0,)
    cold_wall_temp: float = 300.0
    radiation: bool = True
    emissivity: float = 0.8


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
