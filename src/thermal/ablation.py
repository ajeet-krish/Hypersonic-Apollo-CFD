"""Charring ablation model with pyrolysis kinetics and surface recession.

Implements Arrhenius-based pyrolysis, density evolution, char formation
(property switching), blowing correction to convective heat flux, and
surface recession from energy balance.

The ablation model is designed to be coupled with the 1D or 2D thermal
solver. It operates on the density field alongside the temperature field,
computing density change rates, heat sinks from endothermic pyrolysis,
and surface recession.
"""
import numpy as np

from .config import AblationConfig
from .materials import ThermalMaterial


class AblationModel:
    """Charring ablation model with pyrolysis kinetics and surface recession.

    Computes:
        - Pyrolysis rate (Arrhenius)
        - Density evolution
        - Char formation (property switching via density interpolation)
        - Blowing correction to convective heat flux
        - Surface recession (energy balance)
        - Heat sink from endothermic pyrolysis

    Attributes:
        material: Thermal material with char properties defined.
        config: Ablation configuration parameters.
    """

    def __init__(self, material: ThermalMaterial, config: AblationConfig) -> None:
        """Initialize ablation model.

        Args:
            material: Thermal material with char properties defined.
            config: Ablation configuration parameters.
        """
        self.material = material
        self.config = config
        self.heat_of_pyrolysis = material.heat_of_pyrolysis

    def compute_pyrolysis_rate(
        self,
        rho: np.ndarray,
        T: np.ndarray,
    ) -> np.ndarray:
        """Compute drho/dt from Arrhenius kinetics.

        Uses the relation:
            drho/dt = -A * exp(-Ea/(R*T)) * rho

        Only active above the material's decomposition temperature.

        Args:
            rho: Current density field (kg/m^3).
            T: Current temperature field (K).

        Returns:
            drho_dt: Rate of density change (negative, kg/(m^3*s)).
        """
        A = self.config.pyrolysis_A
        Ea = self.config.pyrolysis_Ea
        R = self.config.pyrolysis_R

        # Arrhenius rate (only active above decomposition temperature)
        rate = np.where(
            T > self.material.decomposition_temperature,
            A * np.exp(-Ea / (R * np.maximum(T, 1.0))),
            0.0,
        )
        drho_dt = -rate * rho
        return drho_dt

    def update_density(
        self,
        rho: np.ndarray,
        T: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance density field by one time step.

        Computes the pyrolysis rate and integrates forward with explicit
        Euler, clamping density to the char density floor.

        Args:
            rho: Current density field (kg/m^3).
            T: Current temperature field (K).
            dt: Time step (s).

        Returns:
            Tuple of (rho_new, drho_dt): Updated density and rate of change.
        """
        drho_dt = self.compute_pyrolysis_rate(rho, T)
        rho_new = rho + drho_dt * dt
        # Clamp to char density
        char_rho = self.material.char_density if self.material.char_density is not None else 0.0
        rho_new = np.maximum(rho_new, char_rho)
        return rho_new, drho_dt

    def compute_blowing_factor(
        self,
        rho_surface: np.ndarray,
    ) -> np.ndarray:
        """Compute blowing correction factor B.

        The blowing correction accounts for the blockage of convective
        heat transfer by pyrolysis gases exiting the surface.

        B = 1 / (1 + B_star)
        B_star = blow_coeff * (rho_virgin - rho_surface) / rho_virgin

        Args:
            rho_surface: Surface density at each point (kg/m^3).

        Returns:
            B: Blowing factor in (0, 1]. B=1 means no blowing reduction.
        """
        rho_v = self.material.density
        mass_loss = np.maximum(0.0, rho_v - rho_surface) / rho_v
        B_star = self.config.blow_coeff * mass_loss
        B = 1.0 / (1.0 + B_star)
        return np.maximum(B, 0.1)  # Never reduce heating below 10%

    def compute_pyrolysis_heat_sink(
        self,
        drho_dt: np.ndarray,
    ) -> np.ndarray:
        """Compute energy sink from endothermic pyrolysis.

        The heat of pyrolysis represents energy absorbed per unit mass
        converted to char. This becomes a volumetric heat sink:

            Q_sink = H_pyr * |drho_dt|  [W/m^3]

        Args:
            drho_dt: Rate of density change (kg/(m^3*s)).

        Returns:
            Volumetric heat sink (W/m^3).
        """
        return self.heat_of_pyrolysis * np.abs(drho_dt)

    def compute_recession_rate(
        self,
        T_surface: np.ndarray,
        q_net: np.ndarray,
        rho_surface: np.ndarray,
    ) -> np.ndarray:
        """Compute surface recession rate from energy balance.

        Uses a simplified energy balance where the net heat flux drives
        surface recession after subtracting re-radiation losses:

            v_recess = max(0, q_net - q_rad) / (rho * H_pyr)

        Args:
            T_surface: Surface temperature at each point (K).
            q_net: Net convective heat flux at surface (W/m^2).
            rho_surface: Surface density at each point (kg/m^3).

        Returns:
            Surface recession rate (m/s) at each point.
        """
        sigma = 5.67e-8
        eps = self.config.surface_emissivity
        q_rad = eps * sigma * T_surface**4
        q_available = np.maximum(0.0, q_net - q_rad)
        v_recess = q_available / (np.maximum(rho_surface, 1.0) * self.heat_of_pyrolysis)
        return v_recess
