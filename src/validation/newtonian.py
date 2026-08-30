"""Modified Newtonian pressure coefficient distribution."""

import numpy as np

from .shock_relations import normal_shock


def modified_newtonian_cp(
    theta: float | np.ndarray,
    M: float,
    gamma: float = 1.4,
) -> float | np.ndarray:
    """Modified Newtonian pressure coefficient.

    Cp = Cp_max * sin^2(theta)

    where Cp_max is computed from the normal shock stagnation pressure ratio:
        Cp_max = 2 / (gamma * M^2) * (p02/p1 - 1)
    and p02/p1 is derived from normal shock total pressure ratio p02/p01.

    Args:
        theta: Surface angle from freestream direction (radians).
               theta=pi/2 is stagnation point, theta=0 is tangent to flow.
        M: Freestream Mach number
        gamma: Ratio of specific heats

    Returns:
        Pressure coefficient (float if scalar, ndarray if array)
    """
    shock = normal_shock(M, gamma)

    # p02/p01 is the total pressure ratio across the shock
    # p02/p1 = (p02/p01) * (p01/p1)
    # For an ideal gas: p01/p1 = (1 + (gamma-1)/2 * M^2)^(gamma/(gamma-1))
    p01_over_p1 = (1.0 + (gamma - 1.0) / 2.0 * M**2) ** (gamma / (gamma - 1.0))
    p02_over_p1 = shock.p0_ratio * p01_over_p1

    cp_max = 2.0 / (gamma * M**2) * (p02_over_p1 - 1.0)

    return cp_max * np.sin(theta) ** 2


def stagnation_cp(M: float, gamma: float = 1.4) -> float:
    """Stagnation point pressure coefficient (theta = pi/2).

    Args:
        M: Freestream Mach number
        gamma: Ratio of specific heats

    Returns:
        Cp at stagnation point
    """
    return float(modified_newtonian_cp(np.pi / 2.0, M, gamma))
