"""Sutton-Graves stagnation point heating correlation.

References:
    Sutton, K. and Graves, R. A. (1985), "A General Stagnation-Point
    Convective-Heating Equation for Arbitrary Gas Mixtures," NASA TR R-376.
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FayRiddellResult:
    """Result from Sutton-Graves stagnation heating calculation.

    Attributes:
        q_stag: Stagnation heat flux (W/m^2)
        q_stag_kw: Stagnation heat flux (kW/m^2)
        R_nose: Nose radius (m)
        rho_inf: Freestream density (kg/m^3)
        V_inf: Freestream velocity (m/s)
        formula: Correlation formula string
    """
    q_stag: float
    q_stag_kw: float
    R_nose: float
    rho_inf: float
    V_inf: float
    formula: str


def sutton_graves(
    rho_inf: float,
    V_inf: float,
    R_nose: float,
    C: float = 1.83e-4,
) -> FayRiddellResult:
    """Sutton-Graves stagnation point heating rate.

    This is the Sutton-Graves form: q = C * sqrt(rho/R) * V^3
    NOT the Chapman form: q = C * sqrt(rho^3/R) * V^3

    Reference: Sutton & Graves (1985), NASA TR R-376.

    Args:
        rho_inf: Freestream density (kg/m^3)
        V_inf: Freestream velocity (m/s)
        R_nose: Nose sphere radius (m)
        C: Correlation constant (default 1.83e-4 SI units)

    Returns:
        FayRiddellResult with heating rate and inputs
    """
    if R_nose <= 0:
        raise ValueError("R_nose must be > 0")
    if rho_inf < 0:
        raise ValueError("rho_inf must be >= 0")

    q_stag = C * math.sqrt(rho_inf / R_nose) * V_inf**3
    q_stag_kw = q_stag / 1000.0

    return FayRiddellResult(
        q_stag=q_stag,
        q_stag_kw=q_stag_kw,
        R_nose=R_nose,
        rho_inf=rho_inf,
        V_inf=V_inf,
        formula="q = C * sqrt(rho_inf / R_nose) * V_inf^3",
    )
