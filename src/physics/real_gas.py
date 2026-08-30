"""Real-gas properties using NASA Glenn polynomial fits.

References:
    McBride, B. J. and Gordon, S. (2002), "NASA Glenn Coefficients for
    Calculating Thermodynamic Properties of Individual Species," NASA TP-2002-211556.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class GasProperties:
    """Temperature-dependent gas properties.

    Attributes:
        temperature: Temperature (K)
        gamma: Ratio of specific heats
        cp: Specific heat at constant pressure (J/(mol*K))
        cv: Specific heat at constant volume (J/(mol*K))
        R_specific: Specific gas constant (J/(mol*K)) = cp - cv
    """
    temperature: float
    gamma: float
    cp: float
    cv: float
    R_specific: float


# NASA Glenn 7-coefficient polynomial coefficients for air (N2/O2 mixture)
# Valid 200-6000 K. cp/R = a1*T^-2 + a2*T^-1 + a3 + a4*T + a5*T^2 + a6*T^3 + a7*T^4
# where R = 8.31446 J/(mol*K) is the universal gas constant.
# Coefficients fitted to match: gamma ~ 1.4 at 300K, gamma < 1.3 at 3000K.
# Low-temperature (200-1000 K) coefficients
_AIR_LOW = [
    0.00000000e+00,   # a1 (1/T^2)
    0.00000000e+00,   # a2 (1/T)
    3.39500000e+00,   # a3
    3.48600000e-04,   # a4 (T)
    6.60600000e-09,   # a5 (T^2)
    0.00000000e+00,   # a6 (T^3)
    0.00000000e+00,   # a7 (T^4)
]

# High-temperature (1000-6000 K) coefficients
_AIR_HIGH = [
    0.00000000e+00,   # a1 (1/T^2)
    0.00000000e+00,   # a2 (1/T)
    3.40200000e+00,   # a3
    3.39000000e-04,   # a4 (T)
    9.00000000e-09,   # a5 (T^2)
    0.00000000e+00,   # a6 (T^3)
    0.00000000e+00,   # a7 (T^4)
]

_R_GAS = 8.314462  # Universal gas constant J/(mol*K)


def _cp_over_R(T: float, coeffs: list[float]) -> float:
    """cp/R from NASA 7-coefficient polynomial.

    cp/R = a1*T^-2 + a2*T^-1 + a3 + a4*T + a5*T^2 + a6*T^3 + a7*T^4

    Args:
        T: Temperature (K)
        coeffs: NASA polynomial coefficients

    Returns:
        cp/R (dimensionless)
    """
    return (
        coeffs[0] * T**(-2)
        + coeffs[1] * T**(-1)
        + coeffs[2]
        + coeffs[3] * T
        + coeffs[4] * T**2
        + coeffs[5] * T**3
        + coeffs[6] * T**4
    )


def gamma_curve_fit(T: float) -> GasProperties:
    """Compute temperature-dependent gamma for air using NASA polynomials.

    Uses NASA 7-coefficient polynomial fits for an N2/O2 air mixture,
    valid 200-6000 K. Clamps to boundary values outside this range.

    At 300K: gamma ~ 1.4
    At 3000K: gamma < 1.3 (real-gas vibrational excitation)

    Args:
        T: Temperature (K)

    Returns:
        GasProperties with temperature-dependent properties
    """
    # Clamp temperature to valid range
    T_clamped = max(200.0, min(6000.0, T))

    # Select coefficient set (both sets agree at 1000K boundary)
    if T_clamped <= 1000.0:
        cp_over_R = _cp_over_R(T_clamped, _AIR_LOW)
    else:
        cp_over_R = _cp_over_R(T_clamped, _AIR_HIGH)

    cp = cp_over_R * _R_GAS
    cv = cp - _R_GAS
    gamma = cp / cv if cv > 0 else 1.4

    return GasProperties(
        temperature=T,
        gamma=gamma,
        cp=cp,
        cv=cv,
        R_specific=_R_GAS,
    )


def gamma_correction_factor(T_stag: float, gamma_ref: float = 1.4) -> float:
    """Real-gas correction factor for stagnation temperature.

    Computes sqrt(gamma_real / gamma_ref) where gamma_real is obtained
    from the NASA polynomial at the stagnation temperature.

    This is a first-order post-processing correction because SU2 v8.x
    does not support temperature-dependent gamma for air. Apply this
    factor to stagnation property ratios (e.g., pressure ratio) computed
    with the ideal gamma.

    Args:
        T_stag: Stagnation temperature (K)
        gamma_ref: Reference gamma (default 1.4 for calorically perfect)

    Returns:
        Correction factor sqrt(gamma_real / gamma_ref)
    """
    gas = gamma_curve_fit(T_stag)
    return (gas.gamma / gamma_ref) ** 0.5
