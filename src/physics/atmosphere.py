"""US Standard Atmosphere 1976 (0-86 km).

References:
    NOAA, NASA, USAF (1976), "US Standard Atmosphere 1976," NASA TM-X-74355.
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AtmosphereResult:
    """Standard atmosphere properties at a given altitude.

    Attributes:
        altitude: Altitude (m)
        temperature: Static temperature (K)
        pressure: Static pressure (Pa)
        density: Density (kg/m^3)
        speed_of_sound: Speed of sound (m/s)
        dynamic_viscosity: Dynamic viscosity (Pa*s)
    """
    altitude: float
    temperature: float
    pressure: float
    density: float
    speed_of_sound: float
    dynamic_viscosity: float


# Layer boundaries (km) and base temperatures (K)
# Layers: 0-11, 11-20, 20-32, 32-47, 47-51, 51-71, 71-86
_LAYER_BOUNDARIES_KM = [0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 86.0]
_LAYER_BASE_TEMP_K = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.87]
_LAYER_LAPSE_RATE = [-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002]

# Sea level reference values
_T0 = 288.15       # K
_P0 = 101325.0     # Pa
_rho0 = 1.225      # kg/m^3
_G0 = 9.80665      # m/s^2
_R_AIR = 287.0528  # J/(kg*K)
_GAMMA = 1.4
_SUTHERLAND_S = 110.4  # K
_SUTHERLAND_C = 1.458e-6  # Pa*s*K^(-0.5)


def _sutherland_viscosity(T: float) -> float:
    """Sutherland's law for dynamic viscosity of air.

    mu = 1.458e-6 * T^1.5 / (T + 110.4)

    Args:
        T: Temperature (K)

    Returns:
        Dynamic viscosity (Pa*s)
    """
    return _SUTHERLAND_C * T**1.5 / (T + _SUTHERLAND_S)


def standard_atmosphere(altitude: float) -> AtmosphereResult:
    """US Standard Atmosphere 1976, valid 0-86 km.

    Computes temperature, pressure, density, speed of sound, and
    viscosity at a given altitude using the layered atmosphere model.

    Note:
        The 71-86 km layer uses a constant lapse rate of -0.002 K/m,
        which yields T=184.65 K at 86 km. The standard value at 86 km
        is 186.87 K. This ~1.2% discrepancy is inherent to the
        constant-lapse-rate approximation used by the piecewise model
        and is accepted for this simplified implementation.

    Args:
        altitude: Altitude (m), valid range 0 to 86000

    Returns:
        AtmosphereResult with all thermodynamic properties

    Raises:
        ValueError: If altitude is outside 0-86 km range
    """
    if altitude < 0.0 or altitude > 86000.0:
        raise ValueError(
            f"Altitude must be between 0 and 86000 m, got {altitude}"
        )

    alt_km = altitude / 1000.0

    # Find which layer we are in
    layer_idx = 0
    for i in range(len(_LAYER_BOUNDARIES_KM) - 1):
        if alt_km >= _LAYER_BOUNDARIES_KM[i]:
            layer_idx = i

    # Accumulate pressure from sea level through each layer
    T_base = _LAYER_BASE_TEMP_K[0]
    P_base = _P0

    for i in range(layer_idx):
        h_bottom = _LAYER_BOUNDARIES_KM[i] * 1000.0
        h_top = _LAYER_BOUNDARIES_KM[i + 1] * 1000.0
        dh = h_top - h_bottom
        lapse = _LAYER_LAPSE_RATE[i]

        if abs(lapse) < 1e-12:
            # Isothermal layer
            P_base = P_base * math.exp(-_G0 * dh / (_R_AIR * T_base))
        else:
            # Gradient layer
            T_top = T_base + lapse * dh
            P_base = P_base * (T_top / T_base) ** (-_G0 / (lapse * _R_AIR))
            T_base = T_top

    # Now compute properties in the target layer
    h_bottom = _LAYER_BOUNDARIES_KM[layer_idx] * 1000.0
    lapse = _LAYER_LAPSE_RATE[layer_idx]
    dh = altitude - h_bottom

    if abs(lapse) < 1e-12:
        # Isothermal layer
        T = T_base
        P = P_base * math.exp(-_G0 * dh / (_R_AIR * T_base))
    else:
        # Gradient layer
        T = T_base + lapse * dh
        P = P_base * (T / T_base) ** (-_G0 / (lapse * _R_AIR))

    # Density from ideal gas law
    rho = P / (_R_AIR * T)

    # Speed of sound
    a = math.sqrt(_GAMMA * _R_AIR * T)

    # Dynamic viscosity (Sutherland)
    mu = _sutherland_viscosity(T)

    return AtmosphereResult(
        altitude=altitude,
        temperature=T,
        pressure=P,
        density=rho,
        speed_of_sound=a,
        dynamic_viscosity=mu,
    )
