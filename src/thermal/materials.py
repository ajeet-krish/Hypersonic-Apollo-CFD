"""Material database for heat shield ablatives.

Contains thermal property definitions for AVCOAT and PICA materials
with temperature-dependent property tables. Uses piecewise linear
interpolation between tabulated values (no polynomial fitting to avoid
oscillation risk).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ThermalMaterial:
    """Thermal properties of a heat shield material.

    All properties are temperature-dependent. Use piecewise linear
    interpolation between tabulated values.

    Attributes:
        name: Material name.
        density: Density at reference temperature (kg/m^3).
        thermal_conductivity: Thermal conductivity at reference temperature (W/(m*K)).
        specific_heat: Specific heat at reference temperature (J/(kg*K)).
        decomposition_temperature: Pyrolysis onset temperature (K).
        char_temperature: Full char temperature (K).
        heat_of_pyrolysis: Energy absorbed during decomposition (J/kg).
        emissivity: Surface emissivity for radiation.
        k_table: Temperature-dependent conductivity table (T, k) pairs, or None.
        cp_table: Temperature-dependent specific heat table (T, cp) pairs, or None.
    """

    name: str
    density: float  # kg/m^3
    thermal_conductivity: float  # W/(m*K)
    specific_heat: float  # J/(kg*K)
    decomposition_temperature: float  # K
    char_temperature: float  # K
    heat_of_pyrolysis: float  # J/kg
    emissivity: float

    # Temperature-dependent tables (optional overrides)
    # If None, use constant values above
    k_table: tuple[tuple[float, float], ...] | None = None
    cp_table: tuple[tuple[float, float], ...] | None = None

    def k_at(self, T: float) -> float:
        """Thermal conductivity at temperature T (K).

        Uses piecewise linear interpolation between tabulated values.
        Falls back to constant value if no table is provided. Clamps to
        table endpoints for extrapolation.

        Args:
            T: Temperature (K).

        Returns:
            Thermal conductivity (W/(m*K)).
        """
        if self.k_table is None:
            return self.thermal_conductivity
        return _piecewise_interpolate(T, self.k_table)

    def cp_at(self, T: float) -> float:
        """Specific heat at temperature T (K).

        Uses piecewise linear interpolation between tabulated values.
        Falls back to constant value if no table is provided. Clamps to
        table endpoints for extrapolation.

        Args:
            T: Temperature (K).

        Returns:
            Specific heat (J/(kg*K)).
        """
        if self.cp_table is None:
            return self.specific_heat
        return _piecewise_interpolate(T, self.cp_table)


def _piecewise_interpolate(
    T: float,
    table: tuple[tuple[float, float], ...],
) -> float:
    """Piecewise linear interpolation with clamping.

    Args:
        T: Temperature (K) at which to evaluate.
        table: Sorted (T, value) pairs.

    Returns:
        Interpolated value, clamped to table endpoints.
    """
    if T <= table[0][0]:
        return table[0][1]
    if T >= table[-1][0]:
        return table[-1][1]

    for i in range(len(table) - 1):
        t0, v0 = table[i]
        t1, v1 = table[i + 1]
        if t0 <= T <= t1:
            frac = (T - t0) / (t1 - t0)
            return v0 + frac * (v1 - v0)

    # Fallback (should not reach here with valid table)
    return table[-1][1]


# Pre-defined materials
AVCOAT = ThermalMaterial(
    name="AVCOAT-5026",
    density=512.0,
    thermal_conductivity=0.5,
    specific_heat=1000.0,
    decomposition_temperature=600.0,
    char_temperature=3000.0,
    heat_of_pyrolysis=1.5e6,
    emissivity=0.8,
    k_table=((300, 0.5), (600, 0.8), (1000, 1.2), (1500, 1.8), (2000, 2.5)),
    cp_table=((300, 1000), (600, 1200), (1000, 1500), (1500, 1800), (2000, 2000)),
)

PICA = ThermalMaterial(
    name="PICA-X",
    density=240.0,
    thermal_conductivity=0.5,
    specific_heat=1000.0,
    decomposition_temperature=500.0,
    char_temperature=3200.0,
    heat_of_pyrolysis=2.0e6,
    emissivity=0.85,
    k_table=((300, 0.5), (500, 0.7), (1000, 1.0), (1500, 1.5), (2000, 2.0)),
    cp_table=((300, 1000), (500, 1100), (1000, 1300), (1500, 1600), (2000, 1800)),
)

MATERIALS: dict[str, ThermalMaterial] = {"avcoat": AVCOAT, "pica": PICA}


def get_material(name: str) -> ThermalMaterial:
    """Look up material by name (case-insensitive).

    Args:
        name: Material name (e.g., 'avcoat', 'PICA').

    Returns:
        Matching ThermalMaterial instance.

    Raises:
        ValueError: If material name is not in the database.
    """
    key = name.lower().strip()
    if key not in MATERIALS:
        available = ", ".join(sorted(MATERIALS.keys()))
        raise ValueError(
            f"Unknown material '{name}'. Available: {available}"
        )
    return MATERIALS[key]
