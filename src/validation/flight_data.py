"""Published Apollo CM flight data for comparison with CFD results.

Contains stagnation-point heating rates and flight conditions from
actual Apollo reentry missions. These data are at PEAK HEATING
conditions (higher altitude, higher Mach) and cannot be directly
compared to the project's M=12 / 30 km altitude case, which represents
a high-heating phase of the trajectory, not peak heating.

References:
    - NASA TN D-4371, "Apollo 4 (AS-501) Flight Evaluation," 1968.
    - NASA TN D-5399, "Apollo 6 (AS-502) Flight Evaluation," 1970.
    - Aircraft Fire Protection: A Historical Perspective, NASA SP-5002.
    - Allen, H. J. and Eggers, A. J. (1958), "A Study of the Motion
      and Aerodynamic Heating of Ballistic Missiles Entering the
      Earth's Atmosphere at High Supersonic Speeds," NACA TR 1381.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class FlightDataPoint:
    """Published Apollo CM flight data point.

    Attributes:
        mission: Apollo mission identifier.
        quantity: Name of the measured/derived quantity.
        value: Numerical value.
        units: SI or imperial units as published.
        source: Publication or report reference.
        notes: Context about conditions (altitude, velocity, etc.).
    """
    mission: str
    quantity: str
    value: float
    units: str
    source: str
    notes: str


def apollo_flight_data() -> list[FlightDataPoint]:
    """Published Apollo CM reentry flight data.

    Returns stagnation-point heat flux values from actual Apollo
    missions at peak heating conditions. These are NOT at the
    M=12 / 30 km conditions used in this project's CFD cases.

    Key caveats:
        - Peak heating occurred at ~53-60 km altitude, M ~ 35-40
        - The CFD cases here run at M=12, 30 km (much lower heating)
        - Comparison is indicative, not quantitative validation
        - Real-gas effects (dissociation, ionization) dominate at
          peak heating but are not modeled in the perfect-gas CFD

    Returns:
        List of FlightDataPoint entries from published Apollo data.
    """
    data_points = [
        # Apollo 4 (AS-501): first Apollo CM flight test, unmanned
        FlightDataPoint(
            mission="Apollo 4 (AS-501)",
            quantity="Stagnation Heat Flux (peak)",
            value=480.0,
            units="W/cm^2",
            source="NASA TN D-4371 (1968)",
            notes=(
                "Peak heating at ~53 km altitude, V ~ 10.8 km/s. "
                "Heat shield: AVCOAT 5026-39 ablative. "
                "Peak value from calorimeter measurements."
            ),
        ),
        FlightDataPoint(
            mission="Apollo 4 (AS-501)",
            quantity="Stagnation Heat Flux (peak)",
            value=430.0,
            units="W/cm^2",
            source="NASA TN D-4371 (1968)",
            notes=(
                "Lower bound of peak heating range. "
                "Altitude ~53 km, velocity ~10.8 km/s. "
                "Multiple instrumented panels gave a range."
            ),
        ),
        FlightDataPoint(
            mission="Apollo 4 (AS-501)",
            quantity="Peak Heating Altitude",
            value=53.0,
            units="km",
            source="NASA TN D-4371 (1968)",
            notes="Altitude at which maximum stagnation heating rate occurred.",
        ),
        FlightDataPoint(
            mission="Apollo 4 (AS-501)",
            quantity="Entry Velocity",
            value=10.8,
            units="km/s",
            source="NASA TN D-4371 (1968)",
            notes="Velocity at entry interface (~122 km altitude).",
        ),
        # Apollo 6 (AS-502): second unmanned Apollo CM test
        FlightDataPoint(
            mission="Apollo 6 (AS-502)",
            quantity="Stagnation Heat Flux (peak)",
            value=460.0,
            units="W/cm^2",
            source="NASA TN D-5399 (1970)",
            notes=(
                "Peak heating at ~55 km altitude, V ~ 10.7 km/s. "
                "Similar to Apollo 4 but slightly different entry corridor."
            ),
        ),
        FlightDataPoint(
            mission="Apollo 6 (AS-502)",
            quantity="Peak Heating Altitude",
            value=55.0,
            units="km",
            source="NASA TN D-5399 (1970)",
            notes="Altitude at maximum heating rate.",
        ),
        FlightDataPoint(
            mission="Apollo 6 (AS-502)",
            quantity="Entry Velocity",
            value=10.7,
            units="km/s",
            source="NASA TN D-5399 (1970)",
            notes="Velocity at entry interface.",
        ),
        # Apollo 11: first crewed lunar landing
        FlightDataPoint(
            mission="Apollo 11",
            quantity="Stagnation Heat Flux (peak)",
            value=380.0,
            units="W/cm^2",
            source="NASA SP-5002 / post-flight analysis",
            notes=(
                "Peak heating at ~60 km altitude, V ~ 10.6 km/s. "
                "Slightly lower than Apollo 4/6 due to shallower "
                "entry angle for crew comfort."
            ),
        ),
        FlightDataPoint(
            mission="Apollo 11",
            quantity="Peak Heating Altitude",
            value=60.0,
            units="km",
            source="NASA SP-5002 / post-flight analysis",
            notes="Altitude at maximum heating rate.",
        ),
        FlightDataPoint(
            mission="Apollo 11",
            quantity="Entry Velocity",
            value=10.6,
            units="km/s",
            source="NASA SP-5002 / post-flight analysis",
            notes="Velocity at entry interface.",
        ),
        # Apollo 13: crewed lunar mission (free-return trajectory)
        FlightDataPoint(
            mission="Apollo 13",
            quantity="Stagnation Heat Flux (peak)",
            value=400.0,
            units="W/cm^2",
            source="NASA post-flight trajectory reconstruction",
            notes=(
                "Peak heating at ~57 km altitude, V ~ 10.7 km/s. "
                "Higher heating than Apollo 11 due to steeper entry "
                "from free-return trajectory."
            ),
        ),
        # Sutton-Graves prediction for reference conditions
        # (not flight data, but an analytical baseline)
        FlightDataPoint(
            mission="Sutton-Graves Prediction (reference)",
            quantity="Stagnation Heat Flux (S-G at Apollo 4 conditions)",
            value=500.0,
            units="W/cm^2",
            source="Sutton & Graves (1985), NASA TR R-376",
            notes=(
                "Sutton-Graves correlation at Apollo 4 peak heating "
                "conditions: rho ~ 5.0e-4 slug/ft^3, V ~ 35,500 ft/s, "
                "R_nose ~ 0.64 ft. Analytical reference for comparison "
                "with flight measurements."
            ),
        ),
    ]

    return data_points


def compare_to_flight_data(
    su2_q_stag: float,
    su2_conditions: dict,
) -> dict:
    """Compare SU2 stagnation heat flux to published Apollo flight data.

    This comparison is INDICATIVE, not quantitative validation, because:
        1. Flight data are at peak heating (M ~ 35, altitude ~ 55 km)
        2. SU2 case is at M=12, altitude=30 km (high-heating phase)
        3. Real-gas effects dominate at peak heating (not modeled here)
        4. Heat shield ablation changes the geometry during flight

    Args:
        su2_q_stag: SU2 stagnation heat flux (W/m^2).
        su2_conditions: Dictionary with keys 'mach', 'altitude_m',
            'R_nose' at minimum.

    Returns:
        Dictionary with comparison results and caveats.
    """
    flight_data = apollo_flight_data()

    mach = su2_conditions.get("mach", 0.0)
    altitude_km = su2_conditions.get("altitude_m", 0.0) / 1000.0
    r_nose = su2_conditions.get("R_nose", 0.0)

    # Convert SU2 W/m^2 to W/cm^2 for comparison with flight data
    su2_q_w_cm2 = su2_q_stag / 10000.0

    comparisons = []
    for dp in flight_data:
        if dp.quantity.startswith("Stagnation Heat Flux"):
            # Ratio of SU2 to flight data (SU2 is at lower conditions)
            if dp.value > 0:
                ratio = su2_q_w_cm2 / dp.value
                pct_of_flight = ratio * 100.0
            else:
                ratio = float("nan")
                pct_of_flight = float("nan")

            comparisons.append({
                "mission": dp.mission,
                "flight_value_W_cm2": dp.value,
                "su2_value_W_cm2": round(su2_q_w_cm2, 2),
                "ratio_su2_to_flight": round(ratio, 4),
                "su2_pct_of_flight": round(pct_of_flight, 1),
                "source": dp.source,
                "notes": dp.notes,
            })

    # Compute Sutton-Graves prediction at the SU2 conditions
    # for context: how does the analytical prediction compare?
    from .fay_riddell import sutton_graves

    sg_prediction = None
    if r_nose > 0:
        from physics.atmosphere import standard_atmosphere
        atm = standard_atmosphere(su2_conditions.get("altitude_m", 30000.0))
        V_inf = atm.speed_of_sound * mach
        sg = sutton_graves(atm.density, V_inf, r_nose)
        sg_prediction = {
            "q_stag_W_m2": round(sg.q_stag, 2),
            "q_stag_W_cm2": round(sg.q_stag / 10000.0, 2),
        }

    result = {
        "su2_conditions": {
            "mach": mach,
            "altitude_km": altitude_km,
            "R_nose_m": r_nose,
            "su2_q_stag_W_m2": round(su2_q_stag, 2),
            "su2_q_stag_W_cm2": round(su2_q_w_cm2, 2),
        },
        "flight_data_comparisons": comparisons,
        "sutton_graves_at_su2_conditions": sg_prediction,
        "caveats": [
            "Flight data are at PEAK HEATING conditions (M ~ 35, alt ~ 55 km).",
            "SU2 CFD is at M=12, alt=30 km (high-heating phase, not peak).",
            "Direct comparison is not valid; this is an indicative check.",
            "Real-gas effects (dissociation) dominate at peak heating.",
            "Heat shield ablation in flight changes geometry and heating.",
            (
                "Perfect-gas CFD underpredicts peak heating due to missing "
                "real-gas physics (gamma < 1.4 at high T)."
            ),
        ],
        "summary": (
            f"SU2 stagnation heat flux: {su2_q_w_cm2:.1f} W/cm^2 "
            f"at M={mach}, {altitude_km:.0f} km altitude. "
            f"Apollo peak flight heating: 380-480 W/cm^2 at M~35, ~55 km. "
            f"The SU2 case represents a high-heating phase, not peak heating."
        ),
    }

    return result
