# Apollo Command Module: Hypersonic Aerothermodynamics Case

## Overview

The Apollo Command Module (CM) is the headline validation case for this
hypersonic blunt body CFD project. The 50-degree sphere-cone geometry
with a 0.196 m nose radius and 1.955 m base radius represents one of
the most extensively studied reentry vehicles in aerospace history.

## Geometry

| Parameter | Value | Source |
|-----------|-------|--------|
| Nose sphere radius (R_nose) | 0.196 m | Graves & Witte (1963) |
| Cone half-angle | 50.0 deg | Apollo CM specification |
| Base radius | 1.955 m | Apollo CM specification |
| Heat shield material | AVCOAT 5026-39 | NASA specs |

The geometry is a spherically-blunted cone: a sphere of radius R_nose
tangent to a 50-degree cone, extending to a base radius of 1.955 m.
This produces a body length of approximately 3.25 m.

## Flight Conditions

| Parameter | Value |
|-----------|-------|
| Freestream Mach number | 12.0 |
| Altitude | 30 km |
| Freestream temperature | 226.51 K (US Std Atm) |
| Freestream pressure | 1197.0 Pa |
| Freestream density | 0.01841 kg/m^3 |
| Wall temperature (isothermal) | 300 K |

### Why M=12 at 30 km?

The project uses perfect-gas (calorically perfect air, gamma=1.4)
CFD with the SU2 RANS solver. This limits the applicable regime:

- **Peak heating** for Apollo CM occurred at ~55 km altitude, M~35-40
  (velocity ~10.8 km/s), where real-gas effects (dissociation, ionization)
  dominate and perfect-gas models break down.
- **M=12 at 30 km** represents a high-heating phase later in the trajectory,
  where the atmosphere is denser and stagnation heating is still significant
  (~100-300 kW/m^2), but the flow is still within the perfect-gas regime
  (stagnation temperature ~1700 K, below significant dissociation onset).
- This is a **representative validation case**, not a direct match to peak
  flight heating.

## Convergence Strategy

M=12 is difficult to converge with SU2 RANS. The mach-ramp strategy is used:

1. **Stage 1**: First-order RANS at M=5 (5000 iterations, CFL=0.01)
   - Establishes the bow shock structure
   - Provides a well-conditioned restart file
2. **Stage 2**: Second-order RANS at M=12 (20000 iterations, CFL=0.01-0.05)
   - Restarts from the M=5 solution
   - Refines to second-order spatial accuracy
   - SA turbulence model with freestream initialization

## Analytical Predictions

At M=12, 30 km altitude:

- **Sutton-Graves stagnation heat flux**: ~150-250 kW/m^2 (depends on correlation constant)
- **Billig shock standoff**: delta/R ~ 0.148, delta ~ 29 mm
- **Modified Newtonian Cp_max**: ~1.94
- **Normal shock**: M2 ~ 0.395, p2/p1 ~ 152, T2/T1 ~ 6.3

## Flight Data Comparison

Published Apollo CM reentry data from NASA reports:

| Mission | Peak Heat Flux (W/cm^2) | Altitude (km) | Velocity (km/s) | Source |
|---------|------------------------|---------------|-----------------|--------|
| Apollo 4 (AS-501) | 430-480 | ~53 | ~10.8 | NASA TN D-4371 |
| Apollo 6 (AS-502) | ~460 | ~55 | ~10.7 | NASA TN D-5399 |
| Apollo 11 | ~380 | ~60 | ~10.6 | NASA SP-5002 |
| Apollo 13 | ~400 | ~57 | ~10.7 | NASA trajectory reconstruction |

**Important caveat**: These flight data are at PEAK HEATING conditions
(M~35, altitude ~55 km). The CFD case at M=12, 30 km is at a
high-heating phase but NOT at peak heating. The SU2 stagnation heat
flux will be significantly lower than the flight peak values, which is
expected and physically correct.

## Real-Gas Caveats

At M=12 and 30 km altitude, the stagnation temperature is approximately:

    T_stag = T_inf * (1 + (gamma-1)/2 * M^2)
           = 226.5 * (1 + 0.2 * 144)
           = 226.5 * 29.8
           = 6750 K (perfect gas)

This is well above the dissociation threshold (~2000 K for O2, ~4000 K
for N2). The perfect-gas model overpredicts the stagnation temperature
and consequently overpredicts the stagnation heating rate. A real-gas
correction factor is applied as a post-processing step using NASA Glenn
polynomial fits (gamma_correction_factor).

At the actual peak heating conditions (M~35), the stagnation temperature
exceeds 10,000 K and ionization becomes significant. These conditions
require nonequilibrium thermochemical models beyond the scope of this
project.

## Results

Results are saved to:
- `output/apollo-cm/apollo_results.json`: Combined results
- `output/apollo-cm/postprocess/postprocess.json`: Post-processed data
- `output/apollo-cm/validation/validation.json`: Validation report
- `docs/assets/images/apollo-cm/flight_data_comparison.png`: Comparison plot

## References

1. Graves, R. A. and Witte, D. W. (1963), "Flight-Test Heat-Transfer
   Data for a 10 deg Sphere-Cone with Various Nose Radii," NASA TN D-2142.
2. NASA TN D-4371 (1968), "A Review of the Apollo 4 Flight Results."
3. NASA TN D-5399 (1970), "A Review of the Apollo 6 Flight Results."
4. Sutton, K. and Graves, R. A. (1985), "A General Stagnation-Point
   Convective-Heating Equation for Arbitrary Gas Mixtures," NASA TR R-376.
5. Billig, F. S. (1967), "Shock-Wave Shapes Around Unswept- and
   Swept-Nose Bodies," J. Spacecraft and Rockets, 4(6), 822-823.
