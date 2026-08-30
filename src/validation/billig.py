"""Billig shock standoff distance correlations.

References:
    Billig, F. S. (1967), "Shock-Wave Shapes Around Unswept- and
    Swept-Nose Bodies," Journal of Spacecraft and Rockets, 4(6), 822-823.
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BilligResult:
    """Result from Billig shock standoff calculation.

    Attributes:
        delta_over_R: Standoff distance / nose radius
        delta: Standoff distance (m)
        R_nose: Nose radius (m)
        M: Freestream Mach number
        formula: Correlation formula string
    """
    delta_over_R: float
    delta: float
    R_nose: float
    M: float
    formula: str


def billig_blunted_cone(R_nose: float, M: float) -> BilligResult:
    """Billig correlation for blunted cone shock standoff.

    delta/R = 0.143 * exp(3.24 / M^2)

    As M -> infinity, delta/R -> 0.143.

    Reference: Billig (1967).

    Args:
        R_nose: Nose sphere radius (m)
        M: Freestream Mach number

    Returns:
        BilligResult with standoff distance

    Raises:
        ValueError: If M <= 1.5
    """
    if M <= 1.5:
        raise ValueError(
            f"M must be > 1.5 for Billig correlation, got {M}"
        )

    delta_over_R = 0.143 * math.exp(3.24 / M**2)
    delta = delta_over_R * R_nose

    return BilligResult(
        delta_over_R=delta_over_R,
        delta=delta,
        R_nose=R_nose,
        M=M,
        formula="delta/R = 0.143 * exp(3.24 / M^2)",
    )


def billig_sphere(R_nose: float, M: float) -> BilligResult:
    """Billig correlation for sphere shock standoff.

    delta/R = 0.78 * exp(3.24 / M^2)

    As M -> infinity, delta/R -> 0.78.
    This is a SEPARATE correlation from the blunted-cone form.

    Reference: Billig (1967).

    Args:
        R_nose: Nose sphere radius (m)
        M: Freestream Mach number

    Returns:
        BilligResult with standoff distance

    Raises:
        ValueError: If M <= 1.5
    """
    if M <= 1.5:
        raise ValueError(
            f"M must be > 1.5 for Billig correlation, got {M}"
        )

    delta_over_R = 0.78 * math.exp(3.24 / M**2)
    delta = delta_over_R * R_nose

    return BilligResult(
        delta_over_R=delta_over_R,
        delta=delta,
        R_nose=R_nose,
        M=M,
        formula="delta/R = 0.78 * exp(3.24 / M^2)",
    )
