"""Normal shock relations (Rankine-Hugoniot)."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ShockResult:
    """Result from normal shock calculation.

    Attributes:
        M2: Downstream Mach number
        p_ratio: Static pressure ratio (p2/p1)
        T_ratio: Static temperature ratio (T2/T1)
        rho_ratio: Density ratio (rho2/rho1)
        p0_ratio: Total pressure ratio (p02/p01)
    """
    M2: float
    p_ratio: float
    T_ratio: float
    rho_ratio: float
    p0_ratio: float


def normal_shock(M1: float, gamma: float = 1.4) -> ShockResult:
    """Normal shock relations (Rankine-Hugoniot).

    Args:
        M1: Upstream Mach number
        gamma: Ratio of specific heats

    Returns:
        ShockResult with downstream properties

    Raises:
        ValueError: If M1 <= 1.0
    """
    if M1 <= 1.0:
        raise ValueError(
            f"M1 must be > 1.0 for normal shock, got {M1}"
        )

    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    # Static density ratio: rho2/rho1
    rho_ratio = gp1 * M1**2 / (2.0 + gm1 * M1**2)

    # Static pressure ratio: p2/p1
    p_ratio = 1.0 + 2.0 * gamma / gp1 * (M1**2 - 1.0)

    # Static temperature ratio: T2/T1
    T_ratio = p_ratio / rho_ratio

    # Downstream Mach number
    M2_sq = (2.0 + gm1 * M1**2) / (2.0 * gamma * M1**2 - gm1)
    M2 = math.sqrt(M2_sq)

    # Total pressure ratio: p02/p01
    p0_ratio = (
        (gp1 * M1**2 / (2.0 + gm1 * M1**2)) ** (gamma / gm1)
        * (gp1 / (2.0 * gamma * M1**2 - gm1)) ** (1.0 / gm1)
    )

    return ShockResult(
        M2=M2,
        p_ratio=p_ratio,
        T_ratio=T_ratio,
        rho_ratio=rho_ratio,
        p0_ratio=p0_ratio,
    )
