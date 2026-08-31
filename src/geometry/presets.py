"""Preset blunt body configurations for common vehicles."""

import math

from .config import BluntBodyConfig


def apollo_cm() -> BluntBodyConfig:
    """Apollo Command Module blunt body geometry.

    Source: Graves & Witte (1963).  The geometry uses the sphere + toroidal
    fillet + cone model with the actual Apollo CM shoulder radius.

    Returns:
        BluntBodyConfig matching Apollo CM dimensions.
    """
    return BluntBodyConfig(
        R_shield=4.694,
        R_fillet=0.196,
        cone_half_angle=33.0,
        max_radius=1.955,
        base_radius=1.5,
        num_points=400,
    )


def generic(
    R_nose: float = 0.5,
    cone_half_angle: float = 30.0,
    base_radius: float = 0.3,
) -> BluntBodyConfig:
    """Generic spherically-blunted cone for parametric studies.

    Args:
        R_nose: Nose sphere radius (m).
        cone_half_angle: Cone half-angle (degrees).
        base_radius: Base radius (m).

    Returns:
        BluntBodyConfig with specified geometry.
    """
    max_r = R_nose * (1.0 - math.cos(math.radians(cone_half_angle)))
    if base_radius >= max_r:
        base_radius = max_r * 0.5
    return BluntBodyConfig(
        R_shield=R_nose,
        R_fillet=0.0,
        cone_half_angle=cone_half_angle,
        max_radius=max_r,
        base_radius=base_radius,
        num_points=300,
    )
