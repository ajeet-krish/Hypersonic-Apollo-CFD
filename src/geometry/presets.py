"""Preset blunt body configurations for common vehicles."""

import math

from .config import BluntBodyConfig


def apollo_cm() -> BluntBodyConfig:
    """Apollo Command Module blunt body geometry.

    Source: NASA TN D-6028 (1970), NASA TN D-4185 (1967).
    Spherical heat shield (R=4.694m) + 33-deg conical afterbody.
    Max diameter 3.91m (radius 1.955m), base radius 1.5m.
    Junction fillet R=0.196m, base fillet R=0.231m.
    Total axial length: 133.5 in = 3.391m (NASA TN D-6028).

    Returns:
        BluntBodyConfig matching Apollo CM dimensions.
    """
    return BluntBodyConfig(
        R_shield=4.694,
        R_fillet=0.196,
        cone_half_angle=33.0,
        max_radius=1.955,
        base_radius=1.5,
        base_fillet_radius=0.231,
        body_length=3.391,  # NASA TN D-6028: 133.5 inches
        num_points=600,
    )


def generic(
    R_shield: float = 0.1,
    cone_half_angle: float = 45.0,
    base_radius: float = 0.5,
    num_points: int = 300,
) -> BluntBodyConfig:
    """Generic spherically-blunted cone for parametric studies.

    Args:
        R_shield: Nose sphere radius (m).
        cone_half_angle: Cone half-angle (degrees).
        base_radius: Base radius (m).
        num_points: Number of contour points.

    Returns:
        BluntBodyConfig with specified geometry.
    """
    max_r = R_shield * (1.0 - math.cos(math.radians(cone_half_angle)))
    return BluntBodyConfig(
        R_shield=R_shield,
        R_fillet=0.0,
        cone_half_angle=cone_half_angle,
        max_radius=max_r,
        base_radius=base_radius,
        num_points=num_points,
    )
