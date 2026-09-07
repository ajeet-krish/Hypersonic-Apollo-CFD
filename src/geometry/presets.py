"""Preset blunt body configurations for common vehicles."""

import math

from .config import BluntBodyConfig


def apollo_cm() -> BluntBodyConfig:
    """Apollo Command Module blunt body geometry.

    Source: DXF-verified dimensions from geometry/apollo_2d.dxf.
    Concave spherical heat shield (R=4.694m) + 33-deg conical afterbody.
    Max radius 1.924m at fillet-cone junction (DXF point (0.661, 1.924)).
    Base radius 0.219m at cone-base junction (DXF point (3.287, 0.219)).
    Shoulder fillet R=0.196m, base fillet R=0.231m.
    Total axial length: 3.391m.

    Sphere center is at (R, 0) = (4.694, 0) on the axis, ahead of the nose.
    The heat shield is a concave spherical dish curving inward from the nose.

    Returns:
        BluntBodyConfig matching Apollo CM dimensions.
    """
    return BluntBodyConfig(
        R_shield=4.694,
        R_fillet=0.196,
        cone_half_angle=33.0,
        max_radius=1.924,
        base_radius=0.219,
        base_fillet_radius=0.231,
        body_length=3.391,
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
