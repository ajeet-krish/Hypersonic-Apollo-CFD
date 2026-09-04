"""Preset blunt body configurations for common vehicles."""

import math

from .config import BluntBodyConfig


def apollo_cm() -> BluntBodyConfig:
    """Apollo Command Module blunt body geometry.

    Source: Fusion 360 DXF export (apollo_2d.dxf) + NASA TN D-6028.
    Concave spherical heat shield + 33-deg conical afterbody.
    Total axial length: 3.391m (133.5 in, NASA TN D-6028).

    DXF geometry:
        - Sphere: Center (5.2366, 0), R=5.2366, arc 160.8-199.2 deg
        - Cone: 33-deg half-angle, from junction to base
        - Junction fillet: R=0.196m
        - Base fillet: R=0.231m

    Returns:
        BluntBodyConfig matching Apollo CM dimensions.
    """
    return BluntBodyConfig(
        R_shield=5.2366,
        R_fillet=0.196,
        cone_half_angle=33.0,
        max_radius=1.8232,
        base_radius=0.194,
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
