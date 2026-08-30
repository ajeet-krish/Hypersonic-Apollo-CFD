"""Preset blunt body configurations for common vehicles."""
from .config import BluntBodyConfig


def generic(
    R_nose: float = 0.1,
    half_angle: float = 45.0,
    base_radius: float = 0.5,
) -> BluntBodyConfig:
    """Generic spherically-blunted cone for parametric studies.

    Args:
        R_nose: Nose sphere radius (m)
        half_angle: Cone half-angle (degrees)
        base_radius: Base radius (m)

    Returns:
        BluntBodyConfig with specified geometry.
    """
    return BluntBodyConfig(
        R_nose=R_nose,
        half_angle=half_angle,
        base_radius=base_radius,
        num_points=300,
    )


def apollo_cm() -> BluntBodyConfig:
    """Apollo Command Module blunt body geometry.

    Source: Graves & Witte (1963). The 30 km altitude case used here is
    below the peak heating altitude (~55-60 km) but represents a
    high-heating phase of the trajectory where stagnation heating
    rates are significant and representative for code validation.

    Returns:
        BluntBodyConfig matching Apollo CM dimensions.
    """
    return BluntBodyConfig(
        R_nose=0.196,
        half_angle=50.0,
        base_radius=1.955,
        num_points=400,
    )
