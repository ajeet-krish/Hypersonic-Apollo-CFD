"""Blunt body geometry configuration."""
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BluntBodyConfig:
    """Spherically-blunted cone configuration.

    Attributes:
        R_nose: Nose sphere radius (m)
        half_angle: Cone half-angle (degrees)
        base_radius: Base radius of the cone (m)
        body_length: Total body length (m, 0 = auto-compute from base_radius)
        num_points: Number of contour points
    """
    R_nose: float = 0.196        # m
    half_angle: float = 50.0     # degrees
    base_radius: float = 1.955   # m
    body_length: float = 0.0     # 0 = auto-compute
    num_points: int = 300

    @property
    def half_angle_rad(self) -> float:
        """Cone half-angle in radians."""
        return math.radians(self.half_angle)

    @property
    def junction_x(self) -> float:
        """Axial coordinate at sphere-cone junction (m).

        x_junction = R_nose * sin(theta)
        """
        return self.R_nose * math.sin(self.half_angle_rad)

    @property
    def junction_r(self) -> float:
        """Radial coordinate at sphere-cone junction (m).

        r_junction = R_nose * (1 - cos(theta))
        """
        return self.R_nose * (1.0 - math.cos(self.half_angle_rad))

    @property
    def computed_body_length(self) -> float:
        """Total body length from nose tip to base (m).

        L = (base_radius - junction_r) / tan(theta) + junction_x
        """
        return (self.base_radius - self.junction_r) / math.tan(self.half_angle_rad) + self.junction_x

    @property
    def computed_base_radius(self) -> float:
        """Base radius from body_length (if body_length is set).

        R_base = junction_r + (L - junction_x) * tan(theta)
        """
        if self.body_length > 0:
            return self.junction_r + (self.body_length - self.junction_x) * math.tan(self.half_angle_rad)
        return self.base_radius

    @classmethod
    def validate(cls, **kwargs: Any) -> "BluntBodyConfig":
        """Create and validate BluntBodyConfig.

        Args:
            **kwargs: Keyword arguments passed to BluntBodyConfig constructor.

        Returns:
            Validated BluntBodyConfig instance.

        Raises:
            ValueError: If any parameter is out of range.
        """
        config = cls(**kwargs)
        if config.R_nose <= 0:
            raise ValueError(
                f"R_nose must be > 0, got {config.R_nose}"
            )
        if config.half_angle <= 0 or config.half_angle >= 85:
            raise ValueError(
                f"half_angle must be > 0 and < 85 degrees, got {config.half_angle}"
            )
        if config.base_radius <= config.junction_r:
            raise ValueError(
                f"base_radius must be > junction_r ({config.junction_r:.6f}), "
                f"got {config.base_radius}"
            )
        if config.num_points < 10:
            raise ValueError(
                f"num_points must be >= 10, got {config.num_points}"
            )
        return config
