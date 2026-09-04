"""Blunt body geometry configuration.

Sphere + optional toroidal fillet + conical frustum body shape for
hypersonic aerothermodynamics analysis.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BluntBodyConfig:
    """Spherically-blunted cone with optional toroidal shoulder fillet.

    The body profile is defined by three sections:
      1. Sphere: x(phi) = R*sin(phi), r(phi) = R*(1-cos(phi))
      2. Toroidal fillet (optional): smooth blend from sphere to cone
      3. Cone: straight frustum to base

    Attributes:
        R_shield: Heat shield sphere radius (m).
        R_fillet: Toroidal shoulder fillet radius (m). 0 = no fillet.
        cone_half_angle: Cone half-angle (degrees).
        max_radius: Maximum body radius (m) at the shoulder.
        base_radius: Base (aft) radius (m).
        body_length: Total body length (m). 0 = auto-compute from geometry.
        num_points: Number of contour points.
    """

    R_shield: float = 0.196  # m
    R_fillet: float = 0.0  # m, 0 = no fillet
    cone_half_angle: float = 50.0  # degrees
    max_radius: float = 0.196  # m
    base_radius: float = 1.955  # m
    body_length: float = 0.0  # 0 = auto-compute
    num_points: int = 400

    # ------------------------------------------------------------------
    # Backward-compatible aliases
    # ------------------------------------------------------------------
    @property
    def R_nose(self) -> float:
        """Nose sphere radius (m). Alias for R_shield."""
        return self.R_shield

    @property
    def half_angle(self) -> float:
        """Cone half-angle (degrees). Alias for cone_half_angle."""
        return self.cone_half_angle

    @property
    def half_angle_rad(self) -> float:
        """Cone half-angle in radians."""
        return math.radians(self.cone_half_angle)

    @property
    def computed_body_length(self) -> float:
        """Total body length from nose tip to base (m).

        If body_length > 0 the user-specified value is returned.
        Otherwise the length is computed from the junction geometry
        so that the cone reaches base_radius.
        """
        if self.body_length > 0:
            return self.body_length

        theta = self.half_angle_rad

        if self.R_fillet > 0:
            # Sphere-fillet junction angle
            cos_phi = (self.R_shield + self.R_fillet - self.max_radius) / (
                self.R_shield + self.R_fillet
            )
            phi_sf = math.acos(max(-1.0, min(1.0, cos_phi)))
            # Fillet center
            x_f = (self.R_shield - self.R_fillet) * math.sin(phi_sf)
            r_f = self.R_shield - (self.R_shield + self.R_fillet) * math.cos(phi_sf)
            # Fillet-cone junction (cone tangent point)
            x_tc = x_f + self.R_fillet * math.sin(theta)
            r_tc = r_f + self.R_fillet * math.cos(theta)
            return x_tc + abs(r_tc - self.base_radius) / math.tan(theta)
        else:
            phi_j = math.acos(1.0 - self.max_radius / self.R_shield)
            x_j = self.R_shield * math.sin(phi_j)
            return x_j + abs(self.max_radius - self.base_radius) / math.tan(theta)

    @property
    def junction_x(self) -> float:
        """x at sphere-fillet or sphere-cone junction (m)."""
        if self.R_fillet > 0:
            cos_phi = (self.R_shield + self.R_fillet - self.max_radius) / (
                self.R_shield + self.R_fillet
            )
            phi_sf = math.acos(max(-1.0, min(1.0, cos_phi)))
            return self.R_shield * math.sin(phi_sf)
        else:
            phi_j = math.acos(1.0 - self.max_radius / self.R_shield)
            return self.R_shield * math.sin(phi_j)

    @property
    def junction_r(self) -> float:
        """r at sphere-fillet or sphere-cone junction (m)."""
        if self.R_fillet > 0:
            cos_phi = (self.R_shield + self.R_fillet - self.max_radius) / (
                self.R_shield + self.R_fillet
            )
            phi_sf = math.acos(max(-1.0, min(1.0, cos_phi)))
            return self.R_shield * (1.0 - math.cos(phi_sf))
        else:
            return self.max_radius

    @classmethod
    def validate(cls, **kwargs) -> "BluntBodyConfig":
        """Create and validate a BluntBodyConfig.

        Args:
            **kwargs: Keyword arguments passed to the constructor.

        Returns:
            Validated BluntBodyConfig instance.

        Raises:
            ValueError: If any parameter is out of range.
        """
        R_shield = kwargs.get("R_shield", 0.196)
        R_fillet = kwargs.get("R_fillet", 0.0)
        max_radius = kwargs.get("max_radius", R_shield)
        base_radius = kwargs.get("base_radius", max_radius * 0.8)
        cone_half_angle = kwargs.get("cone_half_angle", 50.0)
        num_points = kwargs.get("num_points", 400)

        if R_shield <= 0:
            raise ValueError(f"R_shield must be > 0, got {R_shield}")
        if R_fillet < 0:
            raise ValueError(f"R_fillet must be >= 0, got {R_fillet}")
        if cone_half_angle <= 0 or cone_half_angle >= 90:
            raise ValueError(
                f"cone_half_angle must be in (0, 90), got {cone_half_angle}"
            )
        if max_radius <= 0:
            raise ValueError(f"max_radius must be > 0, got {max_radius}")
        if base_radius <= 0:
            raise ValueError(
                f"base_radius must be > 0, got {base_radius}"
            )
        if num_points < 10:
            raise ValueError(f"num_points must be >= 10, got {num_points}")
        if R_fillet > 0:
            cos_phi = (R_shield + R_fillet - max_radius) / (R_shield + R_fillet)
            if cos_phi < -1 or cos_phi > 1:
                raise ValueError("Invalid geometry: fillet cannot reach max_radius")
        return cls(**kwargs)
