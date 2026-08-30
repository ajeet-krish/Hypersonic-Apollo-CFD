"""Blunt body contour generation: spherical nose + conical frustum."""
import math

import numpy as np

from .config import BluntBodyConfig


def generate_contour(config: BluntBodyConfig) -> tuple[np.ndarray, np.ndarray]:
    """Generate (x, r) contour points for a spherically-blunted cone.

    The contour consists of two sections:
      1. Spherical nose: x(phi) = R_nose*sin(phi), r(phi) = R_nose*(1-cos(phi))
      2. Conical frustum: linear from junction to base

    C1 continuity at the junction is guaranteed by construction: the sphere
    tangent at half_angle matches the cone generator line.

    Args:
        config: Blunt body geometry parameters

    Returns:
        x: axial coordinates (m), shape (num_points,)
        r: radial coordinates (m), shape (num_points,)
    """
    n_total = config.num_points
    theta = config.half_angle_rad

    # Split points between sphere and cone based on arc-length fraction
    sphere_arc = config.R_nose * theta  # arc length of spherical nose
    cone_length = (config.base_radius - config.junction_r) / math.tan(theta)
    total_arc = sphere_arc + cone_length
    n_sphere = max(int(n_total * sphere_arc / total_arc), 10)
    n_cone = n_total - n_sphere

    x_sphere, r_sphere = _sphere_nose(config.R_nose, theta, n_sphere)
    x_cone, r_cone = _cone_frustum(
        config.junction_x, config.junction_r,
        config.computed_body_length, config.base_radius,
        n_cone,
    )

    # Concatenate (skip duplicate junction point in cone section)
    x = np.concatenate([x_sphere, x_cone[1:]])
    r = np.concatenate([r_sphere, r_cone[1:]])

    return x, r


def _sphere_nose(
    R_nose: float,
    half_angle: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate spherical nose cap contour.

    Parametric equations:
        x(phi) = R_nose * sin(phi)
        r(phi) = R_nose * (1 - cos(phi))
    where phi goes from 0 (nose tip) to half_angle (junction).

    Args:
        R_nose: Nose sphere radius (m)
        half_angle: Cone half-angle (radians)
        n_points: Number of points

    Returns:
        (x, r) coordinate arrays
    """
    phi = np.linspace(0.0, half_angle, n_points)
    x = R_nose * np.sin(phi)
    r = R_nose * (1.0 - np.cos(phi))
    return x, r


def _cone_frustum(
    x_start: float,
    r_start: float,
    x_end: float,
    r_end: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate conical frustum contour (linear interpolation).

    Args:
        x_start: Axial start coordinate (junction)
        r_start: Radial start coordinate (junction)
        x_end: Axial end coordinate (base)
        r_end: Radial end coordinate (base)
        n_points: Number of points

    Returns:
        (x, r) coordinate arrays
    """
    x = np.linspace(x_start, x_end, n_points)
    r = np.linspace(r_start, r_end, n_points)
    return x, r
