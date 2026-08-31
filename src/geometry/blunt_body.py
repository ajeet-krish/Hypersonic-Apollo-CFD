"""Blunt body contour generation: sphere + optional toroidal fillet + cone.

The contour is axisymmetric (x, r) where x is the axial coordinate and r
is the radial distance from the symmetry axis.  Three body sections:
  1. Sphere:        parametric arc from nose tip to sphere-fillet junction
  2. Toroidal fillet: circular arc blending sphere to cone (optional)
  3. Cone:          straight frustum from fillet-cone junction to base
"""

import math

import numpy as np

from .config import BluntBodyConfig


def generate_contour(config: BluntBodyConfig) -> tuple[np.ndarray, np.ndarray]:
    """Generate (x, r) contour points for a blunt body.

    Dispatches to the appropriate internal generator based on whether a
    fillet is present.

    Args:
        config: Blunt body geometry parameters.

    Returns:
        x: axial coordinates (m), shape (num_points,)
        r: radial coordinates (m), shape (num_points,)
    """
    if config.R_fillet > 0:
        return _contour_with_fillet(config)
    return _contour_simple(config)


def _contour_simple(config: BluntBodyConfig) -> tuple[np.ndarray, np.ndarray]:
    """Sphere-cone contour (no fillet).

    Sphere arc: phi from 0 (nose tip) to phi_j (junction).
    Cone:       linear from junction to base.
    C1 continuity at the junction is guaranteed by construction.
    """
    R = config.R_shield
    max_r = config.max_radius

    # Sphere-cone junction angle
    phi_j = math.acos(1.0 - max_r / R)
    x_j = R * math.sin(phi_j)
    r_j = max_r

    # Cone end
    L = config.computed_body_length

    n_sphere = int(config.num_points * 0.4)
    n_cone = config.num_points - n_sphere

    # Sphere arc: phi from 0 to phi_j
    phi = np.linspace(0, phi_j, n_sphere)
    x_sphere = R * np.sin(phi)
    r_sphere = R * (1.0 - np.cos(phi))

    # Cone: from (x_j, r_j) to (L, base_radius)
    x_cone = np.linspace(x_j, L, n_cone)
    r_cone = r_j + (config.base_radius - r_j) * (x_cone - x_j) / (L - x_j)

    # Skip first cone point (duplicate of junction)
    x = np.concatenate([x_sphere, x_cone[1:]])
    r = np.concatenate([r_sphere, r_cone[1:]])

    return x, r


def _contour_with_fillet(config: BluntBodyConfig) -> tuple[np.ndarray, np.ndarray]:
    """Sphere + toroidal fillet + cone contour.

    The fillet is a circular arc of radius R_fillet whose center lies
    on the line tangent to both the sphere and the cone at their
    respective contact points.

    Fillet arc angles (measured from vertical at fillet center):
        alpha goes from phi_sf (sphere tangent) to theta (cone tangent).
    """
    R = config.R_shield
    Rf = config.R_fillet
    theta = config.half_angle_rad
    max_r = config.max_radius

    # Sphere-fillet junction angle
    cos_phi = (R + Rf - max_r) / (R + Rf)
    phi_sf = math.acos(max(-1.0, min(1.0, cos_phi)))

    # Fillet center
    x_f = (R - Rf) * math.sin(phi_sf)
    r_f = R - (R + Rf) * math.cos(phi_sf)

    # Fillet-cone junction (cone tangent point)
    x_tc = x_f + Rf * math.sin(theta)
    r_tc = r_f + Rf * math.cos(theta)

    # Body length
    L = config.computed_body_length

    # Point allocation
    n_sphere = int(config.num_points * 0.3)
    n_fillet = int(config.num_points * 0.2)
    n_cone = config.num_points - n_sphere - n_fillet

    # Sphere arc: phi from 0 to phi_sf
    phi = np.linspace(0, phi_sf, n_sphere)
    x_sphere = R * np.sin(phi)
    r_sphere = R * (1.0 - np.cos(phi))

    # Fillet arc: alpha from phi_sf down to theta
    alpha = np.linspace(phi_sf, theta, n_fillet)
    x_fillet = x_f + Rf * np.sin(alpha)
    r_fillet_curve = r_f + Rf * np.cos(alpha)

    # Cone: from (x_tc, r_tc) to (L, base_radius)
    x_cone = np.linspace(x_tc, L, n_cone)
    r_cone = r_tc + (config.base_radius - r_tc) * (x_cone - x_tc) / (L - x_tc)

    # Concatenate (skip duplicate junction points)
    x = np.concatenate([x_sphere, x_fillet[1:], x_cone[1:]])
    r = np.concatenate([r_sphere, r_fillet_curve[1:], r_cone[1:]])

    return x, r
