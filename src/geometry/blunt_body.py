"""Blunt body contour generation: sphere + optional toroidal fillet + cone + base fillet.

The contour is axisymmetric (x, r) where x is the axial coordinate and r
is the radial distance from the symmetry axis.  Four body sections:
  1. Sphere:        parametric arc from nose tip to sphere-fillet junction
  2. Toroidal fillet: circular arc blending sphere to cone (optional)
  3. Cone:          straight frustum from fillet-cone junction to base
  4. Base fillet:   circular arc at base edge (optional)
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
    """Sphere-cone contour (no shoulder fillet), with optional base fillet.

    The heat shield is a CONCAVE spherical dish (sphere center ahead of nose
    on the axis). The cone tapers inward from junction to base.

    Sphere arc: phi from 0 (nose tip) to phi_j (junction).
    Cone:       linear from junction to base.
    Base fillet: circular arc at base edge (optional).
    C1 continuity at the junction is guaranteed by construction.
    """
    R = config.R_shield
    max_r = config.max_radius
    theta = config.half_angle_rad

    # Sphere-cone junction angle
    # For concave sphere (center at (R, 0)): r = R*sin(phi) = max_r
    phi_j = math.asin(max_r / R)

    # Cone end
    L = config.computed_body_length

    # Point allocation
    n_sphere = int(config.num_points * 0.4)
    n_cone = config.num_points - n_sphere

    # If base fillet, reserve points for it
    Rbf = config.base_fillet_radius
    if Rbf > 0:
        n_cone = int(config.num_points * 0.45)
        n_base_fillet = config.num_points - n_sphere - n_cone
    else:
        n_base_fillet = 0

    # Sphere arc: CONCAVE heat shield
    # Sphere center is at (R, 0) on the axis, ahead of the nose.
    # The surface curves INWARD from the nose toward the axis.
    # x = R * (1 - cos(phi)), r = R * sin(phi)
    phi = np.linspace(0, phi_j, n_sphere)
    x_sphere = R * (1.0 - np.cos(phi))
    r_sphere = R * np.sin(phi)

    # Cone: from junction to base (tapers inward)
    x_j = x_sphere[-1]
    r_j = r_sphere[-1]

    # Base fillet is concave: center near axis, tangent to cone and base face
    if Rbf > 0:
        bf_center_x = L - Rbf
        bf_center_r = config.base_radius - Rbf * math.cos(theta)
        x_cone_end = bf_center_x + Rbf * math.sin(theta)
    else:
        x_cone_end = L

    x_cone = np.linspace(x_j, x_cone_end, n_cone)
    r_cone = r_j + (config.base_radius - r_j) * (x_cone - x_j) / (x_cone_end - x_j)

    # Base fillet: concave arc from cone tangent point to base face
    if Rbf > 0 and n_base_fillet > 0:
        bf_center_x = L - Rbf
        bf_center_r = config.base_radius - Rbf * math.cos(theta)
        # Arc from cone-tangent point (alpha = pi/2 - theta) to base point (alpha = 0)
        alpha = np.linspace(math.pi / 2 - theta, 0, n_base_fillet)
        x_base_fillet = bf_center_x + Rbf * np.cos(alpha)
        r_base_fillet = bf_center_r + Rbf * np.sin(alpha)

        x = np.concatenate([x_sphere, x_cone[1:], x_base_fillet[1:]])
        r = np.concatenate([r_sphere, r_cone[1:], r_base_fillet[1:]])
    else:
        # Skip first cone point (duplicate of junction)
        x = np.concatenate([x_sphere, x_cone[1:]])
        r = np.concatenate([r_sphere, r_cone[1:]])

    return x, r


def _contour_with_fillet(config: BluntBodyConfig) -> tuple[np.ndarray, np.ndarray]:
    """Sphere + toroidal fillet + cone + base fillet contour.

    The heat shield is a CONCAVE spherical dish (sphere center ahead of nose).
    The fillet blends the sphere to the cone. The cone tapers inward.
    The base fillet rounds the base edge.

    Sphere arc: phi from 0 (nose tip) to phi_sf (sphere-fillet junction).
    Fillet arc: circular arc from sphere tangent to cone tangent.
    Cone:       linear from fillet-cone junction to base.
    Base fillet: circular arc at base edge (optional).
    """
    R = config.R_shield
    Rf = config.R_fillet
    theta = config.half_angle_rad
    max_r = config.max_radius

    # Sphere-fillet junction angle
    # Internal tangency for concave sphere: fillet center at distance (R - Rf)
    # from sphere center.  The fillet-cone junction reaches max_r:
    #   r_f + Rf*cos(theta) = max_r, where r_f = (R - Rf)*sin(phi_sf)
    sin_phi_sf = (max_r - Rf * math.cos(theta)) / (R - Rf)
    phi_sf = math.asin(max(-1.0, min(1.0, sin_phi_sf)))

    # Fillet center: internally tangent to concave sphere
    x_f = R - (R - Rf) * math.cos(phi_sf)
    r_f = (R - Rf) * math.sin(phi_sf)

    # Fillet-cone junction (cone tangent point)
    x_tc = x_f + Rf * math.sin(theta)
    r_tc = r_f + Rf * math.cos(theta)

    # Body length
    L = config.computed_body_length

    # Point allocation
    n_sphere = int(config.num_points * 0.25)
    n_fillet = int(config.num_points * 0.15)
    n_cone = int(config.num_points * 0.45)

    # Base fillet
    Rbf = config.base_fillet_radius
    if Rbf > 0:
        n_base_fillet = config.num_points - n_sphere - n_fillet - n_cone
    else:
        n_base_fillet = 0
        n_cone = config.num_points - n_sphere - n_fillet

    # Sphere arc: CONCAVE heat shield
    phi = np.linspace(0, phi_sf, n_sphere)
    x_sphere = R * (1.0 - np.cos(phi))
    r_sphere = R * np.sin(phi)

    # Fillet arc: from sphere tangent to cone tangent
    # At sphere tangent, alpha = phi_sf - pi/2; at cone tangent, alpha = theta
    alpha = np.linspace(phi_sf - math.pi / 2, theta, n_fillet)
    x_fillet = x_f + Rf * np.sin(alpha)
    r_fillet_curve = r_f + Rf * np.cos(alpha)

    # Cone: from (x_tc, r_tc) to base (tapers inward)
    # Base fillet is concave: center near axis, tangent to cone and base face
    if Rbf > 0:
        bf_center_x = L - Rbf
        bf_center_r = config.base_radius - Rbf * math.cos(theta)
        x_cone_end = bf_center_x + Rbf * math.sin(theta)
    else:
        x_cone_end = L

    x_cone = np.linspace(x_tc, x_cone_end, n_cone)
    r_cone = r_tc + (config.base_radius - r_tc) * (x_cone - x_tc) / (x_cone_end - x_tc)

    # Base fillet: concave arc from cone tangent point to base face
    if Rbf > 0 and n_base_fillet > 0:
        bf_center_x = L - Rbf
        bf_center_r = config.base_radius - Rbf * math.cos(theta)
        # Arc from cone-tangent point (beta = pi/2 - theta) to base point (beta = 0)
        beta = np.linspace(math.pi / 2 - theta, 0, n_base_fillet)
        x_base_fillet = bf_center_x + Rbf * np.cos(beta)
        r_base_fillet = bf_center_r + Rbf * np.sin(beta)

        x = np.concatenate([x_sphere, x_fillet[1:], x_cone[1:], x_base_fillet[1:]])
        r = np.concatenate([r_sphere, r_fillet_curve[1:], r_cone[1:], r_base_fillet[1:]])
    else:
        x = np.concatenate([x_sphere, x_fillet[1:], x_cone[1:]])
        r = np.concatenate([r_sphere, r_fillet_curve[1:], r_cone[1:]])

    return x, r
