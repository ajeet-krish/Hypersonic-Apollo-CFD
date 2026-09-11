"""1D implicit thermal solver for heat shield wall conduction.

Solves the 1D heat equation with temperature-dependent properties:

    rho(T) * cp(T) * dT/dt = d/dz(k(T) * dT/dz)

Uses implicit backward Euler for unconditional stability.
Thomas algorithm (tridiagonal solver) for efficiency.
"""
import numpy as np

from .config import ThermalConfig
from .materials import get_material
from .results import ThermalResult1D


class ThermalSolver1D:
    """1D thermal solver for heat shield wall conduction.

    Solves the 1D heat equation with temperature-dependent thermal
    conductivity and specific heat using implicit backward Euler.

    Grid: n_cells+1 nodes through wall thickness, uniform spacing.
    Hot side BC: convective heat flux (q_flux = -k * dT/dz at z=0).
    Cold side BC: fixed temperature or radiation (q_rad = eps * sigma * T^4).
    """

    def __init__(self, config: ThermalConfig) -> None:
        """Initialize solver with thermal configuration.

        Args:
            config: Thermal analysis configuration.
        """
        self.config = config
        self.material = get_material(config.material)
        self.z = np.linspace(0.0, config.wall_thickness, config.n_cells + 1)
        self.T = np.full(config.n_cells + 1, config.cold_wall_temp)
        self.t = 0.0

    def solve(self) -> ThermalResult1D:
        """Run thermal simulation from t=0 to t_end.

        Steps through time using implicit backward Euler with the Thomas
        algorithm for the tridiagonal system.

        Returns:
            ThermalResult1D with temperature history and profiles.
        """
        config = self.config
        n_cells = config.n_cells
        n_nodes = n_cells + 1
        dt = config.dt
        t_end = config.t_end
        dz = config.wall_thickness / n_cells

        # Initialize
        T = self.T.copy()
        T_initial = T.copy()
        t = 0.0

        # Storage for history (pre-allocate for efficiency)
        n_steps = int(t_end / dt) + 1
        T_history = np.zeros((n_steps, n_nodes))
        t_history = np.zeros(n_steps)
        T_history[0] = T.copy()
        t_history[0] = t

        step = 1
        while t < t_end - 1e-12:
            # Ensure we don't overshoot
            dt_actual = min(dt, t_end - t)

            # Build and solve tridiagonal system
            a, b, c, d = self._build_tridiagonal(T, dt_actual)
            T_new = _thomas_solve(a, b, c, d)

            # Apply boundary conditions
            T_new[0] = T[1]  # placeholder, overwritten by BC
            T_new[-1] = T[-2]  # placeholder, overwritten by BC

            self._apply_bc_hot_to_vec(T_new, config.q_stagnation, dz)
            self._apply_bc_cold_to_vec(T_new)

            t += dt_actual
            T = T_new

            if step < n_steps:
                T_history[step] = T.copy()
                t_history[step] = t
            step += 1

        # Trim history arrays to actual steps
        T_history = T_history[:step]
        t_history = t_history[:step]

        # Compute integrated heat flux (trapezoidal rule)
        q_total = self._compute_total_heat_flux(T_history, t_history, dz)

        return ThermalResult1D(
            z=self.z,
            T_initial=T_initial,
            T_final=T.copy(),
            T_history=T_history,
            t_history=t_history,
            t_end=t,
            T_max_wall=float(T[0]),
            T_max_back=float(T[-1]),
            q_total=q_total,
            material=self.material.name,
            wall_thickness=config.wall_thickness,
        )

    def _build_tridiagonal(
        self,
        T: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build tridiagonal system for implicit backward Euler.

        The discretized equation at interior node i is:

            (rho*cp/dt) * T_i_new = (k/dz^2) * (T_{i-1} + T_{i+1} - 2*T_i)
                                     + (rho*cp/dt) * T_i_old

        Rearranged as: a_i * T_{i-1} + b_i * T_i + c_i * T_{i+1} = d_i

        Args:
            T: Current temperature profile (K).
            dt: Time step (s).

        Returns:
            (a_lower, b_main, c_upper, d_rhs) arrays of size n_nodes.
        """
        n_nodes = len(T)
        dz = self.config.wall_thickness / (n_nodes - 1)
        mat = self.material

        a = np.zeros(n_nodes)
        b = np.zeros(n_nodes)
        c = np.zeros(n_nodes)
        d = np.zeros(n_nodes)

        # Compute temperature-dependent properties at each node
        k = np.array([mat.k_at(Ti) for Ti in T])
        cp = np.array([mat.cp_at(Ti) for Ti in T])
        rho = mat.density

        for i in range(1, n_nodes - 1):
            # Average k at interfaces
            k_left = 0.5 * (k[i - 1] + k[i])
            k_right = 0.5 * (k[i] + k[i + 1])

            alpha = rho * cp[i] / dt

            a[i] = k_left / dz**2
            c[i] = k_right / dz**2
            b[i] = -(a[i] + c[i] + alpha)
            d[i] = -alpha * T[i]

        # Boundary conditions: Dirichlet placeholders
        # Hot side (i=0): will be overwritten by convective BC
        b[0] = 1.0
        c[0] = 0.0
        d[0] = T[0]

        # Cold side (i=n-1): will be overwritten by cold BC
        a[-1] = 0.0
        b[-1] = 1.0
        d[-1] = T[-1]

        return a, b, c, d

    def _apply_bc_hot_to_vec(
        self,
        T_new: np.ndarray,
        q_flux: float,
        dz: float,
    ) -> None:
        """Apply convective heat flux BC on hot side (z=0).

        Uses forward difference: q_flux = -k * (T_1 - T_0) / dz
        => T_0 = T_1 + q_flux * dz / k(T_0)

        Args:
            T_new: Temperature vector being solved.
            q_flux: Applied heat flux (W/m^2), positive into wall.
            dz: Grid spacing (m).
        """
        k0 = self.material.k_at(T_new[0])
        if k0 > 0:
            T_new[0] = T_new[1] + q_flux * dz / k0

    def _apply_bc_cold_to_vec(self, T_new: np.ndarray) -> None:
        """Apply cold wall BC (fixed temperature or radiation).

        Fixed temperature: T[-1] = T_cold.
        Radiation: q_rad = eps * sigma * (T[-1]^4 - T_env^4) = -k * dT/dz.
        Uses backward difference at the cold face.

        Args:
            T_new: Temperature vector being solved.
        """
        if not self.config.radiation:
            T_new[-1] = self.config.cold_wall_temp
            return

        # Radiation BC: q_rad = eps * sigma * (T_wall^4 - T_env^4)
        # This is nonlinear; use one Newton iteration for linearization
        sigma = 5.670374419e-8  # Stefan-Boltzmann constant
        eps = self.config.emissivity
        T_env = self.config.cold_wall_temp
        dz = self.config.wall_thickness / self.config.n_cells
        k_n = self.material.k_at(T_new[-1])

        T_wall = T_new[-1]
        # Linearized: q_rad ≈ q_rad_0 + dq/dT * (T - T_wall_0)
        # dq/dT = 4 * eps * sigma * T_wall^3
        dq_dt = 4.0 * eps * sigma * T_wall**3
        T_next = T_new[-2]

        # q_rad = -k * (T_wall - T_next) / dz
        # => T_wall = T_next - q_rad * dz / k
        # Using linearized form with Newton linearization:
        q_rad_0 = eps * sigma * (T_wall**4 - T_env**4)
        T_new[-1] = (k_n * T_next - (q_rad_0 - dq_dt * T_wall) * dz) / (k_n + dq_dt * dz)


    def _compute_total_heat_flux(
        self,
        T_history: np.ndarray,
        t_history: np.ndarray,
        dz: float,
    ) -> float:
        """Compute total heat input by integrating hot-face heat flux over time.

        Uses forward difference for the temperature gradient at z=0:
            q(t) = k(T_0) * (T_1 - T_0) / dz

        Then integrates over time using the trapezoidal rule.

        Args:
            T_history: Temperature history array (n_time x n_nodes).
            t_history: Time array (s).
            dz: Grid spacing (m).

        Returns:
            Total heat input (J/m^2).
        """
        mat = self.material
        n_steps = len(t_history)

        if n_steps < 2:
            return 0.0

        # Compute hot-face heat flux at each time step
        q_wall = np.zeros(n_steps)
        for step_idx in range(n_steps):
            T = T_history[step_idx]
            k0 = mat.k_at(T[0])
            q_wall[step_idx] = k0 * (T[1] - T[0]) / dz

        # Trapezoidal integration over time: integral(q_wall dt)
        q_total = float(np.trapezoid(q_wall, t_history))
        return abs(q_total)


def _thomas_solve(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """Solve tridiagonal system using Thomas algorithm.

    Solves: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]

    Args:
        a: Lower diagonal (size n, a[0] unused).
        b: Main diagonal (size n).
        c: Upper diagonal (size n, c[-1] unused).
        d: Right-hand side (size n).

    Returns:
        Solution vector x of size n.
    """
    n = len(b)
    # Forward elimination
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)

    c_prime[0] = c[0] / b[0] if abs(b[0]) > 1e-30 else 0.0
    d_prime[0] = d[0] / b[0] if abs(b[0]) > 1e-30 else 0.0

    for i in range(1, n):
        denom = b[i] - a[i] * c_prime[i - 1]
        if abs(denom) < 1e-30:
            denom = 1e-30
        c_prime[i] = c[i] / denom if i < n - 1 else 0.0
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

    # Back substitution
    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x
