"""1D implicit thermal solver for heat shield wall conduction.

Solves the 1D heat equation with temperature-dependent properties:

    rho(T) * cp(T) * dT/dt = d/dz(k(T) * dT/dz)

Uses implicit backward Euler for unconditional stability.
Thomas algorithm (tridiagonal solver) for efficiency.

Optionally couples with a charring ablation model for pyrolysis
kinetics, density evolution, and surface recession.
"""
import numpy as np

from .config import AblationConfig, ThermalConfig
from .materials import get_material
from .results import AblationResult1D, ThermalResult1D

# Import AblationModel at module level for type checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ablation import AblationModel


class ThermalSolver1D:
    """1D thermal solver for heat shield wall conduction.

    Solves the 1D heat equation with temperature-dependent thermal
    conductivity and specific heat using implicit backward Euler.

    Grid: n_cells+1 nodes through wall thickness, uniform spacing.
    Hot side BC: convective heat flux (q_flux = -k * dT/dz at z=0).
    Cold side BC: fixed temperature or radiation (q_rad = eps * sigma * T^4).

    When ablation_config is provided, couples with the AblationModel
    for pyrolysis-driven density evolution and heat sink.
    """

    def __init__(
        self,
        config: ThermalConfig,
        ablation_config: AblationConfig | None = None,
    ) -> None:
        """Initialize solver with thermal configuration.

        Args:
            config: Thermal analysis configuration.
            ablation_config: Optional ablation configuration. When provided,
                the solver couples density evolution with temperature.
        """
        self.config = config
        self.material = get_material(config.material)
        self.z = np.linspace(0.0, config.wall_thickness, config.n_cells + 1)
        self.T = np.full(config.n_cells + 1, config.cold_wall_temp)
        self.t = 0.0

        # Ablation coupling
        self.ablation_config = ablation_config
        self.ablation = None
        self.rho: np.ndarray | None = None
        if ablation_config is not None:
            from .ablation import AblationModel
            self.ablation = AblationModel(self.material, ablation_config)
            self.rho = np.full(config.n_cells + 1, self.material.density)

    def solve(self) -> ThermalResult1D | AblationResult1D:
        """Run thermal simulation from t=0 to t_end.

        Steps through time using implicit backward Euler with the Thomas
        algorithm for the tridiagonal system. When ablation is enabled,
        couples density evolution and applies pyrolysis heat sink.

        Returns:
            ThermalResult1D or AblationResult1D depending on whether
            ablation is enabled.
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

        # Ablation state
        rho = self.rho.copy() if self.rho is not None else None
        rho_initial = rho.copy() if rho is not None else None

        # Storage for history (pre-allocate for efficiency)
        n_steps = int(t_end / dt) + 1
        T_history = np.zeros((n_steps, n_nodes))
        t_history = np.zeros(n_steps)
        T_history[0] = T.copy()
        t_history[0] = t

        rho_history = None
        if rho is not None:
            rho_history = np.zeros((n_steps, n_nodes))
            rho_history[0] = rho.copy()

        step = 1
        while t < t_end - 1e-12:
            # Ensure we don't overshoot
            dt_actual = min(dt, t_end - t)

            # Build and solve tridiagonal system
            a, b, c, d = self._build_tridiagonal(T, dt_actual, rho)

            # Apply pyrolysis heat sink to RHS if ablation is active
            if self.ablation is not None and rho is not None:
                drho_dt = self.ablation.compute_pyrolysis_rate(rho, T)
                q_sink = self.ablation.compute_pyrolysis_heat_sink(drho_dt)
                # Add heat sink to RHS (interior nodes only)
                mat = self.material
                for i in range(1, n_nodes - 1):
                    cp_i = mat.cp_at_with_ablation(T[i], rho[i]) if rho is not None else mat.cp_at(T[i])
                    rho_val = rho[i] if rho is not None else mat.density
                    d[i] -= q_sink[i] * dt_actual / (rho_val * cp_i)

            T_new = _thomas_solve(a, b, c, d)

            # Apply boundary conditions
            T_new[0] = T[1]  # placeholder, overwritten by BC
            T_new[-1] = T[-2]  # placeholder, overwritten by BC

            self._apply_bc_hot_to_vec(T_new, config.q_stagnation, dz)
            self._apply_bc_cold_to_vec(T_new)

            t += dt_actual
            T = T_new

            # Update density if ablation is active
            if self.ablation is not None and rho is not None:
                rho, _drho_dt = self.ablation.update_density(rho, T, dt_actual)

            if step < n_steps:
                T_history[step] = T.copy()
                t_history[step] = t
                if rho_history is not None and rho is not None:
                    rho_history[step] = rho.copy()
            step += 1

        # Trim history arrays to actual steps
        T_history = T_history[:step]
        t_history = t_history[:step]
        if rho_history is not None:
            rho_history = rho_history[:step]

        # Compute integrated heat flux (trapezoidal rule)
        q_total = self._compute_total_heat_flux(T_history, t_history, dz)

        if self.ablation is not None and rho is not None and rho_initial is not None:
            # Compute ablation metrics
            rho_v = self.material.density
            rho_c = self.material.char_density if self.material.char_density is not None else 0.0

            # Mass loss per unit area: integral of (rho_v - rho_final) dz
            mass_loss = float(np.trapezoid(rho_v - rho, self.z))

            # Recession: total mass loss / (rho_v * L) approximated
            recession = mass_loss / rho_v if rho_v > 0 else 0.0

            # Char depth: fraction of wall that has charred
            char_threshold = (rho_v + rho_c) / 2.0
            char_mask = rho < char_threshold
            char_depth = float(self.z[-1] * np.sum(char_mask) / n_nodes) if np.any(char_mask) else 0.0

            # Ablation rate
            ablation_rate_mm_s = (recession * 1000.0 / t) if t > 0 else 0.0

            self.rho = rho

            return AblationResult1D(
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
                rho_initial=rho_initial,
                rho_final=rho.copy(),
                rho_history=rho_history if rho_history is not None else np.zeros((1, n_nodes)),
                recession_m=recession,
                char_depth_m=char_depth,
                mass_loss_kg_m2=mass_loss,
                ablation_rate_mm_s=ablation_rate_mm_s,
            )

        self.T = T
        self.t = t

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
        rho: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build tridiagonal system for implicit backward Euler.

        The discretized equation at interior node i is:

            (rho*cp/dt) * T_i_new = (k/dz^2) * (T_{i-1} + T_{i+1} - 2*T_i)
                                     + (rho*cp/dt) * T_i_old

        Rearranged as: a_i * T_{i-1} + b_i * T_i + c_i * T_{i+1} = d_i

        Args:
            T: Current temperature profile (K).
            dt: Time step (s).
            rho: Optional current density field for ablation-coupled solve.

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
        if rho is not None:
            k = np.array([mat.k_at_with_ablation(Ti, rhoi) for Ti, rhoi in zip(T, rho)])
            cp = np.array([mat.cp_at_with_ablation(Ti, rhoi) for Ti, rhoi in zip(T, rho)])
        else:
            k = np.array([mat.k_at(Ti) for Ti in T])
            cp = np.array([mat.cp_at(Ti) for Ti in T])
        rho_val = mat.density

        for i in range(1, n_nodes - 1):
            # Average k at interfaces
            k_left = 0.5 * (k[i - 1] + k[i])
            k_right = 0.5 * (k[i] + k[i + 1])

            rho_i = rho[i] if rho is not None else rho_val
            alpha = rho_i * cp[i] / dt

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
        # Linearized: q_rad approx q_rad_0 + dq/dT * (T - T_wall_0)
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
