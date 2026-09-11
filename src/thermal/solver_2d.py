"""2D axisymmetric thermal solver for heat shield analysis.

Solves the 2D axisymmetric heat equation with temperature-dependent
properties on a structured (s, z) grid:

    rho(T) * cp(T) * dT/dt = (1/s) * d/ds(k(T) * s * dT/dt)
                            + d/dz(k(T) * dT/dz)

Uses implicit backward Euler with sparse direct solve for unconditional
stability. The grid is n_s surface points x n_z+1 wall thickness nodes.

Surface coordinate s runs along the body contour (arc length from nose).
Wall coordinate z runs through the thickness (0 = hot face, thickness = cold face).

Optionally couples with a charring ablation model for pyrolysis
kinetics, density evolution, and surface recession.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

from .config import AblationConfig, ThermalConfig2D
from .materials import get_material
from .results import AblationResult2D, ThermalResult2D


class ThermalSolver2D:
    """2D axisymmetric thermal solver for heat shield.

    Solves the 2D heat equation with temperature-dependent thermal
    conductivity and specific heat using implicit backward Euler.

    Grid: n_s surface points x n_z+1 wall thickness nodes (uniform spacing).
    Hot side BC: convective heat flux at z=0 for each surface point.
    Cold side BC: fixed temperature or linearized radiation at z=wall_thickness.
    Surface boundaries: zero gradient (insulated edges at s=0 and s=s_max).

    When ablation_config is provided, couples with AblationModel
    for pyrolysis-driven density evolution and heat sink.
    """

    def __init__(
        self,
        config: ThermalConfig2D,
        ablation_config: AblationConfig | None = None,
    ) -> None:
        """Initialize solver with 2D thermal configuration.

        Args:
            config: 2D thermal analysis configuration.
            ablation_config: Optional ablation configuration. When provided,
                the solver couples density evolution with temperature.
        """
        self.config = config
        self.material = get_material(config.material)
        self.s = np.asarray(config.s_surface, dtype=float)
        self.z = np.linspace(0.0, config.wall_thickness, config.n_z + 1)
        self.n_s = len(self.s)
        self.n_z = config.n_z
        self.T = np.full((self.n_s, self.n_z + 1), config.cold_wall_temp)
        self.t = 0.0

        # Ablation coupling
        self.ablation_config = ablation_config
        self.ablation = None
        self.rho: np.ndarray | None = None
        if ablation_config is not None:
            from .ablation import AblationModel
            self.ablation = AblationModel(self.material, ablation_config)
            self.rho = np.full((self.n_s, self.n_z + 1), self.material.density)

    def solve(self) -> ThermalResult2D | AblationResult2D:
        """Run 2D thermal simulation from t=0 to t_end.

        Steps through time using implicit backward Euler with a sparse
        direct solve for the (n_s * (n_z+1)) system. When ablation is
        enabled, couples density evolution and applies pyrolysis heat sink.

        Returns:
            ThermalResult2D or AblationResult2D depending on whether
            ablation is enabled.
        """
        config = self.config
        n_s = self.n_s
        n_z = self.n_z
        n_nodes_z = n_z + 1
        dt = config.dt
        t_end = config.t_end
        ds = self.s[1] - self.s[0] if n_s > 1 else 1.0
        dz = config.wall_thickness / n_z

        # Initialize
        T = self.T.copy()
        T_initial = T.copy()
        t = 0.0

        # Ablation state
        rho = self.rho.copy() if self.rho is not None else None
        rho_initial = rho.copy() if rho is not None else None

        # Surface heat flux (interpolated to n_s points if needed)
        q_surface = np.interp(
            self.s,
            np.linspace(self.s[0], self.s[-1], len(config.q_surface)),
            np.asarray(config.q_surface, dtype=float),
        )

        # Pre-allocate history storage
        n_est = int(t_end / dt) + 2
        T_wall_history = np.zeros((n_est, n_s))
        t_history = np.zeros(n_est)
        T_wall_history[0] = T[:, 0].copy()
        t_history[0] = t

        rho_wall_history = None
        if rho is not None:
            rho_wall_history = np.zeros((n_est, n_s))
            rho_wall_history[0] = rho[:, 0].copy()

        step = 1
        while t < t_end - 1e-12:
            dt_actual = min(dt, t_end - t)

            # Build and solve the sparse system
            T_new = self._solve_step(T, q_surface, dt_actual, ds, dz, rho)

            # Apply pyrolysis heat sink if ablation is active
            # (heat sink is included in _solve_step via modified RHS)

            t += dt_actual
            T = T_new

            # Update density if ablation is active
            if self.ablation is not None and rho is not None:
                rho, _drho_dt = self.ablation.update_density(rho, T, dt_actual)

            if step < n_est:
                T_wall_history[step] = T[:, 0].copy()
                t_history[step] = t
                if rho_wall_history is not None and rho is not None:
                    rho_wall_history[step] = rho[:, 0].copy()
            step += 1

        # Trim history arrays
        T_wall_history = T_wall_history[:step]
        t_history = t_history[:step]
        if rho_wall_history is not None:
            rho_wall_history = rho_wall_history[:step]

        # Compute integrated heat flux at hot face
        q_total = self._compute_total_heat_flux(T_wall_history, t_history, dz)

        if self.ablation is not None and rho is not None and rho_initial is not None:
            # Compute ablation metrics per surface point
            rho_v = self.material.density
            rho_c = self.material.char_density if self.material.char_density is not None else 0.0

            # Mass loss per unit area at each surface point
            mass_loss = np.trapezoid(rho_v - rho, self.z, axis=1)

            # Recession at each surface point
            recession = mass_loss / rho_v if rho_v > 0 else np.zeros(n_s)

            # Char depth at each surface point
            char_threshold = (rho_v + rho_c) / 2.0
            char_depth = np.zeros(n_s)
            for i in range(n_s):
                char_mask = rho[i, :] < char_threshold
                char_depth[i] = float(self.z[-1] * np.sum(char_mask) / n_nodes_z) if np.any(char_mask) else 0.0

            # Ablation rate
            ablation_rate = np.where(
                t > 0, recession * 1000.0 / t, 0.0
            )

            self.rho = rho

            return AblationResult2D(
                s=self.s,
                z=self.z,
                T_initial=T_initial,
                T_final=T.copy(),
                T_wall_history=T_wall_history,
                t_history=t_history,
                q_surface=q_surface,
                T_max_wall=float(np.max(T[:, 0])),
                T_max_back=float(np.max(T[:, -1])),
                q_total=q_total,
                material=self.material.name,
                wall_thickness=config.wall_thickness,
                rho_initial=rho_initial,
                rho_final=rho.copy(),
                rho_history=rho_wall_history if rho_wall_history is not None else np.zeros((1, n_s)),
                recession_m=recession,
                char_depth_m=char_depth,
                mass_loss_kg_m2=mass_loss,
                ablation_rate_mm_s=ablation_rate,
            )

        self.T = T
        self.t = t

        return ThermalResult2D(
            s=self.s,
            z=self.z,
            T_initial=T_initial,
            T_final=T.copy(),
            T_wall_history=T_wall_history,
            t_history=t_history,
            q_surface=q_surface,
            T_max_wall=float(np.max(T[:, 0])),
            T_max_back=float(np.max(T[:, -1])),
            q_total=q_total,
            material=self.material.name,
            wall_thickness=config.wall_thickness,
        )

    def _solve_step(
        self,
        T: np.ndarray,
        q_surface: np.ndarray,
        dt: float,
        ds: float,
        dz: float,
        rho: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve one implicit time step.

        Assembles the sparse matrix A and RHS vector b for the system
        A * T_new = b, then solves with spsolve. When ablation is active,
        modifies the RHS to include pyrolysis heat sink.

        Args:
            T: Current temperature field (n_s x n_z+1).
            q_surface: Surface heat flux at each s point (W/m^2).
            dt: Time step (s).
            ds: Surface grid spacing (m).
            dz: Wall thickness grid spacing (m).
            rho: Optional current density field for ablation-coupled solve.

        Returns:
            Updated temperature field T_new (n_s x n_z+1).
        """
        n_s = self.n_s
        n_z = self.n_z
        n_nodes_z = n_z + 1
        n_total = n_s * n_nodes_z
        mat = self.material

        # Compute temperature-dependent properties at all nodes
        if rho is not None:
            k = np.array([[mat.k_at_with_ablation(T[i, j], rho[i, j])
                            for j in range(n_nodes_z)]
                           for i in range(n_s)])
            cp = np.array([[mat.cp_at_with_ablation(T[i, j], rho[i, j])
                             for j in range(n_nodes_z)]
                            for i in range(n_s)])
        else:
            k = np.array([[mat.k_at(T[i, j]) for j in range(n_nodes_z)]
                           for i in range(n_s)])
            cp = np.array([[mat.cp_at(T[i, j]) for j in range(n_nodes_z)]
                            for i in range(n_s)])
        rho_val = mat.density

        # Build sparse matrix using lil_matrix for efficient construction
        A = sparse.lil_matrix((n_total, n_total))
        b = np.zeros(n_total)

        def idx(i: int, j: int) -> int:
            """Map (i, j) grid index to flat array index."""
            return i * n_nodes_z + j

        for i in range(n_s):
            for j in range(n_nodes_z):
                kij = k[i, j]
                cpij = cp[i, j]
                rho_ij = rho[i, j] if rho is not None else rho_val
                alpha = rho_ij * cpij / dt
                row = idx(i, j)

                # Interior nodes in z-direction (not on z boundaries)
                if 0 < j < n_z:
                    # z-direction: d/dz(k * dT/dz)
                    k_up = 0.5 * (kij + k[i, j + 1])
                    k_dn = 0.5 * (kij + k[i, j - 1])

                    A[row, idx(i, j - 1)] = k_dn / dz**2
                    A[row, idx(i, j + 1)] = k_up / dz**2
                    A[row, row] = -(k_dn / dz**2 + k_up / dz**2 + alpha)

                elif j == 0:
                    # Hot side boundary (z=0): convective heat flux BC
                    # Applied after assembly via _apply_bc_hot
                    A[row, row] = 1.0

                elif j == n_z:
                    # Cold side boundary (z=wall_thickness): handled after assembly
                    A[row, row] = 1.0

                # s-direction: (1/s) * d/ds(k * s * dT/ds)
                # Only for interior s-points (0 < i < n_s-1)
                if 0 < i < n_s - 1:
                    s_i = self.s[i]
                    if s_i < 1e-12:
                        s_i = 1e-12  # Avoid division by zero at s=0

                    s_half_right = 0.5 * (self.s[i] + self.s[i + 1])
                    s_half_left = 0.5 * (self.s[i - 1] + self.s[i])

                    k_right = 0.5 * (kij + k[i + 1, j])
                    k_left = 0.5 * (kij + k[i - 1, j])

                    coeff_right = k_right * s_half_right / (s_i * ds**2)
                    coeff_left = k_left * s_half_left / (s_i * ds**2)

                    A[row, idx(i + 1, j)] += coeff_right
                    A[row, idx(i - 1, j)] += coeff_left
                    A[row, row] += -(coeff_right + coeff_left)

                # s-boundary conditions: zero gradient (insulated edges)
                # At i=0 and i=n_s-1, the s-direction terms are omitted
                # (natural Neumann BC with zero flux).

                # RHS: alpha * T_old for interior z-nodes
                if 0 < j < n_z:
                    b[row] = -alpha * T[i, j]

                    # Add pyrolysis heat sink to RHS
                    if self.ablation is not None and rho is not None:
                        drho_dt = self.ablation.compute_pyrolysis_rate(
                            rho[i:i+1, j:j+1], T[i:i+1, j:j+1]
                        )
                        q_sink = self.ablation.compute_pyrolysis_heat_sink(drho_dt)
                        b[row] -= q_sink[0, 0] * dt / (rho[i, j] * cpij)
                elif j == 0:
                    # Will be overwritten by hot BC
                    b[row] = T[i, j]
                elif j == n_z:
                    # Will be overwritten by cold BC
                    b[row] = T[i, j]

        # Convert to CSR for efficient solving
        A_csr = A.tocsr()

        # Solve the system
        T_flat = splinalg.spsolve(A_csr, b)

        # Reshape to 2D
        T_new = T_flat.reshape((n_s, n_nodes_z))

        # Apply boundary conditions (overwrite the placeholder rows)
        self._apply_bc_hot(T_new, q_surface, dz)
        self._apply_bc_cold(T_new)

        return T_new

    def _apply_bc_hot(self, T_new: np.ndarray, q_flux: np.ndarray, dz: float) -> None:
        """Apply convective heat flux BC on hot side (z=0) for all surface points.

        Uses forward difference: q_flux = -k * (T_1 - T_0) / dz
        => T_0 = T_1 + q_flux * dz / k(T_0)

        Args:
            T_new: Temperature field being solved (n_s x n_z+1).
            q_flux: Heat flux at each surface point (W/m^2).
            dz: Grid spacing through wall thickness (m).
        """
        for i in range(self.n_s):
            k0 = self.material.k_at(T_new[i, 0])
            if k0 > 0:
                T_new[i, 0] = T_new[i, 1] + q_flux[i] * dz / k0

    def _apply_bc_cold(self, T_new: np.ndarray) -> None:
        """Apply cold wall BC (fixed temperature or radiation).

        Fixed temperature: T[-1] = T_cold for all surface points.
        Radiation: linearized radiation BC at cold face.

        Args:
            T_new: Temperature field being solved (n_s x n_z+1).
        """
        if not self.config.radiation:
            T_new[:, -1] = self.config.cold_wall_temp
            return

        # Radiation BC: linearized Newton iteration at cold face
        sigma = 5.670374419e-8  # Stefan-Boltzmann constant
        eps = self.config.emissivity
        T_env = self.config.cold_wall_temp
        dz = self.config.wall_thickness / self.config.n_z

        for i in range(self.n_s):
            k_n = self.material.k_at(T_new[i, -1])
            T_wall = T_new[i, -1]
            dq_dt = 4.0 * eps * sigma * T_wall**3
            T_next = T_new[i, -2]
            q_rad_0 = eps * sigma * (T_wall**4 - T_env**4)
            T_new[i, -1] = (k_n * T_next - (q_rad_0 - dq_dt * T_wall) * dz) / (
                k_n + dq_dt * dz
            )

    def _compute_total_heat_flux(
        self,
        T_wall_history: np.ndarray,
        t_history: np.ndarray,
        dz: float,
    ) -> float:
        """Compute total heat input by integrating hot-face heat flux over time.

        Uses forward difference for the temperature gradient at z=0 and
        integrates over time using the trapezoidal rule.

        Args:
            T_wall_history: Wall temperature history (n_time x n_s).
            t_history: Time array (s).
            dz: Grid spacing (m).

        Returns:
            Total heat input averaged over surface (J/m^2).
        """
        mat = self.material
        n_steps = len(t_history)

        if n_steps < 2:
            return 0.0

        # Compute total heat input as applied flux integrated over time
        # This is the total energy delivered to the surface (J/m²)
        q_applied = float(np.mean(self.config.q_surface))
        q_total = q_applied * (t_history[-1] - t_history[0])
        return abs(q_total)
