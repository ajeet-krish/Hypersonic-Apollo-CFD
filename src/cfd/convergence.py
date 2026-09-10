"""Multi-stage convergence strategy for hypersonic SU2 simulations.

Implements Mach ramping from low Mach numbers to the target, using
restart files between stages. Each stage can have different solver
settings (CFL, MUSCL, linear solver, flux method) optimized for its
Mach range.

Strategy for M > 10 (e.g., M=15.6):
    Stage 1: M=2.0 (subsonic/supersonic transition)
    Stage 2: M=5.0 (supersonic)
    Stage 3: M=10.0 (hypersonic)
    Stage 4: M=target (full hypersonic)

Each stage uses:
    - Very conservative CFL (0.001-0.003) with adaptation enabled
    - Tight linear solver tolerance (1e-4, 50 iterations)
    - Roe flux for better shock stability
    - First-order spatial accuracy (no MUSCL) for stability
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConvergenceStage:
    """Single stage in a multi-stage convergence workflow.

    Attributes:
        name: Human-readable stage name (e.g., "M=5 first-order")
        mach: Freestream Mach number for this stage
        cfl: CFL number
        iterations: Maximum iterations
        muscl: Whether to enable MUSCL reconstruction
        conv_method: Convective flux method (ROE, AUSM, HLLC)
        linear_solver: Linear solver type (BCGSTAB, FGMRES)
        linear_solver_error: Linear solver tolerance
        linear_solver_iter: Linear solver max iterations
        cfl_adapt_min: Minimum CFL during adaptation
        cfl_adapt_max: Maximum CFL during adaptation
        cfl_adapt_decrease: CFL decrease factor
        cfl_adapt_increase: CFL increase factor
    """

    name: str
    mach: float
    cfl: float
    iterations: int
    muscl: bool = False
    conv_method: str = "ROE"
    linear_solver: str = "BCGSTAB"
    linear_solver_error: float = 1e-4
    linear_solver_iter: int = 50
    cfl_adapt_min: float = 0.0005
    cfl_adapt_max: float = 1.0  # Must be >= 1.0 for SU2 v8.4
    cfl_adapt_decrease: float = 0.5
    cfl_adapt_increase: float = 1.2


@dataclass
class ConvergenceStrategy:
    """Multi-stage convergence strategy for hypersonic SU2.

    Contains a list of stages that run sequentially, each using the
    previous stage's solution as a restart.
    """

    stages: list[ConvergenceStage] = field(default_factory=list)

    @classmethod
    def for_mach(cls, target_mach: float) -> ConvergenceStrategy:
        """Create appropriate strategy based on target Mach number.

        For M <= 2: single stage (direct solve)
        For 2 < M <= 5: two stages (M=2 ramp)
        For 5 < M <= 10: three stages (M=2 -> M=5 ramp)
        For M > 10: four stages (M=2 -> M=5 -> M=10 ramp)

        Args:
            target_mach: Target freestream Mach number.

        Returns:
            ConvergenceStrategy with appropriate stages.
        """
        if target_mach <= 2.0:
            return cls._single_stage(target_mach)
        elif target_mach <= 5.0:
            return cls._two_stage(target_mach)
        elif target_mach <= 10.0:
            return cls._three_stage(target_mach)
        else:
            return cls._four_stage(target_mach)

    @classmethod
    def mach_ramp(cls, target_mach: float, n_stages: int = 4) -> ConvergenceStrategy:
        """Create a Mach ramping strategy.

        Generates intermediate Mach numbers evenly spaced from M=2 to target_mach.

        Args:
            target_mach: Target freestream Mach number.
            n_stages: Number of stages (minimum 2).

        Returns:
            ConvergenceStrategy with evenly spaced Mach ramp.
        """
        n_stages = max(2, n_stages)
        start_mach = 2.0
        # Generate intermediate Mach numbers
        mach_numbers = []
        for i in range(n_stages):
            frac = i / (n_stages - 1)
            mach_val = start_mach + frac * (target_mach - start_mach)
            mach_numbers.append(mach_val)

        stages = []
        for i, mach_val in enumerate(mach_numbers):
            is_last = i == len(mach_numbers) - 1
            # Progressive CFL: start very conservative, allow growth
            cfl_val = 0.001 + 0.001 * i
            # Progressive iterations: more for later stages
            iters = 5000 if is_last else 3000 + 1000 * i
            stage = ConvergenceStage(
                name=f"Stage {i + 1}: M={mach_val:.1f}",
                mach=mach_val,
                cfl=cfl_val,
                iterations=iters,
                muscl=False,
                conv_method="ROE",
                linear_solver="BCGSTAB",
                linear_solver_error=1e-4,
                linear_solver_iter=50,
                cfl_adapt_min=0.0005,
                cfl_adapt_max=1.0,
                cfl_adapt_decrease=0.5,
                cfl_adapt_increase=1.2,
            )
            stages.append(stage)

        return cls(stages=stages)

    @classmethod
    def _single_stage(cls, target_mach: float) -> ConvergenceStrategy:
        """Create a single-stage direct solve for M <= 2.

        Args:
            target_mach: Target freestream Mach number.

        Returns:
            ConvergenceStrategy with one stage.
        """
        stage = ConvergenceStage(
            name=f"M={target_mach:.1f} direct",
            mach=target_mach,
            cfl=0.01,
            iterations=5000,
            muscl=False,
            conv_method="ROE",
            linear_solver="BCGSTAB",
            linear_solver_error=1e-4,
            linear_solver_iter=50,
            cfl_adapt_min=0.0005,
                cfl_adapt_max=1.0,
            cfl_adapt_decrease=0.5,
            cfl_adapt_increase=1.2,
        )
        return cls(stages=[stage])

    @classmethod
    def _two_stage(cls, target_mach: float) -> ConvergenceStrategy:
        """Create a two-stage ramp for 2 < M <= 5.

        Stage 1: M=2.0, Stage 2: target_mach with restart.

        Args:
            target_mach: Target freestream Mach number.

        Returns:
            ConvergenceStrategy with two stages.
        """
        stage1 = ConvergenceStage(
            name="M=2.0 first-order",
            mach=2.0,
            cfl=0.001,
            iterations=5000,
            muscl=False,
            conv_method="ROE",
            linear_solver="BCGSTAB",
            linear_solver_error=1e-4,
            linear_solver_iter=50,
                cfl_adapt_min=0.0005,
                cfl_adapt_max=1.0,
                cfl_adapt_decrease=0.5,
                cfl_adapt_increase=1.2,
        )
        stage2 = ConvergenceStage(
            name=f"M={target_mach:.1f} restart",
            mach=target_mach,
            cfl=0.002,
            iterations=8000,
            muscl=False,
            conv_method="ROE",
            linear_solver="BCGSTAB",
            linear_solver_error=1e-4,
            linear_solver_iter=50,
                cfl_adapt_min=0.001,
                cfl_adapt_max=1.0,
                cfl_adapt_decrease=0.5,
                cfl_adapt_increase=1.2,
        )
        return cls(stages=[stage1, stage2])

    @classmethod
    def _three_stage(cls, target_mach: float) -> ConvergenceStrategy:
        """Create a three-stage ramp for 5 < M <= 10.

        Stage 1: M=2.0, Stage 2: M=5.0, Stage 3: target_mach with restart.

        Args:
            target_mach: Target freestream Mach number.

        Returns:
            ConvergenceStrategy with three stages.
        """
        stage1 = ConvergenceStage(
            name="M=2.0 first-order",
            mach=2.0,
            cfl=0.001,
            iterations=5000,
            muscl=False,
            conv_method="ROE",
            linear_solver="BCGSTAB",
            linear_solver_error=1e-4,
            linear_solver_iter=50,
            cfl_adapt_min=0.0005,
                cfl_adapt_max=1.0,
            cfl_adapt_decrease=0.5,
            cfl_adapt_increase=1.2,
        )
        stage2 = ConvergenceStage(
            name="M=5.0 restart",
            mach=5.0,
            cfl=0.002,
            iterations=5000,
            muscl=False,
            conv_method="ROE",
            linear_solver="BCGSTAB",
            linear_solver_error=1e-4,
            linear_solver_iter=50,
            cfl_adapt_min=0.001,
                cfl_adapt_max=1.0,
            cfl_adapt_decrease=0.5,
            cfl_adapt_increase=1.2,
        )
        stage3 = ConvergenceStage(
            name=f"M={target_mach:.1f} restart",
            mach=target_mach,
            cfl=0.003,
            iterations=8000,
            muscl=False,
            conv_method="ROE",
            linear_solver="BCGSTAB",
            linear_solver_error=1e-4,
            linear_solver_iter=50,
            cfl_adapt_min=0.001,
                cfl_adapt_max=1.0,
            cfl_adapt_decrease=0.5,
            cfl_adapt_increase=1.2,
        )
        return cls(stages=[stage1, stage2, stage3])

    @classmethod
    def _four_stage(cls, target_mach: float) -> ConvergenceStrategy:
        """Create a four-stage ramp for M > 10.

        Stage 1: M=2.0, Stage 2: M=5.0, Stage 3: M=10.0, Stage 4: target_mach.

        Args:
            target_mach: Target freestream Mach number.

        Returns:
            ConvergenceStrategy with four stages.
        """
        stage1 = ConvergenceStage(
            name="M=2.0 first-order",
            mach=2.0,
            cfl=0.001,
            iterations=5000,
            muscl=False,
            conv_method="ROE",
            linear_solver="BCGSTAB",
            linear_solver_error=1e-4,
            linear_solver_iter=50,
            cfl_adapt_min=0.0005,
                cfl_adapt_max=1.0,
            cfl_adapt_decrease=0.5,
            cfl_adapt_increase=1.2,
        )
        stage2 = ConvergenceStage(
            name="M=5.0 restart",
            mach=5.0,
            cfl=0.002,
            iterations=5000,
            muscl=False,
            conv_method="ROE",
            linear_solver="BCGSTAB",
            linear_solver_error=1e-4,
            linear_solver_iter=50,
            cfl_adapt_min=0.001,
                cfl_adapt_max=1.0,
            cfl_adapt_decrease=0.5,
            cfl_adapt_increase=1.2,
        )
        stage3 = ConvergenceStage(
            name="M=10.0 restart",
            mach=10.0,
            cfl=0.003,
            iterations=8000,
            muscl=False,
            conv_method="ROE",
            linear_solver="BCGSTAB",
            linear_solver_error=1e-4,
            linear_solver_iter=50,
            cfl_adapt_min=0.001,
                cfl_adapt_max=1.0,
            cfl_adapt_decrease=0.5,
            cfl_adapt_increase=1.2,
        )
        stage4 = ConvergenceStage(
            name=f"M={target_mach:.1f} restart",
            mach=target_mach,
            cfl=0.003,
            iterations=15000,
            muscl=False,
            conv_method="ROE",
            linear_solver="BCGSTAB",
            linear_solver_error=1e-4,
            linear_solver_iter=50,
            cfl_adapt_min=0.001,
                cfl_adapt_max=1.0,
            cfl_adapt_decrease=0.5,
            cfl_adapt_increase=1.2,
        )
        return cls(stages=[stage1, stage2, stage3, stage4])

    @classmethod
    def for_3d_mach(cls, target_mach: float, n_stages: int = 4) -> "ConvergenceStrategy":
        """Create a convergence strategy optimized for 3D meshes.

        3D meshes are 10-50x larger than 2D, requiring:
        - Conservative CFL (0.001-0.01) for stability
        - Higher iteration counts per stage (8k, 12k, 16k, 25k for 4-stage)
        - FGMRES linear solver (better for 3D systems)
        - Tighter linear solver tolerance (1e-6)

        For the default 4-stage ramp (M=2 -> M=5 -> M=10 -> target):
            Stage 1 (M=2.0):   8,000 iterations
            Stage 2 (M=5.0):  12,000 iterations
            Stage 3 (M=10.0): 16,000 iterations
            Stage 4 (target): 25,000 iterations
            Total: 61,000 iterations

        Args:
            target_mach: Target freestream Mach number.
            n_stages: Number of stages (minimum 2, default 4).

        Returns:
            ConvergenceStrategy optimized for 3D meshes.
        """
        n_stages = max(2, n_stages)
        start_mach = 2.0

        # Generate intermediate Mach numbers
        mach_numbers = []
        for i in range(n_stages):
            frac = i / (n_stages - 1)
            mach_val = start_mach + frac * (target_mach - start_mach)
            mach_numbers.append(mach_val)

        # Iteration counts per stage for 3D (higher than 2D for larger meshes)
        # Default 4-stage: 8000, 12000, 16000, 25000 = 61000 total
        three_d_iterations = [8000, 12000, 16000, 25000]

        stages = []
        for i, mach_val in enumerate(mach_numbers):
            # CFL progression: 0.001 -> 0.003 -> 0.005 -> 0.010
            cfl_val = 0.001 + 0.002 * i

            # Use explicit iteration counts for known stage counts,
            # otherwise scale linearly for custom stage counts
            if n_stages <= len(three_d_iterations):
                iters = three_d_iterations[i]
            else:
                # For custom stage counts, scale from 8k to 25k linearly
                iters = int(8000 + (25000 - 8000) * i / (n_stages - 1))

            stage = ConvergenceStage(
                name=f"Stage {i + 1}: M={mach_val:.1f}",
                mach=mach_val,
                cfl=cfl_val,
                iterations=iters,
                muscl=False,
                conv_method="ROE",
                linear_solver="FGMRES",  # Better for 3D systems
                linear_solver_error=1e-6,  # Tighter tolerance for 3D
                linear_solver_iter=100,  # More iterations for 3D
                cfl_adapt_min=0.0001,
                cfl_adapt_max=1.0,  # Must be >= 1.0 for SU2 v8.4
                cfl_adapt_decrease=0.5,
                cfl_adapt_increase=1.2,  # More aggressive growth when stable
            )
            stages.append(stage)

        return cls(stages=stages)
