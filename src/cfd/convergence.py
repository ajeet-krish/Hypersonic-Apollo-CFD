"""Multi-stage convergence strategy for hypersonic SU2 simulations.

Implements Mach ramping from low Mach numbers to the target, using
restart files between stages. Each stage can have different solver
settings (CFL, MUSCL, linear solver) optimized for its Mach range.
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
    linear_solver: str = "BCGSTAB"
    linear_solver_error: float = 1e-2
    linear_solver_iter: int = 20
    cfl_adapt_min: float = 0.0005
    cfl_adapt_max: float = 0.05
    cfl_adapt_decrease: float = 0.5
    cfl_adapt_increase: float = 1.5


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

        For M <= 5: single stage (direct solve)
        For M <= 10: two stages (M=5 ramp)
        For M > 10: three stages (M=5 -> M=10 ramp)

        Args:
            target_mach: Target freestream Mach number.

        Returns:
            ConvergenceStrategy with appropriate stages.
        """
        if target_mach <= 5.0:
            return cls._single_stage(target_mach)
        elif target_mach <= 10.0:
            return cls._two_stage(target_mach)
        else:
            return cls._three_stage(target_mach)

    @classmethod
    def mach_ramp(cls, target_mach: float, n_stages: int = 3) -> ConvergenceStrategy:
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
            stage = ConvergenceStage(
                name=f"Stage {i + 1}: M={mach_val:.1f}",
                mach=mach_val,
                cfl=0.005,
                iterations=10000 if is_last else 3000,
                muscl=False,
                linear_solver="BCGSTAB",
                linear_solver_error=1e-4 if is_last else 1e-2,
                linear_solver_iter=40 if is_last else 20,
                cfl_adapt_min=0.0005,
                cfl_adapt_max=0.05,
                cfl_adapt_decrease=0.5,
                cfl_adapt_increase=1.5,
            )
            stages.append(stage)

        return cls(stages=stages)

    @classmethod
    def _single_stage(cls, target_mach: float) -> ConvergenceStrategy:
        """Create a single-stage direct solve for M <= 5.

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
            linear_solver="BCGSTAB",
            linear_solver_error=1e-2,
            linear_solver_iter=20,
            cfl_adapt_min=0.0005,
            cfl_adapt_max=0.05,
            cfl_adapt_decrease=0.5,
            cfl_adapt_increase=1.5,
        )
        return cls(stages=[stage])

    @classmethod
    def _two_stage(cls, target_mach: float) -> ConvergenceStrategy:
        """Create a two-stage ramp for 5 < M <= 10.

        Stage 1: M=5.0, Stage 2: target_mach with restart.

        Args:
            target_mach: Target freestream Mach number.

        Returns:
            ConvergenceStrategy with two stages.
        """
        stage1 = ConvergenceStage(
            name="M=5.0 first-order",
            mach=5.0,
            cfl=0.005,
            iterations=3000,
            muscl=False,
            linear_solver="BCGSTAB",
            linear_solver_error=1e-2,
            linear_solver_iter=20,
            cfl_adapt_min=0.0005,
            cfl_adapt_max=0.05,
            cfl_adapt_decrease=0.5,
            cfl_adapt_increase=1.5,
        )
        stage2 = ConvergenceStage(
            name=f"M={target_mach:.1f} restart",
            mach=target_mach,
            cfl=0.005,
            iterations=5000,
            muscl=False,
            linear_solver="BCGSTAB",
            linear_solver_error=1e-2,
            linear_solver_iter=20,
            cfl_adapt_min=0.0005,
            cfl_adapt_max=0.05,
            cfl_adapt_decrease=0.5,
            cfl_adapt_increase=1.5,
        )
        return cls(stages=[stage1, stage2])

    @classmethod
    def _three_stage(cls, target_mach: float) -> ConvergenceStrategy:
        """Create a three-stage ramp for M > 10.

        Stage 1: M=5.0, Stage 2: M=10.0, Stage 3: target_mach with restart.

        Args:
            target_mach: Target freestream Mach number.

        Returns:
            ConvergenceStrategy with three stages.
        """
        stage1 = ConvergenceStage(
            name="M=5.0 first-order",
            mach=5.0,
            cfl=0.005,
            iterations=3000,
            muscl=False,
            linear_solver="BCGSTAB",
            linear_solver_error=1e-2,
            linear_solver_iter=20,
            cfl_adapt_min=0.0005,
            cfl_adapt_max=0.05,
            cfl_adapt_decrease=0.5,
            cfl_adapt_increase=1.5,
        )
        stage2 = ConvergenceStage(
            name="M=10.0 restart",
            mach=10.0,
            cfl=0.005,
            iterations=5000,
            muscl=False,
            linear_solver="BCGSTAB",
            linear_solver_error=1e-2,
            linear_solver_iter=20,
            cfl_adapt_min=0.0005,
            cfl_adapt_max=0.05,
            cfl_adapt_decrease=0.5,
            cfl_adapt_increase=1.5,
        )
        stage3 = ConvergenceStage(
            name=f"M={target_mach:.1f} restart",
            mach=target_mach,
            cfl=0.005,
            iterations=10000,
            muscl=False,
            linear_solver="BCGSTAB",
            linear_solver_error=1e-2,
            linear_solver_iter=20,
            cfl_adapt_min=0.0005,
            cfl_adapt_max=0.05,
            cfl_adapt_decrease=0.5,
            cfl_adapt_increase=1.5,
        )
        return cls(stages=[stage1, stage2, stage3])
