"""Case configuration for the hypersonic blunt body pipeline."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from geometry.config import BluntBodyConfig

if TYPE_CHECKING:
    from cfd.convergence import ConvergenceStrategy


class PipelineStage(Enum):
    """Pipeline execution stages."""
    GEOMETRY = "geometry"
    MESH = "mesh"
    SU2 = "su2"
    POSTPROCESS = "postprocess"
    VALIDATION = "validation"
    GCI = "gci"
    SWEEP = "sweep"
    SITE = "site"
    APOLLO = "apollo"


@dataclass(frozen=True)
class CaseConfig:
    """Per-case pipeline configuration.

    Attributes:
        name: Case slug for directory naming (e.g. 'apollo-cm')
        label: Human-readable name (e.g. 'Apollo CM')
        preset_fn: Callable returning BluntBodyConfig
        mach: Freestream Mach number
        altitude: Flight altitude (m)
        aoa: Angle of attack in degrees (default 0.0). When nonzero the
            mesh domain is forced to full2d and the body contour is rotated.
        gamma: Ratio of specific heats
        mesh_tier: Mesh refinement tier
        su2_iterations: SU2 max iterations (used when strategy is 'direct')
        su2_cfl: SU2 CFL number
        su2_strategy: Convergence strategy ('direct', 'euler-rans', 'mach-ramp')
        su2_euler_iterations: Iterations for the Euler stage (euler-rans strategy)
        su2_rans_iterations: Iterations for the RANS restart stage (euler-rans strategy)
        su2_mach_ramp_start: Starting Mach for mach-ramp strategy
    """
    name: str
    label: str
    preset_fn: Callable[[], BluntBodyConfig]
    mach: float = 8.0
    altitude: float = 30000.0
    aoa: float = 0.0
    gamma: float = 1.4
    mesh_tier: str = "standard"
    su2_iterations: int = 5000
    su2_cfl: float = 1.0
    su2_strategy: str = "direct"
    su2_euler_iterations: int = 3000
    su2_rans_iterations: int = 10000
    su2_mach_ramp_start: float = 5.0
    convergence_strategy: ConvergenceStrategy | None = None

    @property
    def output_dir(self) -> str:
        """Output directory for simulation artifacts."""
        return f"output/{self.name}"

    @property
    def images_dir(self) -> str:
        """Images directory for plots."""
        return f"docs/assets/images/{self.name}"
