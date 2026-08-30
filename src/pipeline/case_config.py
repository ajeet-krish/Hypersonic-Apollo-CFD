"""Case configuration for the hypersonic blunt body pipeline."""
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from geometry.config import BluntBodyConfig


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


@dataclass(frozen=True)
class CaseConfig:
    """Per-case pipeline configuration.

    Attributes:
        name: Case slug for directory naming (e.g. 'apollo-cm')
        label: Human-readable name (e.g. 'Apollo CM')
        preset_fn: Callable returning BluntBodyConfig
        mach: Freestream Mach number
        altitude: Flight altitude (m)
        gamma: Ratio of specific heats
        mesh_tier: Mesh refinement tier
        su2_iterations: SU2 max iterations
        su2_cfl: SU2 CFL number
    """
    name: str
    label: str
    preset_fn: Callable[[], BluntBodyConfig]
    mach: float = 8.0
    altitude: float = 30000.0
    gamma: float = 1.4
    mesh_tier: str = "standard"
    su2_iterations: int = 5000
    su2_cfl: float = 1.0

    @property
    def output_dir(self) -> str:
        """Output directory for simulation artifacts."""
        return f"output/{self.name}"

    @property
    def images_dir(self) -> str:
        """Images directory for plots."""
        return f"docs/assets/images/{self.name}"
