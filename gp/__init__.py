"""Gaussian Process belief module for CleaningEnv.

Public API:
    GPConfig               — frozen configuration dataclass
    IndependentBayesianGrid — per-cell Kalman GP with optional spatial blur
"""

from .gp_config import GPConfig
from .grid_gp import IndependentBayesianGrid

__all__ = ["GPConfig", "IndependentBayesianGrid"]
