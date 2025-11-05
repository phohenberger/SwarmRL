"""
Engine module for SwarmRL.

This module contains different simulation engines that can be used
for reinforcement learning applications.
"""

from .engine import Engine
from .template_engine import TemplateEngine
from .discrete_lattice import DiscreteLatticeEngine, LatticeAgent

try:
    from .espresso import EspressoMD, MDParams
except ImportError:
    # EspressoMD requires espressomd which may not be available
    pass

try:
    from .real_experiment import RealExperiment
except (ImportError, AttributeError):
    # RealExperiment may have specific dependencies or circular import issues
    RealExperiment = None

# Build __all__ list dynamically based on what's available
__all__ = [
    "Engine",
    "TemplateEngine", 
    "DiscreteLatticeEngine",
    "LatticeAgent",
]

# Add optional imports if they're available
try:
    if 'EspressoMD' in locals():
        __all__.extend(["EspressoMD", "MDParams"])
except NameError:
    pass

try:
    if 'RealExperiment' in locals() and RealExperiment is not None:
        __all__.append("RealExperiment")
except NameError:
    pass