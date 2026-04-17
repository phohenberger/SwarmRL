"""
Data class for the colloid agent.
"""

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class Colloid:
    """
    Wrapper class for a colloid object.
    """

    pos: np.ndarray
    director: np.ndarray
    id: int
    velocity: np.ndarray = None
    type: int = 0

    def __repr__(self):
        return (
            f"Colloid(pos={self.pos}, director={self.director}, id={self.id},"
            f" velocity={self.velocity}, type={self.type})"
        )

    def __eq__(self, other):
        return self.id == other.id
