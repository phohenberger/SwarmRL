"""
Class for the Swarm data container.
"""

from __future__ import annotations

import dataclasses
from typing import List

import numpy as np

from swarmrl.components.colloid import Colloid
from swarmrl.utils.colloid_utils import get_colloid_indices


@dataclasses.dataclass(frozen=True)
class Swarm:
    """
    Wrapper class for a collection of colloids.
    """

    pos: np.ndarray
    director: np.ndarray
    id: int
    velocity: np.ndarray = None
    type: int = 0
    type_indices: dict = None

    def __repr__(self) -> str:
        return (
            f"Swarm(pos={self.pos}, director={self.director}, id={self.id},"
            f" velocity={self.velocity}, type={self.type})"
        )

    def __eq__(self, other):
        return self.id == other.id

    def get_species_swarm(self, species: int) -> Swarm:
        """
        Return a Swarm containing only the colloids of one species.
        """
        indices = self.type_indices[species]
        return Swarm(
            pos=np.take(self.pos, indices, axis=0),
            director=np.take(self.director, indices, axis=0),
            id=np.take(self.id, indices, axis=0),
            velocity=np.take(self.velocity, indices, axis=0),
            type=np.take(self.type, indices, axis=0),
            type_indices=None,
        )


def create_swarm(colloids: List[Colloid]) -> Swarm:
    """
    Create a Swarm from a list of Colloid objects.
    """
    pos = np.array([c.pos for c in colloids]).reshape(-1, colloids[0].pos.shape[0])
    director = np.array([c.director for c in colloids]).reshape(
        -1, colloids[0].director.shape[0]
    )
    id = np.array([c.id for c in colloids]).reshape(-1, 1)
    velocity = np.array([c.velocity for c in colloids]).reshape(
        -1, colloids[0].velocity.shape[0]
    )
    type = np.array([c.type for c in colloids]).reshape(-1, 1)

    type_indices = {}
    for t in np.unique(type):
        type_indices[t] = np.array(get_colloid_indices(colloids, t))

    return Swarm(pos, director, id, velocity, type, type_indices)
