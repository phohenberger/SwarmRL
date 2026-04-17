"""
Class for the species search task.
"""

from typing import List

import numpy as np

from swarmrl.components.colloid import Colloid
from swarmrl.tasks.task import Task


class SpeciesSearch(Task):
    """
    Class for the species search task.
    """

    def __init__(
        self,
        decay_fn: callable,
        box_length: np.ndarray,
        sensing_type: int = 0,
        avoid: bool = False,
        scale_factor: int = 100,
        particle_type: int = 0,
    ):
        super().__init__(particle_type=particle_type)
        self.decay_fn = decay_fn
        self.box_length = box_length
        self.sensing_type = sensing_type
        self.scale_factor = scale_factor
        self.avoid = avoid
        self.historical_field = {}

    def initialize(self, colloids: List[Colloid]):
        reference_ids = self.get_colloid_indices(colloids)
        test_points = np.array([
            c.pos for c in colloids if c.type == self.sensing_type
        ])
        for index in reference_ids:
            colloid = colloids[index]
            _, _, field_value = self._compute_single(
                colloid.id, colloid.pos, test_points, 0.0
            )
            self.historical_field[str(colloid.id)] = field_value

    def _compute_single(
        self,
        index: int,
        reference_position: np.ndarray,
        test_positions: np.ndarray,
        historic_value: float,
    ) -> tuple:
        distances = np.linalg.norm(
            (test_positions - reference_position) / self.box_length, axis=-1
        )
        non_self = distances[distances != 0]
        field_value = float(self.decay_fn(non_self).sum())
        return index, field_value - historic_value, field_value

    def __call__(self, colloids: List[Colloid]):
        if not self.historical_field:
            raise ValueError(
                f"{type(self).__name__} requires initialization. Please set the "
                "initialize attribute of the gym to true and try again."
            )

        reference_ids = self.get_colloid_indices(colloids)
        test_points = np.array([
            c.pos for c in colloids if c.type == self.sensing_type
        ])

        delta_values = []
        for index in reference_ids:
            colloid = colloids[index]
            historic = self.historical_field[str(colloid.id)]
            _, delta, field_value = self._compute_single(
                colloid.id, colloid.pos, test_points, historic
            )
            self.historical_field[str(colloid.id)] = field_value
            delta_values.append(delta)

        rewards = np.array(delta_values)
        if self.avoid:
            rewards = np.clip(rewards, None, 0)
        else:
            rewards = np.clip(rewards, 0, None)

        return self.scale_factor * rewards
