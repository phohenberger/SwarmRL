"""
Observable for particle sensing.
"""

from typing import List

import numpy as np

from swarmrl.components.colloid import Colloid
from swarmrl.observables.observable import Observable


class ParticleSensing(Observable):
    """
    Class for particle sensing.
    """

    def __init__(
        self,
        decay_fn: callable,
        box_length: np.ndarray,
        sensing_type: int = 0,
        scale_factor: int = 100,
        particle_type: int = 0,
    ):
        """
        Parameters
        ----------
        decay_fn : callable
            Decay function of the field.
        box_length : np.ndarray
            Array for scaling of the distances.
        sensing_type : int
            Type of particle to sense.
        scale_factor : int
            Scaling factor for the observable.
        particle_type : int
            Particle type to compute the observable for.
        """
        super().__init__(particle_type=particle_type)
        self.decay_fn = decay_fn
        self.box_length = box_length
        self.sensing_type = sensing_type
        self.scale_factor = scale_factor
        self.historical_field = {}

    def initialize(self, colloids: List[Colloid]):
        """
        Initialize the observable with starting positions of the colloids.
        """
        reference_ids = self.get_colloid_indices(colloids)
        sensed_positions = np.array([
            c.pos for c in colloids if c.type == self.sensing_type
        ])

        for index in reference_ids:
            colloid = colloids[index]
            _, _, field_value = self._compute_single(
                colloid.id,
                colloid.pos,
                sensed_positions,
                historic_value=0.0,
            )
            self.historical_field[str(colloid.id)] = field_value

    def _compute_single(
        self,
        index: int,
        reference_position: np.ndarray,
        test_positions: np.ndarray,
        historic_value: float,
    ) -> tuple:
        """
        Compute the field observable for a single colloid.

        Returns
        -------
        (index, delta_value, field_value)
        """
        distances = np.linalg.norm(
            (test_positions - reference_position) / self.box_length, axis=-1
        )
        # Exclude the self-distance (distance == 0)
        non_self = distances[distances != 0]
        field_value = float(self.decay_fn(non_self).sum())
        return index, field_value - historic_value, field_value

    def compute_observable(self, colloids: List[Colloid]):
        """
        Compute the observable for all colloids of the relevant type.

        Parameters
        ----------
        colloids : List[Colloid]

        Returns
        -------
        np.ndarray (n_colloids, 1)
            Delta field values (current minus previous).
        """
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

        return self.scale_factor * np.array(delta_values).reshape(-1, 1)
