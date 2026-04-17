"""
Computes vision cone(s).
"""

from typing import List

import numpy as np

from swarmrl.components.colloid import Colloid
from swarmrl.observables.observable import Observable
from swarmrl.utils.utils import calc_signed_angle_between_directors


class SubdividedVisionCones(Observable):
    """
    The vision cone acts like a camera for the particles. It can either output all
    colloids within its view, a function of the distances between one colloid and all
    other colloids, or the normalised distance to the source.
    """

    def __init__(
        self,
        vision_range: float,
        vision_half_angle: float,
        n_cones: int,
        radii: List[float],
        detected_types=None,
        particle_type: int = 0,
    ):
        """
        Parameters
        ----------
        vision_range : float
            How far the particles can see.
        vision_half_angle : float
            Half-width of the field of view in radians.
        n_cones : int
            Number of subdivisions of the field of view.
        radii : list
            Radii of the colloids, ordered by colloid id.
        detected_types : list or None
            Colloid types to detect. If None, all types are detected.
        particle_type : int
            Particle type to compute the observable for.
        """
        super().__init__(particle_type=particle_type)
        self.vision_range = vision_range
        self.vision_half_angle = vision_half_angle
        self.n_cones = n_cones
        self.radii = radii
        self.detected_types = detected_types
        self.angle_fn = calc_signed_angle_between_directors

    def _detect_all_things_to_see(self, colloids: List[Colloid]):
        all_types = []
        for c in colloids:
            if c.type not in all_types:
                all_types.append(c.type)
        self.detected_types = np.array(np.sort(all_types))

    def _calculate_cones_single_object(
        self,
        my_pos: np.ndarray,
        my_director: np.ndarray,
        c_type: int,
        c_pos: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        """
        Compute the vision cone contribution from one other colloid.

        Returns
        -------
        np.ndarray (n_cones, n_detected_types)
        """
        vision_val_out = np.ones((self.n_cones, len(self.detected_types)))

        dist = c_pos - my_pos
        dist_norm = np.linalg.norm(dist)
        in_range = dist_norm < self.vision_range
        vision_val_out *= in_range

        if dist_norm > 0:
            vision_val_out *= min(1.0, 2 * radius / dist_norm)
        else:
            vision_val_out *= 0.0

        type_mask = (
            np.ones((self.n_cones, len(self.detected_types)))
            * np.array(self.detected_types)[np.newaxis, :]
        )
        correct_type_mask = type_mask == c_type
        vision_val_out *= correct_type_mask

        if dist_norm > 0:
            angle = self.angle_fn(my_director, dist / dist_norm)
        else:
            return np.zeros((self.n_cones, len(self.detected_types)))

        rims = (
            -self.vision_half_angle
            + np.arange(self.n_cones + 1) * self.vision_half_angle * 2 / self.n_cones
        )
        in_left_rim = rims[:-1] < angle
        in_right_rim = rims[1:] > angle
        in_a_cone = in_left_rim & in_right_rim
        vision_val_out *= in_a_cone[:, np.newaxis]

        return vision_val_out

    def _calculate_cones(self, my_pos, my_director, other_colloids: List[Colloid]):
        """
        Compute total vision cones for one colloid by summing over all others.
        """
        result = np.zeros((self.n_cones, len(self.detected_types)))
        for i, c in enumerate(other_colloids):
            result += self._calculate_cones_single_object(
                my_pos, my_director, c.type, c.pos, self.radii[i]
            )
        return result

    def compute_single_observable(
        self, index: int, colloids: List[Colloid]
    ) -> np.ndarray:
        colloid = colloids[index]

        if self.detected_types is None:
            self._detect_all_things_to_see(colloids)

        my_pos, my_director = colloid.pos, colloid.director

        of_others = [
            [c, self.radii[i]] for i, c in enumerate(colloids) if c is not index
        ]
        other_colloids = [of_others[i][0] for i in range(len(of_others))]
        self.radii = [of_others[i][1] for i in range(len(of_others))]

        return self._calculate_cones(my_pos, my_director, other_colloids)

    def compute_observable(self, colloids: List[Colloid]):
        reference_ids = self.get_colloid_indices(colloids)
        return [
            self.compute_single_observable(index, colloids) for index in reference_ids
        ]
