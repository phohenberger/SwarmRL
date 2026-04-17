"""
Random exploration module.
"""

from abc import ABC

import torch

from swarmrl.exploration_policies.exploration_policy import ExplorationPolicy


class RandomExploration(ExplorationPolicy, ABC):
    """
    Epsilon-greedy style exploration: with probability ``probability`` a random
    action replaces the model's chosen action.
    """

    def __init__(self, probability: float = 0.1):
        """
        Parameters
        ----------
        probability : float
            Probability that a random action is substituted. Range [0, 1].
        """
        self.probability = probability

    def __call__(
        self, model_actions: torch.Tensor, action_space_length: int, seed: int
    ) -> torch.Tensor:
        """
        Randomly replace a fraction of model actions.

        Parameters
        ----------
        model_actions : torch.Tensor (n_colloids,)
        action_space_length : int
            Number of possible discrete actions.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        actions : torch.Tensor (n_colloids,) dtype int64
        """
        if not isinstance(model_actions, torch.Tensor):
            model_actions = torch.tensor(
                model_actions, dtype=torch.float32
            )

        generator = torch.Generator()
        generator.manual_seed(int(seed))

        sample = torch.rand(model_actions.shape, generator=generator)

        to_be_changed = torch.clamp(sample - self.probability, min=0.0, max=1.0)
        to_be_changed = torch.clamp(to_be_changed * 1e6, min=0.0, max=1.0)
        not_to_be_changed = torch.clamp(to_be_changed * -10.0 + 1.0, 0.0, 1.0)

        exploration_actions = torch.randint(
            0,
            action_space_length,
            size=(model_actions.shape[0],),
            generator=generator,
        )

        actions = (
            model_actions * to_be_changed + exploration_actions * not_to_be_changed
        ).long()
        return actions
