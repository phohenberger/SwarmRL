"""
Module for the categorical distribution.
"""

from abc import ABC

import torch

from swarmrl.sampling_strategies.sampling_strategy import SamplingStrategy


class CategoricalDistribution(SamplingStrategy, ABC):
    """
    Categorical sampling with optional additive noise.
    """

    def __init__(self, noise: str = "none"):
        """
        Parameters
        ----------
        noise : str
            Noise type to add to logits before sampling.
            Options: 'none', 'uniform', 'gaussian'.
        """
        noise_dict = {
            "uniform": lambda shape: torch.rand(shape),
            "gaussian": lambda shape: torch.randn(shape),
            "none": None,
        }
        if noise not in noise_dict:
            raise KeyError(
                f"Noise method '{noise}' is not implemented. "
                "Choose from 'none', 'gaussian', 'uniform'."
            )
        self.noise_fn = noise_dict[noise]

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample from the categorical distribution.

        Parameters
        ----------
        logits : torch.Tensor (n_colloids, n_actions)

        Returns
        -------
        indices : torch.Tensor (n_colloids,)
        """
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits, dtype=torch.float32)

        if self.noise_fn is not None:
            logits = logits + self.noise_fn(logits.shape)

        probs = torch.softmax(logits, dim=-1)
        indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return indices
