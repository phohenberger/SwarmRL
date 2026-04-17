"""
Module for the Gumbel distribution.
"""

from abc import ABC

import torch

from swarmrl.sampling_strategies.sampling_strategy import SamplingStrategy


class GumbelDistribution(SamplingStrategy, ABC):
    """
    Gumbel-max sampling for categorical distributions.

    Notes
    -----
    See https://arxiv.org/abs/1611.01144 for more information.
    """

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample actions using the Gumbel-max trick.

        Parameters
        ----------
        logits : torch.Tensor (n_colloids, n_actions)

        Returns
        -------
        indices : torch.Tensor (n_colloids,)
        """
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits, dtype=torch.float32)
        noise = torch.rand_like(logits)
        indices = torch.argmax(logits - torch.log(-torch.log(noise + 1e-20)), dim=-1)
        return indices
