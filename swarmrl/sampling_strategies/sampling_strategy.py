"""
Parent class for sampling strategies.
"""

import torch


class SamplingStrategy:
    """
    Parent class for sampling strategies.
    """

    def compute_entropy(self, probabilities) -> torch.Tensor:
        """
        Compute the Shannon entropy of the probabilities.

        Parameters
        ----------
        probabilities : torch.Tensor (n_colloids, n_actions)
        """
        if not isinstance(probabilities, torch.Tensor):
            probabilities = torch.tensor(probabilities, dtype=torch.float32)
        eps = 1e-8
        p = probabilities + eps
        return -(p * torch.log(p)).sum()

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample from the distribution.

        Parameters
        ----------
        logits : torch.Tensor (n_colloids, n_dimensions)

        Returns
        -------
        indices : torch.Tensor (n_colloids,)
        """
        raise NotImplementedError("Implemented in child classes.")
