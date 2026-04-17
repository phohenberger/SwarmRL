"""
Module for the expected returns value function.
"""

import torch


class ExpectedReturns:
    """
    Compute discounted cumulative returns for each time step.
    """

    def __init__(self, gamma: float = 0.99, standardize: bool = True):
        """
        Parameters
        ----------
        gamma : float
            Discount factor.
        standardize : bool
            If True, standardize returns to zero mean / unit variance per agent.
        """
        self.gamma = gamma
        self.standardize = standardize
        self.eps = torch.finfo(torch.float32).eps

    def __call__(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute expected returns.

        Parameters
        ----------
        rewards : torch.Tensor (n_steps, n_agents)

        Returns
        -------
        expected_returns : torch.Tensor (n_steps, n_agents)
        """
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32)
        n_steps, n_agents = rewards.shape
        expected_returns = torch.zeros_like(rewards)

        for t in range(n_steps):
            exponents = torch.arange(n_steps - t, dtype=torch.float32)
            gamma_array = self.gamma ** exponents  # (n_steps - t,)
            gamma_array = gamma_array.unsqueeze(1).expand(-1, n_agents)
            expected_returns[t] = (rewards[t:] * gamma_array).sum(dim=0)

        if self.standardize:
            mean = expected_returns.mean(dim=0)
            std = expected_returns.std(dim=0, correction=0) + self.eps
            expected_returns = (expected_returns - mean) / std

        return expected_returns
