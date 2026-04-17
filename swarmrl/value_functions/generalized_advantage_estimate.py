"""
Module for the Generalized Advantage Estimate value function.
"""

import torch


class GAE:
    """
    Generalized Advantage Estimation (GAE-λ).

    Notes
    -----
    See https://arxiv.org/pdf/1506.02438.pdf for more information.
    """

    def __init__(self, gamma: float = 0.99, lambda_: float = 0.95):
        """
        Parameters
        ----------
        gamma : float
            Discount factor.
        lambda_ : float
            GAE smoothing parameter (trade-off between bias and variance).
        """
        self.gamma = gamma
        self.lambda_ = lambda_
        self.eps = torch.finfo(torch.float32).eps

    def __call__(
        self, rewards: torch.Tensor, values: torch.Tensor
    ) -> tuple:
        """
        Compute advantages and returns.

        Parameters
        ----------
        rewards : torch.Tensor (n_steps, n_agents)
        values : torch.Tensor (n_steps, n_agents)
            Critic predictions (detached from graph before calling).

        Returns
        -------
        advantages : torch.Tensor (n_steps, n_agents)
        returns : torch.Tensor (n_steps, n_agents)
        """
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32)
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=torch.float32)
        n_steps = len(rewards)
        advantages = torch.zeros_like(rewards)
        gae = 0.0

        for t in reversed(range(n_steps)):
            if t != n_steps - 1:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            else:
                delta = rewards[t] - values[t]
            gae = delta + self.gamma * self.lambda_ * gae
            advantages[t] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (
            advantages.std(correction=0) + self.eps
        )
        return advantages, returns
