"""
Loss functions based on Proximal Policy Optimization.

Notes
-----
https://spinningup.openai.com/en/latest/algorithms/ppo.html
"""

from abc import ABC

import torch
import torch.nn.functional as F

from swarmrl.losses.loss import Loss
from swarmrl.networks.network import Network
from swarmrl.sampling_strategies.gumbel_distribution import GumbelDistribution
from swarmrl.sampling_strategies.sampling_strategy import SamplingStrategy
from swarmrl.value_functions.generalized_advantage_estimate import GAE


class ProximalPolicyLoss(Loss, ABC):
    """
    Proximal Policy Optimization actor-critic loss.
    """

    def __init__(
        self,
        value_function: GAE = GAE(),
        sampling_strategy: SamplingStrategy = GumbelDistribution(),
        n_epochs: int = 20,
        epsilon: float = 0.2,
        entropy_coefficient: float = 0.01,
    ):
        """
        Parameters
        ----------
        value_function : GAE
            Computes advantages and returns from rewards and values.
        n_epochs : int
            Number of PPO gradient steps per episode.
        epsilon : float
            PPO clipping ratio.
        entropy_coefficient : float
            Weight for the entropy bonus.
        """
        self.value_function = value_function
        self.sampling_strategy = sampling_strategy
        self.n_epochs = n_epochs
        self.epsilon = epsilon
        self.entropy_coefficient = entropy_coefficient
        self.eps = 1e-8

    def _calculate_loss(
        self,
        network: Network,
        feature_data: torch.Tensor,
        action_indices: torch.Tensor,
        rewards: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the PPO loss for one gradient step.

        Parameters
        ----------
        network : TorchModel
        feature_data : torch.Tensor (n_steps, n_agents, feature_dim)
        action_indices : torch.Tensor (n_steps, n_agents)
        rewards : torch.Tensor (n_steps, n_agents)
        old_log_probs : torch.Tensor (n_steps, n_agents)

        Returns
        -------
        loss : torch.Tensor (scalar)
        """
        new_logits, predicted_values = network(feature_data)
        predicted_values = predicted_values.squeeze(-1)  # (n_steps, n_agents)

        advantages, returns = self.value_function(
            rewards=rewards, values=predicted_values.detach()
        )

        new_probabilities = torch.softmax(new_logits, dim=-1)
        entropy = self.sampling_strategy.compute_entropy(new_probabilities).sum()

        chosen_log_probs = torch.log(
            torch.gather(new_probabilities, -1, action_indices.long().unsqueeze(-1)).squeeze(-1)
            + self.eps
        )

        ratio = torch.exp(chosen_log_probs - old_log_probs)

        critic_loss = F.huber_loss(predicted_values, returns, reduction="sum")

        advantages = advantages.detach()
        clipped_loss = -torch.minimum(
            ratio * advantages,
            torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages,
        )
        actor_loss = clipped_loss.sum()

        return actor_loss - self.entropy_coefficient * entropy + 0.5 * critic_loss

    def compute_loss(self, network: Network, episode_data):
        """
        Run ``n_epochs`` PPO updates on the network.

        Parameters
        ----------
        network : TorchModel
        episode_data : TrajectoryInformation
        """
        old_log_probs = torch.tensor(episode_data.log_probs, dtype=torch.float32)
        feature_data = torch.tensor(episode_data.features, dtype=torch.float32)
        action_data = torch.tensor(episode_data.actions, dtype=torch.float32)
        reward_data = torch.tensor(episode_data.rewards, dtype=torch.float32)

        for _ in range(self.n_epochs):
            loss = self._calculate_loss(
                network=network,
                feature_data=feature_data,
                action_indices=action_data,
                rewards=reward_data,
                old_log_probs=old_log_probs,
            )
            network.update_model(loss)
