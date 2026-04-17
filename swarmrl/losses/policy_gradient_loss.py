"""
Policy gradient (REINFORCE + baseline) loss.

Notes
-----
https://spinningup.openai.com/en/latest/algorithms/vpg.html
"""

import torch
import torch.nn.functional as F

from swarmrl.losses.loss import Loss
from swarmrl.networks.network import Network
from swarmrl.value_functions.expected_returns import ExpectedReturns


class PolicyGradientLoss(Loss):
    """
    Vanilla policy gradient with a value-function baseline.
    """

    def __init__(self, value_function: ExpectedReturns = ExpectedReturns()):
        super(Loss, self).__init__()
        self.value_function = value_function

    def _calculate_loss(
        self,
        network: Network,
        feature_data: torch.Tensor,
        action_indices: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        network : TorchModel
        feature_data : torch.Tensor (n_steps, n_agents, feature_dim)
        action_indices : torch.Tensor (n_steps, n_agents)
        rewards : torch.Tensor (n_steps, n_agents)

        Returns
        -------
        loss : torch.Tensor (scalar)
        """
        logits, predicted_values = network(feature_data)
        predicted_values = predicted_values.squeeze(-1)

        probabilities = torch.softmax(logits, dim=-1)
        log_probs = torch.log(
            torch.gather(probabilities, -1, action_indices.long().unsqueeze(-1)).squeeze(-1)
            + 1e-8
        )

        returns = self.value_function(rewards)

        advantage = (returns - predicted_values).detach()

        critic_loss = F.huber_loss(predicted_values, returns, reduction="sum")
        actor_loss = -1 * (log_probs * advantage).sum()

        return actor_loss + critic_loss

    def compute_loss(self, network: Network, episode_data):
        """
        Compute a single gradient step.

        Parameters
        ----------
        network : TorchModel
        episode_data : TrajectoryInformation
        """
        feature_data = torch.tensor(episode_data.features, dtype=torch.float32)
        action_data = torch.tensor(episode_data.actions, dtype=torch.float32)
        reward_data = torch.tensor(episode_data.rewards, dtype=torch.float32)

        loss = self._calculate_loss(
            network=network,
            feature_data=feature_data,
            action_indices=action_data,
            rewards=reward_data,
        )
        network.update_model(loss)
