"""
Reward functions based on Random Network Distillation.

Notes
-----
https://arxiv.org/abs/1810.12894
"""

import numpy as np
import torch
import torch.nn as nn

from swarmrl.intrinsic_reward.intrinsic_reward import IntrinsicReward
from swarmrl.intrinsic_reward.rnd_configs import RNDArchitecture, RNDConfig
from swarmrl.utils.colloid_utils import TrajectoryInformation


class RNDReward(IntrinsicReward):
    """
    Intrinsic reward based on Random Network Distillation.

    The reward is the MSE between a fixed random target network and a
    trainable predictor network, both applied to the agent's observations.
    High prediction error → high novelty → high intrinsic reward.
    """

    def __init__(self, rnd_config: RNDConfig):
        input_dim = int(np.prod(rnd_config.input_shape))

        self.target_network = RNDArchitecture(
            input_dim=input_dim, hidden_dim=rnd_config.hidden_dim
        )
        self.predictor_network = RNDArchitecture(
            input_dim=input_dim, hidden_dim=rnd_config.hidden_dim
        )

        # Target network is fixed — no gradient updates.
        for p in self.target_network.parameters():
            p.requires_grad_(False)

        self.optimizer = rnd_config.optimizer_class(
            self.predictor_network.parameters(), **rnd_config.optimizer_kwargs
        )
        self.loss_fn = nn.MSELoss()

        self.n_epochs = rnd_config.n_epochs
        self.batch_size = rnd_config.batch_size
        self.clip_rewards = rnd_config.clip_rewards

    @staticmethod
    def _reshape_data(x: np.ndarray) -> torch.Tensor:
        """Flatten (n_steps, n_agents, features) → (n_steps * n_agents, features)."""
        arr = np.array(x)
        return torch.tensor(arr.reshape(-1, *arr.shape[2:]), dtype=torch.float32)

    def compute_distance(self, points: np.ndarray) -> float:
        """
        Compute the mean MSE between target and predictor representations.

        Parameters
        ----------
        points : np.ndarray (n_steps, n_agents, features)
        """
        x = self._reshape_data(points)
        with torch.no_grad():
            target_out = self.target_network(x)
            predictor_out = self.predictor_network(x)
        return float(self.loss_fn(predictor_out, target_out).item())

    def update(self, episode_data: TrajectoryInformation):
        """
        Train the predictor network to match the target on episode features.
        """
        x = self._reshape_data(episode_data.features)
        with torch.no_grad():
            targets = self.target_network(x)

        dataset_size = x.shape[0]
        for _ in range(self.n_epochs):
            idx = torch.randperm(dataset_size)[: self.batch_size]
            x_batch, t_batch = x[idx], targets[idx]
            pred = self.predictor_network(x_batch)
            loss = self.loss_fn(pred, t_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_reward(self, episode_data: TrajectoryInformation) -> np.ndarray:
        """
        Return the intrinsic reward for the last step of the episode.

        Parameters
        ----------
        episode_data : TrajectoryInformation

        Returns
        -------
        reward : float
        """
        reward = self.compute_distance(episode_data.features[-1:])
        if self.clip_rewards is not None:
            reward = float(np.clip(reward, *self.clip_rewards))
        return reward
