"""
PyTorch model for reinforcement learning.
"""

import os
from abc import ABC
from typing import List, Type

import numpy as onp
import torch
import torch.nn as nn
from loguru import logger

from swarmrl.exploration_policies.exploration_policy import ExplorationPolicy
from swarmrl.exploration_policies.random_exploration import RandomExploration
from swarmrl.networks.network import Network
from swarmrl.sampling_strategies.gumbel_distribution import GumbelDistribution
from swarmrl.sampling_strategies.sampling_strategy import SamplingStrategy


class TorchModel(Network, ABC):
    """
    PyTorch actor-critic network wrapper.

    The wrapped ``torch_model`` must return a tuple ``(logits, values)`` with shapes
    ``(batch, n_actions)`` and ``(batch, 1)`` respectively.

    Attributes
    ----------
    epoch_count : int
        Current training epoch, used when saving checkpoints.
    """

    def __init__(
        self,
        torch_model: nn.Module,
        input_shape: tuple,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict = None,
        exploration_policy: ExplorationPolicy = RandomExploration(probability=0.0),
        sampling_strategy: SamplingStrategy = GumbelDistribution(),
        deployment_mode: bool = False,
    ):
        """
        Parameters
        ----------
        torch_model : nn.Module
            Actor-critic network returning (logits, values).
        input_shape : tuple
            Shape of a single observation (used for documentation/validation).
        optimizer_class : type
            Optimizer class from torch.optim. Defaults to Adam.
        optimizer_kwargs : dict
            Keyword arguments forwarded to the optimizer constructor.
        deployment_mode : bool
            If True, no optimizer is created and no training can be performed.
        """
        self.model = torch_model
        self.input_shape = input_shape
        self.sampling_strategy = sampling_strategy
        self.model_state = None

        if not deployment_mode:
            kwargs = optimizer_kwargs or {}
            self.optimizer = optimizer_class(self.model.parameters(), **kwargs)
            self.exploration_policy = exploration_policy
            self.epoch_count = 0

    def reinitialize_network(self):
        """Re-initialize network weights using each layer's default reset."""
        for layer in self.model.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def update_model(self, loss: torch.Tensor):
        """
        Backpropagate ``loss`` and step the optimizer.

        Parameters
        ----------
        loss : torch.Tensor
            Scalar loss tensor with a grad_fn attached.
        """
        logger.debug(f"loss={loss.item():.6f}")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epoch_count += 1

    def compute_action(self, observables: List) -> tuple:
        """
        Compute actions for all colloids from their observables.

        Parameters
        ----------
        observables : List (n_agents, observable_dimension)
            Observable for each colloid.

        Returns
        -------
        indices : np.ndarray (n_agents,)
            Chosen action index per agent.
        log_probs : np.ndarray (n_agents,)
            Log-probability of the chosen action.
        """
        obs = torch.tensor(onp.array(observables), dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(obs)  # (n_agents, n_actions)

        logger.debug(f"logits={logits}")

        indices = self.sampling_strategy(logits)
        eps = 1e-8
        log_probs = torch.log(torch.softmax(logits, dim=-1) + eps)

        indices = self.exploration_policy(
            indices, logits.shape[-1], onp.random.randint(8759865)
        )

        chosen_log_probs = log_probs[
            torch.arange(len(indices)), indices.long()
        ]
        return indices, chosen_log_probs.detach().cpu().numpy()

    def __call__(self, features: torch.Tensor) -> tuple:
        """
        Forward pass over a full episode batch.

        Parameters
        ----------
        features : torch.Tensor (n_steps, n_agents, feature_dim)

        Returns
        -------
        logits : torch.Tensor (n_steps, n_agents, n_actions)
        values : torch.Tensor (n_steps, n_agents, 1)
        """
        self.model.train()
        n_steps, n_agents, feat_dim = features.shape
        flat = features.reshape(n_steps * n_agents, feat_dim)
        logits, values = self.model(flat)
        n_actions = logits.shape[-1]
        return (
            logits.reshape(n_steps, n_agents, n_actions),
            values.reshape(n_steps, n_agents, 1),
        )

    def export_model(self, filename: str = "model", directory: str = "Models"):
        """Save model and optimizer state to ``directory/filename.pt``."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename + ".pt")
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch_count,
            },
            path,
        )

    def restore_model_state(self, filename: str, directory: str):
        """Load model and optimizer state from ``directory/filename.pt``."""
        path = os.path.join(directory, filename + ".pt")
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epoch_count = checkpoint.get("epoch", 0)
