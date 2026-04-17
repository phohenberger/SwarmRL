"""
Graph Neural Network model for reinforcement learning.

Builds a torch_geometric Data graph from colloid observations before each
forward pass, enabling message-passing over the particle neighborhood.
"""

from abc import ABC
from typing import List, Type

import numpy as onp
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from swarmrl.exploration_policies.exploration_policy import ExplorationPolicy
from swarmrl.exploration_policies.random_exploration import RandomExploration
from swarmrl.networks.torch_network import TorchModel
from swarmrl.sampling_strategies.gumbel_distribution import GumbelDistribution
from swarmrl.sampling_strategies.sampling_strategy import SamplingStrategy


class GNNModel(TorchModel, ABC):
    """
    Actor-critic model that operates on a fully-connected graph of colloids.

    Each colloid becomes a node whose features are its observable vector.
    Edges connect every pair of colloids (fully connected graph). The
    ``gnn_model`` receives a ``torch_geometric.data.Batch`` and must return
    ``(logits, values)`` with shapes ``(n_nodes, n_actions)`` and
    ``(n_nodes, 1)`` respectively.

    Parameters
    ----------
    gnn_model : nn.Module
        PyG-compatible model. Forward signature: ``forward(data: Batch)``.
    input_shape : tuple
        Shape of a single node feature vector.
    """

    def __init__(
        self,
        gnn_model: nn.Module,
        input_shape: tuple,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict = None,
        exploration_policy: ExplorationPolicy = RandomExploration(probability=0.0),
        sampling_strategy: SamplingStrategy = GumbelDistribution(),
        deployment_mode: bool = False,
    ):
        super().__init__(
            torch_model=gnn_model,
            input_shape=input_shape,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            exploration_policy=exploration_policy,
            sampling_strategy=sampling_strategy,
            deployment_mode=deployment_mode,
        )

    @staticmethod
    def _build_graph(obs: torch.Tensor) -> Data:
        """Build a fully-connected graph from a node feature matrix."""
        n = obs.shape[0]
        src = torch.arange(n).repeat_interleave(n)
        dst = torch.arange(n).repeat(n)
        mask = src != dst
        edge_index = torch.stack([src[mask], dst[mask]], dim=0)
        return Data(x=obs, edge_index=edge_index)

    def compute_action(self, observables: List) -> tuple:
        obs = torch.tensor(onp.array(observables), dtype=torch.float32)
        graph = self._build_graph(obs)

        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(graph)

        indices = self.sampling_strategy(logits)
        eps = 1e-8
        log_probs = torch.log(torch.softmax(logits, dim=-1) + eps)
        indices = self.exploration_policy(
            indices, logits.shape[-1], onp.random.randint(8759865)
        )
        chosen_log_probs = log_probs[torch.arange(len(indices)), indices.long()]
        return indices, chosen_log_probs.detach().cpu().numpy()

    def __call__(self, features: torch.Tensor) -> tuple:
        """
        Forward pass over an episode batch, building one graph per time step.

        Parameters
        ----------
        features : torch.Tensor (n_steps, n_agents, feature_dim)
        """
        self.model.train()
        n_steps, n_agents, _ = features.shape
        graphs = [self._build_graph(features[t]) for t in range(n_steps)]
        batch = Batch.from_data_list(graphs)
        logits, values = self.model(batch)
        n_actions = logits.shape[-1]
        return (
            logits.reshape(n_steps, n_agents, n_actions),
            values.reshape(n_steps, n_agents, 1),
        )
