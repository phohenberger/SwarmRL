"""
Configurations for the Random Network Distillation (RND) intrinsic reward.

Notes
-----
https://arxiv.org/abs/1810.12894
"""

from dataclasses import dataclass, field
from typing import Optional, Type

import torch
import torch.nn as nn


class RNDArchitecture(nn.Module):
    """Default 3-layer MLP for the RND target and predictor networks."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class RNDConfig:
    """
    Configuration for the RND intrinsic reward.

    Parameters
    ----------
    input_shape : tuple
        Shape of a single observation (used to size the network input).
    hidden_dim : int
        Hidden layer size for the default RND architecture.
    optimizer_class : type
        Optimizer class for the predictor network.
    optimizer_kwargs : dict
        Keyword arguments forwarded to the optimizer.
    n_epochs : int
        Number of gradient steps when updating the predictor.
    batch_size : int
        Mini-batch size for predictor training.
    clip_rewards : tuple or None
        (min, max) reward clipping range. None disables clipping.
    training_kwargs : dict
        Extra keyword arguments passed to the training loop (reserved).
    """

    input_shape: tuple
    hidden_dim: int = 32
    optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam
    optimizer_kwargs: dict = field(default_factory=lambda: {"lr": 1e-3})
    n_epochs: int = 100
    batch_size: int = 8
    clip_rewards: Optional[tuple] = (-5.0, 5.0)
    training_kwargs: Optional[dict] = field(default_factory=dict)
