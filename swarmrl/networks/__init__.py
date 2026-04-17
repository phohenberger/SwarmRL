"""
Helper module to instantiate several modules.
"""

from swarmrl.networks.flax_network import FlaxModel
from swarmrl.networks.gnn_network import GNNModel
from swarmrl.networks.torch_network import TorchModel

__all__ = [FlaxModel.__name__, TorchModel.__name__, GNNModel.__name__]
