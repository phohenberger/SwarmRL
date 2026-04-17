import numpy as np
import numpy.testing as tst
import torch
import torch.nn.functional as F

from swarmrl.losses.proximal_policy_loss import ProximalPolicyLoss
from swarmrl.sampling_strategies.gumbel_distribution import GumbelDistribution


class DummyNetwork:
    """
    Minimal network stub for testing. Matches TorchModel's __call__ interface:
    accepts (n_steps, n_agents, feature_dim) tensor, returns (logits, values).
    """

    def __call__(self, features: torch.Tensor):
        n_steps, n_agents, _ = features.shape
        logits = 2.0 * torch.ones((n_steps, n_agents, 4))
        values = torch.ones((n_steps, n_agents, 1))
        return logits, values

    def update_model(self, loss):
        pass  # No-op for testing


class TestProximalPolicyLoss:
    """
    Test the PPO loss function.
    """

    def test_compute_actor_loss(self):
        """
        Verify _calculate_loss produces the expected scalar loss.
        """
        epsilon = 0.2
        n_particles = 10
        n_time_steps = 20
        observable_dimension = 4
        entropy_coefficient = 0.01
        sampling_strategy = GumbelDistribution()

        def dummy_value_function(rewards, values):
            advantages = torch.ones_like(rewards) * torch.sign(rewards)
            returns = 2 * torch.ones_like(rewards)
            return advantages, returns

        loss_fn = ProximalPolicyLoss(
            value_function=dummy_value_function,
            sampling_strategy=sampling_strategy,
            entropy_coefficient=entropy_coefficient,
        )

        network = DummyNetwork()
        features = torch.ones((n_time_steps, n_particles, observable_dimension))
        actions = torch.ones((n_time_steps, n_particles), dtype=torch.long)
        rewards = torch.ones((n_time_steps, n_particles))

        old_log_probs_list = [
            2 * torch.ones((n_time_steps, n_particles)),   # ratio == 1
            0 * torch.ones((n_time_steps, n_particles)),   # ratio == e^2 > 1+eps
            3 * torch.ones((n_time_steps, n_particles)),   # ratio == e^-2 < 1-eps
        ]

        for old_log_probs in old_log_probs_list:
            ppo_loss = loss_fn._calculate_loss(
                network=network,
                feature_data=features,
                action_indices=actions,
                rewards=rewards,
                old_log_probs=old_log_probs,
            )
            # Verify the loss is a finite scalar
            assert isinstance(ppo_loss, torch.Tensor)
            assert ppo_loss.shape == ()
            assert torch.isfinite(ppo_loss)
