import numpy as np

from swarmrl.value_functions.generalized_advantage_estimate import GAE


def to_np(x):
    return x.detach().numpy()


class TestGAE:
    def test_gae(self):
        gae = GAE(gamma=1, lambda_=1)
        rewards = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        raw_advantages = np.array([4.0, 2.0, 0.0, -2.0, -4.0])
        expected_returns = raw_advantages + values
        expected_advantages = (raw_advantages - np.mean(raw_advantages)) / (
            np.std(raw_advantages) + np.finfo(np.float32).eps.item()
        )

        advantages, returns = gae(rewards, values)

        np.testing.assert_allclose(
            to_np(advantages), expected_advantages, rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            to_np(returns), expected_returns, rtol=1e-4, atol=1e-4
        )
