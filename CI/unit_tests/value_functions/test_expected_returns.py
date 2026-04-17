"""
Test the expected return module.
"""

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from swarmrl.value_functions.expected_returns import ExpectedReturns


def to_np(x):
    return x.detach().numpy()


class TestExpectedReturns:
    """
    Test suite for the expected returns module.
    """

    def test_unstandardized_returns(self):
        """
        Test that unstandardized returns work correctly.
        """
        true_values = np.array([[6, 15], [5, 11], [3, 6]])

        value_function = ExpectedReturns(gamma=1.0, standardize=False)

        rewards = np.array([[1, 4], [2, 5], [3, 6]])

        expected_returns = to_np(value_function(rewards))

        assert_array_equal(expected_returns, true_values)

    def test_standardized_returns(self):
        """
        Test that the standardization of the return is correct.
        """
        value_function = ExpectedReturns(gamma=0.79, standardize=True)

        true_mean = np.array([0.0, 0.0])
        true_std = np.array([1.0, 1.0])

        rewards = np.array([[1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10]])

        expected_returns = to_np(value_function(rewards))

        mean_vector = np.mean(expected_returns, axis=0)
        std_vector = np.std(expected_returns, axis=0)

        assert_array_almost_equal(mean_vector, true_mean, decimal=6)
        assert_array_almost_equal(std_vector, true_std, decimal=6)
