"""algorithms
*************
Provides unit tests for the algorithms module
"""

import unittest

from mtfit.tests.unit.algorithms.test_base import test_suite as base_test_suite
from mtfit.tests.unit.algorithms.test_monte_carlo import test_suite as monte_carlo_test_suite
from mtfit.tests.unit.algorithms.test_markov_chain_monte_carlo import test_suite as markov_chain_monte_carlo_test_suite


from mtfit.utilities.unittest_utils import run_tests as _run_tests
from mtfit.utilities.unittest_utils import debug_tests as _debug_tests

__all__ = ['run_tests']


def test_suite(verbosity=2):
    return unittest.TestSuite([base_test_suite(verbosity),
                               monte_carlo_test_suite(verbosity),
                               markov_chain_monte_carlo_test_suite(verbosity)])


def run_tests(verbosity=2):
    """Run the modules unit tests"""
    spacer = "----------------------------------------------------------------------"
    if verbosity > 1:
        print('Running mtfit/algorithms/base.py Tests')
        print(spacer)
        _run_tests(base_test_suite(verbosity), verbosity)
        print(spacer)
        print('Running mtfit/algorithms/monte_carlo.py Tests')
        print(spacer)
        _run_tests(monte_carlo_test_suite(verbosity), verbosity)
        print(spacer)
        print('Running mtfit/algorithms/markov_chain_monte_carlo.py Tests')
        print(spacer)
        _run_tests(markov_chain_monte_carlo_test_suite(verbosity), verbosity)
    else:
        print('Running unit tests')
        print(spacer)
        _run_tests(test_suite(verbosity), verbosity)


def debug_tests(verbosity=3):
    _debug_tests(test_suite(verbosity))


if __name__ == "__main__":
    run_tests(1)
