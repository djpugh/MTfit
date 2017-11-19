import unittest
from types import MethodType
import sys

import numpy as np

from mtfit.utilities.unittest_utils import run_tests as _run_tests
from mtfit.utilities.unittest_utils import debug_tests as _debug_tests

VERBOSITY = 2


class BaseAlgorithmTestCase(unittest.TestCase):

    def setUp(self, **kwargs):
        from mtfit.algorithms.base import BaseAlgorithm
        self.base_algorithm = BaseAlgorithm(**kwargs)

    def tearDown(self):
        del self.base_algorithm

    def test_initialise(self):
        self.assertFalse(self.base_algorithm.initialise()[1])

    def test_iterate(self):
        self.assertTrue(self.base_algorithm.iterate({})[1])

    def test___init__(self):

        def mock_get_sampling_model(self, kwargs, file_sample, file_safe):
            self.get_sampling_model_called = True
            self.get_sampling_model_args = (kwargs, file_sample, file_safe)

        self.base_algorithm.get_sampling_model = MethodType(mock_get_sampling_model, self.base_algorithm)
        self.assertFalse(self.base_algorithm.dc)
        self.assertEqual(self.base_algorithm.number_samples, 10000)
        self.assertFalse(self.base_algorithm.mcmc)
        self.assertFalse(self.base_algorithm.basic_cdc)
        self.assertFalse(self.base_algorithm.quality_check)
        self.assertFalse(self.base_algorithm.generate)
        self.assertEqual(self.base_algorithm.number_events, 1)
        self.assertFalse(self.base_algorithm._model)
        self.base_algorithm.__init__()
        # Test that mock_get_sampling has been called.
        self.assertTrue(self.base_algorithm.get_sampling_model_called)
        self.assertEqual(self.base_algorithm.get_sampling_model_args[0], {})
        self.assertFalse(self.base_algorithm.get_sampling_model_args[1])
        self.assertTrue(self.base_algorithm.get_sampling_model_args[2])

    def test_clvd_sampling(self):
        self.tearDown()
        self.setUp(sample_distribution='clvd')
        if sys.version_info.major > 2:
            self.assertEqual(self.base_algorithm.random_model.__name__, self.base_algorithm.random_clvd.__name__)
        else:
            self.assertEqual(self.base_algorithm.random_model, self.base_algorithm.random_clvd)

    def test_random_mt(self):
        self.assertTrue(self.base_algorithm.random_mt().shape,
                        (6, self.base_algorithm.number_samples))

    def test_random_dc(self):
        self.assertTrue(self.base_algorithm.random_dc().shape,
                        (6, self.base_algorithm.number_samples))

    def test_random_clvd(self):
        self.assertTrue(self.base_algorithm.random_clvd().shape,
                        (6, self.base_algorithm.number_samples))

    def test_random_type(self):
        self.base_algorithm.number_samples = 1
        self.assertAlmostEqual(
            self.base_algorithm.random_type(np.array([[1], [1], [1]]))[0, 0], 1/np.sqrt(3))

    def test_random_orthogonal_eigenvectors(self):
        self.base_algorithm.number_samples = 1
        [a, b, c] = self.base_algorithm.random_orthogonal_eigenvectors()
        self.assertAlmostEqual(c[0, 0], np.cross(a.transpose(), b.transpose()).transpose()[0, 0])
        self.assertAlmostEqual(c[1, 0], np.cross(a.transpose(), b.transpose()).transpose()[1, 0])
        self.assertAlmostEqual(c[2, 0], np.cross(a.transpose(), b.transpose()).transpose()[2, 0])

    def test_eigenvectors_mt_2_mt6(self):
        self.base_algorithm.number_samples = 1
        self.assertAlmostEqual(self.base_algorithm.eigenvectors_mt_2_mt6(np.array(
            [[1], [1], [1]]), *self.base_algorithm.random_orthogonal_eigenvectors())[0, 0], 1/np.sqrt(3))


def test_suite(verbosity=2):
    """Return test suite"""
    global VERBOSITY
    VERBOSITY = verbosity
    suite = [unittest.TestLoader().loadTestsFromTestCase(BaseAlgorithmTestCase),
             ]
    suite = unittest.TestSuite(suite)
    return suite


def run_tests(verbosity=2):
    """Run tests"""
    _run_tests(test_suite(verbosity), verbosity)


def debug_tests(verbosity=2):
    """Run tests with debugging on errors"""
    _debug_tests(test_suite(verbosity))


if __name__ == "__main__":
    # Run tests
    run_tests(verbosity=2)
