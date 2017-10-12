"""
test_unittest_utils.py
**********************

Tests for src/utils/unittest_utils.py
"""

import unittest

import numpy as np

from mtfit.utilities.unittest_utils import TestCase
from mtfit.utilities.unittest_utils import run_tests as _run_tests
from mtfit.utilities.unittest_utils import debug_tests as _debug_tests


class TestCaseTestCase(TestCase):

    def test_assertAlmostEquals(self):
        self.assertAlmostEquals(1.000000, 0.9999999999)
        self.assertAlmostEquals(np.array([1.000000]), np.array([0.9999999999]))
        self.assertAlmostEquals(np.array([1.000000]), [0.9999999999])
        self.assertAlmostEquals({'x': 1.000000}, {'x': 0.9999999999})

    def test_assertEquals(self):
        self.assertEquals(1.000000, 1.000000)
        self.assertEquals(np.array([1.000000]), np.array([1.000000]))
        self.assertEquals(np.array([1.000000]), [1.000000])
        self.assertEquals({'x': 1.000000}, {'x': 1.000000})


def test_suite(verbosity=2):
    global VERBOSITY
    VERBOSITY = verbosity
    suite = [
        unittest.TestLoader().loadTestsFromTestCase(TestCaseTestCase),
    ]
    suite = unittest.TestSuite(suite)
    return suite


def run_tests(verbosity=2):
    """Run tests"""
    _run_tests(test_suite(verbosity), verbosity)


def debug_tests(verbosity=2):
    """Runs tests with debugging on errors"""
    _debug_tests(test_suite(verbosity))

if __name__ == "__main__":
    # Run tests
    run_tests(verbosity=2)
