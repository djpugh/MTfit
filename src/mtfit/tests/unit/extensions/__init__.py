"""
extensions
*****

Tests for extensions
"""

import unittest

from mtfit.utilities.unittest_utils import run_tests as _run_tests
from mtfit.utilities.unittest_utils import debug_tests as _debug_tests
from mtfit.tests.unit.extensions.test_scatangle import test_suite as scatangle_test_suite

__all__ = ['run_tests']


def test_suite(verbosity=2):
    return unittest.TestSuite([scatangle_test_suite(verbosity)])


def run_tests(verbosity=2):
    """Run the modules unit tests"""
    spacer = "----------------------------------------------------------------------"
    if verbosity > 1:
        print('Running extensions/scatangle.py Tests')
        print(spacer)
        test_result = _run_tests(scatangle_test_suite(verbosity), verbosity)
    else:
        print('Running unit tests')
        print(spacer)
        test_result = _run_tests(test_suite(verbosity), verbosity)
    return test_result


def debug_tests(verbosity=2):
    """Runs tests with debugging on errors"""
    _debug_tests(test_suite(verbosity))


if __name__ == "__main__":
    run_tests(1)
