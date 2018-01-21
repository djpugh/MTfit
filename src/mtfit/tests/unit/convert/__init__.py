"""convert
*************
Provides unit tests for the convert module
"""

import unittest

from MTfit.tests.unit.convert.test_moment_tensor_conversion import test_suite as moment_tensor_conversion_test_suite
from MTfit.tests.unit.convert.test_cmoment_tensor_conversion import test_suite as cmoment_tensor_conversion_test_suite

from MTfit.utilities.unittest_utils import run_tests as _run_tests
from MTfit.utilities.unittest_utils import debug_tests as _debug_tests

__all__ = ['run_tests']


def test_suite(verbosity=2):
    return unittest.TestSuite([moment_tensor_conversion_test_suite(verbosity),
                               cmoment_tensor_conversion_test_suite(verbosity)
                               ])


def run_tests(verbosity=2):
    """Run the modules unit tests"""
    spacer = "----------------------------------------------------------------------"
    if verbosity > 1:
        print('Running convert/moment_tensor_conversion.py Tests')
        print(spacer)
        test_result = _run_tests(moment_tensor_conversion_test_suite(verbosity), verbosity)
        print(spacer)
        print('Running convert/cmoment_tensor_conversion.py Tests')
        print(spacer)
        test_result = _run_tests(cmoment_tensor_conversion_test_suite(verbosity), verbosity, test_result=test_result)
    else:
        print('Running unit tests')
        print(spacer)
        test_result = _run_tests(test_suite(verbosity), verbosity)
    return test_result


def debug_tests(verbosity=3):
    _debug_tests(test_suite(verbosity))


if __name__ == "__main__":
    run_tests(1)
