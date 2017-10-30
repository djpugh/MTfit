"""
utils
*****

Tests for src/utilities
"""

import unittest

from mtfit.utilities.unittest_utils import run_tests as _run_tests
from mtfit.utilities.unittest_utils import debug_tests as _debug_tests
from mtfit.tests.unit.utilities.test_unittest_utils import test_suite as unittest_utils_test_suite
from mtfit.tests.unit.utilities.test_argparser import test_suite as argparser_test_suite
from mtfit.tests.unit.utilities.test_file_io import test_suite as file_io_test_suite
from mtfit.tests.unit.utilities.test_multiprocessing_helper import test_suite as multiprocessing_helper_test_suite
from mtfit.tests.unit.utilities.test_extensions import test_suite as extensions_test_suite

__all__ = ['run_tests']


def test_suite(verbosity=2):
    return unittest.TestSuite([argparser_test_suite(verbosity),
                               extensions_test_suite(verbosity),
                               file_io_test_suite(verbosity),
                               multiprocessing_helper_test_suite(verbosity)
                               ])


def run_tests(verbosity=2):
    """Run the modules unit tests"""
    spacer = "----------------------------------------------------------------------"
    if verbosity > 1:
        print('Running utilities/unittest_utils.py Tests')
        print(spacer)
        test_result = _run_tests(unittest_utils_test_suite(verbosity), verbosity)
        print('\n'+spacer)
        print('Running utilities/argparser.py Tests')
        print(spacer)
        test_result = _run_tests(argparser_test_suite(verbosity), verbosity, test_result=test_result)
        print('\n'+spacer)
        print('Running utilities/exception_handler.py Tests')
        print(spacer)
        test_result = _run_tests(extensions_test_suite(verbosity), verbosity, test_result=test_result)
        print('\n'+spacer)
        print('Running utilities/file_io.py Tests')
        print(spacer)
        test_result = _run_tests(file_io_test_suite(verbosity), verbosity, test_result=test_result)
        print('\n'+spacer)
        print('Running utilities/multiprocessing_helper.py Tests')
        print(spacer)
        test_result = _run_tests(multiprocessing_helper_test_suite(verbosity), verbosity, test_result=test_result)
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
