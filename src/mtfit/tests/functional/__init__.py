import unittest
import sys

from MTINV.src.utils.unittest_utils import run_tests as _run_tests
from MTINV.src.utils.unittest_utils import debug_tests as _debug_tests

__all__ = ['run_tests']


def test_suite(verbosity=2):
    return unittest.TestSuite([])


def run_tests(verbosity=2):
    """Run the modules unit tests"""
    spacer = "----------------------------------------------------------------------"
    if verbosity > 1:
        pass
    else:
        import unittest
        print('Running unit tests')
        print(spacer)
        _run_tests(test_suite(verbosity), verbosity)

def debug_tests(verbosity=2):
    """Runs tests with debugging on errors"""
    _debug_tests(test_suite(verbosity))


if __name__ == "__main__":
    run_tests(2)