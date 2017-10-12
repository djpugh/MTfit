import unittest

from mtfit.utilities.unittest_utils import run_tests as _run_tests
from mtfit.utilities.unittest_utils import debug_tests as _debug_tests


class SphericalProjectionTestCase(unittest.TestCase):

    def test_equal_area(self):
        raise NotImplementedError()

    def test_equal_angle(self):
        raise NotImplementedError()

    def test__project(self):
        raise NotImplementedError()


def test_suite(verbosity=2):
    """Returns test suite"""
    global VERBOSITY
    VERBOSITY = verbosity
    suite = [unittest.TestLoader().loadTestsFromTestCase(SphericalProjectionTestCase),
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
