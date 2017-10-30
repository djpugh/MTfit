"""
plot
****
Provides unit tests for the plot module
"""

import unittest

from mtfit.tests.unit.plot.test_core import test_suite as core_test_suite
from mtfit.tests.unit.plot.test_plot_classes import test_suite as plot_classes_test_suite
from mtfit.tests.unit.plot.test_spherical_projection import test_suite as spherical_projection_test_suite


from mtfit.utilities.unittest_utils import run_tests as _run_tests
from mtfit.utilities.unittest_utils import debug_tests as _debug_tests


__all__ = ['run_tests']


def test_suite(verbosity=2):
    return unittest.TestSuite([core_test_suite(verbosity),
                               plot_classes_test_suite(verbosity),
                               spherical_projection_test_suite(verbosity)])


def run_tests(verbosity=2):
    """Run the modules unit tests"""
    spacer = "----------------------------------------------------------------------"
    if verbosity > 1:
        print('Running src/plot/core.py Tests')
        print(spacer)
        test_result = _run_tests(core_test_suite(verbosity), verbosity)
        print(spacer)
        print('Running src/plot/plot_classes.py Tests')
        print(spacer)
        test_result = _run_tests(plot_classes_test_suite(verbosity), verbosity, test_result=test_result)
        print(spacer)
        print('Running src/plot/spherical_projection.py Tests')
        print(spacer)
        test_result = _run_tests(spherical_projection_test_suite(verbosity), verbosity, test_result=test_result)
    else:
        print('Running unit tests')
        print(spacer)
        test_result = _run_tests(test_suite(verbosity), verbosity)
    return test_result


def debug_tests(verbosity=3):
    _debug_tests(test_suite(verbosity))


if __name__ == "__main__":
    run_tests(1)
