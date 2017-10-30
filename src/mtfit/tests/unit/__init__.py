import unittest

from mtfit.utilities.unittest_utils import run_tests as _run_tests
from mtfit.utilities.unittest_utils import debug_tests as _debug_tests
from mtfit.tests.unit.utilities import test_suite as utilities_test_suite
from mtfit.tests.unit.extensions import test_suite as extensions_test_suite
from mtfit.tests.unit.algorithms import test_suite as algorithms_test_suite
from mtfit.tests.unit.probability import test_suite as probability_test_suite
from mtfit.tests.unit.test_sampling import test_suite as sampling_test_suite
from mtfit.tests.unit.test_inversion import test_suite as inversion_test_suite
from mtfit.tests.unit.test_run import test_suite as run_test_suite
from mtfit.tests.unit.convert import test_suite as convert_test_suite
from mtfit.tests.unit.plot import test_suite as plot_test_suite

__all__ = ['run_tests']


def test_suite(verbosity=2):
    return unittest.TestSuite([utilities_test_suite(verbosity),
                               algorithms_test_suite(verbosity),
                               probability_test_suite(verbosity),
                               sampling_test_suite(verbosity),
                               inversion_test_suite(verbosity),
                               run_test_suite(verbosity),
                               convert_test_suite(verbosity),
                               plot_test_suite(verbosity),
                               extensions_test_suite(verbosity),
                               ])


def run_tests(verbosity=2):
    """Run the modules unit tests"""
    spacer = "----------------------------------------------------------------------"
    if verbosity > 1:
        print('Running src/utilities/ Tests')
        print(spacer)
        test_result = _run_tests(utilities_test_suite(verbosity), verbosity)
        print('\n'+spacer)
        print('Running algorithms Tests')
        print(spacer)
        test_result = _run_tests(algorithms_test_suite(verbosity), verbosity, test_result=test_result)
        print('\n'+spacer)
        print('Running probability Tests')
        print(spacer)
        test_result = _run_tests(probability_test_suite(verbosity), verbosity, test_result=test_result)
        print('\n'+spacer)
        print('Running sampling.py Tests')
        print(spacer)
        test_result = _run_tests(sampling_test_suite(verbosity), verbosity, test_result=test_result)
        print('\n'+spacer)
        print('Running inversion.py Tests')
        print(spacer)
        test_result = _run_tests(inversion_test_suite(verbosity), verbosity, test_result=test_result)
        print('\n'+spacer)
        print('Running run.py Tests')
        print(spacer)
        test_result = _run_tests(run_test_suite(verbosity), verbosity, test_result=test_result)
        print('\n'+spacer)
        print('Running convert Tests')
        print(spacer)
        test_result = _run_tests(convert_test_suite(verbosity), verbosity, test_result=test_result)
        print('\n'+spacer)
        print('Running plot Tests')
        print(spacer)
        test_result = _run_tests(plot_test_suite(verbosity), verbosity, test_result=test_result)
        print('\n'+spacer)
        print('Running extensions Tests')
        print(spacer)
        test_result = _run_tests(extensions_test_suite(verbosity), verbosity, test_result=test_result)
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
