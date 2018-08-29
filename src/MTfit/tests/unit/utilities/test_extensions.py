"""
test_extensions.py
******************

Tests for src/utils/extensions.py
"""

import unittest

from MTfit.utilities.unittest_utils import TestCase
from MTfit.utilities.unittest_utils import run_tests as _run_tests
from MTfit.utilities.unittest_utils import debug_tests as _debug_tests


class ExtensionsTestCase(TestCase):

    def test_evaluate_extensions(self):
        from MTfit.utilities.extensions import evaluate_extensions
        extensions = evaluate_extensions('MTfit.cmd_defaults')
        self.assertTrue('bin_size' in extensions[0][0].keys())
        self.assertTrue('bin_scatangle' in extensions[0][0].keys())

    def test_get_extensions(self):
        from MTfit.utilities.extensions import get_extensions
        (name, extensions) = get_extensions('MTfit.parsers')
        self.assertTrue(any(['.hyp' in u for u in name]))
        self.assertTrue(any(['.csv' in u for u in name]))


def test_suite(verbosity=2):
    global VERBOSITY
    VERBOSITY = verbosity
    suite = [unittest.TestLoader().loadTestsFromTestCase(ExtensionsTestCase), ]
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
