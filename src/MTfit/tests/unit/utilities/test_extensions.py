"""
test_extensions.py
******************

Tests for src/utils/extensions.py
"""

import unittest

from MTfit.utilities.unittest_utils import TestCase


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
