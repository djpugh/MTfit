"""
test_unittest_utils.py
**********************

Tests for src/utils/unittest_utils.py
"""

import numpy as np

from MTfit.utilities.unittest_utils import TestCase


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
