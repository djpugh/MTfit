"""unittest_utils
******************
Provides test functions for running and debugging unit tests.
"""

# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.

import unittest
import sys
import importlib
import traceback

import numpy as np


def get_extension_skip_if_args(module):
    reason = 'No C extension available'
    try:
        c_extension = importlib.import_module(module)
    except ImportError:
        c_extension = False
    except Exception:
        reason += '\n=======\nException loading C extension = \n{}\n=======\n'.format(traceback.format_exc())
        c_extension = False
    return (not c_extension, reason)


# class PythonOnly(object):

#     def __init__(self, *args):
#         self.c_extensions = []
#         self.original_values = []
#         for arg in args:
#             self.c_extensions.append(arg)
#             self.original_values = globals()[arg]

#     def __enter__(self, *args, **kwargs):
#         for extension in self.c_extensions:
#             globals()[extension] = False

#     def __exit__(self, *args, **kwargs):
#         for i, extension in enumerate(self.c_extensions):
#             globals()[extension] = self.original_values[i]


class TestCase(unittest.TestCase):

    def assertAlmostEqual(self, first, second, places=None, msg=None, delta=None):

        if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
            # Handle array and vectors
            if isinstance(second, (list, float, int)):
                second = np.array(second)
            if isinstance(first, (list, float, int)):
                first = np.array(first)
            if len([u for u in second.shape if u != 1]) == len([u for u in first.shape if u != 1]):
                if places is not None:
                    np.testing.assert_array_almost_equal(np.array(first).squeeze(), np.array(second).squeeze(), places)
                else:
                    np.testing.assert_array_almost_equal(np.array(first).squeeze(), np.array(second).squeeze())
            else:
                if places is not None:
                    np.testing.assert_array_almost_equal(first, second, places)
                else:
                    np.testing.assert_array_almost_equal(first, second)
            return
        elif isinstance(first, dict) and isinstance(second, dict):
            self.assertEqual(sorted(first.keys()), sorted(second.keys()), 'Dictionary keys do not match')
            for key in first.keys():
                self.assertAlmostEqual(first[key], second[key])
            return
        elif isinstance(first, list) and isinstance(second, list):
            super(TestCase, self).assertAlmostEqual(set(first), set(second), msg, delta)
        else:
            super(TestCase, self).assertAlmostEqual(first, second, places, msg, delta)

    def assertAlmostEquals(self, *args, **kwargs):
        self.assertAlmostEqual(*args, **kwargs)

    def assertEqual(self, first, second, msg=None):
        if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
            # Handle matrix and vector

            np.testing.assert_array_equal(first, second)
            return
        elif isinstance(first, dict) and isinstance(second, dict):
            self.assertEqual(
                sorted(first.keys()), sorted(second.keys()), 'Dictionary keys do not match')
            for key in first.keys():
                self.assertEqual(first[key], second[key])
            return
        elif isinstance(first, list) and isinstance(second, list):
            super(TestCase, self).assertEqual(first, second, msg)

        else:
            super(TestCase, self).assertAlmostEqual(first, second, msg)

    def assertEquals(self, *args, **kwargs):
        self.assertEqual(*args, **kwargs)

    def assertVectorEquals(self, first, second, *args):
        try:
            first_norm = np.sqrt(np.sum(np.multiply(first, first)))
            second_norm = np.sqrt(np.sum(np.multiply(second, second)))
            return self.assertAlmostEquals(first/first_norm, second/second_norm, *args)
        except AssertionError as e1:
            try:
                return self.assertAlmostEquals(-first/first_norm, second/second_norm, *args)
            except AssertionError as e2:
                if sys.version_info.major <= 2 and sys.version_info.minor <= 6:
                    raise AssertionError(e1.message+' or '+e2.message)
                else:
                    raise AssertionError('{} or {}'.format(e1.args, e2.args))
