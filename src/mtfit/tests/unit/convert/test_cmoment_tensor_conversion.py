"""
test_cmoment_tensor_conversion.py
*****************

Tests for src/convert/cmoment_tensor_conversion.py
"""

import unittest


from mtfit.utilities.unittest_utils import run_tests as _run_tests
from mtfit.utilities.unittest_utils import debug_tests as _debug_tests


class cMomentTensorConvertSkipTestCase(unittest.TestCase):

    def test_c_tape2mt6(self):
        raise unittest.SkipTest("No cmoment_tensor_conversion module")

    def test_c_E_tk(self):
        raise unittest.SkipTest("No cmoment_tensor_conversion module")

    def test_c_tk_uv(self):
        raise unittest.SkipTest("No cmoment_tensor_conversion module")

    def test_c_E_gd(self):
        raise unittest.SkipTest("No cmoment_tensor_conversion module")

    def test_c_N_sdr(self):
        raise unittest.SkipTest("No cmoment_tensor_conversion module")

    def test_c_TP_SDR(self):
        raise unittest.SkipTest("No cmoment_tensor_conversion module")

    def test_c_MT62TNPE(self):
        raise unittest.SkipTest("No cmoment_tensor_conversion module")

    def test_c_E2GD(self):
        raise unittest.SkipTest("No cmoment_tensor_conversion module")

    def test_c_TP2SDR(self):
        raise unittest.SkipTest("No cmoment_tensor_conversion module")

    def test_c_SDR2SDR(self):
        raise unittest.SkipTest("No cmoment_tensor_conversion module")


def test_suite(verbosity=2):
    try:
        from mtfit.convert.cmoment_tensor_conversion import cMomentTensorConvertTestCase
        suite = [unittest.TestLoader().loadTestsFromTestCase(cMomentTensorConvertTestCase)]
    except ImportError:
        suite = [unittest.TestLoader().loadTestsFromTestCase(cMomentTensorConvertSkipTestCase)]
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
