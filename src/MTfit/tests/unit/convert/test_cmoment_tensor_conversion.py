"""
test_cmoment_tensor_conversion.py
*****************

Tests for src/convert/cmoment_tensor_conversion.py
"""

import unittest

try:
    from MTfit.convert.cmoment_tensor_conversion import cMomentTensorConvertTestCase
except ImportError:
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
