"""
test_moment_tensor_conversion.py
*****************

Tests for src/convert/moment_tensor_conversion.py
"""

import unittest
import sys

import numpy as np

from MTfit.convert import moment_tensor_conversion as mtc
from MTfit.convert import MT33_MT6
from MTfit.convert import MT6_MT33
from MTfit.convert import MT6_TNPE
from MTfit.convert import MT6_Tape
from MTfit.convert import MT33_TNPE
from MTfit.convert import MT33_SDR
from MTfit.convert import MT33_GD
from MTfit.convert import E_tk
from MTfit.convert import tk_uv
from MTfit.convert import E_uv
from MTfit.convert import E_GD
from MTfit.convert import GD_basic_cdc
from MTfit.convert import TNP_SDR
from MTfit.convert import TP_FP
from MTfit.convert import FP_SDR
from MTfit.convert import basic_cdc_GD
from MTfit.convert import GD_E
from MTfit.convert import SDR_TNP
from MTfit.convert import SDR_SDR
from MTfit.convert import FP_TNP
from MTfit.convert import SDSD_FP
from MTfit.convert import SDR_FP
from MTfit.convert import SDR_SDSD
from MTfit.convert import FP_SDSD
from MTfit.convert import Tape_MT33
from MTfit.convert import Tape_MT6
from MTfit.convert import Tape_TNPE
from MTfit.convert import normal_SD
from MTfit.convert import toa_vec
from MTfit.convert import output_convert
from MTfit.convert import isotropic_c
from MTfit.convert import MT6_biaxes
from MTfit.convert import MT6c_D6
from MTfit.convert import is_isotropic_c
from MTfit.convert import c21_cvoigt
from MTfit.convert import c_norm
from MTfit.convert import moment_tensor_conversion
from MTfit.utilities import C_EXTENSION_FALLBACK_LOG_MSG
from MTfit.utilities.unittest_utils import TestCase
from MTfit.utilities.unittest_utils import get_extension_skip_if_args

if sys.version_info >= (3, 3):
    from unittest import mock
else:
    import mock


C_EXTENSIONS = get_extension_skip_if_args('MTfit.convert.cmoment_tensor_conversion')


class PythonOnly(object):

    def __enter__(self, *args, **kwargs):
        self.cmoment_tensor_conversion = moment_tensor_conversion.cmoment_tensor_conversion
        moment_tensor_conversion.cmoment_tensor_conversion = False

    def __exit__(self, *args, **kwargs):
        moment_tensor_conversion.cmoment_tensor_conversion = self.cmoment_tensor_conversion


class MomentTensorConvertTestCase(TestCase):

    def setUp(self):
        self.DC6 = np.array([[0.], [0.], [0.], [1.], [0.], [0.]])
        self.MT6 = np.array([[1.], [0.], [-2.], [1.], [0.], [2.]])
        self.MT6norm = np.sqrt(np.sum(self.MT6*self.MT6))
        self.DC33 = np.array([[0, 1/np.sqrt(2), 0], [1/np.sqrt(2), 0, 0], [0, 0, 0]])
        self.MT33 = np.array([[1, 1/np.sqrt(2), 0], [1/np.sqrt(2), 0, 2/np.sqrt(2)], [0, 2/np.sqrt(2), -2]])
        self.MT33norm = np.sqrt(np.sum(self.MT33*self.MT33))
        # Values from matlab code
        self.Tmt = np.matrix([0.78187414360087504, 0.57856722491374268, 0.23223434247330654]).T
        self.Nmt = np.matrix([0.61708838355029516, -0.66519145682767544, -0.42038345905941465]).T
        self.Pmt = np.matrix([0.088739790712409491, -0.47199607, 0.87712311]).T
        self.Emt = np.matrix([1.5232412549475853, 0.23777306063666762, -2.7610143155842528]).T
        self.amt = 1.9822156229199277
        self.vmt = -0.19209745970264186
        self.adc = np.pi/2
        self.vdc = 0.5
        self.Tdc = np.matrix([1/np.sqrt(2), 1/np.sqrt(2), 0]).T
        self.Ndc = np.matrix([0., 0., 1.]).T
        self.Pdc = np.matrix([-1/np.sqrt(2), 1/np.sqrt(2), 0]).T
        self.Edc = np.matrix([1/np.sqrt(2), 0, -1/np.sqrt(2)]).T
        self.Gmt = 0.22691207246521938
        self.Dmt = -0.18360401027891848
        self.Umt = 0.41369317844275666
        self.Vmt = -0.12072857842564184
        self.tmt = 0.41369317844275666
        self.kmt = -0.12072857842564184
        self.Gdc = 0.
        self.Ddc = 0.
        self.Udc = 0.
        self.Vdc = 0.
        self.tdc = 0.
        self.kdc = 0.
        self.N1mt = np.matrix([0.61561701812768677, 0.075357186487224101, 0.78443417916119973]).T
        self.N2mt = np.matrix([0.49012000239468767, 0.7428604317087556, -0.4560052217399625]).T
        self.N1dc = np.matrix([0, 1., 0]).T
        self.N2dc = np.matrix([1., 0, 0]).T
        self.S1mt = 1.69259957
        self.D1mt = 0.66901303
        self.R1mt = -2.31557077
        self.S2mt = 5.69996974
        self.D2mt = 1.09729495
        self.R2mt = -1.07883773
        self.S1dc = 0.
        self.D1dc = 90.*np.pi/180
        self.R1dc = 0.
        self.S2dc = 270.*np.pi/180
        self.D2dc = 90.*np.pi/180
        self.R2dc = -180.*np.pi/180
        self.Kmt = 5.69996974
        self.Hmt = 0.45600522
        self.Omt = -1.07883773
        self.Kdc = 0.
        self.Hdc = 0.
        self.Odc = 0.
        self.T2 = np.array([[1., 0.99931011, 0.99753384, 1., 0.98559856, 1., 1., 1.],
                            [0., 0., 0.0701871, 0., 0.16910198, 0., 0., 0.],
                            [0., 0.0371391, 0., 0., 0., 0., 0., 0.]])
        self.N2 = np.array([[0., 0., -0.0701871, 0., -0.16910198, 0., 0., 0.],
                            [1., 1., 0.99753384, -0.98559856,  0.98559856, 1., 1., 1.],
                            [0., 0., 0., -0.16910198, 0., 0., 0., 0.]])
        self.P2 = np.array([[0., -0.0371391, 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0.16910198, 0., 0., 0., 0.],
                            [1., 0.99931011, 1., -0.98559856, 1., 1., 1., 1.]])
        self.S21 = np.array([90., 90., 94.02473349, 279.5980303, 99.73561032, 90., 90., 90.])*np.pi/180.
        self.S22 = np.array([270., 270., 274.02473349, 80.4019697, 279.73561032, 270., 270., 270.])*np.pi/180.
        self.D21 = np.array([45., 42.87159666, 45., 45.81931182, 45., 45., 45., 45.])*np.pi/180.
        self.D22 = np.array([45., 47.12840334, 45., 45.81931182, 45., 45., 45., 45.])*np.pi/180.
        self.R21 = np.array([-90., -90., -90., -76.36129238206001, -90., -90., -90., -90.])*np.pi/180.
        self.R22 = np.array([-90., -90., -90., -103.63870728, -90., -90., -90., -90.])*np.pi/180.

    def tearDown(self):
        del self.DC6
        del self.MT6
        del self.DC33
        del self.MT33
        del self.MT6norm
        del self.MT33norm
        del self.Tmt
        del self.Nmt
        del self.Pmt
        del self.Emt
        del self.Tdc
        del self.Ndc
        del self.Pdc
        del self.Edc
        del self.amt
        del self.vmt
        del self.adc
        del self.vdc
        del self.Gmt
        del self.Dmt
        del self.Umt
        del self.Vmt
        del self.tmt
        del self.kmt
        del self.Gdc
        del self.Ddc
        del self.Udc
        del self.Vdc
        del self.tdc
        del self.kdc
        del self.N1mt
        del self.N2mt
        del self.N1dc
        del self.N2dc
        del self.S1mt
        del self.D1mt
        del self.R1mt
        del self.S2mt
        del self.D2mt
        del self.R2mt
        del self.S1dc
        del self.D1dc
        del self.R1dc
        del self.S2dc
        del self.D2dc
        del self.R2dc
        del self.Kmt
        del self.Hmt
        del self.Omt
        del self.Kdc
        del self.Hdc
        del self.Odc
        del self.T2
        del self.P2
        del self.N2
        del self.S21
        del self.D21
        del self.R21
        del self.S22
        del self.D22
        del self.R22

    def assertSigmaEquals(self, first, second, *args):
        try:
            return self.assertAlmostEquals(first, second, *args)
        except AssertionError as e1:
            try:
                self.assertAlmostEquals(np.abs(first), np.pi, *args)
                return self.assertAlmostEquals(-first, second, *args)
            except Exception:
                raise e1

    def test_MT33_MT6(self):
        DC6 = MT33_MT6(self.DC33)
        MT6 = MT33_MT6(self.MT33)
        self.assertAlmostEqual(DC6, self.DC6)
        self.assertAlmostEqual(MT6, self.MT6/self.MT6norm)

    def test_MT6_MT33(self):
        DC33 = MT6_MT33(self.DC6)
        MT33 = MT6_MT33(self.MT6)
        self.assertAlmostEqual(DC33, self.DC33)
        self.assertAlmostEqual(MT33, self.MT33)

    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_MT6_TNPE(self, logger):
        with PythonOnly():
            [Tdc, Ndc, Pdc, Edc] = MT6_TNPE(self.DC6)
            logger.info.assert_called_once_with(C_EXTENSION_FALLBACK_LOG_MSG)
            [Tmt, Nmt, Pmt, Emt] = MT6_TNPE(self.MT6)
            self.assertVectorEquals(Tdc, self.Tdc)
            self.assertVectorEquals(Ndc, self.Ndc)
            self.assertVectorEquals(Pdc, self.Pdc)
            self.assertAlmostEquals(Edc, self.Edc)
            self.assertVectorEquals(Tmt, self.Tmt)
            self.assertVectorEquals(Nmt, self.Nmt)
            self.assertVectorEquals(Pmt, self.Pmt)
            self.assertAlmostEquals(Emt, self.Emt)
            [T, N, P, E] = MT6_TNPE(np.append(self.MT6, self.DC6, 1))
            self.assertVectorEquals(T[:, 1], self.Tdc)
            self.assertVectorEquals(N[:, 1], self.Ndc)
            self.assertVectorEquals(P[:, 1], self.Pdc)
            self.assertAlmostEquals(E[:, 1], self.Edc)
            self.assertVectorEquals(T[:, 0], self.Tmt)
            self.assertVectorEquals(N[:, 0], self.Nmt)
            self.assertVectorEquals(P[:, 0], self.Pmt)
            self.assertAlmostEquals(E[:, 0], self.Emt)
            self.assertEqual(logger.info.call_args_list, [mock.call(C_EXTENSION_FALLBACK_LOG_MSG),
                                                          mock.call(C_EXTENSION_FALLBACK_LOG_MSG),
                                                          mock.call(C_EXTENSION_FALLBACK_LOG_MSG)])

    @unittest.skipIf(*C_EXTENSIONS)
    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_MT6_TNPE_cython(self, logger):
        [Tdc, Ndc, Pdc, Edc] = MT6_TNPE(self.DC6)
        [Tmt, Nmt, Pmt, Emt] = MT6_TNPE(self.MT6)
        self.assertVectorEquals(Tdc, self.Tdc)
        self.assertVectorEquals(Ndc, self.Ndc)
        self.assertVectorEquals(Pdc, self.Pdc)
        self.assertAlmostEquals(Edc, self.Edc)
        self.assertVectorEquals(Tmt, self.Tmt)
        self.assertVectorEquals(Nmt, self.Nmt)
        self.assertVectorEquals(Pmt, self.Pmt)
        self.assertAlmostEquals(Emt, self.Emt)
        [T, N, P, E] = MT6_TNPE(np.append(self.MT6, self.DC6, 1))
        self.assertVectorEquals(T[:, 1], self.Tdc)
        self.assertVectorEquals(N[:, 1], self.Ndc)
        self.assertVectorEquals(P[:, 1], self.Pdc)
        self.assertAlmostEquals(E[:, 1], self.Edc)
        self.assertVectorEquals(T[:, 0], self.Tmt)
        self.assertVectorEquals(N[:, 0], self.Nmt)
        self.assertVectorEquals(P[:, 0], self.Pmt)
        self.assertAlmostEquals(E[:, 0], self.Emt)
        logger.info.assert_not_called()

    def test_MT6_Tape(self):
        mtc._CYTHON = False
        [Gmt, Dmt, Kmt, Hmt, Omt] = MT6_Tape(self.MT6)
        self.assertAlmostEquals(Gmt, self.Gmt)
        self.assertAlmostEquals(Dmt, self.Dmt)
        self.assertAlmostEquals(Kmt, self.Kmt)
        self.assertAlmostEquals(Hmt, self.Hmt)
        self.assertAlmostEquals(Omt, self.Omt)
        [Gdc, Ddc, Kdc, Hdc, Odc] = MT6_Tape(self.DC6)
        self.assertAlmostEquals(Gdc, self.Gdc)
        self.assertAlmostEquals(Ddc, self.Ddc)
        try:
            self.assertAlmostEquals(Kdc, self.Kdc)
        except Exception:
            self.assertAlmostEquals(Kdc-np.pi, self.Kdc)
        self.assertAlmostEquals(Hdc, self.Hdc)
        self.assertAlmostEquals(Odc, self.Odc)

    def test_MT6_Tape_multiple(self):
        [G, D, K, H, O] = MT6_Tape(np.append(self.MT6, self.DC6, 1))
        self.assertAlmostEquals(G[0], self.Gmt)
        self.assertAlmostEquals(D[0], self.Dmt)
        self.assertAlmostEquals(K[0], self.Kmt)
        self.assertAlmostEquals(H[0], self.Hmt)
        self.assertAlmostEquals(O[0], self.Omt)
        self.assertAlmostEquals(G[1], self.Gdc)
        self.assertAlmostEquals(D[1], self.Ddc)
        try:
            self.assertAlmostEquals(K[1], self.Kdc)
        except Exception:
            self.assertAlmostEquals(K[1]-np.pi, self.Kdc)
        self.assertAlmostEquals(H[1], self.Hdc)
        self.assertAlmostEquals(O[1], self.Odc)

    def test_MT33_TNPE(self):
        [Tdc, Ndc, Pdc, Edc] = MT33_TNPE(self.DC33)
        [Tmt, Nmt, Pmt, Emt] = MT33_TNPE(self.MT33)
        self.assertVectorEquals(Tdc, self.Tdc)
        self.assertVectorEquals(Ndc, self.Ndc)
        self.assertVectorEquals(Pdc, self.Pdc)
        self.assertAlmostEquals(Edc, self.Edc)
        self.assertVectorEquals(Tmt, self.Tmt)
        self.assertVectorEquals(Nmt, self.Nmt)
        self.assertVectorEquals(Pmt, self.Pmt)
        self.assertAlmostEquals(Emt, self.Emt)

    def test_MT33_SDR(self):
        [Sdc, Ddc, Rdc] = MT33_SDR(self.DC33)
        [Smt, Dmt, Rmt] = MT33_SDR(self.MT33)
        try:
            self.assertAlmostEquals(Sdc, self.S1dc)
            self.assertAlmostEquals(Ddc, self.D1dc)
            self.assertAlmostEquals(Rdc, self.R1dc)
        except Exception:
            self.assertAlmostEquals(Sdc, self.S2dc)
            self.assertAlmostEquals(Ddc, self.D2dc)
            self.assertAlmostEquals(Rdc, self.R2dc)
        try:
            self.assertAlmostEquals(Smt, self.S1mt)
            self.assertAlmostEquals(Dmt, self.D1mt)
            self.assertAlmostEquals(Rmt, self.R1mt)
        except Exception:
            self.assertAlmostEquals(Smt, self.S2mt)
            self.assertAlmostEquals(Dmt, self.D2mt)
            self.assertAlmostEquals(Rmt, self.R2mt)

    def test_MT33_GD(self):
        [Gmt, Dmt] = MT33_GD(self.MT33)
        self.assertAlmostEquals(Gmt, self.Gmt)
        self.assertAlmostEquals(Dmt, self.Dmt)

    def test_E_tk(self):
        tdc, kdc = E_tk(self.Edc)
        tmt, kmt = E_tk(self.Emt)
        self.assertAlmostEquals(tdc, self.tdc)
        self.assertAlmostEquals(kdc, self.kdc)
        self.assertAlmostEquals(tmt, self.tmt)
        self.assertAlmostEquals(kmt, self.kmt)

    def test_E_tk_multiple(self):
        t, k = E_tk(np.append(self.Emt, self.Edc, 1))
        self.assertAlmostEquals(t[1], self.tdc)
        self.assertAlmostEquals(k[1], self.kdc)
        self.assertAlmostEquals(t[0], self.tmt)
        self.assertAlmostEquals(k[0], self.kmt)

    def test_tk_uv(self):
        Udc, Vdc = tk_uv(self.tdc, self.kdc)
        Umt, Vmt = tk_uv(self.tmt, self.kmt)
        self.assertAlmostEquals(Udc, self.Udc)
        self.assertAlmostEquals(Vdc, self.Vdc)
        self.assertAlmostEquals(Umt, self.Umt)
        self.assertAlmostEquals(Vmt, self.Vmt)

    def test_tk_uv_multiple(self):
        u, v = tk_uv(np.array([self.tmt, self.tdc]), np.array([self.kmt, self.kdc]))
        self.assertAlmostEquals(u[1], self.Udc)
        self.assertAlmostEquals(v[1], self.Vdc)
        self.assertAlmostEquals(u[0], self.Umt)
        self.assertAlmostEquals(v[0], self.Vmt)

    def test_E_uv(self):
        Udc, Vdc = E_uv(self.Edc)
        Umt, Vmt = E_uv(self.Emt)
        self.assertAlmostEquals(Udc, self.Udc)
        self.assertAlmostEquals(Vdc, self.Vdc)
        self.assertAlmostEquals(Umt, self.Umt)
        self.assertAlmostEquals(Vmt, self.Vmt)

    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_E_GD_python(self, logger):
        with PythonOnly():
            Gdc, Ddc = E_GD(self.Edc)
            logger.info.assert_called_once_with(C_EXTENSION_FALLBACK_LOG_MSG)
            Gmt, Dmt = E_GD(self.Emt)
            self.assertAlmostEquals(Gdc, self.Gdc)
            self.assertAlmostEquals(Ddc, self.Ddc)
            self.assertAlmostEquals(Gmt, self.Gmt)
            self.assertAlmostEquals(Dmt, self.Dmt)

    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_E_GD_multiple_python(self, logger):
        with PythonOnly():
            t, k = E_GD(np.append(self.Emt, self.Edc, 1))
            self.assertAlmostEquals(t[1], self.Gdc)
            self.assertAlmostEquals(k[1], self.Ddc)
            self.assertAlmostEquals(t[0], self.Gmt)
            self.assertAlmostEquals(k[0], self.Dmt)
            logger.info.assert_called_with(C_EXTENSION_FALLBACK_LOG_MSG)

    @unittest.skipIf(*C_EXTENSIONS)
    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_E_GD_cython(self, logger):
        Gdc, Ddc = E_GD(self.Edc)
        Gmt, Dmt = E_GD(self.Emt)
        self.assertAlmostEquals(Gdc, self.Gdc)
        self.assertAlmostEquals(Ddc, self.Ddc)
        self.assertAlmostEquals(Gmt, self.Gmt)
        self.assertAlmostEquals(Dmt, self.Dmt)
        logger.info.assert_not_called()

    @unittest.skipIf(*C_EXTENSIONS)
    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_E_GD_multiple_cython(self, logger):
        t, k = E_GD(np.append(self.Emt, self.Edc, 1))
        self.assertAlmostEquals(t[1], self.Gdc)
        self.assertAlmostEquals(k[1], self.Ddc)
        self.assertAlmostEquals(t[0], self.Gmt)
        self.assertAlmostEquals(k[0], self.Dmt)
        logger.info.assert_not_called()

    def test_GD_basic_cdc(self):
        adc, vdc = GD_basic_cdc(self.Gdc, self.Ddc)
        amt, vmt = GD_basic_cdc(self.Gmt, self.Dmt)
        self.assertAlmostEquals(adc, self.adc)
        self.assertAlmostEquals(vdc, self.vdc)
        self.assertAlmostEquals(amt, self.amt)
        self.assertAlmostEquals(vmt, self.vmt)
        a, v = GD_basic_cdc(
            np.array([self.Gmt, self.Gdc]), np.array([self.Dmt, self.Ddc]))
        self.assertAlmostEquals(a[1], self.adc)
        self.assertAlmostEquals(v[1], self.vdc)
        self.assertAlmostEquals(a[0], self.amt)
        self.assertAlmostEquals(v[0], self.vmt)

    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_TNP_SDR_python(self, logger):
        with PythonOnly():
            [Sdc, Ddc, Rdc] = TNP_SDR(self.Tdc, self.Ndc, self.Pdc)
            logger.info.assert_called_once_with(C_EXTENSION_FALLBACK_LOG_MSG)
            [Smt, Dmt, Rmt] = TNP_SDR(self.Tmt, self.Nmt, self.Pmt)
            try:
                self.assertAlmostEquals(Sdc, self.S1dc)
                self.assertAlmostEquals(Ddc, self.D1dc)
                self.assertAlmostEquals(Rdc, self.R1dc)
            except Exception:
                self.assertAlmostEquals(Sdc, self.S2dc)
                self.assertAlmostEquals(Ddc, self.D2dc)
                self.assertAlmostEquals(Rdc, self.R2dc)
            try:
                self.assertAlmostEquals(Smt, self.S1mt)
                self.assertAlmostEquals(Dmt, self.D1mt)
                self.assertAlmostEquals(Rmt, self.R1mt)
            except Exception:
                self.assertAlmostEquals(Smt, self.S2mt)
                self.assertAlmostEquals(Dmt, self.D2mt)
                self.assertAlmostEquals(Rmt, self.R2mt)

    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_TNP_SDR_multiple_python(self, logger):
        with PythonOnly():
            [s, d, r] = TNP_SDR(np.append(self.Tmt, self.Tdc, 1),
                                np.append(self.Nmt, self.Ndc, 1),
                                np.append(self.Pmt, self.Pdc, 1))
            logger.info.assert_called_once_with(C_EXTENSION_FALLBACK_LOG_MSG)
            try:
                self.assertAlmostEquals(s[0], self.S1mt)
                self.assertAlmostEquals(d[0], self.D1mt)
                self.assertAlmostEquals(r[0], self.R1mt)
            except Exception:
                self.assertAlmostEquals(s[0], self.S2mt)
                self.assertAlmostEquals(d[0], self.D2mt)
                self.assertAlmostEquals(r[0], self.R2mt)
            try:
                self.assertAlmostEquals(s[1], self.S1dc)
                self.assertAlmostEquals(d[1], self.D1dc)
                self.assertAlmostEquals(r[1], self.R1dc)
            except Exception:
                self.assertAlmostEquals(s[1], self.S2dc)
                self.assertAlmostEquals(d[1], self.D2dc)
                self.assertAlmostEquals(r[1], self.R2dc)
            [S2, D2, R2] = TNP_SDR(self.T2, self.N2, self.P2)
            for i, s in enumerate(S2):
                try:
                    self.assertAlmostEquals(s, self.S21[i])
                    self.assertAlmostEquals(D2[i], self.D21[i])
                    self.assertAlmostEquals(R2[i], self.R21[i])
                except Exception:
                    self.assertAlmostEquals(s, self.S22[i])
                    self.assertAlmostEquals(D2[i], self.D22[i])
                    self.assertAlmostEquals(R2[i], self.R22[i])

            [S2, D2, R2] = TNP_SDR(-self.T2, -self.N2, -self.P2)
            for i, s in enumerate(S2):
                try:
                    self.assertAlmostEquals(s, self.S21[i])
                    self.assertAlmostEquals(D2[i], self.D21[i])
                    self.assertAlmostEquals(R2[i], self.R21[i])
                except Exception:
                    self.assertAlmostEquals(s, self.S22[i])
                    self.assertAlmostEquals(D2[i], self.D22[i])
                    self.assertAlmostEquals(R2[i], self.R22[i])

    @unittest.skipIf(*C_EXTENSIONS)
    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_TNP_SDR_cython(self, logger):
        [Sdc, Ddc, Rdc] = TNP_SDR(self.Tdc, self.Ndc, self.Pdc)
        [Smt, Dmt, Rmt] = TNP_SDR(self.Tmt, self.Nmt, self.Pmt)
        logger.info.assert_not_called()
        try:
            self.assertAlmostEquals(Sdc, self.S1dc)
            self.assertAlmostEquals(Ddc, self.D1dc)
            self.assertAlmostEquals(Rdc, self.R1dc)
        except Exception:
            self.assertAlmostEquals(Sdc, self.S2dc)
            self.assertAlmostEquals(Ddc, self.D2dc)
            self.assertAlmostEquals(Rdc, self.R2dc)
        try:
            self.assertAlmostEquals(Smt, self.S1mt)
            self.assertAlmostEquals(Dmt, self.D1mt)
            self.assertAlmostEquals(Rmt, self.R1mt)
        except Exception:
            self.assertAlmostEquals(Smt, self.S2mt)
            self.assertAlmostEquals(Dmt, self.D2mt)
            self.assertAlmostEquals(Rmt, self.R2mt)

    @unittest.skipIf(*C_EXTENSIONS)
    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_TNP_SDR_multiple_cython(self, logger):
        [s, d, r] = TNP_SDR(np.append(self.Tmt, self.Tdc, 1),
                            np.append(self.Nmt, self.Ndc, 1),
                            np.append(self.Pmt, self.Pdc, 1))
        logger.info.assert_not_called()
        try:
            self.assertAlmostEquals(s[0], self.S1mt)
            self.assertAlmostEquals(d[0], self.D1mt)
            self.assertAlmostEquals(r[0], self.R1mt)
        except Exception:
            self.assertAlmostEquals(s[0], self.S2mt)
            self.assertAlmostEquals(d[0], self.D2mt)
            self.assertAlmostEquals(r[0], self.R2mt)
        try:
            self.assertAlmostEquals(s[1], self.S1dc)
            self.assertAlmostEquals(d[1], self.D1dc)
            self.assertAlmostEquals(r[1], self.R1dc)
        except Exception:
            self.assertAlmostEquals(s[1], self.S2dc)
            self.assertAlmostEquals(d[1], self.D2dc)
            self.assertAlmostEquals(r[1], self.R2dc)
        [S2, D2, R2] = TNP_SDR(self.T2, self.N2, self.P2)
        for i, s in enumerate(S2):
            try:
                self.assertAlmostEquals(s, self.S21[i])
                self.assertAlmostEquals(D2[i], self.D21[i])
                self.assertAlmostEquals(R2[i], self.R21[i])
            except Exception:
                self.assertAlmostEquals(s, self.S22[i])
                self.assertAlmostEquals(D2[i], self.D22[i])
                self.assertAlmostEquals(R2[i], self.R22[i])
        [S2, D2, R2] = TNP_SDR(-self.T2, -self.N2, -self.P2)
        for i, s in enumerate(S2):
            try:
                self.assertAlmostEquals(s, self.S21[i])
                self.assertAlmostEquals(D2[i], self.D21[i])
                self.assertAlmostEquals(R2[i], self.R21[i])
            except Exception:
                self.assertAlmostEquals(s, self.S22[i])
                self.assertAlmostEquals(D2[i], self.D22[i])
                self.assertAlmostEquals(R2[i], self.R22[i])

    def test_TP_FP(self):
        [N1dc, N2dc] = TP_FP(self.Tdc, self.Pdc)
        [N1mt, N2mt] = TP_FP(self.Tmt, self.Pmt)
        self.assertVectorEquals(N1dc, self.N1dc)
        self.assertVectorEquals(N2dc, self.N2dc)
        self.assertVectorEquals(N1mt, self.N1mt)
        self.assertVectorEquals(N2mt, self.N2mt)
        [N1, N2] = TP_FP(
            np.append(self.Tmt, self.Tdc, 1), np.append(self.Pmt, self.Pdc, 1))
        self.assertVectorEquals(N1[:, 0], self.N1mt)
        self.assertVectorEquals(N2[:, 0], self.N2mt)
        self.assertVectorEquals(N1[:, 1], self.N1dc)
        self.assertVectorEquals(N2[:, 1], self.N2dc)

    def test_FP_SDR(self):
        [Sdc, Ddc, Rdc] = FP_SDR(np.array(self.N1dc), np.array(self.N2dc))
        [Smt, Dmt, Rmt] = FP_SDR(np.array(self.N1mt), np.array(self.N2mt))
        self.assertAlmostEquals(Sdc, self.S1dc)
        self.assertAlmostEquals(Ddc, self.D1dc)
        self.assertAlmostEquals(Rdc, self.R1dc)
        self.assertAlmostEquals(Smt, self.S1mt)
        self.assertAlmostEquals(Dmt, self.D1mt)
        self.assertAlmostEquals(Rmt, self.R1mt)

    def test_FP_SDR_matrix(self):
        [Sdc, Ddc, Rdc] = FP_SDR(np.matrix(self.N1dc), np.matrix(self.N2dc))
        [Smt, Dmt, Rmt] = FP_SDR(np.matrix(self.N1mt), np.matrix(self.N2mt))
        self.assertAlmostEquals(Sdc, self.S1dc)
        self.assertAlmostEquals(Ddc, self.D1dc)
        self.assertAlmostEquals(Rdc, self.R1dc)
        self.assertAlmostEquals(Smt, self.S1mt)
        self.assertAlmostEquals(Dmt, self.D1mt)
        self.assertAlmostEquals(Rmt, self.R1mt)

    def test_FP_SDR_multiple(self):
        [s, d, r] = FP_SDR(np.append(self.N1mt, self.N1dc, 1),
                           np.append(self.N2mt, self.N2dc, 1))
        self.assertAlmostEquals(s[0], self.S1mt)
        self.assertAlmostEquals(d[0], self.D1mt)
        self.assertAlmostEquals(r[0], self.R1mt)
        self.assertAlmostEquals(s[1], self.S1dc)
        self.assertAlmostEquals(d[1], self.D1dc)
        self.assertAlmostEquals(r[1], self.R1dc)
        [Sdc, Ddc, Rdc] = FP_SDR(self.N2dc, self.N1dc)
        [Smt, Dmt, Rmt] = FP_SDR(self.N2mt, self.N1mt)
        self.assertAlmostEquals(Sdc, self.S2dc)
        self.assertAlmostEquals(Ddc, self.D2dc)
        self.assertAlmostEquals(Rdc, self.R2dc)
        self.assertAlmostEquals(Smt, self.S2mt)
        self.assertAlmostEquals(Dmt, self.D2mt)
        self.assertAlmostEquals(Rmt, self.R2mt)
        [s, d, r] = FP_SDR(np.append(self.N2mt, self.N2dc, 1),
                           np.append(self.N1mt, self.N1dc, 1))
        self.assertAlmostEquals(s[0], self.S2mt)
        self.assertAlmostEquals(d[0], self.D2mt)
        self.assertAlmostEquals(r[0], self.R2mt)
        self.assertAlmostEquals(s[1], self.S2dc)
        self.assertAlmostEquals(d[1], self.D2dc)
        self.assertAlmostEquals(r[1], self.R2dc)

    def test_FP_SDR_array(self):
        [Sdc, Ddc, Rdc] = FP_SDR(np.array(2*self.N1dc), np.array(2*self.N2dc))
        [Smt, Dmt, Rmt] = FP_SDR(np.array(2*self.N1mt), np.array(2*self.N2mt))
        self.assertAlmostEquals(Sdc, self.S1dc)
        self.assertAlmostEquals(Ddc, self.D1dc)
        self.assertAlmostEquals(Rdc, self.R1dc)
        self.assertAlmostEquals(Smt, self.S1mt)
        self.assertAlmostEquals(Dmt, self.D1mt)
        self.assertAlmostEquals(Rmt, self.R1mt)

    def test_basic_cdc_GD(self):
        Gdc, Ddc = basic_cdc_GD(self.adc, self.vdc)
        Gmt, Dmt = basic_cdc_GD(self.amt, self.vmt)
        self.assertAlmostEquals(Gdc, self.Gdc)
        self.assertAlmostEquals(Ddc, self.Ddc)
        self.assertAlmostEquals(Gmt, self.Gmt)
        self.assertAlmostEquals(Dmt, self.Dmt)
        G, D = basic_cdc_GD(np.array([self.amt, self.adc]),
                            np.array([self.vmt, self.vdc]))
        self.assertAlmostEquals(G[1], self.Gdc)
        self.assertAlmostEquals(D[1], self.Ddc)
        self.assertAlmostEquals(G[0], self.Gmt)
        self.assertAlmostEquals(D[0], self.Dmt)

    def test_basic_cdc_GD_pi_2(self):
        G2, D2 = basic_cdc_GD(np.pi/2, self.vmt)
        self.assertEqual(D2, 0)
        G, D = basic_cdc_GD(np.array([np.pi/2, np.pi/2]),
                            np.array([self.vmt, self.vdc]))
        self.assertAlmostEquals(D[1], 0)
        self.assertAlmostEquals(D[0], 0)

    def test_GD_E(self):
        mtc._CYTHON = False
        Edc = GD_E(self.Gdc, self.Ddc)
        Emt = GD_E(self.Gmt, self.Dmt)
        self.assertAlmostEquals(Edc, self.Edc)
        self.assertAlmostEquals(Emt, self.Emt/np.sqrt(np.diag(self.Emt.T*self.Emt)))
        E = GD_E(np.array([self.Gmt, self.Gdc]),
                 np.array([self.Dmt, self.Ddc]))
        self.assertAlmostEquals(E[:, 1], self.Edc)
        self.assertAlmostEquals(E[:, 0], self.Emt/np.sqrt(np.diag(self.Emt.T*self.Emt)))

    def test_SDR_TNP(self):
        [Tdc, Ndc, Pdc] = SDR_TNP(self.S1dc, self.D1dc, self.R1dc)
        [Tmt, Nmt, Pmt] = SDR_TNP(self.S1mt, self.D1mt, self.R1mt)
        self.assertVectorEquals(Tdc, self.Tdc)
        self.assertVectorEquals(Ndc, self.Ndc)
        self.assertVectorEquals(Pdc, self.Pdc)
        self.assertVectorEquals(Tmt, self.Tmt)
        self.assertVectorEquals(Nmt, self.Nmt)
        self.assertVectorEquals(Pmt, self.Pmt)
        [T, N, P] = SDR_TNP(np.array([self.S1mt, self.S1dc]), np.array(
            [self.D1mt, self.D1dc]), np.array([self.R1mt, self.R1dc]))
        self.assertVectorEquals(T[:, 0], self.Tmt)
        self.assertVectorEquals(N[:, 0], self.Nmt)
        self.assertVectorEquals(P[:, 0], self.Pmt)
        self.assertVectorEquals(T[:, 1], self.Tdc)
        self.assertVectorEquals(N[:, 1], self.Ndc)
        self.assertVectorEquals(P[:, 1], self.Pdc)
        [Tdc, Ndc, Pdc] = SDR_TNP(self.S2dc, self.D2dc, self.R2dc)
        [Tmt, Nmt, Pmt] = SDR_TNP(self.S2mt, self.D2mt, self.R2mt)
        self.assertVectorEquals(Tdc, self.Tdc)
        self.assertVectorEquals(Ndc, self.Ndc)
        self.assertVectorEquals(Pdc, self.Pdc)
        self.assertVectorEquals(Tmt, self.Tmt)
        self.assertVectorEquals(Nmt, self.Nmt)
        self.assertVectorEquals(Pmt, self.Pmt)
        [T, N, P] = SDR_TNP(np.array([self.S2mt, self.S2dc]), np.array(
            [self.D2mt, self.D2dc]), np.array([self.R2mt, self.R2dc]))
        self.assertVectorEquals(T[:, 0], self.Tmt)
        self.assertVectorEquals(N[:, 0], self.Nmt)
        self.assertVectorEquals(P[:, 0], self.Pmt)
        self.assertVectorEquals(T[:, 1], self.Tdc)
        self.assertVectorEquals(N[:, 1], self.Ndc)
        self.assertVectorEquals(P[:, 1], self.Pdc)

    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_SDR_SDR_python(self, logger):
        with PythonOnly():
            [S2dc, D2dc, R2dc] = SDR_SDR(self.S1dc, self.D1dc, self.R1dc)
            logger.info.assert_called_once_with(C_EXTENSION_FALLBACK_LOG_MSG)
            [S2mt, D2mt, R2mt] = SDR_SDR(self.S1mt, self.D1mt, self.R1mt)
            self.assertAlmostEquals(S2dc, self.S2dc)
            self.assertAlmostEquals(D2dc, self.D2dc)
            self.assertSigmaEquals(R2dc, self.R2dc)
            self.assertAlmostEquals(S2mt, self.S2mt)
            self.assertAlmostEquals(D2mt, self.D2mt)
            self.assertSigmaEquals(R2mt, self.R2mt)
            [S, D, R] = SDR_SDR(np.array([self.S1mt, self.S1dc]), np.array(
                [self.D1mt, self.D1dc]), np.array([self.R1mt, self.R1dc]))
            self.assertAlmostEquals(S[0], self.S2mt)
            self.assertAlmostEquals(D[0], self.D2mt)
            self.assertSigmaEquals(R[0], self.R2mt)
            self.assertAlmostEquals(S[1], self.S2dc)
            self.assertAlmostEquals(D[1], self.D2dc)
            self.assertSigmaEquals(R[1], self.R2dc)
            [S1dc, D1dc, R1dc] = SDR_SDR(self.S2dc, self.D2dc, self.R2dc)
            [S1mt, D1mt, R1mt] = SDR_SDR(self.S2mt, self.D2mt, self.R2mt)
            try:
                self.assertAlmostEquals(S1dc, self.S1dc)
            except AssertionError as e:
                if self.S1dc in [0, np.pi]:
                    self.assertAlmostEquals(S1dc, np.mod(self.S1dc+np.pi, 2*np.pi))
                else:
                    raise e
            self.assertAlmostEquals(D1dc, self.D1dc)
            self.assertSigmaEquals(R1dc, self.R1dc)
            self.assertAlmostEquals(S1mt, self.S1mt)
            self.assertAlmostEquals(D1mt, self.D1mt)
            self.assertSigmaEquals(R1mt, self.R1mt)
            [S, D, R] = SDR_SDR(np.array([self.S2mt, self.S2dc]), np.array(
                [self.D2mt, self.D2dc]), np.array([self.R2mt, self.R2dc]))
            self.assertAlmostEquals(S[0], self.S1mt)
            self.assertAlmostEquals(D[0], self.D1mt)
            self.assertSigmaEquals(R[0], self.R1mt)
            try:
                self.assertAlmostEquals(S[1], self.S1dc)
            except AssertionError as e:
                if self.S1dc in [0, np.pi]:
                    self.assertAlmostEquals(S[1], np.mod(self.S1dc+np.pi, 2*np.pi))
                else:
                    raise e
            self.assertAlmostEquals(D[1], self.D1dc)
            self.assertSigmaEquals(R[1], self.R1dc)

    @unittest.skipIf(*C_EXTENSIONS)
    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_SDR_SDR_cython(self, logger):
        [S2dc, D2dc, R2dc] = SDR_SDR(self.S1dc, self.D1dc, self.R1dc)
        [S2mt, D2mt, R2mt] = SDR_SDR(self.S1mt, self.D1mt, self.R1mt)
        self.assertAlmostEquals(S2dc, self.S2dc)
        self.assertAlmostEquals(D2dc, self.D2dc)
        self.assertSigmaEquals(R2dc, self.R2dc)
        self.assertAlmostEquals(S2mt, self.S2mt)
        self.assertAlmostEquals(D2mt, self.D2mt)
        self.assertSigmaEquals(R2mt, self.R2mt)
        [S, D, R] = SDR_SDR(np.array([self.S1mt, self.S1dc]), np.array(
            [self.D1mt, self.D1dc]), np.array([self.R1mt, self.R1dc]))
        self.assertAlmostEquals(S[0], self.S2mt)
        self.assertAlmostEquals(D[0], self.D2mt)
        self.assertSigmaEquals(R[0], self.R2mt)
        self.assertAlmostEquals(S[1], self.S2dc)
        self.assertAlmostEquals(D[1], self.D2dc)
        self.assertSigmaEquals(R[1], self.R2dc)
        [S1dc, D1dc, R1dc] = SDR_SDR(self.S2dc, self.D2dc, self.R2dc)
        [S1mt, D1mt, R1mt] = SDR_SDR(self.S2mt, self.D2mt, self.R2mt)
        try:
            self.assertAlmostEquals(S1dc, self.S1dc)
        except AssertionError as e:
            if self.S1dc in [0, np.pi]:
                self.assertAlmostEquals(S1dc, np.mod(self.S1dc+np.pi, 2*np.pi))
            else:
                raise e
        self.assertAlmostEquals(D1dc, self.D1dc)
        self.assertSigmaEquals(R1dc, self.R1dc)
        self.assertAlmostEquals(S1mt, self.S1mt)
        self.assertAlmostEquals(D1mt, self.D1mt)
        self.assertSigmaEquals(R1mt, self.R1mt)
        [S, D, R] = SDR_SDR(np.array([self.S2mt, self.S2dc]), np.array(
            [self.D2mt, self.D2dc]), np.array([self.R2mt, self.R2dc]))
        self.assertAlmostEquals(S[0], self.S1mt)
        self.assertAlmostEquals(D[0], self.D1mt)
        self.assertSigmaEquals(R[0], self.R1mt)
        try:
            self.assertAlmostEquals(S[1], self.S1dc)
        except AssertionError as e:
            if self.S1dc in [0, np.pi]:
                self.assertAlmostEquals(S[1], np.mod(self.S1dc+np.pi, 2*np.pi))
            else:
                raise e
        self.assertAlmostEquals(D[1], self.D1dc)
        self.assertSigmaEquals(R[1], self.R1dc)
        logger.info.assert_not_called()

    def test_FP_TNP(self):
        [Tdc, Ndc, Pdc] = FP_TNP(self.N1dc, self.N2dc)
        [Tmt, Nmt, Pmt] = FP_TNP(self.N1mt, self.N2mt)
        self.assertVectorEquals(Tdc, self.Tdc)
        self.assertVectorEquals(Ndc, self.Ndc)
        self.assertVectorEquals(Pdc, self.Pdc)
        self.assertVectorEquals(Tmt, self.Tmt)
        self.assertVectorEquals(Nmt, self.Nmt)
        self.assertVectorEquals(Pmt, self.Pmt)
        [T, N, P] = FP_TNP(np.append(self.N1mt, self.N1dc, 1),
                           np.append(self.N2mt, self.N2dc, 1))
        self.assertVectorEquals(T[:, 0], self.Tmt)
        self.assertVectorEquals(N[:, 0], self.Nmt)
        self.assertVectorEquals(P[:, 0], self.Pmt)
        self.assertVectorEquals(T[:, 1], self.Tdc)
        self.assertVectorEquals(N[:, 1], self.Ndc)
        self.assertVectorEquals(P[:, 1], self.Pdc)

    def test_SDSD_FP(self):
        mtc._CYTHON = False
        [N1dc, N2dc] = SDSD_FP(self.S1dc, self.D1dc, self.S2dc, self.D2dc)
        [N1mt, N2mt] = SDSD_FP(self.S1mt, self.D1mt, self.S2mt, self.D2mt)
        try:
            self.assertVectorEquals(N1dc, self.N1dc)
            self.assertVectorEquals(N2dc, self.N2dc)
        except AssertionError as e1:
            try:
                self.assertVectorEquals(N2dc, self.N1dc)
                self.assertVectorEquals(N1dc, self.N2dc)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)
        try:
            self.assertVectorEquals(N1mt, self.N1mt)
            self.assertVectorEquals(N2mt, self.N2mt)
        except AssertionError as e1:
            try:
                self.assertVectorEquals(N2mt, self.N1mt)
                self.assertVectorEquals(N1mt, self.N2mt)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)

    def test_SDSD_FP_multiple(self):
        [N1, N2] = SDSD_FP(np.array([self.S1mt, self.S1dc]),
                           np.array([self.D1mt, self.D1dc]),
                           np.array([self.S2mt, self.S2dc]),
                           np.array([self.D2mt, self.D2dc]))
        try:
            self.assertVectorEquals(N1[:, 1], self.N1dc)
            self.assertVectorEquals(N2[:, 1], self.N2dc)
        except AssertionError as e1:
            try:
                self.assertVectorEquals(N2[:, 1], self.N1dc)
                self.assertVectorEquals(N1[:, 1], self.N2dc)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)
        try:
            self.assertVectorEquals(N1[:, 0], self.N1mt)
            self.assertVectorEquals(N2[:, 0], self.N2mt)
        except AssertionError as e1:
            try:
                self.assertVectorEquals(N2[:, 0], self.N1mt)
                self.assertVectorEquals(N1[:, 0], self.N2mt)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)

    def test_SDR_FP(self):
        [N1dc, N2dc] = SDR_FP(self.S1dc, self.D1dc, self.R1dc)
        [N1mt, N2mt] = SDR_FP(self.S1mt, self.D1mt, self.R1mt)
        try:
            self.assertVectorEquals(N1dc, self.N1dc)
            self.assertVectorEquals(N2dc, self.N2dc)
        except AssertionError as e1:
            try:
                self.assertVectorEquals(N2dc, self.N1dc)
                self.assertVectorEquals(N1dc, self.N2dc)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)
        try:
            self.assertVectorEquals(N1mt, self.N1mt)
            self.assertVectorEquals(N2mt, self.N2mt)
        except AssertionError as e1:
            try:
                self.assertVectorEquals(N2mt, self.N1mt)
                self.assertVectorEquals(N1mt, self.N2mt)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)

    def test_SDR_FP_multiple(self):
        [N1, N2] = SDR_FP(np.array([self.S1mt, self.S1dc]),
                          np.array([self.D1mt, self.D1dc]),
                          np.array([self.R1mt, self.R1dc]))
        try:
            self.assertVectorEquals(N1[:, 1], self.N1dc)
            self.assertVectorEquals(N2[:, 1], self.N2dc)
        except AssertionError as e1:
            try:
                self.assertVectorEquals(N2[:, 1], self.N1dc)
                self.assertVectorEquals(N1[:, 1], self.N2dc)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)
        try:
            self.assertVectorEquals(N1[:, 0], self.N1mt)
            self.assertVectorEquals(N2[:, 0], self.N2mt)
        except AssertionError as e1:
            try:
                self.assertVectorEquals(N2[:, 0], self.N1mt)
                self.assertVectorEquals(N1[:, 0], self.N2mt)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)
        [N1dc, N2dc] = SDR_FP(self.S2dc, self.D2dc, self.R2dc)
        [N1mt, N2mt] = SDR_FP(self.S2mt, self.D2mt, self.R2mt)
        try:
            self.assertVectorEquals(N1dc, self.N1dc)
            self.assertVectorEquals(N2dc, self.N2dc)
        except AssertionError as e1:
            try:
                self.assertVectorEquals(N2dc, self.N1dc)
                self.assertVectorEquals(N1dc, self.N2dc)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)
        try:
            self.assertVectorEquals(N1mt, self.N1mt)
            self.assertVectorEquals(N2mt, self.N2mt)
        except AssertionError as e1:
            try:
                self.assertVectorEquals(N2mt, self.N1mt)
                self.assertVectorEquals(N1mt, self.N2mt)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)
        [N1, N2] = SDR_FP(np.array([self.S2mt, self.S2dc]), np.array(
            [self.D2mt, self.D2dc]), np.array([self.R2mt, self.R2dc]))
        try:
            self.assertVectorEquals(N1[:, 1], self.N1dc)
            self.assertVectorEquals(N2[:, 1], self.N2dc)
        except AssertionError as e1:
            try:
                self.assertVectorEquals(N2[:, 1], self.N1dc)
                self.assertVectorEquals(N1[:, 1], self.N2dc)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)
        try:
            self.assertVectorEquals(N1[:, 0], self.N1mt)
            self.assertVectorEquals(N2[:, 0], self.N2mt)
        except AssertionError as e1:
            try:
                self.assertVectorEquals(N2[:, 0], self.N1mt)
                self.assertVectorEquals(N1[:, 0], self.N2mt)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)

    def test_SDR_SDSD(self):
        [S1dc, D1dc, S2dc, D2dc] = SDR_SDSD(self.S1dc, self.D1dc, self.R1dc)
        [S1mt, D1mt, S2mt, D2mt] = SDR_SDSD(self.S1mt, self.D1mt, self.R1mt)
        try:
            self.assertAlmostEquals(S1dc, self.S1dc)
            self.assertAlmostEquals(S2dc, self.S2dc)
            self.assertAlmostEquals(D1dc, self.D1dc)
            self.assertAlmostEquals(D2dc, self.D2dc)
        except AssertionError as e1:
            try:
                self.assertAlmostEquals(S2dc, self.S1dc)
                self.assertAlmostEquals(S1dc, self.S2dc)
                self.assertAlmostEquals(D2dc, self.D1dc)
                self.assertAlmostEquals(D1dc, self.D2dc)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)
        try:
            self.assertAlmostEquals(S1mt, self.S1mt)
            self.assertAlmostEquals(S2mt, self.S2mt)
            self.assertAlmostEquals(D1mt, self.D1mt)
            self.assertAlmostEquals(D2mt, self.D2mt)
        except AssertionError as e1:
            try:
                self.assertAlmostEquals(S2mt, self.S1mt)
                self.assertAlmostEquals(S1mt, self.S2mt)
                self.assertAlmostEquals(D2mt, self.D1mt)
                self.assertAlmostEquals(D1mt, self.D2mt)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)

    def test_SDR_SDSD_multiple(self):
        [S1, D1, S2, D2] = SDR_SDSD(np.array([self.S1mt, self.S1dc]),
                                    np.array([self.D1mt, self.D1dc]),
                                    np.array([self.R1mt, self.R1dc]))
        try:
            self.assertAlmostEquals(S1[1], self.S1dc)
            self.assertAlmostEquals(S2[1], self.S2dc)
            self.assertAlmostEquals(D1[1], self.D1dc)
            self.assertAlmostEquals(D2[1], self.D2dc)
        except AssertionError as e1:
            try:
                self.assertAlmostEquals(S2[1], self.S1dc)
                self.assertAlmostEquals(S1[1], self.S2dc)
                self.assertAlmostEquals(D2[1], self.D1dc)
                self.assertAlmostEquals(D1[1], self.D2dc)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)
        try:
            self.assertAlmostEquals(S1[0], self.S1mt)
            self.assertAlmostEquals(S2[0], self.S2mt)
            self.assertAlmostEquals(D1[0], self.D1mt)
            self.assertAlmostEquals(D2[0], self.D2mt)
        except AssertionError as e1:
            try:
                self.assertAlmostEquals(S2[0], self.S1mt)
                self.assertAlmostEquals(S1[0], self.S2mt)
                self.assertAlmostEquals(D2[0], self.D1mt)
                self.assertAlmostEquals(D1[0], self.D2mt)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)
        [S1dc, D1dc, S2dc, D2dc] = SDR_SDSD(self.S2dc, self.D2dc, self.R2dc)
        [S1mt, D1mt, S2mt, D2mt] = SDR_SDSD(self.S2mt, self.D2mt, self.R2mt)
        try:
            try:
                self.assertAlmostEquals(S1dc, self.S1dc)
            except AssertionError as e:
                if self.S1dc in [0, np.pi]:
                    self.assertAlmostEquals(
                        S1dc, np.mod(self.S1dc+np.pi, 2*np.pi))
                else:
                    raise e
            try:
                self.assertAlmostEquals(S2dc, self.S2dc)
            except AssertionError as e:
                if self.S2dc in [0, np.pi]:
                    self.assertAlmostEquals(
                        S2dc, np.mod(self.S2dc+np.pi, 2*np.pi))
                else:
                    raise e
            self.assertAlmostEquals(D1dc, self.D1dc)
            self.assertAlmostEquals(D2dc, self.D2dc)
        except AssertionError as e1:
            try:
                try:
                    self.assertAlmostEquals(S2dc, self.S1dc)
                except AssertionError as e:
                    if self.S1dc in [0, np.pi]:
                        self.assertAlmostEquals(
                            S2dc, np.mod(self.S1dc+np.pi, 2*np.pi))
                    else:
                        raise e
                try:
                    self.assertAlmostEquals(S1dc, self.S2dc)
                except AssertionError as e:
                    if self.S2dc in [0, np.pi]:
                        self.assertAlmostEquals(
                            S1dc, np.mod(self.S2dc+np.pi, 2*np.pi))
                    else:
                        raise e
                self.assertAlmostEquals(D2dc, self.D1dc)
                self.assertAlmostEquals(D1dc, self.D2dc)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)
        try:
            self.assertAlmostEquals(S1mt, self.S1mt)
            self.assertAlmostEquals(S2mt, self.S2mt)
            self.assertAlmostEquals(D1mt, self.D1mt)
            self.assertAlmostEquals(D2mt, self.D2mt)
        except AssertionError as e1:
            try:
                self.assertAlmostEquals(S2mt, self.S1mt)
                self.assertAlmostEquals(S1mt, self.S2mt)
                self.assertAlmostEquals(D2mt, self.D1mt)
                self.assertAlmostEquals(D1mt, self.D2mt)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)
        [S1, D1, S2, D2] = SDR_SDSD(np.array([self.S2mt, self.S2dc]),
                                    np.array([self.D2mt, self.D2dc]),
                                    np.array([self.R2mt, self.R2dc]))
        try:
            try:
                self.assertAlmostEquals(S1[1], self.S1dc)
            except AssertionError as e:
                if self.S1dc in [0, np.pi]:
                    self.assertAlmostEquals(
                        S1[1], np.mod(self.S1dc+np.pi, 2*np.pi))
                else:
                    raise e
            try:
                self.assertAlmostEquals(S2[1], self.S2dc)
            except AssertionError as e:
                if self.S2dc in [0, np.pi]:
                    self.assertAlmostEquals(
                        S2[1], np.mod(self.S2dc+np.pi, 2*np.pi))
                else:
                    raise e
            self.assertAlmostEquals(D1[1], self.D1dc)
            self.assertAlmostEquals(D2[1], self.D2dc)
        except AssertionError as e1:
            try:
                try:
                    self.assertAlmostEquals(S2[1], self.S1dc)
                except AssertionError as e:
                    if self.S1dc in [0, np.pi]:
                        self.assertAlmostEquals(
                            S2[1], np.mod(self.S1dc+np.pi, 2*np.pi))
                    else:
                        raise e
                try:
                    self.assertAlmostEquals(S1[1], self.S2dc)
                except AssertionError as e:
                    if self.S2dc in [0, np.pi]:
                        self.assertAlmostEquals(
                            S1[1], np.mod(self.S2dc+np.pi, 2*np.pi))
                    else:
                        raise e
                self.assertAlmostEquals(D2[1], self.D1dc)
                self.assertAlmostEquals(D1[1], self.D2dc)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)
        try:
            self.assertAlmostEquals(S1[0], self.S1mt)
            self.assertAlmostEquals(S2[0], self.S2mt)
            self.assertAlmostEquals(D1[0], self.D1mt)
            self.assertAlmostEquals(D2[0], self.D2mt)
        except AssertionError as e1:
            try:
                self.assertAlmostEquals(S2[0], self.S1mt)
                self.assertAlmostEquals(S1[0], self.S2mt)
                self.assertAlmostEquals(D2[0], self.D1mt)
                self.assertAlmostEquals(D1[0], self.D2mt)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)

    def test_FP_SDSD(self):
        [S1dc, D1dc, S2dc, D2dc] = FP_SDSD(self.N1dc, self.N2dc)
        [S1mt, D1mt, S2mt, D2mt] = FP_SDSD(self.N1mt, self.N2mt)
        try:
            self.assertAlmostEquals(S1dc, self.S1dc)
            self.assertAlmostEquals(S2dc, self.S2dc)
            self.assertAlmostEquals(D1dc, self.D1dc)
            self.assertAlmostEquals(D2dc, self.D2dc)
        except AssertionError as e1:
            try:
                self.assertAlmostEquals(S2dc, self.S1dc)
                self.assertAlmostEquals(S1dc, self.S2dc)
                self.assertAlmostEquals(D2dc, self.D1dc)
                self.assertAlmostEquals(D1dc, self.D2dc)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)
        try:
            self.assertAlmostEquals(S1mt, self.S1mt)
            self.assertAlmostEquals(S2mt, self.S2mt)
            self.assertAlmostEquals(D1mt, self.D1mt)
            self.assertAlmostEquals(D2mt, self.D2mt)
        except AssertionError as e1:
            try:
                self.assertAlmostEquals(S2mt, self.S1mt)
                self.assertAlmostEquals(S1mt, self.S2mt)
                self.assertAlmostEquals(D2mt, self.D1mt)
                self.assertAlmostEquals(D1mt, self.D2mt)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)

    def test_FP_SDSD_multiple(self):
        [S1, D1, S2, D2] = FP_SDSD(np.append(self.N1mt, self.N1dc, 1),
                                   np.append(self.N2mt, self.N2dc, 1))
        try:
            self.assertAlmostEquals(S1[1], self.S1dc)
            self.assertAlmostEquals(S2[1], self.S2dc)
            self.assertAlmostEquals(D1[1], self.D1dc)
            self.assertAlmostEquals(D2[1], self.D2dc)
        except AssertionError as e1:
            try:
                self.assertAlmostEquals(S2[1], self.S1dc)
                self.assertAlmostEquals(S1[1], self.S2dc)
                self.assertAlmostEquals(D2[1], self.D1dc)
                self.assertAlmostEquals(D1[1], self.D2dc)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)
        try:
            self.assertAlmostEquals(S1[0], self.S1mt)
            self.assertAlmostEquals(S2[0], self.S2mt)
            self.assertAlmostEquals(D1[0], self.D1mt)
            self.assertAlmostEquals(D2[0], self.D2mt)
        except AssertionError as e1:
            try:
                self.assertAlmostEquals(S2[0], self.S1mt)
                self.assertAlmostEquals(S1[0], self.S2mt)
                self.assertAlmostEquals(D2[0], self.D1mt)
                self.assertAlmostEquals(D1[0], self.D2mt)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)
        [S1dc, D1dc, S2dc, D2dc] = FP_SDSD(self.N2dc, self.N1dc)
        [S1mt, D1mt, S2mt, D2mt] = FP_SDSD(self.N2mt, self.N1mt)
        try:
            self.assertAlmostEquals(S1dc, self.S1dc)
            self.assertAlmostEquals(S2dc, self.S2dc)
            self.assertAlmostEquals(D1dc, self.D1dc)
            self.assertAlmostEquals(D2dc, self.D2dc)
        except AssertionError as e1:
            try:
                self.assertAlmostEquals(S2dc, self.S1dc)
                self.assertAlmostEquals(S1dc, self.S2dc)
                self.assertAlmostEquals(D2dc, self.D1dc)
                self.assertAlmostEquals(D1dc, self.D2dc)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)
        try:
            self.assertAlmostEquals(S1mt, self.S1mt)
            self.assertAlmostEquals(S2mt, self.S2mt)
            self.assertAlmostEquals(D1mt, self.D1mt)
            self.assertAlmostEquals(D2mt, self.D2mt)
        except AssertionError as e1:
            try:
                self.assertAlmostEquals(S2mt, self.S1mt)
                self.assertAlmostEquals(S1mt, self.S2mt)
                self.assertAlmostEquals(D2mt, self.D1mt)
                self.assertAlmostEquals(D1mt, self.D2mt)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)
        [S1, D1, S2, D2] = FP_SDSD(np.append(self.N2mt, self.N2dc, 1),
                                   np.append(self.N1mt, self.N1dc, 1))
        try:
            self.assertAlmostEquals(S1[1], self.S1dc)
            self.assertAlmostEquals(S2[1], self.S2dc)
            self.assertAlmostEquals(D1[1], self.D1dc)
            self.assertAlmostEquals(D2[1], self.D2dc)
        except AssertionError as e1:
            try:
                self.assertAlmostEquals(S2[1], self.S1dc)
                self.assertAlmostEquals(S1[1], self.S2dc)
                self.assertAlmostEquals(D2[1], self.D1dc)
                self.assertAlmostEquals(D1[1], self.D2dc)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)
        try:
            self.assertAlmostEquals(S1[0], self.S1mt)
            self.assertAlmostEquals(S2[0], self.S2mt)
            self.assertAlmostEquals(D1[0], self.D1mt)
            self.assertAlmostEquals(D2[0], self.D2mt)
        except AssertionError as e1:
            try:
                self.assertAlmostEquals(S2[0], self.S1mt)
                self.assertAlmostEquals(S1[0], self.S2mt)
                self.assertAlmostEquals(D2[0], self.D1mt)
                self.assertAlmostEquals(D1[0], self.D2mt)
            except AssertionError as e2:
                raise AssertionError(e1.message+' or '+e2.message)

    def test_Tape_MT33(self):
        MT33 = Tape_MT33(self.Gmt, self.Dmt, self.Kmt, self.Hmt, self.Omt)
        self.assertAlmostEquals(MT33, self.MT33/self.MT33norm)
        DC33 = Tape_MT33(self.Gdc, self.Ddc, self.Kdc, self.Hdc, self.Odc)
        self.assertAlmostEquals(DC33, self.DC33)

    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_Tape_MT6_python(self, logger):
        with PythonOnly():
            MT6 = Tape_MT6(self.Gmt, self.Dmt, self.Kmt, self.Hmt, self.Omt)
            logger.info.assert_called_once_with(C_EXTENSION_FALLBACK_LOG_MSG)
            self.assertAlmostEquals(MT6, self.MT6/self.MT6norm)
            DC6 = Tape_MT6(self.Gdc, self.Ddc, self.Kdc, self.Hdc, self.Odc)
            self.assertAlmostEquals(DC6, self.DC6)

    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_Tape_MT6_multiple_python(self, logger):
        with PythonOnly():
            MTs = Tape_MT6(np.array([self.Gmt, self.Gdc]),
                           np.array([self.Dmt, self.Ddc]),
                           np.array([self.Kmt, self.Kdc]),
                           np.array([self.Hmt, self.Hdc]),
                           np.array([self.Omt, self.Odc]))
            self.assertAlmostEquals(MTs[:, 0], self.MT6/self.MT6norm)
            self.assertAlmostEquals(MTs[:, 1], self.DC6)

    @unittest.skipIf(*C_EXTENSIONS)
    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_Tape_MT6_cython(self, logger):
        MT6 = Tape_MT6(self.Gmt, self.Dmt, self.Kmt, self.Hmt, self.Omt)
        self.assertAlmostEquals(MT6, self.MT6/self.MT6norm)
        DC6 = Tape_MT6(self.Gdc, self.Ddc, self.Kdc, self.Hdc, self.Odc)
        self.assertAlmostEquals(DC6, self.DC6)
        logger.info.assert_not_called()

    @unittest.skipIf(*C_EXTENSIONS)
    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_Tape_MT6_multiple_cython(self, logger):
        MTs = Tape_MT6(np.array([self.Gmt, self.Gdc]),
                       np.array([self.Dmt, self.Ddc]),
                       np.array([self.Kmt, self.Kdc]),
                       np.array([self.Hmt, self.Hdc]),
                       np.array([self.Omt, self.Odc]))
        self.assertAlmostEquals(MTs[:, 0], self.MT6/self.MT6norm)
        self.assertAlmostEquals(MTs[:, 1], self.DC6)
        logger.info.assert_not_called()

    def test_Tape_TNPE(self):
        [Tmt, Nmt, Pmt, Emt] = Tape_TNPE(self.Gmt, self.Dmt, self.Kmt, self.Hmt, self.Omt)
        self.assertVectorEquals(Tmt, self.Tmt, 4)
        self.assertVectorEquals(Nmt, self.Nmt, 4)
        self.assertVectorEquals(Pmt, self.Pmt, 4)
        Emtnorm = np.sqrt(self.Emt.T*self.Emt)
        self.assertAlmostEquals(Emt, self.Emt/Emtnorm, 4)
        [Tdc, Ndc, Pdc, Edc] = Tape_TNPE(self.Gdc, self.Ddc, self.Kdc, self.Hdc, self.Odc)
        self.assertVectorEquals(Tdc, self.Tdc, 4)
        self.assertVectorEquals(Ndc, self.Ndc, 4)
        self.assertVectorEquals(Pdc, self.Pdc, 4)
        self.assertAlmostEquals(Edc, self.Edc, 4)
        [T, N, P, E] = Tape_TNPE(np.array([self.Gmt, self.Gdc]),
                                 np.array([self.Dmt, self.Ddc]),
                                 np.array([self.Kmt, self.Kdc]),
                                 np.array([self.Hmt, self.Hdc]),
                                 np.array([self.Omt, self.Odc]))
        self.assertVectorEquals(T[:, 0], self.Tmt, 4)
        self.assertVectorEquals(N[:, 0], self.Nmt, 4)
        self.assertVectorEquals(P[:, 0], self.Pmt, 4)
        self.assertAlmostEquals(E[:, 0], self.Emt/Emtnorm, 4)
        self.assertVectorEquals(T[:, 1], self.Tdc, 4)
        self.assertVectorEquals(N[:, 1], self.Ndc, 4)
        self.assertVectorEquals(P[:, 1], self.Pdc, 4)
        self.assertAlmostEquals(E[:, 1], self.Edc, 4)

    def test_normal_SD(self):
        s, d = normal_SD(self.N1dc)
        self.assertAlmostEquals(s, self.S1dc)
        self.assertAlmostEquals(d, self.D1dc)
        s, d = normal_SD(self.N1mt)
        self.assertAlmostEquals(s, self.S1mt)
        self.assertAlmostEquals(d, self.D1mt)
        s, d = normal_SD(self.N2dc)
        self.assertAlmostEquals(s, self.S2dc)
        self.assertAlmostEquals(d, self.D2dc)
        s, d = normal_SD(self.N2mt)
        self.assertAlmostEquals(s, self.S2mt)
        self.assertAlmostEquals(d, self.D2mt)

    def test_normal_SD_array(self):
        s, d = normal_SD(np.array(self.N1dc))
        self.assertAlmostEquals(s, self.S1dc)
        self.assertAlmostEquals(d, self.D1dc)

    def test_toa_vec_radians(self):
        vec = toa_vec(np.pi/4, np.pi/4, True)
        self.assertVectorEquals(vec, np.matrix([[0.5], [0.5], [1/np.sqrt(2)]]))

    def test_toa_vec_degrees(self):
        vec = toa_vec(45., 45., False)
        self.assertVectorEquals(vec, np.matrix([[0.5], [0.5], [1/np.sqrt(2)]]))

    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_output_convert_mt(self, logger):
        with PythonOnly():
            result_mt = output_convert(self.MT6)
            self.assertEqual(logger.info.call_args_list[0], mock.call(C_EXTENSION_FALLBACK_LOG_MSG))
            for key in result_mt.keys():
                new_key = key
                if key == 's':
                    new_key = 'O'
                if key in ['S1', 'D1', 'S2', 'D2']:
                    try:
                        self.assertAlmostEquals(result_mt[key],
                                                getattr(self, new_key.upper()+'mt')*180/np.pi, 5)
                    except AssertionError as e:
                        if key in ['S1', 'D1']:
                            new_key = key.replace('1', '2')
                        else:
                            new_key = key.replace('2', '1')
                        try:
                            self.assertAlmostEquals(result_mt[key],
                                                    getattr(self, new_key.upper()+'mt')*180/np.pi, 5)
                        except AssertionError as e2:
                            try:
                                self.assertAlmostEquals(result_mt[key]-180,
                                                        getattr(self, new_key.upper()+'mt')*180/np.pi, 5)
                            except AssertionError as e3:
                                raise AssertionError('Key:'+str(key)+'\n'+e.message+' or '+e2.message+' or '+e3.message)

                elif key in ['R1', 'R2']:
                    try:
                        self.assertSigmaEquals(result_mt[key],
                                               getattr(self, new_key.upper()+'mt')*180/np.pi, 5)
                    except AssertionError as e:
                        if key in ['R1']:
                            new_key = key.replace('1', '2')
                        else:
                            new_key = key.replace('2', '1')
                        try:
                            self.assertSigmaEquals(result_mt[key],
                                                   getattr(self, new_key.upper()+'mt')*180/np.pi, 5)
                        except AssertionError as e2:
                            raise AssertionError('Key:'+str(key)+'\n'+e.message+' or '+e2.message)

                elif key in ['k']:
                    try:
                        self.assertSigmaEquals(result_mt[key],
                                               getattr(self, new_key.upper()+'mt'), 5)
                    except AssertionError as e:
                        try:
                            self.assertSigmaEquals(result_mt[key]-np.pi,
                                                   getattr(self, new_key.upper()+'mt'), 5)
                        except AssertionError as e2:
                            raise AssertionError('Key:'+str(key)+'\n'+e.message+' or '+e2.message)
                else:
                    try:
                        self.assertAlmostEquals(result_mt[key],
                                                getattr(self, new_key.upper()+'mt'), 5)
                    except AssertionError as e:
                        raise AssertionError('Key:'+str(key)+'\n'+e.message)

    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_output_convert_dc(self, logger):
        with PythonOnly():
            result_dc = output_convert(self.DC6)
            self.assertEqual(logger.info.call_args_list[0], mock.call(C_EXTENSION_FALLBACK_LOG_MSG))
            for key in result_dc.keys():
                new_key = key
                if key == 's':
                    new_key = 'O'
                if key in ['S1', 'D1', 'S2', 'D2']:
                    try:
                        self.assertAlmostEquals(result_dc[key],
                                                getattr(self, new_key.upper()+'dc')*180/np.pi, 5)
                    except AssertionError as e:
                        if key in ['S1', 'D1']:
                            new_key = key.replace('1', '2')
                        else:
                            new_key = key.replace('2', '1')
                        try:
                            self.assertAlmostEquals(result_dc[key],
                                                    getattr(self, new_key.upper()+'dc')*180/np.pi, 5)
                        except AssertionError as e2:
                            try:
                                self.assertAlmostEquals(result_dc[key]-180, getattr(self, new_key.upper()+'dc')*180/np.pi, 5)
                            except AssertionError as e3:
                                raise AssertionError('Key:'+str(key)+'\n'+e.message+' or '+e2.message+' or '+e3.message)

                elif key in ['R1', 'R2']:
                    try:
                        self.assertSigmaEquals(result_dc[key]*np.pi/180., getattr(self, new_key.upper()+'dc'), 5)
                    except AssertionError as e:
                        if key in ['R1']:
                            new_key = key.replace('1', '2')
                        else:
                            new_key = key.replace('2', '1')
                        try:
                            self.assertSigmaEquals(result_dc[key]*np.pi/180., getattr(self, new_key.upper()+'dc'), 5)
                        except AssertionError as e2:
                            raise AssertionError('Key:'+str(key)+'\n'+e.message+' or '+e2.message)

                elif key in ['k']:
                    try:
                        self.assertSigmaEquals(result_dc[key],
                                               getattr(self, new_key.upper()+'dc'), 5)
                    except AssertionError as e:
                        try:
                            self.assertSigmaEquals(result_dc[key]-np.pi,
                                                   getattr(self, new_key.upper()+'dc'), 5)
                        except AssertionError as e2:
                            raise AssertionError('Key:'+str(key)+'\n'+e.message+' or '+e2.message)
                else:
                    try:
                        self.assertAlmostEquals(result_dc[key],
                                                getattr(self, new_key.upper()+'dc'), 5)
                    except AssertionError as e:
                        raise AssertionError('Key:'+str(key)+'\n'+e.message)

    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_output_convert_multiple(self, logger):
        with PythonOnly():
            result = output_convert(np.append(self.MT6, self.DC6, 1))
            self.assertEqual(logger.info.call_args_list[0], mock.call(C_EXTENSION_FALLBACK_LOG_MSG))
            for key in result.keys():
                new_key = key
                if key == 's':
                    new_key = 'O'
                try:
                    self.assertAlmostEquals(result[key][:, 0],
                                            getattr(self, new_key.upper()+'mt'), 5)
                except IndexError:
                    if key in ['S1', 'D1', 'S2', 'D2']:
                        try:
                            self.assertAlmostEquals(result[key][0]*np.pi/180,
                                                    getattr(self, new_key.upper()+'mt'), 5)
                        except AssertionError as e:
                            if key in ['S1', 'D1']:
                                new_key = key.replace('1', '2')
                            else:
                                new_key = key.replace('2', '1')
                            try:
                                self.assertAlmostEquals(result[key][0]*np.pi/180,
                                                        getattr(self, new_key.upper()+'mt'), 5)
                            except AssertionError as e2:
                                try:
                                    self.assertAlmostEquals(result[key][0]-180,
                                                            getattr(self, new_key.upper()+'mt')*180/np.pi, 5)
                                except AssertionError as e3:
                                    raise AssertionError('Key:'+str(key)+'\n'+e.message+' or '+e2.message+' or '+e3.message)

                    elif key in ['R1', 'R2']:
                        try:
                            self.assertSigmaEquals(result[key][0]*np.pi/180,
                                                   getattr(self, new_key.upper()+'mt'), 5)
                        except AssertionError as e:
                            if key in ['R1']:
                                new_key = key.replace('1', '2')
                            else:
                                new_key = key.replace('2', '1')
                            try:
                                self.assertSigmaEquals(result[key][0]*np.pi/180,
                                                       getattr(self, new_key.upper()+'mt'), 5)
                            except AssertionError as e2:
                                raise AssertionError('Key:'+str(key)+'\n'+e.message+' or '+e2.message)

                    elif key in ['k']:
                        try:
                            self.assertSigmaEquals(result[key][0],
                                                   getattr(self, new_key.upper()+'mt'), 5)
                        except AssertionError as e:
                            try:
                                self.assertSigmaEquals(result[key][0]-np.pi,
                                                       getattr(self, new_key.upper()+'mt'), 5)
                            except AssertionError as e2:
                                raise AssertionError('Key:'+str(key)+'\n'+e.message+' or '+e2.message)
                    else:
                        try:
                            self.assertAlmostEquals(result[key][0],
                                                    getattr(self, new_key.upper()+'mt'), 5)
                        except AssertionError as e:
                            raise AssertionError('Key:'+str(key)+'\n'+e.message)
                except AssertionError as e:
                    raise AssertionError('Key:'+str(key)+'\n'+e.message)
                try:
                    self.assertAlmostEquals(result[key][:, 1],
                                            getattr(self, new_key.upper()+'dc'), 5)
                except IndexError:
                    if key in ['S1', 'D1', 'S2', 'D2']:
                        try:
                            self.assertAlmostEquals(result[key][1]*np.pi/180,
                                                    getattr(self, new_key.upper()+'dc'), 5)
                        except AssertionError as e:
                            if key in ['S1', 'D1']:
                                new_key = key.replace('1', '2')
                            else:
                                new_key = key.replace('2', '1')
                            try:
                                self.assertAlmostEquals(result[key][1]*np.pi/180,
                                                        getattr(self, new_key.upper()+'dc'), 5)
                            except AssertionError as e2:
                                try:
                                    self.assertAlmostEquals(result[key][1]-180,
                                                            getattr(self, new_key.upper()+'dc')*180/np.pi, 5)
                                except AssertionError as e3:
                                    raise AssertionError('Key:'+str(key)+'\n'+e.message+' or '+e2.message+' or '+e3.message)

                    elif key in ['R1', 'R2']:
                        try:
                            self.assertSigmaEquals(result[key][1]*np.pi/180,
                                                   getattr(self, new_key.upper()+'dc'), 5)
                        except AssertionError as e:
                            if key in ['R1']:
                                new_key = key.replace('1', '2')
                            else:
                                new_key = key.replace('2', '1')
                            try:
                                self.assertSigmaEquals(result[key][1]*np.pi/180,
                                                       getattr(self, new_key.upper()+'dc'), 5)
                            except AssertionError as e2:
                                raise AssertionError('Key:'+str(key)+'\n'+e.message+' or '+e2.message)

                    elif key in ['k']:
                        try:
                            self.assertSigmaEquals(result[key][1],
                                                   getattr(self, new_key.upper()+'dc'), 5)
                        except AssertionError as e:
                            try:
                                self.assertSigmaEquals(result[key][1]-np.pi,
                                                       getattr(self, new_key.upper()+'dc'), 5)
                            except AssertionError as e2:
                                raise AssertionError('Key:'+str(key)+'\n'+e.message+' or '+e2.message)
                    else:
                        try:
                            self.assertAlmostEquals(result[key][1],
                                                    getattr(self, new_key.upper()+'dc'), 5)
                        except AssertionError as e:
                            raise AssertionError('Key:'+str(key)+'\n'+e.message)
                except AssertionError as e:
                    raise AssertionError('Key:'+str(key)+'\n'+e.message)

    @unittest.skipIf(*C_EXTENSIONS)
    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_output_convert_cython(self, logger):
        result_mt = output_convert(self.MT6)
        logger.info.assert_not_called()
        for key in result_mt.keys():
            new_key = key
            if key == 's':
                new_key = 'O'
            if key in ['S1', 'D1', 'S2', 'D2']:
                try:
                    self.assertAlmostEquals(result_mt[key],
                                            getattr(self, new_key.upper()+'mt')*180/np.pi, 5)
                except AssertionError as e:
                    if key in ['S1', 'D1']:
                        new_key = key.replace('1', '2')
                    else:
                        new_key = key.replace('2', '1')
                    try:
                        self.assertAlmostEquals(result_mt[key],
                                                getattr(self, new_key.upper()+'mt')*180/np.pi, 5)
                    except AssertionError as e2:
                        try:
                            self.assertAlmostEquals(result_mt[key]-180,
                                                    getattr(self, new_key.upper()+'mt')*180/np.pi, 5)
                        except AssertionError as e3:
                            raise AssertionError(
                                'Key:'+str(key)+'\n'+e.message+' or '+e2.message+' or '+e3.message)

            elif key in ['R1', 'R2']:
                try:
                    self.assertSigmaEquals(result_mt[key],
                                           getattr(self, new_key.upper()+'mt')*180/np.pi, 5)
                except AssertionError as e:
                    if key in ['R1']:
                        new_key = key.replace('1', '2')
                    else:
                        new_key = key.replace('2', '1')
                    try:
                        self.assertSigmaEquals(result_mt[key],
                                               getattr(self, new_key.upper()+'mt')*180/np.pi, 5)
                    except AssertionError as e2:
                        raise AssertionError(
                            'Key:'+str(key)+'\n'+e.message+' or '+e2.message)

            elif key in ['k']:
                try:
                    self.assertSigmaEquals(result_mt[key],
                                           getattr(self, new_key.upper()+'mt'), 5)
                except AssertionError as e:
                    try:
                        self.assertSigmaEquals(result_mt[key]-np.pi,
                                               getattr(self, new_key.upper()+'mt'), 5)
                    except AssertionError as e2:
                        raise AssertionError(
                            'Key:'+str(key)+'\n'+e.message+' or '+e2.message)
            else:
                try:
                    self.assertAlmostEquals(result_mt[key],
                                            getattr(self, new_key.upper()+'mt'), 5)
                except AssertionError as e:
                    raise AssertionError('Key:'+str(key)+'\n'+e.message)
        result_dc = output_convert(self.DC6)
        for key in result_dc.keys():
            new_key = key
            if key == 's':
                new_key = 'O'
            if key in ['S1', 'D1', 'S2', 'D2']:
                try:
                    self.assertAlmostEquals(result_dc[key],
                                            getattr(self, new_key.upper()+'dc')*180/np.pi, 5)
                except AssertionError as e:
                    if key in ['S1', 'D1']:
                        new_key = key.replace('1', '2')
                    else:
                        new_key = key.replace('2', '1')
                    try:
                        self.assertAlmostEquals(result_dc[key],
                                                getattr(self, new_key.upper()+'dc')*180/np.pi, 5)
                    except AssertionError as e2:
                        try:
                            self.assertAlmostEquals(result_dc[key]-180,
                                                    getattr(self, new_key.upper()+'dc')*180/np.pi, 5)
                        except AssertionError as e3:
                            raise AssertionError(
                                'Key:'+str(key)+'\n'+e.message+' or '+e2.message+' or '+e3.message)

            elif key in ['R1', 'R2']:
                try:
                    self.assertSigmaEquals(result_dc[key]*np.pi/180.,
                                           getattr(self, new_key.upper()+'dc'), 5)
                except AssertionError as e:
                    if key in ['R1']:
                        new_key = key.replace('1', '2')
                    else:
                        new_key = key.replace('2', '1')
                    try:
                        self.assertSigmaEquals(result_dc[key]*np.pi/180.,
                                               getattr(self, new_key.upper()+'dc'), 5)
                    except AssertionError as e2:
                        raise AssertionError('Key:'+str(key)+'\n'+e.message+' or '+e2.message)

            elif key in ['k']:
                try:
                    self.assertSigmaEquals(result_dc[key],
                                           getattr(self, new_key.upper()+'dc'), 5)
                except AssertionError as e:
                    try:
                        self.assertSigmaEquals(result_dc[key]-np.pi,
                                               getattr(self, new_key.upper()+'dc'), 5)
                    except AssertionError as e2:
                        raise AssertionError('Key:'+str(key)+'\n'+e.message+' or '+e2.message)
            else:
                try:
                    self.assertAlmostEquals(result_dc[key],
                                            getattr(self, new_key.upper()+'dc'), 5)
                except AssertionError as e:
                    raise AssertionError('Key:'+str(key)+'\n'+e.message)
        result = output_convert(np.append(self.MT6, self.DC6, 1))
        for key in result.keys():
            new_key = key
            if key == 's':
                new_key = 'O'
            try:
                self.assertAlmostEquals(result[key][:, 0],
                                        getattr(self, new_key.upper()+'mt'), 5)
            except IndexError:
                if key in ['S1', 'D1', 'S2', 'D2']:
                    try:
                        self.assertAlmostEquals(result[key][0]*np.pi/180,
                                                getattr(self, new_key.upper()+'mt'), 5)
                    except AssertionError as e:
                        if key in ['S1', 'D1']:
                            new_key = key.replace('1', '2')
                        else:
                            new_key = key.replace('2', '1')
                        try:
                            self.assertAlmostEquals(result[key][0]*np.pi/180,
                                                    getattr(self, new_key.upper()+'mt'), 5)
                        except AssertionError as e2:
                            try:
                                self.assertAlmostEquals(result[key][0]-180,
                                                        getattr(self, new_key.upper()+'mt')*180/np.pi, 5)
                            except AssertionError as e3:
                                raise AssertionError(
                                    'Key:'+str(key)+'\n'+e.message+' or '+e2.message+' or '+e3.message)

                elif key in ['R1', 'R2']:
                    try:
                        self.assertSigmaEquals(result[key][0]*np.pi/180,
                                               getattr(self, new_key.upper()+'mt'), 5)
                    except AssertionError as e:
                        if key in ['R1']:
                            new_key = key.replace('1', '2')
                        else:
                            new_key = key.replace('2', '1')
                        try:
                            self.assertSigmaEquals(result[key][0]*np.pi/180,
                                                   getattr(self, new_key.upper()+'mt'), 5)
                        except AssertionError as e2:
                            raise AssertionError(
                                'Key:'+str(key)+'\n'+e.message+' or '+e2.message)

                elif key in ['k']:
                    try:
                        self.assertSigmaEquals(result[key][0],
                                               getattr(self, new_key.upper()+'mt'), 5)
                    except AssertionError as e:
                        try:
                            self.assertSigmaEquals(result[key][0]-np.pi,
                                                   getattr(self, new_key.upper()+'mt'), 5)
                        except AssertionError as e2:
                            raise AssertionError('Key:'+str(key)+'\n'+e.message+' or '+e2.message)
                else:
                    try:
                        self.assertAlmostEquals(result[key][0],
                                                getattr(self, new_key.upper()+'mt'), 5)
                    except AssertionError as e:
                        raise AssertionError('Key:'+str(key)+'\n'+e.message)
            except AssertionError as e:
                raise AssertionError('Key:'+str(key)+'\n'+e.message)
            try:
                self.assertAlmostEquals(result[key][:, 1],
                                        getattr(self, new_key.upper()+'dc'), 5)
            except IndexError:
                if key in ['S1', 'D1', 'S2', 'D2']:
                    try:
                        self.assertAlmostEquals(result[key][1]*np.pi/180,
                                                getattr(self, new_key.upper()+'dc'), 5)
                    except AssertionError as e:
                        if key in ['S1', 'D1']:
                            new_key = key.replace('1', '2')
                        else:
                            new_key = key.replace('2', '1')
                        try:
                            self.assertAlmostEquals(result[key][1]*np.pi/180,
                                                    getattr(self, new_key.upper()+'dc'), 5)
                        except AssertionError as e2:
                            try:
                                self.assertAlmostEquals(result[key][1]-180,
                                                        getattr(self, new_key.upper()+'dc')*180/np.pi, 5)
                            except AssertionError as e3:
                                raise AssertionError(
                                    'Key:'+str(key)+'\n'+e.message+' or '+e2.message+' or '+e3.message)

                elif key in ['R1', 'R2']:
                    try:
                        self.assertSigmaEquals(result[key][1]*np.pi/180,
                                               getattr(self, new_key.upper()+'dc'), 5)
                    except AssertionError as e:
                        if key in ['R1']:
                            new_key = key.replace('1', '2')
                        else:
                            new_key = key.replace('2', '1')
                        try:
                            self.assertSigmaEquals(result[key][1]*np.pi/180,
                                                   getattr(self, new_key.upper()+'dc'), 5)
                        except AssertionError as e2:
                            raise AssertionError(
                                'Key:'+str(key)+'\n'+e.message+' or '+e2.message)

                elif key in ['k']:
                    try:
                        self.assertSigmaEquals(result[key][1],
                                               getattr(self, new_key.upper()+'dc'), 5)
                    except AssertionError as e:
                        try:
                            self.assertSigmaEquals(result[key][1]-np.pi,
                                                   getattr(self, new_key.upper()+'dc'), 5)
                        except AssertionError as e2:
                            raise AssertionError(
                                'Key:'+str(key)+'\n'+e.message+' or '+e2.message)
                else:
                    try:
                        self.assertAlmostEquals(result[key][1],
                                                getattr(self, new_key.upper()+'dc'), 5)
                    except AssertionError as e:
                        raise AssertionError('Key:'+str(key)+'\n'+e.message)
            except AssertionError as e:
                raise AssertionError('Key:'+str(key)+'\n'+e.message)

    @unittest.skipIf(*C_EXTENSIONS)
    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_MT6_biaxes_python(self, logger):
        with PythonOnly():
            # Tested against MATLAB code
            phi, explosion, area_displacement = MT6_biaxes(np.array([2., 1, 0, -1, 0, 0]))
            logger.info.assert_called_with(C_EXTENSION_FALLBACK_LOG_MSG)
            self.assertAlmostEquals(phi, np.array([[0.75983569,  0.75983569],
                                                   [-0.39331989, -0.39331989],
                                                   [0.51763809, -0.51763809]]))
            self.assertAlmostEquals(area_displacement, 1.18301270189222)
            self.assertAlmostEquals(explosion, 0.0849364905389036)
            phi, explosion, area_displacement = MT6_biaxes(
                np.array([[2, 3], [1, 0], [0, -3], [-1, 0], [0, 0], [0, 0]]))
            self.assertAlmostEquals(phi, np.array([[[0.75983569, 0.75983569], [-0.39331989, -0.39331989],
                                                    [0.51763809, -0.51763809]],
                                                   [[0.70710678, 0.70710678], [0., 0.],
                                                    [0.70710678, -0.70710678]]]))
            self.assertAlmostEquals(area_displacement, np.array([1.1830127, 3.]))
            self.assertAlmostEquals(explosion, np.array([0.08493649, 0.]))
            vtic = [1.0000, 2.0000, 0.5000, 0, 0, 0, 1.0000, 0.5000,
                    0, 0, 0, 1.0000, 0, 0, 0, 6.0000, 0, 0, 6.0000, 0, 4.]
            phi, explosion, area_displacement = MT6_biaxes(
                np.array([2, 1, 0, -1, 0, 0]), vtic)
            try:
                self.assertVectorEquals(phi[:, 0], np.array([-0.062501878200605, 0.712609970150016, 0.698770738986823]))
                self.assertVectorEquals(phi[:, 1], np.array(
                    [-0.062501878200605, 0.712609970150016, -0.698770738986823]))
            except AssertionError:
                self.assertVectorEquals(phi[:, 1], np.array([-0.062501878200605, 0.712609970150016, 0.698770738986823]))
                self.assertVectorEquals(phi[:, 0], np.array([-0.062501878200605, 0.712609970150016, -0.698770738986823]))
            self.assertAlmostEquals(area_displacement, 1.984495199420505)
            self.assertAlmostEquals(explosion, 0.461237998551262)

    @unittest.skipIf(*C_EXTENSIONS)
    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_MT6_biaxes_cython(self, logger):
        mtc._CYTHON = True
        # Tested against MATLAB code
        phi, explosion, area_displacement = MT6_biaxes(
            np.array([2, 1, 0, -1, 0, 0]))
        self.assertAlmostEquals(phi, np.array(
            [[0.75983569,  0.75983569], [-0.39331989, -0.39331989], [0.51763809, -0.51763809]]))
        self.assertAlmostEquals(area_displacement, 1.18301270189222)
        self.assertAlmostEquals(explosion, 0.0849364905389036)
        phi, explosion, area_displacement = MT6_biaxes(
            np.array([[2, 3], [1, 0], [0, -3], [-1, 0], [0, 0], [0, 0]]))
        self.assertAlmostEquals(phi, np.array([[[0.75983569, 0.75983569], [-0.39331989, -0.39331989], [0.51763809, -0.51763809]],
                                               [[0.70710678, 0.70710678], [0., 0.], [0.70710678, -0.70710678]]]))
        self.assertAlmostEquals(area_displacement, np.array([1.1830127, 3.]))
        self.assertAlmostEquals(explosion, np.array([0.08493649, 0.]))
        vtic = [1.0000, 2.0000, 0.5000, 0, 0, 0, 1.0000, 0.5000,
                0, 0, 0, 1.0000, 0, 0, 0, 6.0000, 0, 0, 6.0000, 0, 4.]
        phi, explosion, area_displacement = MT6_biaxes(
            np.array([2, 1, 0, -1, 0, 0]), vtic)
        try:
            self.assertVectorEquals(
                phi[:, 0], np.array([-0.062501878200605, 0.712609970150016, 0.698770738986823]))
            self.assertVectorEquals(phi[:, 1], np.array(
                [-0.062501878200605, 0.712609970150016, -0.698770738986823]))
        except AssertionError:
            self.assertVectorEquals(
                phi[:, 1], np.array([-0.062501878200605, 0.712609970150016, 0.698770738986823]))
            self.assertVectorEquals(phi[:, 0], np.array(
                [-0.062501878200605, 0.712609970150016, -0.698770738986823]))
        self.assertAlmostEquals(area_displacement, 1.984495199420505)
        self.assertAlmostEquals(explosion, 0.461237998551262)
        logger.info.assert_not_called()

    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_MT6c_D6_python(self, logger):
        with PythonOnly():
            D6 = MT6c_D6(self.MT6, isotropic_c())
            logger.info.assert_called_with(C_EXTENSION_FALLBACK_LOG_MSG)
            Td, Nd, Pd, Ed = MT6_TNPE(D6)
            try:
                self.assertAlmostEquals(Td, self.Tmt)
            except Exception:
                self.assertAlmostEquals(Td, -self.Tmt)
            try:
                self.assertAlmostEquals(Nd, self.Nmt)
            except Exception:
                self.assertAlmostEquals(Nd, -self.Nmt)
            try:
                self.assertAlmostEquals(Pd, self.Pmt)
            except Exception:
                self.assertAlmostEquals(Pd, -self.Pmt)
            with self.assertRaises(AssertionError):
                self.assertAlmostEquals(Ed, self.Emt)

    @unittest.skipIf(*C_EXTENSIONS)
    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_MT6c_D6_cython(self, logger):
        D6 = MT6c_D6(self.MT6, isotropic_c())
        logger.info.assert_not_called()
        Td, Nd, Pd, Ed = MT6_TNPE(D6)
        try:
            self.assertAlmostEquals(Td, self.Tmt)
        except Exception:
            self.assertAlmostEquals(Td, -self.Tmt)
        try:
            self.assertAlmostEquals(Nd, self.Nmt)
        except Exception:
            self.assertAlmostEquals(Nd, -self.Nmt)
        try:
            self.assertAlmostEquals(Pd, self.Pmt)
        except Exception:
            self.assertAlmostEquals(Pd, -self.Pmt)
        with self.assertRaises(AssertionError):
            self.assertAlmostEquals(Ed, self.Emt)

    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_c_norm_python(self, logger):
        with PythonOnly():
            self.assertEqual(c_norm(isotropic_c()), 6.7082039324993694)
            self.assertEqual(c_norm(isotropic_c(2)), 9.16515138991168)
            self.assertEqual(c_norm(isotropic_c(2, 4)), 22.715633383201094)
            logger.info.assert_called_with(C_EXTENSION_FALLBACK_LOG_MSG)
            # Matlab values

    @unittest.skipIf(*C_EXTENSIONS)
    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_c_norm_cython(self, logger):
        self.assertEqual(c_norm(isotropic_c()), 6.7082039324993694)
        self.assertEqual(c_norm(isotropic_c(2)), 9.16515138991168)
        self.assertEqual(c_norm(isotropic_c(2, 4)), 22.715633383201094)
        # Matlab values
        logger.info.assert_not_called()

    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_isotropic_c_python(self, logger):
        with PythonOnly():
            self.assertEqual(isotropic_c(),
                             [3, 1, 1, 0, 0, 0, 3, 1, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 1, 0, 1])
            self.assertEqual(isotropic_c(2),
                             [4, 2, 2, 0, 0, 0, 4, 2, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 1, 0, 1])
            vtic = [1.0000, 2.0000, 0.5000, 0, 0, 0, 1.0000, 0.5000,
                    0, 0, 0, 1.0000, 0, 0, 0, 6.0000, 0, 0, 6.0000, 0, 4.]
            self.assertAlmostEquals(np.array(isotropic_c(c=vtic)),
                                    np.array([5.2666666666666675, -1.1333333333333333,
                                              -1.1333333333333333, 0, 0, 0,
                                              5.2666666666666675, -1.1333333333333333,
                                              0, 0, 0, 5.2666666666666675, 0, 0, 0,
                                              3.2000000000000002, 0, 0, 3.2000000000000002,
                                              0, 3.2000000000000002]))
            logger.info.assert_called_with(C_EXTENSION_FALLBACK_LOG_MSG)

    @unittest.skipIf(*C_EXTENSIONS)
    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_isotropic_c_cython(self, logger):
        self.assertEqual(isotropic_c(),
                         [3, 1, 1, 0, 0, 0, 3, 1, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 1, 0, 1])
        self.assertEqual(isotropic_c(2),
                         [4, 2, 2, 0, 0, 0, 4, 2, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 1, 0, 1])
        vtic = [1.0000, 2.0000, 0.5000, 0, 0, 0, 1.0000, 0.5000,
                0, 0, 0, 1.0000, 0, 0, 0, 6.0000, 0, 0, 6.0000, 0, 4.]
        self.assertAlmostEquals(np.array(isotropic_c(c=vtic)),
                                np.array([5.2666666666666675, -1.1333333333333333,
                                          -1.1333333333333333, 0, 0, 0, 5.2666666666666675,
                                          -1.1333333333333333, 0, 0, 0, 5.2666666666666675,
                                          0, 0, 0, 3.2000000000000002, 0, 0, 3.2000000000000002,
                                          0, 3.2000000000000002]))
        logger.info.assert_not_called()

    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_is_isotropic_c_python(self, logger):
        with PythonOnly():
            self.assertTrue(is_isotropic_c(isotropic_c()))
            self.assertTrue(is_isotropic_c(isotropic_c(2)))
            vtic = [1.0000, 2.0000, 0.5000, 0, 0, 0, 1.0000, 0.5000,
                    0, 0, 0, 1.0000, 0, 0, 0, 6.0000, 0, 0, 6.0000, 0, 4.]
            self.assertFalse(is_isotropic_c(vtic))
            self.assertTrue(is_isotropic_c(isotropic_c(c=vtic)))
            logger.info.assert_called_with(C_EXTENSION_FALLBACK_LOG_MSG)

    @unittest.skipIf(*C_EXTENSIONS)
    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_is_isotropic_c_cython(self, logger):
        self.assertTrue(is_isotropic_c(isotropic_c()))
        self.assertTrue(is_isotropic_c(isotropic_c(2)))
        vtic = [1.0000, 2.0000, 0.5000, 0, 0, 0, 1.0000, 0.5000,
                0, 0, 0, 1.0000, 0, 0, 0, 6.0000, 0, 0, 6.0000, 0, 4.]
        self.assertFalse(is_isotropic_c(vtic))
        self.assertTrue(is_isotropic_c(isotropic_c(c=vtic)))
        logger.info.assert_not_called()

    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_c21_cvoigt_python(self, logger):
        with PythonOnly():
            self.assertAlmostEquals(c21_cvoigt(isotropic_c()), np.array([[3, 1, 1, 0, 0, 0],
                                                                         [1, 3, 1,
                                                                             0, 0, 0],
                                                                         [1, 1, 3,
                                                                             0, 0, 0],
                                                                         [0, 0, 0,
                                                                             2, 0, 0],
                                                                         [0, 0, 0,
                                                                             0, 2, 0],
                                                                         [0, 0, 0, 0, 0, 2]]))
            self.assertAlmostEquals(c21_cvoigt(isotropic_c(2)), np.array([[4, 2, 2, 0, 0, 0],
                                                                          [2, 4, 2,
                                                                              0, 0, 0],
                                                                          [2, 2, 4,
                                                                              0, 0, 0],
                                                                          [0, 0, 0,
                                                                              2, 0, 0],
                                                                          [0, 0, 0,
                                                                              0, 2, 0],
                                                                          [0, 0, 0, 0, 0, 2]]))
            vtic = [1.0000, 2.0000, 0.5000, 0, 0, 0, 1.0000, 0.5000,
                    0, 0, 0, 1.0000, 0, 0, 0, 6.0000, 0, 0, 6.0000, 0, 4.]
            self.assertAlmostEquals(c21_cvoigt(vtic), np.array([[1, 2, 0.5, 0, 0, 0],
                                                                [2, 1, 0.5,
                                                                    0, 0, 0],
                                                                [0.5, 0.5, 1,
                                                                    0, 0, 0],
                                                                [0, 0,  0,
                                                                    12, 0, 0],
                                                                [0, 0,  0, 0,
                                                                    12, 0],
                                                                [0, 0,  0, 0, 0, 8]]))
            logger.info.assert_called_with(C_EXTENSION_FALLBACK_LOG_MSG)

    @unittest.skipIf(*C_EXTENSIONS)
    @mock.patch('MTfit.convert.moment_tensor_conversion.logger')
    def test_c21_cvoigt_cython(self, logger):
        self.assertAlmostEquals(c21_cvoigt(isotropic_c()), np.array([[3, 1, 1, 0, 0, 0],
                                                                     [1, 3, 1,
                                                                         0, 0, 0],
                                                                     [1, 1, 3,
                                                                         0, 0, 0],
                                                                     [0, 0, 0,
                                                                         2, 0, 0],
                                                                     [0, 0, 0,
                                                                         0, 2, 0],
                                                                     [0, 0, 0, 0, 0, 2]]))
        self.assertAlmostEquals(c21_cvoigt(isotropic_c(2)), np.array([[4, 2, 2, 0, 0, 0],
                                                                      [2, 4, 2,
                                                                          0, 0, 0],
                                                                      [2, 2, 4,
                                                                          0, 0, 0],
                                                                      [0, 0, 0,
                                                                          2, 0, 0],
                                                                      [0, 0, 0,
                                                                          0, 2, 0],
                                                                      [0, 0, 0, 0, 0, 2]]))
        vtic = [1.0000, 2.0000, 0.5000, 0, 0, 0, 1.0000, 0.5000,
                0, 0, 0, 1.0000, 0, 0, 0, 6.0000, 0, 0, 6.0000, 0, 4.]
        self.assertAlmostEquals(c21_cvoigt(vtic), np.array([[1, 2, 0.5, 0, 0, 0],
                                                            [2, 1, 0.5,
                                                                0, 0, 0],
                                                            [0.5, 0.5, 1,
                                                                0, 0, 0],
                                                            [0, 0,  0,
                                                                12, 0, 0],
                                                            [0, 0,  0, 0,
                                                                12, 0],
                                                            [0, 0,  0, 0, 0, 8]]))
        logger.info.assert_not_called()
