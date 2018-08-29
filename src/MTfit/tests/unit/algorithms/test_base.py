import unittest
from types import MethodType
import sys

import numpy as np

from MTfit.utilities import C_EXTENSION_FALLBACK_LOG_MSG
from MTfit.utilities.unittest_utils import get_extension_skip_if_args
import MTfit.algorithms.base as base

if sys.version_info >= (3, 3):
    from unittest import mock
else:
    import mock


C_EXTENSIONS = get_extension_skip_if_args('MTfit.probability.cprobability')


class PythonOnly(object):

    def __enter__(self, *args, **kwargs):
        self.cprobability = base.cprobability
        base.cprobability = False

    def __exit__(self, *args, **kwargs):
        base.cprobability = self.cprobability


class BaseAlgorithmTestCase(unittest.TestCase):

    def setUp(self, **kwargs):
        self.base_algorithm = base.BaseAlgorithm(**kwargs)

    def tearDown(self):
        del self.base_algorithm

    def test_initialise(self):
        self.assertFalse(self.base_algorithm.initialise()[1])

    def test_iterate(self):
        self.assertTrue(self.base_algorithm.iterate({})[1])

    @mock.patch('MTfit.algorithms.base.BaseAlgorithm.get_sampling_model')
    def test___init__(self, get_sampling_model):
        self.base_algorithm = base.BaseAlgorithm()
        self.assertFalse(self.base_algorithm.dc)
        self.assertEqual(self.base_algorithm.number_samples, 10000)
        self.assertFalse(self.base_algorithm.mcmc)
        self.assertFalse(self.base_algorithm.basic_cdc)
        self.assertFalse(self.base_algorithm.quality_check)
        self.assertFalse(self.base_algorithm.generate)
        self.assertEqual(self.base_algorithm.number_events, 1)
        self.assertFalse(self.base_algorithm._model)
        get_sampling_model.assert_called_once_with({}, False, True)

    @mock.patch('MTfit.algorithms.base.BaseAlgorithm.get_sampling_model')
    def test___init___kwargs(self, get_sampling_model):
        self.base_algorithm = base.BaseAlgorithm(basic_cdc=3,
                                                 number_events=5,
                                                 sample_distribution=7,
                                                 file_sample=True,
                                                 file_safe=False)
        self.assertFalse(self.base_algorithm.dc)
        self.assertEqual(self.base_algorithm.number_samples, 10000)
        self.assertFalse(self.base_algorithm.mcmc)
        self.assertEqual(self.base_algorithm.basic_cdc, 3)
        self.assertFalse(self.base_algorithm.quality_check)
        self.assertFalse(self.base_algorithm.generate)
        self.assertEqual(self.base_algorithm.number_events, 5)
        self.assertEqual(self.base_algorithm._model, 7)
        get_sampling_model.assert_called_once_with({'basic_cdc': 3,
                                                    'number_events': 5,
                                                    'sample_distribution': 7},
                                                   True,
                                                   False)

    @unittest.expectedFailure
    @mock.patch('MTfit.algorithms.base.get_extensions')
    def test_get_sampling_model_model(self, get_extensions):
        raise NotImplementedError()

    def test_max_value(self):
        self.assertEqual(self.base_algorithm.max_value(), 'BaseAlgorithm has no max_value')

    def test_random_sample_generate(self):
        self.base_algorithm.generate = True
        self.assertFalse(self.base_algorithm.random_sample())

    @mock.patch('MTfit.algorithms.base.BaseAlgorithm.random_dc')
    def test_random_sample_random_dc(self, random_dc):
        self.base_algorithm.dc = True
        random_dc.return_value = 5
        self.assertEqual(self.base_algorithm.random_sample(), 5)
        random_dc.assert_called_once_with()

    def test_random_sample_random_basic_cdc(self):
        self.base_algorithm.dc = False
        self.base_algorithm.basic_cdc = True
        self.base_algorithm.random_basic_cdc = mock.MagicMock(return_value=5)
        self.assertEqual(self.base_algorithm.random_sample(), 5)
        self.base_algorithm.random_basic_cdc.assert_called_once_with()

    def test_random_sample_random_model(self):
        self.base_algorithm.dc = False
        self.base_algorithm.basic_cdc = False
        self.base_algorithm._model = True
        self.base_algorithm.random_model = mock.MagicMock(return_value=5)
        self.assertEqual(self.base_algorithm.random_sample(), 5)
        self.base_algorithm.random_model.assert_called_once_with(self.base_algorithm.number_samples)

    def test_random_sample_random_mt(self):
        self.base_algorithm.dc = False
        self.base_algorithm.basic_cdc = False
        self.base_algorithm._model = False
        self.base_algorithm.random_mt = mock.MagicMock(return_value=5)
        self.assertEqual(self.base_algorithm.random_sample(), 5)
        self.base_algorithm.random_mt.assert_called_once_with()

    @unittest.expectedFailure
    def test_output(self):
        raise NotImplementedError()

    def test_clvd_sampling(self):
        self.tearDown()
        self.setUp(sample_distribution='clvd')
        if sys.version_info.major > 2:
            self.assertEqual(self.base_algorithm.random_model.__name__, self.base_algorithm.random_clvd.__name__)
        else:
            self.assertEqual(self.base_algorithm.random_model, self.base_algorithm.random_clvd)

    def test_random_mt(self):
        self.assertTrue(self.base_algorithm.random_mt().shape,
                        (6, self.base_algorithm.number_samples))

    @mock.patch('MTfit.algorithms.base.logger')
    def test_random_dc_no_cython(self, logger):
        with PythonOnly():
            self.assertTrue(self.base_algorithm.random_dc().shape,
                            (6, self.base_algorithm.number_samples))
            logger.info.assert_called_once_with(C_EXTENSION_FALLBACK_LOG_MSG)

    @unittest.skipIf(*C_EXTENSIONS)
    @mock.patch('MTfit.algorithms.base.logger')
    def test_random_dc_cython(self, logger):
        self.assertTrue(self.base_algorithm.random_dc().shape,
                        (6, self.base_algorithm.number_samples))
        logger.info.assert_not_called()

    def test_random_clvd(self):
        self.assertTrue(self.base_algorithm.random_clvd().shape,
                        (6, self.base_algorithm.number_samples))

    def test_random_type(self):
        self.base_algorithm.number_samples = 1
        self.assertAlmostEqual(
            self.base_algorithm.random_type(np.array([[1], [1], [1]]))[0, 0], 1/np.sqrt(3))

    def test_random_orthogonal_eigenvectors(self):
        self.base_algorithm.number_samples = 1
        [a, b, c] = self.base_algorithm.random_orthogonal_eigenvectors()
        self.assertAlmostEqual(c[0, 0], np.cross(a.transpose(), b.transpose()).transpose()[0, 0])
        self.assertAlmostEqual(c[1, 0], np.cross(a.transpose(), b.transpose()).transpose()[1, 0])
        self.assertAlmostEqual(c[2, 0], np.cross(a.transpose(), b.transpose()).transpose()[2, 0])

    def test_eigenvectors_mt_2_mt6(self):
        self.base_algorithm.number_samples = 1
        self.assertAlmostEqual(self.base_algorithm.eigenvectors_mt_2_mt6(np.array(
            [[1], [1], [1]]), *self.base_algorithm.random_orthogonal_eigenvectors())[0, 0], 1/np.sqrt(3))

    @unittest.skipIf(*C_EXTENSIONS)
    @mock.patch('MTfit.algorithms.base.logger')
    def test_6sphere_random_mt_cython(self, logger):
        self.base_algorithm.get_sampling_model = MethodType(base._6sphere_random_mt, self.base_algorithm)
        res = self.base_algorithm.get_sampling_model()
        logger.info.assert_not_called()
        self.assertEqual(res.shape, (6, self.base_algorithm.number_samples))
        self.assertAlmostEqual(np.prod(np.sum(np.multiply(res, res), axis=0)), 1)

    @mock.patch('MTfit.algorithms.base.logger')
    def test_6sphere_random_mt_no_cython(self, logger):
        self.base_algorithm.get_sampling_model = MethodType(base._6sphere_random_mt, self.base_algorithm)
        with PythonOnly():
            res = self.base_algorithm.get_sampling_model()
            logger.info.assert_called_once_with(C_EXTENSION_FALLBACK_LOG_MSG)
            self.assertEqual(res.shape, (6, self.base_algorithm.number_samples))
            self.assertAlmostEqual(np.prod(np.sum(np.multiply(res, res), axis=0)), 1)

    @mock.patch('MTfit.algorithms.base.logger')
    def test_6sphere_random_mt_int(self, logger):
        with PythonOnly():
            res = base._6sphere_random_mt(10)
            logger.info.assert_called_once_with(C_EXTENSION_FALLBACK_LOG_MSG)
            self.assertEqual(res.shape, (6, 10))
            self.assertAlmostEqual(np.prod(np.sum(np.multiply(res, res), axis=0)), 1)
