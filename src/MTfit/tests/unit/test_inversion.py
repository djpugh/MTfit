import unittest
import os
import glob
import time
import sys
import tempfile
import shutil
import gc
import logging
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np

from MTfit.utilities.unittest_utils import TestCase
from MTfit.inversion import McMCForwardTask
from MTfit.inversion import ForwardTask
from MTfit.inversion import MultipleEventsMcMCForwardTask
from MTfit.inversion import MultipleEventsForwardTask
from MTfit.inversion import Inversion
from MTfit.inversion import polarity_matrix
from MTfit.inversion import polarity_probability_matrix
from MTfit.inversion import amplitude_ratio_matrix
from MTfit.inversion import relative_amplitude_ratio_matrix
from MTfit.inversion import _intersect_stations
from MTfit.inversion import station_angles
from MTfit.extensions.scatangle import parse_scatangle
from MTfit.algorithms import markov_chain_monte_carlo as mcmc
from MTfit.utilities import C_EXTENSION_FALLBACK_LOG_MSG
from MTfit.utilities.unittest_utils import get_extension_skip_if_args

logger = logging.getLogger('MTfit.tests')

if sys.version_info >= (3, 3):
    from unittest import mock
else:
    import mock


C_EXTENSIONS = get_extension_skip_if_args('MTfit.algorithms.cmarkov_chain_monte_carlo')


class PythonAlgorithms(object):

    def __enter__(self, *args, **kwargs):
        self.cmarkov_chain_monte_carlo = mcmc.cmarkov_chain_monte_carlo
        mcmc.cmarkov_chain_monte_carlo = False

    def __exit__(self, *args, **kwargs):
        mcmc.cmarkov_chain_monte_carlo = self.cmarkov_chain_monte_carlo


class McMCForwardTaskTestCase(TestCase):

    def setUp(self):
        self.cwd = os.getcwd()
        if sys.version_info >= (3, 0):
            self.tempdir = tempfile.TemporaryDirectory()
            os.chdir(self.tempdir.name)
        else:
            self.tempdir = tempfile.mkdtemp()
            os.chdir(self.tempdir)
        self.existing_log_files = glob.glob('*.log')
        data = {'PPolarity': {'Stations': {'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                              'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.001], [0.001]])},
                'PPolarity2': {'Stations': {'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                               'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.001], [0.001]])}}
        a_polarity, error_polarity, incorrect_polarity_prob = polarity_matrix(data)
        a_polarity_prob, polarity_prob, incorrect_polarity_prob = polarity_probability_matrix(data)
        data = {'P/SHRMSAmplitudeRatio': {'Stations': {'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                          'Measured': np.matrix([[1.3664, 1], [1.0038, 1]]), 'Error': np.matrix([[0.001, 0.003], [0.001, 0.002]])},
                'P/SVAmplitudeRatio': {'Stations': {'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                       'Measured': np.matrix([[1.3386, 1], [0.9805, 1]]), 'Error': np.matrix([[0.001, 0.03], [0.001, 0.02]])}}
        a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio = amplitude_ratio_matrix(data)
        self.algorithm_kwargs = {'learning_length': 10, 'chain_length': 4, 'acceptance_rate_window': 20}
        self.mcmc_forward_task = McMCForwardTask(self.algorithm_kwargs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio,
                                                 amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio,
                                                 a_polarity_prob, polarity_prob, incorrect_polarity_prob)

    def tearDown(self):
        for fname in glob.glob('*.log'):
            if fname not in self.existing_log_files:
                try:
                    os.remove(fname)
                except Exception:
                    print('Cannot remove {}'.format(fname))
        del self.mcmc_forward_task
        os.chdir(self.cwd)
        if sys.version_info >= (3, 0):
            self.tempdir.cleanup()
        else:
            try:
                shutil.rmtree(self.tempdir)
            except:
                pass

    @unittest.skipIf(*C_EXTENSIONS)
    @mock.patch('MTfit.algorithms.markov_chain_monte_carlo.logger')
    def test___call__cython(self, logger):
        result = self.mcmc_forward_task()
        self.assertTrue('algorithm_output_data' in result)
        self.assertTrue('event_data' in result)
        del self.mcmc_forward_task
        location_samples = [{'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([91.0, 271.0, 120.1]), 'TakeOffAngle':np.array(
            [31.0, 61.0, 12.1])}, {'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([92.0, 272.0, 122.1]), 'TakeOffAngle':np.array([32.0, 62.0, 13.1])}]
        data = {'PPolarity': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                              'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.9], [0.9]])},
                'PPolarity2': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                               'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.9], [0.9]])}}
        a_polarity, error_polarity, incorrect_polarity_prob = polarity_matrix(data, location_samples)
        data = {'P/SHRMSAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                          'Measured': np.matrix([[1.3664, 1], [1.0038, 1]]), 'Error': np.matrix([[0.9, 0.9], [0.9, 0.9]])},
                'P/SVAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                       'Measured': np.matrix([[1.3386, 1], [0.9805, 1]]), 'Error': np.matrix([[0.9, 0.9], [0.9, 0.9]])}}
        a_polarity_prob, polarity_prob, incorrect_polarity_prob = polarity_probability_matrix(data, location_samples)
        a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio = amplitude_ratio_matrix(
            data, location_samples)
        self.mcmc_forward_task = McMCForwardTask(self.algorithm_kwargs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio,
                                                 amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio,
                                                 a_polarity_prob, polarity_prob, incorrect_polarity_prob)
        result = self.mcmc_forward_task()
        self.assertNotIn(mock.call(C_EXTENSION_FALLBACK_LOG_MSG), logger.info.call_args_list)

    @mock.patch('MTfit.algorithms.markov_chain_monte_carlo.logger')
    def test___call__python(self, logger):
        with PythonAlgorithms():
            result = self.mcmc_forward_task()
            self.assertTrue('algorithm_output_data' in result)
            self.assertTrue('event_data' in result)
            del self.mcmc_forward_task
            location_samples = [{'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([91.0, 271.0, 120.1]), 'TakeOffAngle':np.array(
                [31.0, 61.0, 12.1])}, {'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([92.0, 272.0, 122.1]), 'TakeOffAngle':np.array([32.0, 62.0, 13.1])}]
            data = {'PPolarity': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                  'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.9], [0.9]])},
                    'PPolarity2': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                   'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.9], [0.9]])}}
            a_polarity, error_polarity, incorrect_polarity_prob = polarity_matrix(data, location_samples)
            data = {'P/SHRMSAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                              'Measured': np.matrix([[1.3664, 1], [1.0038, 1]]), 'Error': np.matrix([[0.9, 0.9], [0.9, 0.9]])},
                    'P/SVAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                           'Measured': np.matrix([[1.3386, 1], [0.9805, 1]]), 'Error': np.matrix([[0.9, 0.9], [0.9, 0.9]])}}
            a_polarity_prob, polarity_prob, incorrect_polarity_prob = polarity_probability_matrix(data, location_samples)
            a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio = amplitude_ratio_matrix(
                data, location_samples)
            self.mcmc_forward_task = McMCForwardTask(self.algorithm_kwargs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio,
                                                     amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio,
                                                     a_polarity_prob, polarity_prob, incorrect_polarity_prob)
            result = self.mcmc_forward_task()
            self.assertIn(mock.call(C_EXTENSION_FALLBACK_LOG_MSG), logger.info.call_args_list)


class MultipleEventsMcMCForwardTaskTestCase(TestCase):

    def setUp(self):
        self.cwd = os.getcwd()
        if sys.version_info >= (3, 0):
            self.tempdir = tempfile.TemporaryDirectory()
            os.chdir(self.tempdir.name)
        else:
            self.tempdir = tempfile.mkdtemp()
            os.chdir(self.tempdir)
        self.existing_log_files = glob.glob('*.log')
        data = {'PPolarity': {'Stations': {'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                              'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.9], [0.9]])}}
        a_polarity, error_polarity, incorrect_polarity_prob = polarity_matrix(data)
        a_polarity_prob, polarity_prob, incorrect_polarity_prob = polarity_probability_matrix(data)
        data = {'P/SHRMSAmplitudeRatio': {'Stations': {'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                          'Measured': np.matrix([[1.3664, 1], [1.0038, 1]]), 'Error': np.matrix([[0.9, 0.9], [0.9, 0.9]])},
                'P/SVAmplitudeRatio': {'Stations': {'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                       'Measured': np.matrix([[1.3386, 1], [0.9805, 1]]), 'Error': np.matrix([[0.9, 0.9], [0.9, 0.9]])}}
        a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio = amplitude_ratio_matrix(
            data)
        self.algorithm_kwargs = {'learning_length': 10, 'chain_length': 4, 'acceptance_rate_window': 20, 'number_events': 2, 'min_number_initialisation_samples': 10}
        a_relative_amplitude, relative_amplitude, percentage_error_relative_amplitude, relative_amplitude_stations = relative_amplitude_ratio_matrix(
            data)
        a_polarity = [a_polarity, a_polarity]
        error_polarity = [error_polarity, error_polarity]
        a1_amplitude_ratio = [a1_amplitude_ratio, a1_amplitude_ratio]
        a2_amplitude_ratio = [a2_amplitude_ratio, a2_amplitude_ratio]
        percentage_error1_amplitude_ratio = [percentage_error1_amplitude_ratio, percentage_error1_amplitude_ratio]
        percentage_error2_amplitude_ratio = [percentage_error2_amplitude_ratio, percentage_error2_amplitude_ratio]
        amplitude_ratio = [amplitude_ratio, amplitude_ratio]
        a_polarity_prob = [a_polarity_prob, a_polarity_prob]
        polarity_prob = [polarity_prob, polarity_prob]
        relative_amplitude_stations = [relative_amplitude_stations, relative_amplitude_stations]
        incorrect_polarity_prob = [incorrect_polarity_prob, incorrect_polarity_prob]
        self.multiple_events_mcmc_forward_task = MultipleEventsMcMCForwardTask(self.algorithm_kwargs, a_polarity, error_polarity, a1_amplitude_ratio,
                                                                               a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio,
                                                                               percentage_error2_amplitude_ratio, a_polarity_prob, polarity_prob,
                                                                               a_relative_amplitude, relative_amplitude, percentage_error_relative_amplitude,
                                                                               relative_amplitude_stations, incorrect_polarity_prob)

    def tearDown(self):
        for fname in glob.glob('*.log'):
            if fname not in self.existing_log_files:
                try:
                    os.remove(fname)
                except Exception:
                    print('Cannot remove {}'.format(fname))
        del self.multiple_events_mcmc_forward_task
        os.chdir(self.cwd)
        if sys.version_info >= (3, 0):
            self.tempdir.cleanup()
        else:
            try:
                shutil.rmtree(self.tempdir)
            except:
                pass

    @unittest.skipIf(*C_EXTENSIONS)
    @mock.patch('MTfit.algorithms.markov_chain_monte_carlo.logger')
    def test___call___cython(self, logger):
        result = self.multiple_events_mcmc_forward_task()
        self.assertTrue('algorithm_output_data' in result)
        self.assertTrue('event_data' in result)
        self.assertNotIn(mock.call(C_EXTENSION_FALLBACK_LOG_MSG), logger.info.call_args_list)

    @mock.patch('MTfit.algorithms.markov_chain_monte_carlo.logger')
    def test___call__python(self, logger):
        with PythonAlgorithms():
            result = self.multiple_events_mcmc_forward_task()
            self.assertTrue('algorithm_output_data' in result)
            self.assertTrue('event_data' in result)
            self.assertIn(mock.call(C_EXTENSION_FALLBACK_LOG_MSG), logger.info.call_args_list)

    @unittest.skipIf(*C_EXTENSIONS)
    @mock.patch('MTfit.algorithms.markov_chain_monte_carlo.logger')
    def test___call__location_samples_cython(self, logger):
        location_samples = [{'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([91.0, 271.0, 120.1]), 'TakeOffAngle':np.array(
            [31.0, 61.0, 12.1])}, {'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([92.0, 272.0, 122.1]), 'TakeOffAngle':np.array([32.0, 62.0, 13.1])}]
        data = {'PPolarity': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                              'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.9], [0.9]])},
                'PPolarity2': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                               'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.9], [0.9]])}}
        a_polarity, error_polarity, incorrect_polarity_prob = polarity_matrix(data, location_samples)
        data = {'P/SHRMSAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                          'Measured': np.matrix([[1.3664, 1], [1.0038, 1]]), 'Error': np.matrix([[0.8, 0.9], [0.8, 0.9]])},
                'P/SVAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                       'Measured': np.matrix([[1.3386, 1], [0.9805, 1]]), 'Error': np.matrix([[0.8, 0.9], [0.8, 0.9]])},
                'P/SHRMSAmplitudeRatio2': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                           'Measured': np.matrix([[1.3664, 1], [1.0038, 1]]), 'Error': np.matrix([[0.8, 0.9], [0.8, 0.9]])},
                'P/SVAmplitudeRatio2': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                        'Measured': np.matrix([[1.3386, 1], [0.9805, 1]]), 'Error': np.matrix([[0.8, 0.9], [0.8, 0.9]])}}
        a_polarity_prob, polarity_prob, incorrect_polarity_prob = polarity_probability_matrix(data, location_samples)
        a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio = amplitude_ratio_matrix(
            data, location_samples)
        a_relative_amplitude, relative_amplitude, percentage_error_relative_amplitude, relative_amplitude_stations = relative_amplitude_ratio_matrix(
            data)
        a_polarity = [a_polarity, a_polarity]
        error_polarity = [error_polarity, error_polarity]
        a1_amplitude_ratio = [a1_amplitude_ratio, a1_amplitude_ratio]
        a2_amplitude_ratio = [a2_amplitude_ratio, a2_amplitude_ratio]
        percentage_error1_amplitude_ratio = [percentage_error1_amplitude_ratio, percentage_error1_amplitude_ratio]
        percentage_error2_amplitude_ratio = [percentage_error2_amplitude_ratio, percentage_error2_amplitude_ratio]
        amplitude_ratio = [amplitude_ratio, amplitude_ratio]
        a_polarity_prob = [a_polarity_prob, a_polarity_prob]
        polarity_prob = [polarity_prob, polarity_prob]
        self.multiple_events_mcmc_forward_task = MultipleEventsMcMCForwardTask(self.algorithm_kwargs, a_polarity, error_polarity, a1_amplitude_ratio,
                                                                               a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio,
                                                                               percentage_error2_amplitude_ratio, a_polarity_prob, polarity_prob,
                                                                               a_relative_amplitude, relative_amplitude, percentage_error_relative_amplitude,
                                                                               relative_amplitude_stations, incorrect_polarity_prob)
        result = self.multiple_events_mcmc_forward_task()
        self.assertTrue('algorithm_output_data' in result)
        self.assertTrue('event_data' in result)
        self.assertNotIn(mock.call(C_EXTENSION_FALLBACK_LOG_MSG), logger.info.call_args_list)

    @mock.patch('MTfit.algorithms.markov_chain_monte_carlo.logger')
    def test___call__location_samples_python(self, logger):
        with PythonAlgorithms():
            location_samples = [{'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([91.0, 271.0, 120.1]), 'TakeOffAngle':np.array(
                [31.0, 61.0, 12.1])}, {'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([92.0, 272.0, 122.1]), 'TakeOffAngle':np.array([32.0, 62.0, 13.1])}]
            data = {'PPolarity': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                  'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.9], [0.9]])},
                    'PPolarity2': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                   'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.9], [0.9]])}}
            a_polarity, error_polarity, incorrect_polarity_prob = polarity_matrix(data, location_samples)
            data = {'P/SHRMSAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                              'Measured': np.matrix([[1.3664, 1], [1.0038, 1]]), 'Error': np.matrix([[0.8, 0.9], [0.8, 0.9]])},
                    'P/SVAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                           'Measured': np.matrix([[1.3386, 1], [0.9805, 1]]), 'Error': np.matrix([[0.8, 0.9], [0.8, 0.9]])},
                    'P/SHRMSAmplitudeRatio2': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                               'Measured': np.matrix([[1.3664, 1], [1.0038, 1]]), 'Error': np.matrix([[0.8, 0.9], [0.8, 0.9]])},
                    'P/SVAmplitudeRatio2': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                            'Measured': np.matrix([[1.3386, 1], [0.9805, 1]]), 'Error': np.matrix([[0.8, 0.9], [0.8, 0.9]])}}
            a_polarity_prob, polarity_prob, incorrect_polarity_prob = polarity_probability_matrix(data, location_samples)
            a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio = amplitude_ratio_matrix(
                data, location_samples)
            a_relative_amplitude, relative_amplitude, percentage_error_relative_amplitude, relative_amplitude_stations = relative_amplitude_ratio_matrix(
                data)
            a_polarity = [a_polarity, a_polarity]
            error_polarity = [error_polarity, error_polarity]
            a1_amplitude_ratio = [a1_amplitude_ratio, a1_amplitude_ratio]
            a2_amplitude_ratio = [a2_amplitude_ratio, a2_amplitude_ratio]
            percentage_error1_amplitude_ratio = [percentage_error1_amplitude_ratio, percentage_error1_amplitude_ratio]
            percentage_error2_amplitude_ratio = [percentage_error2_amplitude_ratio, percentage_error2_amplitude_ratio]
            amplitude_ratio = [amplitude_ratio, amplitude_ratio]
            a_polarity_prob = [a_polarity_prob, a_polarity_prob]
            polarity_prob = [polarity_prob, polarity_prob]
            self.multiple_events_mcmc_forward_task = MultipleEventsMcMCForwardTask(self.algorithm_kwargs, a_polarity, error_polarity, a1_amplitude_ratio,
                                                                                   a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio,
                                                                                   percentage_error2_amplitude_ratio, a_polarity_prob, polarity_prob,
                                                                                   a_relative_amplitude, relative_amplitude, percentage_error_relative_amplitude,
                                                                                   relative_amplitude_stations, incorrect_polarity_prob)
            result = self.multiple_events_mcmc_forward_task()
            self.assertTrue('algorithm_output_data' in result)
            self.assertTrue('event_data' in result)
            self.assertIn(mock.call(C_EXTENSION_FALLBACK_LOG_MSG), logger.info.call_args_list)


class MultipleEventsForwardTaskTestCase(TestCase):

    def setUp(self):
        self.cwd = os.getcwd()
        if sys.version_info >= (3, 0):
            self.tempdir = tempfile.TemporaryDirectory()
            os.chdir(self.tempdir.name)
        else:
            self.tempdir = tempfile.mkdtemp()
            os.chdir(self.tempdir)
        data = {'PPolarity': {'Stations': {'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                              'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.001], [0.001]])},
                'PPolarity2': {'Stations': {'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                               'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.001], [0.001]])}}
        a_polarity, error_polarity, incorrect_polarity_prob = polarity_matrix(data)
        a_polarity_prob, polarity_prob, incorrect_polarity_prob = polarity_probability_matrix(data)
        data = {'P/SHRMSAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                          'Measured': np.matrix([[1.3664, 1], [1.0038, 1]]), 'Error': np.matrix([[0.001, 0.003], [0.001, 0.002]])},
                'P/SVAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                       'Measured': np.matrix([[1.3386, 1], [0.9805, 1]]), 'Error': np.matrix([[0.001, 0.03], [0.001, 0.02]])}}
        a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio = amplitude_ratio_matrix(
            data)
        data = {'PRMSQAmplitude': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                   'Measured': np.matrix([[1.3664], [1.0038]]), 'Error': np.matrix([[0.001], [0.002]])}}
        a_relative_amplitude, relative_amplitude, percentage_error_relative_amplitude, relative_amplitude_stations = relative_amplitude_ratio_matrix(
            data)
        MTs = np.matrix([[-3.11183179e-01,   6.28082467e-01,   3.75893170e-01,
                          2.32496521e-01,  -1.91598717e-01,  -1.01887355e-02,
                          -2.71853364e-01,  -3.33336969e-01,   3.79659261e-02,
                          -1.74421393e-01,  -6.60894648e-01,  -3.27058984e-01,
                          -3.65045571e-01,   4.56519075e-01,  -3.11112188e-01,
                          7.41618454e-01,   2.93738581e-01,   3.71018571e-01,
                          -4.43535198e-01,  -6.60453690e-01,  -3.95178552e-01,
                          4.48132136e-01,   3.85049118e-01,  -2.91051116e-01,
                          -4.19363018e-02,   1.49496434e-01,   3.93103350e-01,
                          5.77320333e-01,  -7.89128261e-01,   4.02086850e-01,
                          -8.75080588e-01,  -6.56342704e-01,   7.73729770e-01,
                          1.52122268e-01,   3.95142707e-01,  -4.89791368e-01,
                          -1.41934380e-01,  -4.64999289e-01,  -1.00274668e-01,
                          2.76049224e-01,  -1.30781712e-01,   2.70804196e-01,
                          -7.17410721e-02,  -3.15111639e-01,   2.22802875e-01,
                          7.32200782e-01,   1.33571659e-01,  -1.66079805e-01,
                          -3.47722526e-01,   2.22104260e-01,   3.56355066e-01,
                          1.50009019e-01,   3.74253659e-01,  -1.93817333e-01,
                          1.95867543e-01,  -7.65882439e-01,  -1.58391949e-01,
                          -5.48912849e-01,  -6.72616521e-01,   3.74480351e-02,
                          3.97595466e-01,   4.97397722e-01,   7.00417521e-02,
                          -2.16585918e-01,  -1.84533439e-01,   4.45112005e-01,
                          3.45592583e-01,  -1.42229256e-01,   2.36169780e-01,
                          -4.93825978e-01,  -1.93098202e-01,   2.98000866e-01,
                          4.84558512e-01,   2.00337052e-01,  -5.49797321e-01,
                          -6.85122398e-01,   1.53842425e-01,   1.92430061e-01,
                          -4.30863922e-02,   4.18559270e-01,   1.07106768e-01,
                          4.35950903e-01,   2.37178391e-01,  -1.21451912e-01,
                          -4.95148143e-01,   2.43052776e-01,   3.37991061e-02,
                          4.00339447e-02,  -7.27785189e-01,  -3.49653062e-02,
                          4.27229807e-01,   4.11258644e-01,   3.01426900e-01,
                          6.26059699e-03,  -1.15131196e-01,  -5.55731212e-02,
                          5.83175280e-01,   1.55629260e-01,   4.66899189e-01,
                          1.33622043e-01],
                         [-4.81990410e-01,  -3.43780097e-01,   1.17920180e-01,
                          -4.45276093e-02,   2.75977663e-01,  -5.83045150e-01,
                          2.72970450e-01,   1.18479787e-01,   1.19877505e-01,
                          -3.58285641e-01,  -2.91404968e-01,   5.66458943e-01,
                          -2.98735776e-01,  -1.64113623e-01,  -4.49243775e-01,
                          5.86544422e-02,   2.24878125e-01,   4.08302342e-04,
                          -6.09150416e-01,  -2.31378304e-01,   2.08332332e-01,
                          -7.66764159e-01,  -7.95865698e-01,  -4.54441413e-02,
                          5.61153066e-01,   4.62023576e-01,   1.30169846e-01,
                          1.82812720e-02,   1.90129687e-01,  -6.66965021e-01,
                          -9.63074675e-02,   2.65362148e-01,  -1.50339165e-01,
                          2.85922743e-01,   4.63084318e-01,   2.73031957e-01,
                          4.11464515e-01,  -7.04297722e-01,  -4.48186311e-01,
                          -8.93483767e-02,   2.02609445e-01,   5.70785815e-02,
                          -1.66145913e-01,  -5.13363879e-01,   6.59216469e-01,
                          -7.28055890e-02,   4.56739995e-01,  -2.93321085e-01,
                          3.25872486e-01,  -6.38871034e-01,  -1.85954226e-01,
                          2.57125567e-01,  -5.18904755e-01,  -1.16639367e-01,
                          3.08144148e-01,  -2.85228076e-01,   2.94983589e-01,
                          -4.80652614e-01,   3.34555024e-01,  -4.65517337e-01,
                          1.96379666e-01,   1.16969567e-01,  -9.82696531e-02,
                          7.83520427e-01,  -5.03859811e-02,  -5.67435191e-02,
                          3.83359000e-01,  -7.34575503e-01,  -7.37596573e-01,
                          -6.56213641e-01,  -3.78296664e-01,  -6.57246610e-01,
                          -4.30941668e-01,   2.06386790e-01,  -4.42056936e-02,
                          -2.00698787e-01,  -7.38708780e-01,   1.44589600e-01,
                          2.13149955e-01,   2.26898109e-01,  -1.07195833e-01,
                          -3.12149841e-01,  -3.86197233e-01,   5.39339983e-01,
                          2.36892729e-01,   4.04360057e-02,  -5.52796806e-01,
                          5.52662758e-01,   3.95952901e-01,   6.22711034e-01,
                          1.83525596e-03,  -1.07912080e-01,   1.85782872e-01,
                          -1.66192241e-01,   2.83242525e-01,  -4.56790103e-01,
                          4.79222089e-01,   6.49966877e-04,  -4.48141800e-03,
                          2.37667557e-02],
                         [3.58795227e-01,   3.09934383e-01,   7.24943528e-01,
                          -1.24861314e-01,  -4.87686601e-01,   5.58514119e-01,
                          -6.64015313e-01,   7.12793257e-01,  -4.83475316e-01,
                          9.70552181e-02,  -1.39945431e-01,  -5.00980139e-01,
                          -1.17332249e-01,  -6.72356360e-01,   6.07299657e-01,
                          -5.53722035e-01,   5.65195997e-01,   1.76127236e-01,
                          3.02187498e-01,   1.14129043e-01,  -1.69257622e-01,
                          2.72512636e-01,  -2.94954648e-01,   1.46192237e-01,
                          -2.76998887e-01,  -5.82877112e-01,  -4.63364621e-01,
                          -6.65402559e-01,   5.54200123e-01,   2.95205103e-01,
                          1.75791622e-01,   4.81317107e-01,  -3.07291324e-01,
                          4.65858642e-01,   2.32509422e-01,  -1.50688306e-01,
                          -3.21696229e-01,   2.29425781e-01,   3.70533138e-01,
                          3.10243708e-01,   1.10176464e-01,   8.46453658e-01,
                          1.70278184e-01,  -1.92471902e-01,  -2.69778807e-01,
                          -6.31943728e-01,  -7.59932854e-01,   8.29564470e-01,
                          -3.68159152e-01,  -2.10699197e-01,   1.78401838e-01,
                          7.22950229e-01,   5.26113372e-01,   1.13933734e-01,
                          -6.68767936e-01,   1.26022038e-01,   5.35448137e-02,
                          2.02191581e-02,   1.03664710e-01,   3.59405679e-01,
                          2.93387345e-01,   2.73610747e-01,  -2.40028588e-01,
                          -2.82516384e-01,   4.74226342e-01,   3.03150255e-01,
                          -2.00230764e-01,  -3.87918196e-02,  -2.64254011e-01,
                          4.22271990e-01,   5.04628828e-01,  -6.12669356e-02,
                          -4.69101219e-01,  -9.10406151e-01,  -7.21063912e-01,
                          6.51845746e-01,   5.09083578e-01,   5.52949389e-01,
                          -4.25652382e-01,  -8.69724965e-02,   3.99005623e-01,
                          -2.54151666e-01,  -2.46690909e-01,   1.38495258e-01,
                          -6.38220944e-01,   5.78023620e-01,  -6.12203943e-01,
                          -2.43511010e-01,  -1.08132410e-01,  -3.14182304e-01,
                          5.92569808e-01,   4.75754663e-01,  -2.98290011e-01,
                          7.11312066e-01,   3.41648485e-01,   3.24386957e-01,
                          9.13530018e-02,  -6.91615082e-01,   3.31423035e-01,
                          6.84173449e-01],
                         [-2.78378071e-01,   2.60569829e-01,   2.53366805e-01,
                          -7.33390656e-01,  -5.18561385e-01,  -5.55772684e-01,
                          -1.51606390e-02,   3.45628360e-01,  -7.86421951e-01,
                          -1.65092528e-01,   5.88736201e-01,  -4.26139409e-02,
                          -2.63293899e-01,  -5.55444721e-01,  -2.69657962e-01,
                          2.57846066e-01,  -1.58862372e-01,   3.18501329e-01,
                          1.35923924e-01,   8.04313382e-02,   1.50297672e-01,
                          1.23475042e-01,  -2.49864358e-01,   3.87644601e-01,
                          -5.76521840e-01,  -2.10051271e-01,   1.67346822e-02,
                          -3.08835850e-01,   1.77723602e-01,   2.54329881e-01,
                          -3.24428447e-01,  -2.81153619e-01,  -2.11768213e-01,
                          -2.02646715e-01,   4.48159822e-01,   4.46101952e-01,
                          -4.61468505e-02,  -1.80983683e-01,   7.87366633e-01,
                          1.26136843e-01,   8.62295000e-01,   1.78606246e-01,
                          -4.27309288e-01,  -2.06216556e-01,  -2.99529270e-01,
                          -1.31022923e-01,   4.09103474e-01,  -2.53368878e-01,
                          9.34582718e-02,   4.53053806e-01,  -5.48897839e-01,
                          -5.20566433e-01,   4.55777223e-01,  -4.39839511e-01,
                          -4.68091557e-01,  -3.57825556e-01,  -4.24360310e-01,
                          6.90471023e-02,  -1.78404541e-01,   4.46578311e-02,
                          1.30128057e-01,   7.32746248e-02,   3.26960468e-01,
                          -3.83289492e-01,  -6.38159241e-01,   1.49980124e-01,
                          -4.64257142e-01,   8.65414266e-02,  -2.55058658e-01,
                          7.52890132e-02,  -1.83738208e-01,   5.31896589e-01,
                          -4.16493413e-02,  -1.15670794e-01,   1.80033212e-01,
                          -2.94728740e-03,   5.46513788e-02,  -6.61100220e-01,
                          -4.69035616e-01,   1.68496705e-01,   7.23622400e-01,
                          6.72577294e-01,   3.66378654e-01,  -5.28689295e-01,
                          -7.04212841e-02,  -3.88168080e-01,   1.35384415e-01,
                          1.50561739e-01,   1.48762586e-01,  -5.93972576e-01,
                          3.24103333e-01,  -4.47670419e-01,  -2.24073260e-01,
                          -1.04698492e-01,  -3.65807121e-01,  -1.17145578e-01,
                          5.70177064e-02,  -4.81962471e-01,  -3.42091734e-01,
                          -9.62803328e-02],
                         [-6.51058927e-01,  -5.15604518e-02,   4.71540134e-01,
                          2.46726214e-01,  -5.35375562e-01,  -9.16791046e-02,
                          -5.56037563e-02,   6.54208853e-03,  -2.74700722e-02,
                          6.90865110e-01,   5.44918932e-02,   1.56024461e-02,
                          -3.92788415e-01,   5.52346421e-02,  -3.21903864e-01,
                          -1.27122838e-01,   9.01239391e-02,   4.79050360e-01,
                          1.15946816e-01,  -2.36936195e-01,   7.08788692e-01,
                          -2.23454906e-01,  -9.65032611e-03,   7.67809741e-01,
                          -4.03214944e-01,   5.90999447e-01,  -7.72050782e-01,
                          -1.71104226e-02,   9.12096882e-05,   3.53588880e-01,
                          -2.80541415e-01,   1.31209057e-01,  -2.90025593e-01,
                          5.39778826e-01,   5.69464226e-01,  -6.57833946e-01,
                          1.83114460e-01,   3.71900531e-01,  -5.31745271e-02,
                          8.72043994e-01,   3.85817443e-01,   1.22581135e-01,
                          -8.38662852e-01,  -5.76787769e-02,  -5.88106143e-01,
                          5.86411388e-02,   1.50013227e-01,  -3.65504527e-01,
                          4.22987623e-01,   3.16175094e-01,   6.50783310e-01,
                          2.17973362e-02,   3.03891885e-01,  -8.59332997e-01,
                          -3.24049593e-01,  -2.52065818e-01,  -8.30600639e-01,
                          -3.49389936e-01,   2.74031174e-02,   3.23780672e-01,
                          6.54256579e-01,   6.66854184e-01,   8.48975839e-01,
                          -1.95113033e-01,  -5.06318176e-01,  -5.93207916e-01,
                          -2.92295947e-01,  -3.14676452e-01,   4.71417465e-01,
                          2.79966721e-01,  -7.07754884e-01,   3.41266104e-01,
                          -5.48666390e-01,  -2.64853482e-01,  -1.36189400e-01,
                          2.55479687e-01,  -3.14625200e-01,   4.29170200e-01,
                          -7.22125631e-01,   4.75211821e-01,  -5.18306225e-01,
                          3.90783391e-01,  -7.74002650e-01,   2.89100373e-01,
                          8.46077997e-02,  -5.26592095e-01,   3.34726660e-02,
                          -5.73737671e-01,  -3.28883687e-01,   3.94488701e-01,
                          5.56043445e-01,   5.99223694e-01,  -1.71546449e-01,
                          6.34992787e-01,  -7.96093608e-01,   7.19460709e-01,
                          6.40498564e-01,  -3.07772250e-01,   3.25472775e-01,
                          -4.71450358e-01],
                         [2.01851882e-01,  -5.66315479e-01,   1.80862383e-01,
                          -5.74139907e-01,  -3.06194608e-01,  -1.75294573e-01,
                          6.38235789e-01,  -4.97265539e-01,  -3.62250605e-01,
                          5.72047829e-01,   3.30358902e-01,  -5.64903711e-01,
                          -7.34932238e-01,  -3.20042853e-02,   3.95270604e-01,
                          2.39392613e-01,   7.14381818e-01,  -7.07383846e-01,
                          5.55856130e-01,   6.59265959e-01,   4.96802384e-01,
                          -2.67972004e-01,  -2.62319266e-01,   3.89938166e-01,
                          3.34156367e-01,  -1.76174406e-01,   1.32198094e-01,
                          3.57678787e-01,   4.90288354e-02,  -3.41509866e-01,
                          1.00492978e-01,  -4.13361650e-01,   3.94146926e-01,
                          -5.87925041e-01,  -2.24093271e-01,  -1.76347805e-01,
                          8.19391655e-01,  -2.53059466e-01,   1.70309926e-01,
                          -2.07825176e-01,   1.93129328e-01,   3.99996252e-01,
                          -2.28703930e-01,   7.44489517e-01,  -8.61701969e-02,
                          -1.96526580e-01,  -7.86105946e-02,  -2.03391799e-02,
                          -6.70598262e-01,   4.39198026e-01,  -2.86001963e-01,
                          -3.42435642e-01,   1.17446239e-01,  -6.27471780e-02,
                          3.08730779e-01,  -3.53003080e-01,   1.22680607e-01,
                          5.83453600e-01,   6.26366082e-01,   7.38844074e-01,
                          5.21815826e-01,  -4.62587410e-01,   3.16481252e-01,
                          2.72746848e-01,  -2.73701896e-01,  -5.76511691e-01,
                          -6.26528268e-01,   5.76328184e-01,   2.07498442e-01,
                          2.51308829e-01,   1.74005056e-01,   2.75848156e-01,
                          2.38048103e-01,  -7.00221359e-02,  -3.53378108e-01,
                          1.20773532e-02,   2.63628466e-01,   1.22763719e-01,
                          -1.73409553e-01,   7.15222872e-01,  -1.59878035e-01,
                          2.06988752e-01,  -2.06950131e-02,  -5.58649696e-01,
                          -5.28456755e-01,   4.20957574e-01,  -5.46836617e-01,
                          -5.30867022e-01,  -4.14201177e-01,   6.22343883e-02,
                          -2.28265333e-01,  -1.82759328e-01,  -8.40245564e-01,
                          -2.28469857e-01,  -1.49046671e-01,  -3.89462109e-01,
                          -9.17495648e-02,  -4.12839084e-01,  -6.70212320e-01,
                          5.30991186e-01]])
        self.MTs = MTs
        self.multiple_events_forward_task = MultipleEventsForwardTask([MTs, MTs], [a_polarity, a_polarity], [error_polarity, error_polarity],
                                                                      [a1_amplitude_ratio, a1_amplitude_ratio], [a2_amplitude_ratio, a2_amplitude_ratio],
                                                                      [amplitude_ratio, amplitude_ratio], [percentage_error1_amplitude_ratio, percentage_error1_amplitude_ratio],
                                                                      [percentage_error2_amplitude_ratio, percentage_error2_amplitude_ratio], [a_polarity_prob, a_polarity_prob],
                                                                      [polarity_prob, polarity_prob], [a_relative_amplitude, a_relative_amplitude], [relative_amplitude, relative_amplitude],
                                                                      [percentage_error_relative_amplitude, percentage_error_relative_amplitude],
                                                                      [relative_amplitude_stations, relative_amplitude_stations], incorrect_polarity_prob)
        self.existing_log_files = glob.glob('*.log')

    def tearDown(self):
        del self.multiple_events_forward_task
        del self.MTs
        os.chdir(self.cwd)
        if sys.version_info >= (3, 0):
            self.tempdir.cleanup()
        else:
            try:
                shutil.rmtree(self.tempdir)
            except:
                pass

    def test___call__(self):
        result = self.multiple_events_forward_task()
        self.assertTrue('moment_tensors' in result)
        self.assertTrue('ln_pdf' in result)
        del self.multiple_events_forward_task
        location_samples = [{'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([91.0, 271.0, 120.1]), 'TakeOffAngle':np.array(
            [31.0, 61.0, 12.1])}, {'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([92.0, 272.0, 122.1]), 'TakeOffAngle':np.array([32.0, 62.0, 13.1])}]
        data = {'PPolarity': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                              'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.001], [0.001]])},
                'PPolarity2': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                               'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.001], [0.001]])}}
        a_polarity, error_polarity, incorrect_polarity_prob = polarity_matrix(data, location_samples)
        data = {'P/SHRMSAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                          'Measured': np.matrix([[1.3664, 1], [1.0038, 1]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.02]])},
                'P/SVQAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                        'Measured': np.matrix([[1.3386, 1], [0.9805, 1]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.02]])}}
        a_polarity_prob, polarity_prob, incorrect_polarity_prob = polarity_probability_matrix(data, location_samples)
        a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio = amplitude_ratio_matrix(
            data, location_samples)
        a_relative_amplitude, relative_amplitude, percentage_error_relative_amplitude, relative_amplitude_stations = relative_amplitude_ratio_matrix(
            data, location_samples)
        self.assertFalse(a_relative_amplitude)
        self.multiple_events_forward_task = MultipleEventsForwardTask([self.MTs, self.MTs], [a_polarity, a_polarity], [error_polarity, error_polarity],
                                                                      [a1_amplitude_ratio, a1_amplitude_ratio], [a2_amplitude_ratio, a2_amplitude_ratio],
                                                                      [amplitude_ratio, amplitude_ratio], [percentage_error1_amplitude_ratio, percentage_error1_amplitude_ratio],
                                                                      [percentage_error2_amplitude_ratio, percentage_error2_amplitude_ratio], [a_polarity_prob, a_polarity_prob],
                                                                      [polarity_prob, polarity_prob], a1_amplitude_ratio, [relative_amplitude, relative_amplitude],
                                                                      [percentage_error_relative_amplitude, percentage_error_relative_amplitude],
                                                                      [relative_amplitude_stations, relative_amplitude_stations], incorrect_polarity_prob, relative=False, return_zero=True)
        result = self.multiple_events_forward_task()
        self.assertTrue(result['ln_pdf'].shape, (self.MTs.shape[0], 1))
        self.assertFalse(result['scale_factor'])  # No relative
        self.multiple_events_forward_task = MultipleEventsForwardTask([self.MTs, self.MTs], [a_polarity, a_polarity], [error_polarity, error_polarity],
                                                                      [a1_amplitude_ratio, a1_amplitude_ratio], [a2_amplitude_ratio, a2_amplitude_ratio],
                                                                      [amplitude_ratio, amplitude_ratio], [percentage_error1_amplitude_ratio, percentage_error1_amplitude_ratio],
                                                                      [percentage_error2_amplitude_ratio, percentage_error2_amplitude_ratio], [a_polarity_prob, a_polarity_prob],
                                                                      [polarity_prob, polarity_prob], a1_amplitude_ratio, [relative_amplitude, relative_amplitude],
                                                                      [percentage_error_relative_amplitude, percentage_error_relative_amplitude],
                                                                      [relative_amplitude_stations, relative_amplitude_stations], incorrect_polarity_prob, relative=False, return_zero=False)
        result = self.multiple_events_forward_task()
        self.assertTrue(result['ln_pdf'].shape, (self.MTs.shape[0], 1))
        self.assertTrue(result['ln_pdf'].shape[1], 3)
        data = {'PRMSQAmplitude': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                   'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.001], [0.001]])}}
        a_relative_amplitude, relative_amplitude, percentage_error_relative_amplitude, relative_amplitude_stations = relative_amplitude_ratio_matrix(data, location_samples)
        self.multiple_events_forward_task = MultipleEventsForwardTask([self.MTs, self.MTs], a_polarity, [error_polarity, error_polarity],
                                                                      a1_amplitude_ratio, a2_amplitude_ratio, [amplitude_ratio, amplitude_ratio],
                                                                      [percentage_error1_amplitude_ratio, percentage_error1_amplitude_ratio],
                                                                      [percentage_error2_amplitude_ratio, percentage_error2_amplitude_ratio], a_polarity_prob,
                                                                      [polarity_prob, polarity_prob], False, [relative_amplitude, relative_amplitude],
                                                                      [percentage_error_relative_amplitude, percentage_error_relative_amplitude],
                                                                      [relative_amplitude_stations, relative_amplitude_stations], incorrect_polarity_prob,
                                                                      location_sample_size=2, relative=False, return_zero=True)
        result = self.multiple_events_forward_task()
        self.assertTrue(result['ln_pdf'].shape, (self.MTs.shape[0], 1))
        self.assertTrue(result['scale_factor'].shape, (self.MTs.shape[0], 2, 2))
        self.assertTrue((result['scale_factor'][0]['mu'] == np.array([[1, 0], [0, 1]])).all(), msg=str(result['scale_factor'][0]['mu']))
        self.multiple_events_forward_task = MultipleEventsForwardTask([self.MTs, self.MTs], [a_polarity, a_polarity], [error_polarity, error_polarity],
                                                                      [a1_amplitude_ratio, a1_amplitude_ratio], [a2_amplitude_ratio, a2_amplitude_ratio],
                                                                      [amplitude_ratio, amplitude_ratio], [percentage_error1_amplitude_ratio, percentage_error1_amplitude_ratio],
                                                                      [percentage_error2_amplitude_ratio, percentage_error2_amplitude_ratio], [a_polarity_prob, a_polarity_prob],
                                                                      [polarity_prob, polarity_prob], a1_amplitude_ratio, [relative_amplitude, relative_amplitude],
                                                                      [percentage_error_relative_amplitude, percentage_error_relative_amplitude],
                                                                      [relative_amplitude_stations, relative_amplitude_stations], incorrect_polarity_prob,
                                                                      location_sample_size=2, relative=True, return_zero=False)
        result = self.multiple_events_forward_task()
        self.assertTrue(result['ln_pdf'].shape, (self.MTs.shape[0], 1))
        self.assertTrue(result['ln_pdf'].shape[1], 3)
        self.multiple_events_forward_task = MultipleEventsForwardTask([self.MTs, self.MTs], a_polarity, [error_polarity, error_polarity],
                                                                      a1_amplitude_ratio, a2_amplitude_ratio, [amplitude_ratio, amplitude_ratio],
                                                                      [percentage_error1_amplitude_ratio, percentage_error1_amplitude_ratio],
                                                                      [percentage_error2_amplitude_ratio, percentage_error2_amplitude_ratio], a_polarity_prob,
                                                                      [polarity_prob, polarity_prob], False, [relative_amplitude, relative_amplitude],
                                                                      [percentage_error_relative_amplitude, percentage_error_relative_amplitude],
                                                                      [relative_amplitude_stations, relative_amplitude_stations], incorrect_polarity_prob,
                                                                      location_sample_size=2, relative=False, marginalise_relative=True, return_zero=True)
        result = self.multiple_events_forward_task()
        self.assertTrue(result['ln_pdf'].shape, (self.MTs.shape[0], 1))
        self.assertFalse(result['scale_factor'])
        self.multiple_events_forward_task = MultipleEventsForwardTask([self.MTs, self.MTs], a_polarity, [error_polarity, error_polarity],
                                                                      a1_amplitude_ratio, a2_amplitude_ratio, [amplitude_ratio, amplitude_ratio],
                                                                      [percentage_error1_amplitude_ratio, percentage_error1_amplitude_ratio],
                                                                      [percentage_error2_amplitude_ratio, percentage_error2_amplitude_ratio], a_polarity_prob,
                                                                      [polarity_prob, polarity_prob], a_relative_amplitude, [relative_amplitude, relative_amplitude],
                                                                      [percentage_error_relative_amplitude, percentage_error_relative_amplitude],
                                                                      [relative_amplitude_stations, relative_amplitude_stations], incorrect_polarity_prob,
                                                                      location_sample_size=2, relative=True, marginalise_relative=False, return_zero=True)
        result = self.multiple_events_forward_task()
        self.assertTrue(result['ln_pdf'].shape, (self.MTs.shape[0], 1))
        self.assertTrue(result['scale_factor'].shape, (self.MTs.shape[0]))
        self.assertAlmostEquals(result['scale_factor'][0]['mu'], np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]), places=2, msg=str(result['scale_factor'][0]['mu']))
        # Add valid calculation (non-zero prob)
        #


class ForwardTaskTestCase(TestCase):

    def setUp(self):
        self.cwd = os.getcwd()
        if sys.version_info >= (3, 0):
            self.tempdir = tempfile.TemporaryDirectory()
            os.chdir(self.tempdir.name)
        else:
            self.tempdir = tempfile.mkdtemp()
            os.chdir(self.tempdir)
        data = {'PPolarity': {'Stations': {'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                              'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.1], [0.1]])},
                'PPolarity2': {'Stations': {'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                               'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.1], [0.1]])}}
        a_polarity, error_polarity, incorrect_polarity_prob = polarity_matrix(data)
        a_polarity_prob, polarity_prob, incorrect_polarity_prob = polarity_probability_matrix(data)
        data = {'P/SHRMSAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                          'Measured': np.matrix([[1.3664, 1], [1.0038, 1]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.02]])},
                'P/SVQAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                        'Measured': np.matrix([[1.3386, 1], [0.9805, 1]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.02]])}}
        a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio = amplitude_ratio_matrix(
            data)
        MTs = np.matrix([[-3.11183179e-01,   6.28082467e-01,   3.75893170e-01,
                          2.32496521e-01,  -1.91598717e-01,  -1.01887355e-02,
                          -2.71853364e-01,  -3.33336969e-01,   3.79659261e-02,
                          -1.74421393e-01,  -6.60894648e-01,  -3.27058984e-01,
                          -3.65045571e-01,   4.56519075e-01,  -3.11112188e-01,
                          7.41618454e-01,   2.93738581e-01,   3.71018571e-01,
                          -4.43535198e-01,  -6.60453690e-01,  -3.95178552e-01,
                          4.48132136e-01,   3.85049118e-01,  -2.91051116e-01,
                          -4.19363018e-02,   1.49496434e-01,   3.93103350e-01,
                          5.77320333e-01,  -7.89128261e-01,   4.02086850e-01,
                          -8.75080588e-01,  -6.56342704e-01,   7.73729770e-01,
                          1.52122268e-01,   3.95142707e-01,  -4.89791368e-01,
                          -1.41934380e-01,  -4.64999289e-01,  -1.00274668e-01,
                          2.76049224e-01,  -1.30781712e-01,   2.70804196e-01,
                          -7.17410721e-02,  -3.15111639e-01,   2.22802875e-01,
                          7.32200782e-01,   1.33571659e-01,  -1.66079805e-01,
                          -3.47722526e-01,   2.22104260e-01,   3.56355066e-01,
                          1.50009019e-01,   3.74253659e-01,  -1.93817333e-01,
                          1.95867543e-01,  -7.65882439e-01,  -1.58391949e-01,
                          -5.48912849e-01,  -6.72616521e-01,   3.74480351e-02,
                          3.97595466e-01,   4.97397722e-01,   7.00417521e-02,
                          -2.16585918e-01,  -1.84533439e-01,   4.45112005e-01,
                          3.45592583e-01,  -1.42229256e-01,   2.36169780e-01,
                          -4.93825978e-01,  -1.93098202e-01,   2.98000866e-01,
                          4.84558512e-01,   2.00337052e-01,  -5.49797321e-01,
                          -6.85122398e-01,   1.53842425e-01,   1.92430061e-01,
                          -4.30863922e-02,   4.18559270e-01,   1.07106768e-01,
                          4.35950903e-01,   2.37178391e-01,  -1.21451912e-01,
                          -4.95148143e-01,   2.43052776e-01,   3.37991061e-02,
                          4.00339447e-02,  -7.27785189e-01,  -3.49653062e-02,
                          4.27229807e-01,   4.11258644e-01,   3.01426900e-01,
                          6.26059699e-03,  -1.15131196e-01,  -5.55731212e-02,
                          5.83175280e-01,   1.55629260e-01,   4.66899189e-01,
                          1.33622043e-01],
                         [-4.81990410e-01,  -3.43780097e-01,   1.17920180e-01,
                          -4.45276093e-02,   2.75977663e-01,  -5.83045150e-01,
                          2.72970450e-01,   1.18479787e-01,   1.19877505e-01,
                          -3.58285641e-01,  -2.91404968e-01,   5.66458943e-01,
                          -2.98735776e-01,  -1.64113623e-01,  -4.49243775e-01,
                          5.86544422e-02,   2.24878125e-01,   4.08302342e-04,
                          -6.09150416e-01,  -2.31378304e-01,   2.08332332e-01,
                          -7.66764159e-01,  -7.95865698e-01,  -4.54441413e-02,
                          5.61153066e-01,   4.62023576e-01,   1.30169846e-01,
                          1.82812720e-02,   1.90129687e-01,  -6.66965021e-01,
                          -9.63074675e-02,   2.65362148e-01,  -1.50339165e-01,
                          2.85922743e-01,   4.63084318e-01,   2.73031957e-01,
                          4.11464515e-01,  -7.04297722e-01,  -4.48186311e-01,
                          -8.93483767e-02,   2.02609445e-01,   5.70785815e-02,
                          -1.66145913e-01,  -5.13363879e-01,   6.59216469e-01,
                          -7.28055890e-02,   4.56739995e-01,  -2.93321085e-01,
                          3.25872486e-01,  -6.38871034e-01,  -1.85954226e-01,
                          2.57125567e-01,  -5.18904755e-01,  -1.16639367e-01,
                          3.08144148e-01,  -2.85228076e-01,   2.94983589e-01,
                          -4.80652614e-01,   3.34555024e-01,  -4.65517337e-01,
                          1.96379666e-01,   1.16969567e-01,  -9.82696531e-02,
                          7.83520427e-01,  -5.03859811e-02,  -5.67435191e-02,
                          3.83359000e-01,  -7.34575503e-01,  -7.37596573e-01,
                          -6.56213641e-01,  -3.78296664e-01,  -6.57246610e-01,
                          -4.30941668e-01,   2.06386790e-01,  -4.42056936e-02,
                          -2.00698787e-01,  -7.38708780e-01,   1.44589600e-01,
                          2.13149955e-01,   2.26898109e-01,  -1.07195833e-01,
                          -3.12149841e-01,  -3.86197233e-01,   5.39339983e-01,
                          2.36892729e-01,   4.04360057e-02,  -5.52796806e-01,
                          5.52662758e-01,   3.95952901e-01,   6.22711034e-01,
                          1.83525596e-03,  -1.07912080e-01,   1.85782872e-01,
                          -1.66192241e-01,   2.83242525e-01,  -4.56790103e-01,
                          4.79222089e-01,   6.49966877e-04,  -4.48141800e-03,
                          2.37667557e-02],
                         [3.58795227e-01,   3.09934383e-01,   7.24943528e-01,
                          -1.24861314e-01,  -4.87686601e-01,   5.58514119e-01,
                          -6.64015313e-01,   7.12793257e-01,  -4.83475316e-01,
                          9.70552181e-02,  -1.39945431e-01,  -5.00980139e-01,
                          -1.17332249e-01,  -6.72356360e-01,   6.07299657e-01,
                          -5.53722035e-01,   5.65195997e-01,   1.76127236e-01,
                          3.02187498e-01,   1.14129043e-01,  -1.69257622e-01,
                          2.72512636e-01,  -2.94954648e-01,   1.46192237e-01,
                          -2.76998887e-01,  -5.82877112e-01,  -4.63364621e-01,
                          -6.65402559e-01,   5.54200123e-01,   2.95205103e-01,
                          1.75791622e-01,   4.81317107e-01,  -3.07291324e-01,
                          4.65858642e-01,   2.32509422e-01,  -1.50688306e-01,
                          -3.21696229e-01,   2.29425781e-01,   3.70533138e-01,
                          3.10243708e-01,   1.10176464e-01,   8.46453658e-01,
                          1.70278184e-01,  -1.92471902e-01,  -2.69778807e-01,
                          -6.31943728e-01,  -7.59932854e-01,   8.29564470e-01,
                          -3.68159152e-01,  -2.10699197e-01,   1.78401838e-01,
                          7.22950229e-01,   5.26113372e-01,   1.13933734e-01,
                          -6.68767936e-01,   1.26022038e-01,   5.35448137e-02,
                          2.02191581e-02,   1.03664710e-01,   3.59405679e-01,
                          2.93387345e-01,   2.73610747e-01,  -2.40028588e-01,
                          -2.82516384e-01,   4.74226342e-01,   3.03150255e-01,
                          -2.00230764e-01,  -3.87918196e-02,  -2.64254011e-01,
                          4.22271990e-01,   5.04628828e-01,  -6.12669356e-02,
                          -4.69101219e-01,  -9.10406151e-01,  -7.21063912e-01,
                          6.51845746e-01,   5.09083578e-01,   5.52949389e-01,
                          -4.25652382e-01,  -8.69724965e-02,   3.99005623e-01,
                          -2.54151666e-01,  -2.46690909e-01,   1.38495258e-01,
                          -6.38220944e-01,   5.78023620e-01,  -6.12203943e-01,
                          -2.43511010e-01,  -1.08132410e-01,  -3.14182304e-01,
                          5.92569808e-01,   4.75754663e-01,  -2.98290011e-01,
                          7.11312066e-01,   3.41648485e-01,   3.24386957e-01,
                          9.13530018e-02,  -6.91615082e-01,   3.31423035e-01,
                          6.84173449e-01],
                         [-2.78378071e-01,   2.60569829e-01,   2.53366805e-01,
                          -7.33390656e-01,  -5.18561385e-01,  -5.55772684e-01,
                          -1.51606390e-02,   3.45628360e-01,  -7.86421951e-01,
                          -1.65092528e-01,   5.88736201e-01,  -4.26139409e-02,
                          -2.63293899e-01,  -5.55444721e-01,  -2.69657962e-01,
                          2.57846066e-01,  -1.58862372e-01,   3.18501329e-01,
                          1.35923924e-01,   8.04313382e-02,   1.50297672e-01,
                          1.23475042e-01,  -2.49864358e-01,   3.87644601e-01,
                          -5.76521840e-01,  -2.10051271e-01,   1.67346822e-02,
                          -3.08835850e-01,   1.77723602e-01,   2.54329881e-01,
                          -3.24428447e-01,  -2.81153619e-01,  -2.11768213e-01,
                          -2.02646715e-01,   4.48159822e-01,   4.46101952e-01,
                          -4.61468505e-02,  -1.80983683e-01,   7.87366633e-01,
                          1.26136843e-01,   8.62295000e-01,   1.78606246e-01,
                          -4.27309288e-01,  -2.06216556e-01,  -2.99529270e-01,
                          -1.31022923e-01,   4.09103474e-01,  -2.53368878e-01,
                          9.34582718e-02,   4.53053806e-01,  -5.48897839e-01,
                          -5.20566433e-01,   4.55777223e-01,  -4.39839511e-01,
                          -4.68091557e-01,  -3.57825556e-01,  -4.24360310e-01,
                          6.90471023e-02,  -1.78404541e-01,   4.46578311e-02,
                          1.30128057e-01,   7.32746248e-02,   3.26960468e-01,
                          -3.83289492e-01,  -6.38159241e-01,   1.49980124e-01,
                          -4.64257142e-01,   8.65414266e-02,  -2.55058658e-01,
                          7.52890132e-02,  -1.83738208e-01,   5.31896589e-01,
                          -4.16493413e-02,  -1.15670794e-01,   1.80033212e-01,
                          -2.94728740e-03,   5.46513788e-02,  -6.61100220e-01,
                          -4.69035616e-01,   1.68496705e-01,   7.23622400e-01,
                          6.72577294e-01,   3.66378654e-01,  -5.28689295e-01,
                          -7.04212841e-02,  -3.88168080e-01,   1.35384415e-01,
                          1.50561739e-01,   1.48762586e-01,  -5.93972576e-01,
                          3.24103333e-01,  -4.47670419e-01,  -2.24073260e-01,
                          -1.04698492e-01,  -3.65807121e-01,  -1.17145578e-01,
                          5.70177064e-02,  -4.81962471e-01,  -3.42091734e-01,
                          -9.62803328e-02],
                         [-6.51058927e-01,  -5.15604518e-02,   4.71540134e-01,
                          2.46726214e-01,  -5.35375562e-01,  -9.16791046e-02,
                          -5.56037563e-02,   6.54208853e-03,  -2.74700722e-02,
                          6.90865110e-01,   5.44918932e-02,   1.56024461e-02,
                          -3.92788415e-01,   5.52346421e-02,  -3.21903864e-01,
                          -1.27122838e-01,   9.01239391e-02,   4.79050360e-01,
                          1.15946816e-01,  -2.36936195e-01,   7.08788692e-01,
                          -2.23454906e-01,  -9.65032611e-03,   7.67809741e-01,
                          -4.03214944e-01,   5.90999447e-01,  -7.72050782e-01,
                          -1.71104226e-02,   9.12096882e-05,   3.53588880e-01,
                          -2.80541415e-01,   1.31209057e-01,  -2.90025593e-01,
                          5.39778826e-01,   5.69464226e-01,  -6.57833946e-01,
                          1.83114460e-01,   3.71900531e-01,  -5.31745271e-02,
                          8.72043994e-01,   3.85817443e-01,   1.22581135e-01,
                          -8.38662852e-01,  -5.76787769e-02,  -5.88106143e-01,
                          5.86411388e-02,   1.50013227e-01,  -3.65504527e-01,
                          4.22987623e-01,   3.16175094e-01,   6.50783310e-01,
                          2.17973362e-02,   3.03891885e-01,  -8.59332997e-01,
                          -3.24049593e-01,  -2.52065818e-01,  -8.30600639e-01,
                          -3.49389936e-01,   2.74031174e-02,   3.23780672e-01,
                          6.54256579e-01,   6.66854184e-01,   8.48975839e-01,
                          -1.95113033e-01,  -5.06318176e-01,  -5.93207916e-01,
                          -2.92295947e-01,  -3.14676452e-01,   4.71417465e-01,
                          2.79966721e-01,  -7.07754884e-01,   3.41266104e-01,
                          -5.48666390e-01,  -2.64853482e-01,  -1.36189400e-01,
                          2.55479687e-01,  -3.14625200e-01,   4.29170200e-01,
                          -7.22125631e-01,   4.75211821e-01,  -5.18306225e-01,
                          3.90783391e-01,  -7.74002650e-01,   2.89100373e-01,
                          8.46077997e-02,  -5.26592095e-01,   3.34726660e-02,
                          -5.73737671e-01,  -3.28883687e-01,   3.94488701e-01,
                          5.56043445e-01,   5.99223694e-01,  -1.71546449e-01,
                          6.34992787e-01,  -7.96093608e-01,   7.19460709e-01,
                          6.40498564e-01,  -3.07772250e-01,   3.25472775e-01,
                          -4.71450358e-01],
                         [2.01851882e-01,  -5.66315479e-01,   1.80862383e-01,
                          -5.74139907e-01,  -3.06194608e-01,  -1.75294573e-01,
                          6.38235789e-01,  -4.97265539e-01,  -3.62250605e-01,
                          5.72047829e-01,   3.30358902e-01,  -5.64903711e-01,
                          -7.34932238e-01,  -3.20042853e-02,   3.95270604e-01,
                          2.39392613e-01,   7.14381818e-01,  -7.07383846e-01,
                          5.55856130e-01,   6.59265959e-01,   4.96802384e-01,
                          -2.67972004e-01,  -2.62319266e-01,   3.89938166e-01,
                          3.34156367e-01,  -1.76174406e-01,   1.32198094e-01,
                          3.57678787e-01,   4.90288354e-02,  -3.41509866e-01,
                          1.00492978e-01,  -4.13361650e-01,   3.94146926e-01,
                          -5.87925041e-01,  -2.24093271e-01,  -1.76347805e-01,
                          8.19391655e-01,  -2.53059466e-01,   1.70309926e-01,
                          -2.07825176e-01,   1.93129328e-01,   3.99996252e-01,
                          -2.28703930e-01,   7.44489517e-01,  -8.61701969e-02,
                          -1.96526580e-01,  -7.86105946e-02,  -2.03391799e-02,
                          -6.70598262e-01,   4.39198026e-01,  -2.86001963e-01,
                          -3.42435642e-01,   1.17446239e-01,  -6.27471780e-02,
                          3.08730779e-01,  -3.53003080e-01,   1.22680607e-01,
                          5.83453600e-01,   6.26366082e-01,   7.38844074e-01,
                          5.21815826e-01,  -4.62587410e-01,   3.16481252e-01,
                          2.72746848e-01,  -2.73701896e-01,  -5.76511691e-01,
                          -6.26528268e-01,   5.76328184e-01,   2.07498442e-01,
                          2.51308829e-01,   1.74005056e-01,   2.75848156e-01,
                          2.38048103e-01,  -7.00221359e-02,  -3.53378108e-01,
                          1.20773532e-02,   2.63628466e-01,   1.22763719e-01,
                          -1.73409553e-01,   7.15222872e-01,  -1.59878035e-01,
                          2.06988752e-01,  -2.06950131e-02,  -5.58649696e-01,
                          -5.28456755e-01,   4.20957574e-01,  -5.46836617e-01,
                          -5.30867022e-01,  -4.14201177e-01,   6.22343883e-02,
                          -2.28265333e-01,  -1.82759328e-01,  -8.40245564e-01,
                          -2.28469857e-01,  -1.49046671e-01,  -3.89462109e-01,
                          -9.17495648e-02,  -4.12839084e-01,  -6.70212320e-01,
                          5.30991186e-01]])
        self.MTs = MTs
        self.forward_task = ForwardTask(MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio,
                                        percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio, a_polarity_prob, polarity_prob, False, incorrect_polarity_prob)
        self.existing_log_files = glob.glob('*.log')

    def tearDown(self):
        for fname in glob.glob('*.log'):
            if fname not in self.existing_log_files:
                try:
                    os.remove(fname)
                except Exception:
                    print('Cannot remove {}'.format(fname))
        del self.forward_task
        del self.MTs
        global _CYTHON_TESTS
        _CYTHON_TESTS = False
        global _COMBINED_TESTS
        _COMBINED_TESTS = True
        os.chdir(self.cwd)
        if sys.version_info >= (3, 0):
            self.tempdir.cleanup()
        else:
            try:
                shutil.rmtree(self.tempdir)
            except:
                pass

    def test___call__(self):
        global _COMBINED_TESTS
        global _CYTHON_TESTS
        result = self.forward_task()
        self.assertTrue('moment_tensors' in result)
        self.assertTrue('ln_pdf' in result)
        del self.forward_task
        location_samples = [{'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([91.0, 271.0, 120.1]), 'TakeOffAngle':np.array(
            [31.0, 61.0, 12.1])}, {'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([92.0, 272.0, 122.1]), 'TakeOffAngle':np.array([32.0, 62.0, 13.1])}]
        data = {'PPolarity': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                              'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.1], [0.1]])},
                'PPolarity2': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                               'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.1], [0.1]])}}
        a_polarity, error_polarity, incorrect_polarity_prob = polarity_matrix(data, location_samples)
        data = {'P/SHRMSAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                          'Measured': np.matrix([[1.3664, 1], [1.0038, 1]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.02]])},
                'P/SVQAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                        'Measured': np.matrix([[1.3386, 1], [0.9805, 1]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.02]])}}
        a_polarity_prob, polarity_prob, incorrect_polarity_prob = polarity_probability_matrix(data)
        a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio = amplitude_ratio_matrix(
            data, location_samples)
        self.forward_task = ForwardTask(self.MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio,
                                        percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio, a_polarity_prob, polarity_prob,
                                        [1, 1], incorrect_polarity_prob)
        resultcc = self.forward_task()
        self.tearDown()
        self.setUp()
        _COMBINED_TESTS = True
        result = self.forward_task()
        self.assertTrue('moment_tensors' in result)
        self.assertTrue('ln_pdf' in result)
        del self.forward_task
        location_samples = [{'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([91.0, 271.0, 120.1]), 'TakeOffAngle':np.array(
            [31.0, 61.0, 12.1])}, {'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([92.0, 272.0, 122.1]), 'TakeOffAngle':np.array([32.0, 62.0, 13.1])}]
        data = {'PPolarity': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                              'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.1], [0.1]])},
                'PPolarity2': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                               'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.1], [0.1]])}}
        a_polarity, error_polarity, incorrect_polarity_prob = polarity_matrix(data, location_samples)
        data = {'P/SHRMSAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                          'Measured': np.matrix([[1.3664, 1], [1.0038, 1]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.02]])},
                'P/SVQAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                        'Measured': np.matrix([[1.3386, 1], [0.9805, 1]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.02]])}}
        a_polarity_prob, polarity_prob, incorrect_polarity_prob = polarity_probability_matrix(data)
        a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio = amplitude_ratio_matrix(
            data, location_samples)
        self.forward_task = ForwardTask(self.MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio,
                                        percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio, a_polarity_prob, polarity_prob,
                                        [1, 1], incorrect_polarity_prob)
        resultc = self.forward_task()
        self.assertAlmostEquals(resultcc['ln_pdf']._ln_pdf[resultcc['ln_pdf']._ln_pdf >= resultc['ln_pdf']._ln_pdf.min()], resultc['ln_pdf']._ln_pdf, 2)
        self.tearDown()
        self.setUp()
        _CYTHON_TESTS = True
        _COMBINED_TESTS = True
        result = self.forward_task()
        self.assertTrue('moment_tensors' in result)
        self.assertTrue('ln_pdf' in result)
        del self.forward_task
        location_samples = [{'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([91.0, 271.0, 120.1]), 'TakeOffAngle':np.array([31.0, 61.0, 12.1])},
                            {'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([92.0, 272.0, 122.1]), 'TakeOffAngle':np.array([32.0, 62.0, 13.1])}]
        data = {'PPolarity': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                              'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.1], [0.1]])},
                'PPolarity2': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                               'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.1], [0.1]])}}
        a_polarity, error_polarity, incorrect_polarity_prob = polarity_matrix(data, location_samples)
        data = {'P/SHRMSAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                          'Measured': np.matrix([[1.3664, 1], [1.0038, 1]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.02]])},
                'P/SVQAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                        'Measured': np.matrix([[1.3386, 1], [0.9805, 1]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.02]])}}
        a_polarity_prob, polarity_prob, incorrect_polarity_prob = polarity_probability_matrix(data)
        a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio = amplitude_ratio_matrix(
            data, location_samples)
        self.forward_task = ForwardTask(self.MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio,
                                        percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio, a_polarity_prob, polarity_prob,
                                        [1, 1], incorrect_polarity_prob)
        resulto = self.forward_task()
        self.assertAlmostEquals(resultcc['ln_pdf']._ln_pdf[resultcc['ln_pdf']._ln_pdf >= resulto['ln_pdf']._ln_pdf.min()], resulto['ln_pdf']._ln_pdf, 2)
        self.assertAlmostEquals(resulto['ln_pdf']._ln_pdf, resultc['ln_pdf']._ln_pdf, 2)

    def test_run_times(self):
        raise unittest.SkipTest('Test run times not setup correctly for c vs python')
        combined_times = []
        cython_times = []
        python_times = []
        global _COMBINED_TESTS
        global _CYTHON_TESTS
        sys.stdout.write('{:03.0f} %'.format(0))
        imax = 100
        for i in range(imax):
            sys.stdout.write(
                '\b\b\b\b\b{: 3.0f} %'.format(100.0*float(i)/imax))
            self.tearDown()
            self.setUp()
            _COMBINED_TESTS = False
            _CYTHON_TESTS = False
            _CYTHON = True
            # _COMBINED_TESTS,_CYTHON,_CYTHON_TESTS, _COMBINED_TESTS
            del self.forward_task
            location_samples = [{'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([91.0, 271.0, 120.1]), 'TakeOffAngle':np.array(
                [31.0, 61.0, 12.1])}, {'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([92.0, 272.0, 122.1]), 'TakeOffAngle':np.array([32.0, 62.0, 13.1])}]
            data = {'PPolarity': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                  'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.1], [0.1]])},
                    'PPolarity2': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                   'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.1], [0.1]])}}
            a_polarity, error_polarity, incorrect_polarity_prob = polarity_matrix(data, location_samples)
            data = {'P/SHRMSAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                              'Measured': np.matrix([[1.3664, 1], [1.0038, 1]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.02]])},
                    'P/SVQAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                            'Measured': np.matrix([[1.3386, 1], [0.9805, 1]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.02]])}}
            a_polarity_prob, polarity_prob, incorrect_polarity_prob = polarity_probability_matrix(data)
            a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio = amplitude_ratio_matrix(
                data, location_samples)
            self.forward_task = ForwardTask(self.MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio,
                                            percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio, a_polarity_prob, polarity_prob,
                                            [1, 1], incorrect_polarity_prob)
            t0 = time.time()
            resultcc = self.forward_task()
            combined_times.append(time.time()-t0)
            self.tearDown()
            self.setUp()
            _COMBINED_TESTS = True
            del self.forward_task
            location_samples = [{'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([91.0, 271.0, 120.1]), 'TakeOffAngle':np.array(
                [31.0, 61.0, 12.1])}, {'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([92.0, 272.0, 122.1]), 'TakeOffAngle':np.array([32.0, 62.0, 13.1])}]
            data = {'PPolarity': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                  'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.1], [0.1]])},
                    'PPolarity2': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                   'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.1], [0.1]])}}
            a_polarity, error_polarity, incorrect_polarity_prob = polarity_matrix(data, location_samples)
            data = {'P/SHRMSAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                              'Measured': np.matrix([[1.3664, 1], [1.0038, 1]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.02]])},
                    'P/SVQAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                            'Measured': np.matrix([[1.3386, 1], [0.9805, 1]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.02]])}}
            a_polarity_prob, polarity_prob, incorrect_polarity_prob = polarity_probability_matrix(data)
            a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio = amplitude_ratio_matrix(
                data, location_samples)

            self.forward_task = ForwardTask(self.MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio,
                                            percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio, a_polarity_prob, polarity_prob,
                                            [1, 1], incorrect_polarity_prob)
            t0 = time.time()
            resultc = self.forward_task()
            cython_times.append(time.time()-t0)
            self.tearDown()
            self.setUp()
            _CYTHON_TESTS = True
            _COMBINED_TESTS = True
            del self.forward_task
            location_samples = [{'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([91.0, 271.0, 120.1]), 'TakeOffAngle':np.array(
                [31.0, 61.0, 12.1])}, {'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([92.0, 272.0, 122.1]), 'TakeOffAngle':np.array([32.0, 62.0, 13.1])}]
            data = {'PPolarity': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                  'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.1], [0.1]])},
                    'PPolarity2': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                   'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.1], [0.1]])}}
            a_polarity, error_polarity, incorrect_polarity_prob = polarity_matrix(
                data, location_samples)
            data = {'P/SHRMSAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                              'Measured': np.matrix([[1.3664, 1], [1.0038, 1]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.02]])},
                    'P/SVQAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                            'Measured': np.matrix([[1.3386, 1], [0.9805, 1]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.02]])}}
            a_polarity_prob, polarity_prob, incorrect_polarity_prob = polarity_probability_matrix(data)
            a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio = amplitude_ratio_matrix(data, location_samples)
            self.forward_task = ForwardTask(self.MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio,
                                            percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio, a_polarity_prob, polarity_prob,
                                            [1, 1], incorrect_polarity_prob)
            t0 = time.time()
            resulto = self.forward_task()
            python_times.append(time.time()-t0)
        print(' Combined Time = {}, Cython Time = {}, Python Time = {}'.format(sum(combined_times)/float(len(combined_times)), sum(cython_times)/float(len(cython_times)), sum(python_times)/float(len(python_times))))

    def test_location_sample_multiplier(self):
        location_samples = [{'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([91.0, 271.0, 120.1]), 'TakeOffAngle':np.array(
            [31.0, 61.0, 12.1])}, {'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([92.0, 272.0, 122.1]), 'TakeOffAngle':np.array([32.0, 62.0, 13.1])}]
        data = {'PPolarity': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                              'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.001], [0.001]])},
                'PPolarity2': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                               'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.001], [0.001]])}}
        a_polarity, error_polarity, incorrect_polarity_prob = polarity_matrix(data, location_samples)
        data = {'P/SHRMSAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                          'Measured': np.matrix([[1.3664, 1], [1.0038, 1]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.02]])},
                'P/SVQAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                        'Measured': np.matrix([[1.3386, 1], [0.9805, 1]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.02]])}}
        a_polarity_prob, polarity_prob, incorrect_polarity_prob = polarity_probability_matrix(data)
        a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio = amplitude_ratio_matrix(
            data, location_samples)
        self.forward_task = ForwardTask(self.MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio,
                                        percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio, a_polarity_prob, polarity_prob,
                                        [1, 1], incorrect_polarity_prob)
        result = self.forward_task()
        oldpdf = result['ln_pdf']
        oldmts = result['moment_tensors']
        del self.forward_task
        location_samples = [{'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([91.0, 271.0, 120.1]), 'TakeOffAngle':np.array(
            [31.0, 61.0, 12.1])}, {'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([92.0, 272.0, 122.1]), 'TakeOffAngle':np.array([32.0, 62.0, 13.1])}]
        data = {'PPolarity': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                              'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.001], [0.001]])},
                'PPolarity2': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                               'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.001], [0.001]])}}
        a_polarity, error_polarity, incorrect_polarity_prob = polarity_matrix(data, location_samples)
        data = {'P/SHRMSAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                          'Measured': np.matrix([[1.3664, 1], [1.0038, 1]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.02]])},
                'P/SVQAmplitudeRatio': {'Stations': {'Name': ['S01', 'S02'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                                        'Measured': np.matrix([[1.3386, 1], [0.9805, 1]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.02]])}}
        a_polarity_prob, polarity_prob, incorrect_polarity_prob = polarity_probability_matrix(data)
        a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio = amplitude_ratio_matrix(
            data, location_samples)
        self.forward_task = ForwardTask(self.MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio,
                                        percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio, a_polarity_prob, polarity_prob,
                                        [2, 1], incorrect_polarity_prob)
        result = self.forward_task()
        self.assertFalse((result['ln_pdf'] == oldpdf).all())
        self.assertTrue((result['moment_tensors'] == oldmts).all())
        # Calculated separately
        self.assertAlmostEquals(float(result['ln_pdf'][0, 0]), -1139.61341966, 3)


class InversionTestCase(TestCase):

    def setUp(self):
        self.cwd = os.getcwd()
        if sys.version_info >= (3, 0):
            self.tempdir = tempfile.TemporaryDirectory()
            os.chdir(self.tempdir.name)
        else:
            self.tempdir = tempfile.mkdtemp()
            os.chdir(self.tempdir)
        global _DEBUG
        _DEBUG = True
        self.parallel = not _DEBUG
        self.existing_log_files = glob.glob('*.log')
        self.inversion = Inversion({'PPolarity': {}}, parallel=self.parallel, phy_mem=1, max_time=10, convert=False)

    def tearDown(self):
        for fname in glob.glob('*.log'):
            if fname not in self.existing_log_files:
                try:
                    os.remove(fname)
                except Exception:
                    print('Cannot remove {}'.format(fname))
        import gc
        try:
            del self.inversion
        except Exception:
            pass
        try:
            os.remove('test.scatangle')
        except Exception:
            pass
        try:
            os.remove('MTfitOutputMT.mat')
        except Exception:
            pass
        try:
            os.remove('MTfitOutputMTStationDistribution.mat')
        except Exception:
            pass
        try:
            os.remove('csvtest.csv')
        except Exception:
            pass
        try:
            os.remove('invtest.inv')
        except Exception:
            pass
        try:
            os.remove('MTfitOutput_joint_inversionMT.mat')
        except Exception:
            pass
        gc.collect()
        os.chdir(self.cwd)
        if sys.version_info >= (3, 0):
            self.tempdir.cleanup()
        else:
            try:
                shutil.rmtree(self.tempdir)
            except:
                pass

    def _test_csv_file(self):
        csv = """UID=123,,,,
PPolarity,,,,
Name,Azimuth,TakeOffAngle,Measured,Error
S001,120,70,1,0.01
S002,160,60,-1,0.02
P/SHRMSAmplitudeRatio,,,,
Name,Azimuth,TakeOffAngle,Measured,Error
S003,110,10,1 2,0.05 0.04
S005,140,10,1 2,0.01 0.02
,,,,
PPolarity ,,,,
Name,Azimuth,TakeOffAngle,Measured,Error
S003,110,10,1,0.05"""
        open('csvtest.csv', 'w').write(csv)

    def _test_inv_file(self):
        data = {'UID': '1', 'PPolarity': {'Stations': {'Name': ['S001', 'S002'], 'Azimuth': np.matrix([[120.0], [140.0]]), 'TakeOffAngle': np.matrix(
            [[65.0], [22.0]])}, 'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.01], [0.02]])}}
        pickle.dump(data, open('invtest.inv', 'wb'))

    def test__load(self):
        self._test_csv_file()
        self.assertTrue(os.path.exists('./csvtest.csv'))

        x = self.inversion._load('./csvtest.csv')
        self.assertEqual(type(x), list)
        self.assertEqual(len(x), 2)
        self.assertEqual(x[0]['UID'], '123')
        self.assertEqual(
            x[0]['PPolarity']['Stations']['Name'], ['S001', 'S002'])
        self.assertEqual(x[0]['PPolarity']['Measured'][0, 0], 1)
        self.assertEqual(x[0]['PPolarity']['Measured'][1, 0], -1)
        self.assertEqual(
            sorted(x[0].keys()), ['P/SHRMSAmplitudeRatio', 'PPolarity', 'UID'])
        self.assertEqual(x[0]['P/SHRMSAmplitudeRatio']['Error'][0, 0], 0.05)
        self.assertEqual(x[0]['P/SHRMSAmplitudeRatio']['Error'][0, 1], 0.04)
        self.assertEqual(x[0]['P/SHRMSAmplitudeRatio']['Error'][1, 0], 0.01)
        self.assertEqual(x[0]['P/SHRMSAmplitudeRatio']['Error'][1, 1], 0.02)
        self.assertEqual(x[1]['UID'], '2')
        self._test_inv_file()
        x = self.inversion._load('invtest.inv')
        self.assertEqual(type(x), dict)
        self.assertEqual(sorted(x.keys()), ['PPolarity', 'UID'])
        self.assertEqual(x['PPolarity']['Measured'][0, 0], 1)
        self.assertEqual(x['PPolarity']['Stations']['Name'], ['S001', 'S002'])

    def test___init__(self):
        self.tearDown()
        self._test_csv_file()
        self.assertTrue(os.path.exists('./csvtest.csv'))
        self.inversion = Inversion(data_file='./csvtest.csv', parallel=self.parallel, phy_mem=1, max_time=10, convert=False)
        x = self.inversion.data
        self.assertEqual(type(x), list)
        self.assertEqual(len(x), 2)
        self.assertEqual(x[0]['UID'], '123')
        self.assertEqual(x[0]['PPolarity']['Stations']['Name'], ['S001', 'S002'])
        self.assertEqual(x[0]['PPolarity']['Measured'][0, 0], 1)
        self.assertEqual(x[0]['PPolarity']['Measured'][1, 0], -1)
        self.assertEqual(sorted(x[0].keys()), ['P/SHRMSAmplitudeRatio', 'PPolarity', 'UID'])
        self.assertEqual(x[0]['P/SHRMSAmplitudeRatio']['Error'][0, 0], 0.05)
        self.assertEqual(x[0]['P/SHRMSAmplitudeRatio']['Error'][0, 1], 0.04)
        self.assertEqual(x[0]['P/SHRMSAmplitudeRatio']['Error'][1, 0], 0.01)
        self.assertEqual(x[0]['P/SHRMSAmplitudeRatio']['Error'][1, 1], 0.02)
        self.assertEqual(x[1]['UID'], '2')
        self.tearDown()
        self._test_inv_file()
        self.inversion = Inversion(data_file='invtest.inv', parallel=self.parallel, phy_mem=1, max_time=10, convert=False)
        x = self.inversion.data
        self.assertEqual(type(x), list)
        self.assertEqual(len(x), 1)
        self.assertEqual(x[0]['UID'], '1')
        self.assertEqual(x[0]['PPolarity']['Stations']['Name'], ['S001', 'S002'])
        self.assertEqual(x[0]['PPolarity']['Measured'][0, 0], 1)
        self.assertEqual(x[0]['PPolarity']['Measured'][1, 0], -1)
        self.assertEqual(sorted(x[0].keys()), ['PPolarity', 'UID'])

    def test__set_algorithm(self):
        self.inversion._algorithm_name = 'grid'
        self.inversion._set_algorithm()
        self.assertTrue('IterationSample' in str(self.inversion.algorithm.__class__))
        self.inversion._algorithm_name = 'mcmc'
        self.inversion._set_algorithm()
        self.assertTrue('IterativeMultipleTryMetropolisHastingsGaussianTape' in str(self.inversion.algorithm.__class__), str(self.inversion.algorithm.__class__))

    def test__worker_params(self):
        self.assertTrue(self.inversion.number_samples > 0)
        if len([u for u in os.environ.keys() if 'PBS_' in u]):
            self.assertTrue(self.inversion.PBS)
        else:
            self.assertFalse(self.inversion.PBS)

    def test__close_pool(self):
        self.assertTrue(hasattr(self.inversion, 'pool'))
        self.inversion._close_pool()
        self.assertFalse(self.inversion.pool)
        self.tearDown()
        self.inversion = Inversion({'PPolarity': {}}, parallel=self.parallel, phy_mem=1, max_time=10, convert=False)
        self.assertTrue(hasattr(self.inversion, 'pool'))
        self.inversion._close_pool()
        self.assertFalse(self.inversion.pool)

    def test__trim_data(self):
        self.assertTrue(list(self.inversion.data[0].keys()) == ['PPolarity'])
        self.inversion.inversion_options = ['PAmplitude']
        try:
            self.inversion._trim_data(self.inversion.data[0])
        except Exception as e:
            self.assertIsInstance(e, ValueError)

    def test__fid(self):
        self.assertEqual(self.inversion._fid({'UID': 'A01'}, 1), self.inversion._path+os.path.sep+'A01MT.mat')
        self.inversion.fid = 'Test'
        self.assertEqual(self.inversion._fid({'UID': 'A01'}, 1), 'Test1MT.mat')
        self.inversion.fid = False
        self.assertEqual(self.inversion._fid({}, 1), self.inversion._path+os.path.sep+'MTfitOutputMT.mat')

    def test__recover_test(self):
        self.assertFalse(self.inversion._recover_test('RecoverTestMT'))
        self.inversion = Inversion({'UID': 'RecoverTest', 'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                                                        'Measured': np.matrix([[1], [-1], [-1]]), 'Error': np.matrix([[0.001], [0.5], [0.02]])}},
                                   algorithm='Time', parallel=self.parallel, phy_mem=1, max_time=10, convert=False)
        self.inversion.number_samples = 100
        self.assertFalse(len(self.inversion.algorithm.pdf_sample))
        self.inversion.algorithm.max_time = 10
        self.inversion.forward()
        self.assertTrue(self.inversion.algorithm.pdf_sample.n, str(
            self.inversion.algorithm.pdf_sample.n))
        self.assertTrue(os.path.exists('RecoverTestMT.mat'))
        self.inversion = Inversion({'UID': 'RecoverTest', 'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                                                        'Measured': np.matrix([[1], [-1], [-1]]), 'Error': np.matrix([[0.001], [0.5], [0.02]])}},
                                   algorithm='Time', recover=True, parallel=self.parallel, phy_mem=1, max_time=10, convert=False)
        self.assertTrue(self.inversion._recover_test('RecoverTestMT'))
        try:
            os.remove('RecoverTestMT.mat')
        except Exception:
            pass

    def test__file_sample_test(self):
        try:
            self.tearDown()
        except Exception:
            pass
        try:
            from hdf5storage import loadmat   # nopqa E401
        except Exception:
            print('Cannot run _file_sample test as required hdf5storage and h5py modules')
            return
        self.inversion = Inversion({'UID': 'RecoverTest', 'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                                                        'Measured': np.matrix([[1], [-1], [-1]]), 'Error': np.matrix([[0.001], [0.5], [0.02]])}},
                                   algorithm='Time', parallel=self.parallel, phy_mem=1, max_time=10, file_sample=True, convert=False)
        self.assertFalse(self.inversion._file_sample_test('FileSampleTest'))
        self.inversion.algorithm.pdf_sample.fname = 'FileSampleTest.mat'
        self.assertEqual(self.inversion.algorithm.pdf_sample.fname, 'FileSampleTest.mat')
        self.inversion.algorithm.pdf_sample.append(np.matrix([[1], [2], [1], [2], [1], [1]]), np.matrix([1]), 10)
        self.assertTrue(os.path.exists(self.inversion.algorithm.pdf_sample.fname))
        self.inversion.algorithm.pdf_sample.append(np.matrix([[1], [2], [1], [2], [1], [1]]), np.matrix([1]), 20)
        self.inversion.algorithm.pdf_sample.append(np.matrix([[1], [2], [1], [2], [1], [1]]), np.matrix([-np.inf]), 20)
        self.inversion._file_sample_test('FileSampleTest')
        try:
            os.remove('FileSampleTest.mat')
        except Exception:
            pass
        try:
            os.remove('FileSampleTest.mat~')
        except Exception:
            pass

    def test_file_sample(self):
        try:
            self.tearDown()
        except Exception:
            pass
        self.inversion = Inversion({'UID': 'RecoverTest', 'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                                                        'Measured': np.matrix([[1], [-1], [-1]]), 'Error': np.matrix([[0.001], [0.5], [0.02]])}},
                                   algorithm='Time', parallel=self.parallel, phy_mem=1, max_time=10, file_sample=True, convert=False)
        # File Sample is being used appropriately
        self.assertTrue('fname' in self.inversion.algorithm.pdf_sample.__dict__)

    def test__station_angles(self):
        self.assertAlmostEquals(self.inversion._station_angles({'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                                                              'Measured': np.matrix([[1], [-1], [-1]]), 'Error': np.matrix([[0.001], [0.5], [0.02]])}}, 1)[0], np.array([[[9.37349864e-34,   2.50000000e-01,   7.50000000e-01,
                                                                                                                                                                                           2.16489014e-17,   3.74969972e-17,   6.12372436e-01]],
                                                                                                                                                                                         [[-2.53084463e-32,  -7.50000000e-01,  -2.50000000e-01,
                                                                                                                                                                                           -1.94840113e-16,   1.12490991e-16,   6.12372436e-01]],
                                                                                                                                                                                         [[-3.28989928e-01,  -4.93405863e-33,  -6.71010072e-01,
                                                                                                                                                                                           5.69781642e-17,   6.64463024e-01,  -8.13732516e-17]]]))
        self.assertAlmostEquals(self.inversion._station_angles({'P/SHQRMSAmplitudeRatio': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                                                                           'Measured': np.matrix([[1, 2], [-1, 3], [-1, 5]]), 'Error': np.matrix([[0.001, 0.02], [0.5, 0.02], [0.02, 0.02]])}}, 1)[2], np.array([[[9.37349864e-34,   2.50000000e-01,   7.50000000e-01,
                                                                                                                                                                                                                                   2.16489014e-17,   3.74969972e-17,   6.12372436e-01]],
                                                                                                                                                                                                                                 [[2.53084463e-32,  7.50000000e-01,  2.50000000e-01,
                                                                                                                                                                                                                                   1.94840113e-16,   -1.12490991e-16,   -6.12372436e-01]],
                                                                                                                                                                                                                                 [[3.28989928e-01,  4.93405863e-33,  6.71010072e-01,
                                                                                                                                                                                                                                   - 5.69781642e-17,   -6.64463024e-01,  8.13732516e-17]]]))
        self.inversion.marginalise_relative = True
        with open('test.scatangle', 'w') as f:
            f.write(self.station_angles())
        self.inversion.location_pdf_files = ['test.scatangle', '']
        data = {'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                              'Measured': np.matrix([[1], [-1], [-1]]), 'Error': np.matrix([[0.001], [0.5], [0.02]])}}
        angles = self.inversion._station_angles(data, 0)
        self.assertEqual(angles[0].shape, (3, 2, 6))
        angles = self.inversion._station_angles(data, 1)
        self.assertEqual(angles[0].shape, (3, 1, 6))
        self.inversion.marginalise_relative = False
        self.inversion._relative = True
        angles = self.inversion._station_angles(data, 0)
        self.assertEqual(angles[0].shape, (3, 2, 6))
        angles = self.inversion._station_angles(data, 1)
        self.assertEqual(angles[0].shape, (3, 2, 6))

        try:
            os.remove('test.scatangle')
        except Exception:
            pass

    def test_forward(self):
        self.tearDown()
        self.inversion = Inversion({'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                                  'Measured': np.matrix([[1], [-1], [-1]]), 'Error': np.matrix([[0.001], [0.5], [0.02]])}},
                                   algorithm='Time', parallel=self.parallel, phy_mem=0.1, n=2, max_time=10, convert=False)
        self.inversion.number_samples = 100
        self.assertFalse(len(self.inversion.algorithm.pdf_sample))
        self.inversion.algorithm.max_time = 10
        self.inversion.forward()
        self.assertTrue(self.inversion.algorithm.pdf_sample.n, str(
            self.inversion.algorithm.pdf_sample.n))
        self.tearDown()
        with open('test.scatangle', 'w') as f:
            f.write(self.station_angles())
        self.inversion = Inversion({'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                                  'Measured': np.matrix([[1], [-1], [-1]]), 'Error': np.matrix([[0.001], [0.5], [0.02]])}},
                                   algorithm='Time', parallel=self.parallel, phy_mem=0.1, n=2, max_time=10, convert=False)
        self.inversion.location_pdf_files = ['test.scatangle']
        self.inversion.algorithm.max_time = 10
        self.inversion.forward()
        try:
            os.remove('test.scatangle')
        except Exception:
            pass
        self.assertTrue(len(self.inversion.algorithm.pdf_sample))

    def test__random_sampling_forward(self):
        self.tearDown()
        self.inversion = Inversion({'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                                  'Measured': np.matrix([[1], [-1], [-1]]), 'Error': np.matrix([[0.001], [0.5], [0.02]])}},
                                   algorithm='Time', parallel=self.parallel, phy_mem=0.1, n=2, max_time=10, convert=False)
        self.inversion.number_samples = 100
        self.assertFalse(len(self.inversion.algorithm.pdf_sample))
        self.inversion.algorithm.max_time = 10
        self.inversion._random_sampling_forward()
        self.assertTrue(self.inversion.algorithm.pdf_sample.n, str(
            self.inversion.algorithm.pdf_sample.n))
        self.inversion._close_pool()
        self.tearDown()
        with open('test.scatangle', 'w') as f:
            f.write(self.station_angles())
        self.inversion = Inversion({'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                                  'Measured': np.matrix([[1], [-1], [-1]]), 'Error': np.matrix([[0.001], [0.5], [0.02]])}},
                                   algorithm='Time', parallel=self.parallel, phy_mem=0.1, n=2, max_time=10, convert=False)
        self.inversion.location_pdf_files = ['test.scatangle']
        self.inversion.algorithm.max_time = 10
        self.inversion._random_sampling_forward()
        try:
            os.remove('test.scatangle')
        except Exception:
            pass
        self.assertTrue(len(self.inversion.algorithm.pdf_sample))

    def test__random_sampling_multiple_forward(self):
        data = {'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                              'Measured': np.matrix([[1], [-1], [-1]]), 'Error': np.matrix([[0.001], [0.5], [0.02]])}}
        self.tearDown()
        self.inversion = Inversion([data, data], multiple_events=True, algorithm='Time', parallel=self.parallel, phy_mem=0.1, n=2, max_time=10, convert=False)
        self.inversion.number_samples = 100
        self.assertFalse(len(self.inversion.algorithm.pdf_sample))
        self.inversion.algorithm.max_time = 10
        self.inversion._random_sampling_multiple_forward()
        self.assertTrue(self.inversion.algorithm.pdf_sample.n, str(
            self.inversion.algorithm.pdf_sample.n))
        self.inversion._close_pool()
        self.tearDown()
        with open('test.scatangle', 'w') as f:
            f.write(self.station_angles())
        self.inversion = Inversion([data, data], multiple_events=True, algorithm='Time', parallel=self.parallel, phy_mem=0.1, n=2, max_time=10, convert=False)
        self.inversion.location_pdf_files = ['test.scatangle', 'test.scatangle']
        self.inversion.algorithm.max_time = 10
        self.inversion._random_sampling_multiple_forward()
        try:
            os.remove('test.scatangle')
        except Exception:
            pass
        self.assertTrue(len(self.inversion.algorithm.pdf_sample))
        data = {'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                              'Measured': np.matrix([[1], [-1], [-1]]), 'Error': np.matrix([[0.001], [0.5], [0.02]])},
                'PRMSQAmplitude': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                   'Measured': np.matrix([[1], [-1], [-1]]), 'Error': np.matrix([[0.001], [0.5], [0.02]])}}
        self.tearDown()
        self.inversion = Inversion([data, data], multiple_events=True, algorithm='Time', parallel=self.parallel, phy_mem=0.1, n=2, max_time=10, relative_amplitude=True, convert=False)
        self.inversion.number_samples = 100
        self.assertFalse(len(self.inversion.algorithm.pdf_sample))
        self.inversion.algorithm.max_time = 10
        self.inversion._random_sampling_multiple_forward()
        self.assertTrue(self.inversion.algorithm.pdf_sample.n, str(self.inversion.algorithm.pdf_sample.n))
        self.inversion._close_pool()
        self.tearDown()
        with open('test.scatangle', 'w') as f:
            f.write(self.station_angles())
        self.inversion = Inversion([data, data], multiple_events=True, algorithm='Time', parallel=self.parallel, phy_mem=0.1, n=2, max_time=10, relative_amplitude=True, convert=False)
        self.inversion.location_pdf_files = ['test.scatangle', 'test.scatangle']
        self.inversion.algorithm.max_time = 10
        self.inversion._random_sampling_multiple_forward()
        try:
            os.remove('test.scatangle')
        except Exception:
            pass
        self.assertTrue(len(self.inversion.algorithm.pdf_sample))
        try:
            os.remove('MTfitOutput_joint_inversionMT.mat')
        except Exception:
            pass

    def test__mcmc_forward(self):
        self.tearDown()
        try:
            os.remove('TestAMT.mat')
        except Exception:
            pass
        self.inversion = Inversion({'UID': 'TestA', 'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                                                  'Measured': np.matrix([[1], [-1], [-1]]), 'Error': np.matrix([[0.001], [0.5], [0.02]])}},
                                   algorithm='McMC', parallel=False, learning_length=10, chain_length=100, acceptance_rate_window=5, phy_mem=1, convert=False)
        self.inversion.number_samples = 100
        self.assertFalse(len(self.inversion.algorithm.pdf_sample))
        self.inversion.algorithm.max_time = 10
        self.inversion._mcmc_forward()
        self.assertTrue(os.path.exists('TestAMT.mat'))
        try:
            os.remove('TestAMT.mat')
        except Exception:
            pass
        self.inversion._close_pool()
        self.tearDown()
        with open('test.scatangle', 'w') as f:
            f.write(self.station_angles())
        self.inversion = Inversion({'UID': 'TestA', 'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                                                  'Measured': np.matrix([[1], [-1], [-1]]), 'Error': np.matrix([[0.001], [0.5], [0.02]])}},
                                   algorithm='Time', parallel=False, learning_length=10, chain_length=100, acceptance_rate_window=5, phy_mem=1, max_time=10, convert=False)
        self.inversion.location_pdf_files = ['test.scatangle']
        self.inversion.algorithm.max_time = 10
        self.inversion._mcmc_forward()
        self.assertTrue(os.path.exists('TestAMT.mat'))
        try:
            os.remove('TestAMT.mat')
        except Exception:
            pass
        try:
            os.remove('test.scatangle')
        except Exception:
            pass

    def test__mcmc_multiple_forward(self):
        try:
            os.remove('MTfitOutput_joint_inversionMT.mat')
        except Exception:
            pass
        data = {'UID': 'TestA', 'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                              'Measured': np.matrix([[1], [1], [-1]]), 'Error': np.matrix([[0.1], [0.5], [0.02]])}}
        self.inversion = Inversion([data, data], algorithm='McMC', parallel=False, learning_length=10, chain_length=100, acceptance_rate_window=5, phy_mem=1, multiple_events=True, convert=False)
        self.assertFalse(len(self.inversion.algorithm.pdf_sample))
        self.inversion._mcmc_multiple_forward()
        self.assertTrue(os.path.exists('MTfitOutput_joint_inversionMT.mat'))
        try:
            os.remove('MTfitOutput_joint_inversionMT.mat')
        except Exception:
            pass

    @unittest.expectedFailure  # Marginalise relative with location uncertainty not implemented
    def test__mcmc_multiple_forward_location_uncertainty(self):
        try:
            os.remove('MTfitOutput_joint_inversionMT.mat')
        except Exception:
            pass
        data = {'UID': 'TestA', 'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                              'Measured': np.matrix([[1], [1], [-1]]), 'Error': np.matrix([[0.1], [0.5], [0.02]])}}
        data2 = data.copy()
        data2['UID'] = 'TestB'
        self.inversion = Inversion([data, data], algorithm='McMC', parallel=False, relative_amplitude=True, learning_length=10, chain_length=100, acceptance_rate_window=5, phy_mem=1, multiple_events=True, convert=False)
        self.assertFalse(len(self.inversion.algorithm.pdf_sample))
        with open('test.scatangle', 'w') as f:
            f.write(self.station_angles())
        self.inversion.location_pdf_files = ['test.scatangle', 'test.scatangle']
        self.inversion.algorithm.max_time = 10
        self.inversion._mcmc_multiple_forward()
        try:
            os.remove('test.scatangle')
        except Exception:
            pass
        self.assertTrue(os.path.exists('MTfitOutput_joint_inversionMT.mat'))
        try:
            os.remove('MTfitOutput_joint_inversionMT.mat')
        except Exception:
            pass

    def test__mcmc_multiple_forward_amplitude(self):
        try:
            os.remove('MTfitOutput_joint_inversionMT.mat')
        except Exception:
            pass
        data = {'UID': 'TestA', 'PPolarity': {'Stations': {'Name': ['S0649', "S0162"], 'Azimuth': np.matrix([[90.0], [270.0]]), 'TakeOffAngle': np.matrix([[30.0], [60.0]])},
                                              'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.1], [0.5]])},
                'PRMSQAmplitude': {'Stations': {'Name': ['S0649', "S0162"], 'Azimuth': np.matrix([[90.0], [270.0]]), 'TakeOffAngle': np.matrix([[30.0], [60.0]])},
                                   'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.1], [0.5]])}}
        self.inversion = Inversion([data, data], multiple_events=True, algorithm='Time', parallel=False, learning_length=10, chain_length=100, acceptance_rate_window=5,
                                   phy_mem=1, max_time=10, relative_amplitude=False, convert=False)
        self.assertFalse(len(self.inversion.algorithm.pdf_sample))
        self.inversion._mcmc_multiple_forward()
        self.assertTrue(os.path.exists('MTfitOutput_joint_inversionMT.mat'))
        try:
            os.remove('MTfitOutput_joint_inversionMT.mat')
        except Exception:
            pass

    @unittest.expectedFailure  # Marginalise relative with location uncertainty not implemented
    def test__mcmc_multiple_forward_amplitude_location_uncertainty(self):
        try:
            os.remove('MTfitOutput_joint_inversionMT.mat')
        except Exception:
            pass
        data = {'UID': 'TestA', 'PPolarity': {'Stations': {'Name': ['S0649', "S0162"], 'Azimuth': np.matrix([[90.0], [270.0]]), 'TakeOffAngle': np.matrix([[30.0], [60.0]])},
                                              'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.1], [0.5]])},
                'PRMSQAmplitude': {'Stations': {'Name': ['S0649', "S0162"], 'Azimuth': np.matrix([[90.0], [270.0]]), 'TakeOffAngle': np.matrix([[30.0], [60.0]])},
                                   'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.1], [0.5]])}}
        data2 = data.copy()
        data2['UID'] = 'TestB'
        self.inversion = Inversion([data, data], multiple_events=True, algorithm='Time', parallel=False, learning_length=10, chain_length=100, acceptance_rate_window=5,
                                   phy_mem=1, max_time=10, relative_amplitude=True, convert=False)
        self.assertFalse(len(self.inversion.algorithm.pdf_sample))
        with open('test.scatangle', 'w') as f:
            f.write(self.station_angles())
        self.inversion.location_pdf_files = ['test.scatangle', 'test.scatangle']
        self.inversion.algorithm.max_time = 10
        self.inversion._mcmc_multiple_forward()
        try:
            os.remove('test.scatangle')
        except Exception:
            pass
        self.assertTrue(os.path.exists('MTfitOutput_joint_inversionMT.mat'))
        try:
            os.remove('MTfitOutput_joint_inversionMT.mat')
        except Exception:
            pass

    def test__MATLAB_output(self):
        self.tearDown()
        with open('test.scatangle', 'w') as f:
            f.write(self.station_angles())
        self.inversion = Inversion({'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                                  'Measured': np.matrix([[1], [-1], [-1]]), 'Error': np.matrix([[0.001], [0.001], [0.002]])}},
                                   algorithm='Time', parallel=self.parallel, phy_mem=1, max_time=10, convert=False)
        self.inversion.location_pdf_files = ['test.scatangle']
        self.inversion.algorithm.max_time = 10
        self.inversion.forward()
        try:
            os.remove('test.scatangle')
        except Exception:
            pass
        self.assertTrue(os.path.exists('MTfitOutputMT.mat'))
        try:
            os.remove('MTfitOutputMT.mat')
        except Exception:
            pass

    def test__pickle_output(self):
        self.tearDown()
        with open('test.scatangle', 'w') as f:
            f.write(self.station_angles())
        self.inversion = Inversion({'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                                  'Measured': np.matrix([[1], [-1], [-1]]), 'Error': np.matrix([[0.001], [0.001], [0.002]])}},
                                   algorithm='Time', parallel=self.parallel, phy_mem=1, max_time=10, output_format='pickle', convert=False)
        self.inversion.location_pdf_files = ['test.scatangle']
        self.inversion.algorithm.max_time = 10
        self.inversion.forward()
        try:
            os.remove('test.scatangle')
        except Exception:
            pass
        self.assertTrue(os.path.exists('MTfitOutputMT.out'))
        pickle.load(open('MTfitOutputMT.out', 'rb'))
        try:
            os.remove('MTfitOutputMT.out')
        except Exception:
            pass

    def test__hyp_output(self):
        self.tearDown()
        with open('test.scatangle', 'w') as f:
            f.write(self.station_angles())
        self.inversion = Inversion({'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                                  'Measured': np.matrix([[1], [-1], [-1]]), 'Error': np.matrix([[0.001], [0.001], [0.002]])}},
                                   algorithm='Time', parallel=self.parallel, phy_mem=1, max_time=2, output_format='hyp', convert=False)
        self.inversion.location_pdf_files = ['test.scatangle']
        self.inversion._floatmem = 160000
        self.inversion.max_time = 0.1
        self.inversion.forward()
        try:
            os.remove('test.scatangle')
        except Exception:
            pass
        self.assertTrue(os.path.exists('MTfitOutputMT.hyp'))
        try:
            os.remove('MTfitOutputMT.hyp')
        except Exception:
            pass
        self.assertTrue(os.path.exists('MTfitOutputMT.mt'))
        try:
            os.remove('MTfitOutputMT.mt')
        except Exception:
            pass

    def test_recover(self):
        self.tearDown()
        with open('test.scatangle', 'w') as f:
            f.write(self.station_angles())
        self.inversion = Inversion({'UID': 'TestA', 'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                                                  'Measured': np.matrix([[1], [-1], [-1]]), 'Error': np.matrix([[0.1], [0.1], [0.2]])}},
                                   algorithm='Time', parallel=self.parallel, phy_mem=1, max_time=10, convert=False)
        self.inversion.location_pdf_files = ['test.scatangle']
        self.inversion.algorithm.max_time = 10
        self.inversion.forward()
        try:
            os.remove('test.scatangle')
        except Exception:
            pass
        self.assertTrue(os.path.exists('TestAMT.mat'))
        del self.inversion
        gc.collect()
        self.assertTrue(os.path.exists('TestAMT.mat'))
        with open('test.scatangle', 'w') as f:
            f.write(self.station_angles())
        self.inversion = Inversion({'UID': 'TestA', 'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                                                  'Measured': np.matrix([[1], [-1], [-1]]), 'Error': np.matrix([[0.001], [0.001], [0.02]])}},
                                   algorithm='Time', recover=True, parallel=self.parallel, phy_mem=1, max_time=10, convert=False)
        self.inversion.location_pdf_files = ['test.scatangle']
        self.inversion.algorithm.max_time = 10
        self.inversion.forward()
        # LOG FILE NOT DELETING
        try:
            os.remove('test.scatangle')
        except Exception:
            pass
        try:
            os.remove('TestAMT.mat')
        except Exception:
            pass
        try:
            os.remove('TestAMTStationDistribution.mat')
        except Exception:
            pass

    def station_angles(self):
        out = "504.7\n"
        out += "S0271   231.1   154.7\n"
        out += "S0649   42.9    109.7\n"
        out += "S0484   21.2    145.4\n"
        out += "S0263   256.4   122.7\n"
        out += "S0142   197.4   137.6\n"
        out += "S0244   229.7   148.1\n"
        out += "S0415   75.6    122.8\n"
        out += "S0065   187.5   126.1\n"
        out += "S0362   85.3    128.2\n"
        out += "S0450   307.5   137.7\n"
        out += "S0534   355.8   138.2\n"
        out += "S0641   14.7    120.2\n"
        out += "S0155   123.5   117\n"
        out += "S0162   231.8   127.5\n"
        out += "S0650   45.9    108.2\n"
        out += "S0195   193.8   147.3\n"
        out += "S0517   53.7    124.2\n"
        out += "S0004   218.4   109.8\n"
        out += "S0588   12.9    128.6\n"
        out += "S0377   325.5   165.3\n"
        out += "S0618   29.4    120.5\n"
        out += "S0347   278.9   149.5\n"
        out += "S0529   326.1   131.7\n"
        out += "S0083   223.7   118.2\n"
        out += "S0595   42.6    117.8\n"
        out += "S0236   253.6   118.6\n"
        out += '\n'
        out += "504.7\n"
        out += "S0271   230.9   154.8\n"
        out += "S0649   43      109.8\n"
        out += "S0484   21.3    145.4\n"
        out += "S0263   256.4   122.8\n"
        out += "S0142   197.3   137.6\n"
        out += "S0244   229.6   148.1\n"
        out += "S0415   75.7    122.8\n"
        out += "S0065   187.4   126.1\n"
        out += "S0362   85.3    128.2\n"
        out += "S0450   307.5   137.8\n"
        out += "S0534   355.7   138.3\n"
        out += "S0641   14.8    120.3\n"
        out += "S0155   123.5   117.1\n"
        out += "S0162   231.7   127.6\n"
        out += "S0650   45.9    108.3\n"
        out += "S0195   193.6   147.3\n"
        out += "S0517   53.7    124.2\n"
        out += "S0004   218.3   109.9\n"
        out += "S0588   13      128.7\n"
        out += "S0377   325.7   165.4\n"
        out += "S0618   29.5    120.5\n"
        out += "S0347   278.9   149.6\n"
        out += "S0529   326.1   131.8\n"
        out += "S0083   223.7   118.2\n"
        out += "S0595   42.7    117.9\n"
        out += "S0236   253.5   118.7\n"
        out += "\n"
        return out


class MiscTestCase(TestCase):

    def setUp(self):
        self.cwd = os.getcwd()
        if sys.version_info >= (3, 0):
            self.tempdir = tempfile.TemporaryDirectory()
            os.chdir(self.tempdir.name)
        else:
            self.tempdir = tempfile.mkdtemp()
            os.chdir(self.tempdir)
        self.existing_csv_files = glob.glob('*.csv')
        self.existing_hyp_files = glob.glob('*.hyp')

    def tearDown(self):
        for fname in glob.glob('*.csv'):
            if fname not in self.existing_csv_files:
                try:
                    os.remove(fname)
                except Exception:
                    print('Cannot remove {}'.format(fname))
        for fname in glob.glob('*.hyp'):
            if fname not in self.existing_hyp_files:
                try:
                    os.remove(fname)
                except Exception:
                    print('Cannot remove {}'.format(fname))
        os.chdir(self.cwd)
        if sys.version_info >= (3, 0):
            self.tempdir.cleanup()
        else:
            try:
                shutil.rmtree(self.tempdir)
            except:
                pass

    def station_angles(self):
        out = "504.7\n"
        out += "S0271   231.1   154.7\n"
        out += "S0649   42.9    109.7\n"
        out += "S0484   21.2    145.4\n"
        out += "S0263   256.4   122.7\n"
        out += "S0142   197.4   137.6\n"
        out += "S0244   229.7   148.1\n"
        out += "S0415   75.6    122.8\n"
        out += "S0065   187.5   126.1\n"
        out += "S0362   85.3    128.2\n"
        out += "S0450   307.5   137.7\n"
        out += "S0534   355.8   138.2\n"
        out += "S0641   14.7    120.2\n"
        out += "S0155   123.5   117\n"
        out += "S0162   231.8   127.5\n"
        out += "S0650   45.9    108.2\n"
        out += "S0195   193.8   147.3\n"
        out += "S0517   53.7    124.2\n"
        out += "S0004   218.4   109.8\n"
        out += "S0588   12.9    128.6\n"
        out += "S0377   325.5   165.3\n"
        out += "S0618   29.4    120.5\n"
        out += "S0347   278.9   149.5\n"
        out += "S0529   326.1   131.7\n"
        out += "S0083   223.7   118.2\n"
        out += "S0595   42.6    117.8\n"
        out += "S0236   253.6   118.6\n"
        out += '\n'
        out += "504.7\n"
        out += "S0271   230.9   154.8\n"
        out += "S0649   43      109.8\n"
        out += "S0484   21.3    145.4\n"
        out += "S0263   256.4   122.8\n"
        out += "S0142   197.3   137.6\n"
        out += "S0244   229.6   148.1\n"
        out += "S0415   75.7    122.8\n"
        out += "S0065   187.4   126.1\n"
        out += "S0362   85.3    128.2\n"
        out += "S0450   307.5   137.8\n"
        out += "S0534   355.7   138.3\n"
        out += "S0641   14.8    120.3\n"
        out += "S0155   123.5   117.1\n"
        out += "S0162   231.7   127.6\n"
        out += "S0650   45.9    108.3\n"
        out += "S0195   193.6   147.3\n"
        out += "S0517   53.7    124.2\n"
        out += "S0004   218.3   109.9\n"
        out += "S0588   13      128.7\n"
        out += "S0377   325.7   165.4\n"
        out += "S0618   29.5    120.5\n"
        out += "S0347   278.9   149.6\n"
        out += "S0529   326.1   131.8\n"
        out += "S0083   223.7   118.2\n"
        out += "S0595   42.7    117.9\n"
        out += "S0236   253.5   118.7\n"
        out += "\n"
        return out

    def test_assertAlmostEquals(self):
        self.assertAlmostEquals(1.000000, 0.9999999999)

    def test_station_angles(self):
        self.assertAlmostEquals(station_angles({'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])}, 'P'), np.array(
            [[0.000, 0.2500, 0.7500, 0.000, 0.0000, 0.6124], [0.000, 0.7500, 0.2500, 0.000, -0.0000, -0.6124]]), 3)

    def test_polarity_matrix(self):
        data = {'PPolarity': {'Stations': {'Name': ['S0271', 'S0595'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                              'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.001], [0.001]])},
                'PPolarity2': {'Stations': {'Name': ['S0347', 'S0588'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                               'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.001], [0.001]])}}
        A, Error, IncorrectPolarityProb = polarity_matrix(data)
        self.assertAlmostEquals(A, np.array([[[9.37349864e-34, 2.50000000e-01, 7.50000000e-01, 2.16489014e-17, 3.74969972e-17, 6.12372436e-01]], [[-2.53084463e-32, -7.50000000e-01, -2.50000000e-01, -1.94840113e-16, 1.12490991e-16, 6.12372436e-01]], [
                                [9.37349864e-34, 2.50000000e-01, 7.50000000e-01, 2.16489014e-17, 3.74969972e-17, 6.12372436e-01]], [[-2.53084463e-32, -7.50000000e-01, -2.50000000e-01, -1.94840113e-16, 1.12490991e-16, 6.12372436e-01]]]))
        with open('test.scatangle', 'w') as f:
            f.write(self.station_angles())
        location_samples, StationProb = parse_scatangle('test.scatangle')
        A, Error, IncorrectPolarityProb = polarity_matrix(data, location_samples)
        self.assertEqual(A.shape, (4, 2, 6))
        data = {'PPolarity': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                              'Measured': np.matrix([[1], [-1], [-1]]), 'Error': np.matrix([[0.001], [0.5], [0.02]])}}
        A, Error, IncorrectPolarityProb = polarity_matrix(data, location_samples)
        self.assertEqual(A.shape, (3, 2, 6))
        self.assertEqual(IncorrectPolarityProb, 0)
        os.remove('test.scatangle')
        data = {'PPolarity': {'Stations': {'Name': ['S0271', 'S0595'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                              'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.001], [0.001]]), 'IncorrectPolarityProbability': np.matrix([[0.1], [0.2]])},
                'PPolarity2': {'Stations': {'Name': ['S0347', 'S0588'], 'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0])},
                               'Measured': np.matrix([[1], [-1]]), 'Error': np.matrix([[0.001], [0.001]])}}
        A, Error, IncorrectPolarityProb = polarity_matrix(data)
        self.assertAlmostEquals(A, np.array([[[9.37349864e-34, 2.50000000e-01, 7.50000000e-01, 2.16489014e-17, 3.74969972e-17, 6.12372436e-01]], [[-2.53084463e-32, -7.50000000e-01, -2.50000000e-01, -1.94840113e-16, 1.12490991e-16, 6.12372436e-01]], [
                                [9.37349864e-34, 2.50000000e-01, 7.50000000e-01, 2.16489014e-17, 3.74969972e-17, 6.12372436e-01]], [[-2.53084463e-32, -7.50000000e-01, -2.50000000e-01, -1.94840113e-16, 1.12490991e-16, 6.12372436e-01]]]))
        self.assertEqual(IncorrectPolarityProb[1], 0.2)
        self.assertEqual(IncorrectPolarityProb[0], 0.1)
        self.assertEqual(IncorrectPolarityProb[2], 0)
        self.assertEqual(IncorrectPolarityProb[3], 0)

    def test_amplitude_ratio_matrix(self):
        data = {'P/SHRMSAmplitudeRatio': {'Stations': {'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0]), 'Name': ['S01', 'S02']},
                                          'Measured': np.matrix([[1, 2], [-1, 2]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.02]])},
                'P/SVAmplitudeRatio': {'Stations': {'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0]), 'Name': ['S01', 'S02']},
                                       'Measured': np.matrix([[1, 5], [-1, 3]]), 'Error': np.matrix([[0.001, 0.02], [0.001, 0.03]])}}
        A1, A2, amplitude_ratio, Error1, Error2 = amplitude_ratio_matrix(data)
        A1test = np.array([[[9.37349864e-34, 2.50000000e-01, 7.50000000e-01, 2.16489014e-17, 3.74969972e-17, 6.12372436e-01]],  # P/SH S01
                           [[2.53084463e-32, 7.50000000e-01, 2.50000000e-01,
                               1.94840113e-16, -1.12490991e-16, -6.12372436e-01]],  # P/SH S02
                           [[9.37349864e-34, 2.50000000e-01, 7.50000000e-01,
                               2.16489014e-17, 3.74969972e-17, 6.12372436e-01]],  # P/SV S01
                           [[2.53084463e-32, 7.50000000e-01, 2.50000000e-01, 1.94840113e-16, -1.12490991e-16, -6.12372436e-01]]])  # P/SV S02
        self.assertAlmostEquals(A1, A1test)
        data = {'P/SHRMSAmplitudeRatio': {'Stations': {'Azimuth': np.array([90.0, 270.0]), 'Name': ['S01', 'S02'], 'TakeOffAngle': np.array([30.0, 60.0])},
                                          'Measured': np.matrix([[1, 4], [-1, 2]]), 'Error': np.matrix([[0.001, 0.003], [0.001, 0.04]])},
                'P/SVAmplitudeRatio': {'Stations': {'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0]), 'Name': ['S01', 'S02']},
                                       'Measured': np.matrix([[1, 2], [-1, 3]]), 'Error': np.matrix([[0.001, 0.001], [0.001, 0.02]])}}
        location_samples = [{'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([91.0, 271.0, 120.1]), 'TakeOffAngle':np.array(
            [31.0, 61.0, 12.1])}, {'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([92.0, 272.0, 122.1]), 'TakeOffAngle':np.array([32.0, 62.0, 13.1])}]
        A1, A2, amplitude_ratio, Error1, Error2 = amplitude_ratio_matrix(data, location_samples)
        A1test = np.array([[[8.07958974e-05,   2.65183423e-01,   7.34735781e-01,  # P/SH S01
                             -6.54610306e-03,  -1.08962046e-02,   6.24243141e-01],  # P/SH S01 Sample 1
                            [3.42024915e-04,   2.80472402e-01,   7.19185573e-01,
                             -1.38512490e-02,  -2.21801436e-02,   6.35156209e-01]],  # P/SH S01 Sample 2

                           [[2.32996370e-04,   7.64726636e-01,   2.35040368e-01,  # P/SH S02
                             -1.88774220e-02,   1.04655198e-02,  -5.99569228e-01],  # P/SH S02 Sample 1
                            [9.49528887e-04,   7.78646923e-01,   2.20403548e-01,
                               -3.84538099e-02,   2.04587163e-02,  -5.85860981e-01]],  # P/SH S02 Sample 2

                           [[8.07958974e-05,   2.65183423e-01,   7.34735781e-01,  # P/SV S01
                             -6.54610306e-03,  -1.08962046e-02,   6.24243141e-01],  # P/SV S01 Sample 1
                            [3.42024915e-04,   2.80472402e-01,   7.19185573e-01,
                               -1.38512490e-02,  -2.21801436e-02,   6.35156209e-01]],  # P/SV S01 Sample 2

                           [[2.32996370e-04,   7.64726636e-01,   2.35040368e-01,  # P/SV S02
                             -1.88774220e-02,   1.04655198e-02,  -5.99569228e-01],  # P/SV S02 Sample 1
                            [9.49528887e-04,   7.78646923e-01,   2.20403548e-01,
                               -3.84538099e-02,   2.04587163e-02,  -5.85860981e-01]]])  # P/SV S02 Sample 2
        self.assertAlmostEquals(A1, A1test)
        with open('test.scatangle', 'w') as f:
            f.write(self.station_angles())
        data = {'P/SHRMSAmplitudeRatio': {'Stations': {'Azimuth': np.array([90.0, 270.0]), 'Name': ['S0162', 'S0083'], 'TakeOffAngle': np.array([30.0, 60.0])},
                                          'Measured': np.matrix([[1, 4], [-1, 8]]), 'Error': np.matrix([[0.001, 0.003], [0.001, 0.04]])},
                'P/SVAmplitudeRatio': {'Stations': {'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0]), 'Name': ['S0162', 'S0083']},
                                       'Measured': np.matrix([[1, 2], [-1, 3]]), 'Error': np.matrix([[0.001, 0.001], [0.001, 0.02]])}}
        location_samples, StationProb = parse_scatangle('test.scatangle')
        A1, A2, amplitude_ratio, Error1, Error2 = amplitude_ratio_matrix(
            data, location_samples)
        self.assertEqual(A1.shape, (4, 2, 6))
        data = {'P/SHRMSAmplitudeRatio': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                          'Measured': np.matrix([[1, 1], [-1, 4], [-1, 6]]), 'Error': np.matrix([[0.001, 0.02], [0.5, 0.1], [0.02, 0.03]])}}
        A1, A2, amplitude_ratio, Error1, Error2 = amplitude_ratio_matrix(
            data, location_samples)
        self.assertEqual(A1.shape, (3, 2, 6))
        os.remove('test.scatangle')

    def test_relative_amplitude_ratio_matrix(self):
        data = {'PAmplitude': {'Stations': {'Azimuth': np.array([90.0, 270.0]), 'TakeOffAngle': np.array([30.0, 60.0]), 'Name': ['SA', 'SB']},
                               'Measured': np.matrix([[1], [-2]]), 'Error': np.matrix([[0.001], [0.02]])}}
        a_relative_amplitude, relative_amplitude, percentage_error_relative_amplitude, relative_amplitude_stations = relative_amplitude_ratio_matrix(
            data)
        a_relative_amplitudetest = np.array([[[9.37349864e-34, 2.50000000e-01, 7.50000000e-01, 2.16489014e-17, 3.74969972e-17, 6.12372436e-01]],  # P S01
                                             [[2.53084463e-32, 7.50000000e-01, 2.50000000e-01, 1.94840113e-16, -1.12490991e-16, -6.12372436e-01]]])  # P S02
        self.assertAlmostEquals(a_relative_amplitude, a_relative_amplitudetest)
        data = {'PRMSAmplitude': {'Stations': {'Azimuth': np.array([90.0, 270.0]), 'Name': ['S01', 'S02'], 'TakeOffAngle': np.array([30.0, 60.0])},
                                  'Measured': np.matrix([[4], [-1]]), 'Error': np.matrix([[0.003], [0.04]])}}
        location_samples = [{'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([91.0, 271.0, 120.1]), 'TakeOffAngle':np.array(
            [31.0, 61.0, 12.1])}, {'Name': ['S01', 'S02', 'S03'], 'Azimuth':np.array([92.0, 272.0, 122.1]), 'TakeOffAngle':np.array([32.0, 62.0, 13.1])}]
        a_relative_amplitude, relative_amplitude, percentage_error_relative_amplitude, relative_amplitude_stations = relative_amplitude_ratio_matrix(
            data, location_samples)
        a_relative_amplitudetest = np.array([[[8.07958974e-05,   2.65183423e-01,   7.34735781e-01,  # P S01
                                               -6.54610306e-03,  -1.08962046e-02,   6.24243141e-01],  # P S01 Sample 1
                                              [3.42024915e-04,   2.80472402e-01,   7.19185573e-01,
                                               -1.38512490e-02,  -2.21801436e-02,   6.35156209e-01]],  # P S01 Sample 2

                                             [[2.32996370e-04,   7.64726636e-01,   2.35040368e-01,  # P S02
                                               -1.88774220e-02,   1.04655198e-02,  -5.99569228e-01],  # P S02 Sample 1
                                              [9.49528887e-04,   7.78646923e-01,   2.20403548e-01,
                                                 -3.84538099e-02,   2.04587163e-02,  -5.85860981e-01]]])  # P S02 Sample 2
        self.assertAlmostEquals(a_relative_amplitude, a_relative_amplitudetest)
        with open('test.scatangle', 'w') as f:
            f.write(self.station_angles())
        data = {'PRMSAmplitude': {'Stations': {'Azimuth': np.array([90.0, 270.0]), 'Name': ['S0162', 'S0083'], 'TakeOffAngle': np.array([30.0, 60.0])},
                                  'Measured': np.matrix([[4], [-8]]), 'Error': np.matrix([[0.003], [0.04]])}}

        location_samples, StationProb = parse_scatangle('test.scatangle')
        a_relative_amplitude, relative_amplitude, percentage_error_relative_amplitude, relative_amplitude_stations = relative_amplitude_ratio_matrix(
            data)
        self.assertEqual(a_relative_amplitude.shape, (2, 1, 6))
        data = {'PRMSAmplitude': {'Stations': {'Name': ['S0649', "S0162", "S0083"], 'Azimuth': np.matrix([[90.0], [270.0], [180.]]), 'TakeOffAngle': np.matrix([[30.0], [60.0], [35.]])},
                                  'Measured': np.matrix([[1], [-4], [6]]), 'Error': np.matrix([[0.02], [0.5], [0.03]])}}
        a_relative_amplitude, relative_amplitude, percentage_error_relative_amplitude, relative_amplitude_stations = relative_amplitude_ratio_matrix(
            data)
        self.assertEqual(a_relative_amplitude.shape, (3, 1, 6))
        os.remove('test.scatangle')

    def test__intersect_stations(self):
        relative_amplitude_stations_i = ['SA', 'SB']
        relative_amplitude_stations_j = ['SB', 'SC', ]
        a_relative_amplitudetest = np.matrix([[9.37349864e-34, 2.50000000e-01, 7.50000000e-01, 2.16489014e-17, 3.74969972e-17, 6.12372436e-01],  # P S01
                                              [2.53084463e-32, 7.50000000e-01, 2.50000000e-01, 1.94840113e-16, -1.12490991e-16, -6.12372436e-01]])  # P S02
        relative_amplitude_i = np.matrix([[1], [2]])
        relative_amplitude_j = np.matrix([[3], [4]])
        percentage_error_relative_amplitude_i = np.matrix([[1], [4]])
        percentage_error_relative_amplitude_j = np.matrix([[3], [2]])
        a_relative_amplitudetest = np.array([[[8.07958974e-05,   2.65183423e-01,   7.34735781e-01,  # P S01
                                               -6.54610306e-03,  -1.08962046e-02,   6.24243141e-01],  # P S01 Sample 1
                                              [3.42024915e-04,   2.80472402e-01,   7.19185573e-01,
                                               -1.38512490e-02,  -2.21801436e-02,   6.35156209e-01]],  # P S01 Sample 2

                                             [[2.32996370e-04,   7.64726636e-01,   2.35040368e-01,  # P S02
                                               -1.88774220e-02,   1.04655198e-02,  -5.99569228e-01],  # P S02 Sample 1
                                              [9.49528887e-04,   7.78646923e-01,   2.20403548e-01,
                                                 -3.84538099e-02,   2.04587163e-02,  -5.85860981e-01]]])
        a1_relative_amplitude_ratio, a2_relative_amplitude_ratio, relative_amplitude_1, relative_amplitude_2, percentage_error_relative_amplitude_1, percentage_error_relative_amplitude_2, n_intersections = _intersect_stations(
            relative_amplitude_stations_i, relative_amplitude_stations_j, a_relative_amplitudetest, a_relative_amplitudetest, relative_amplitude_i, relative_amplitude_j, percentage_error_relative_amplitude_i, percentage_error_relative_amplitude_j)
        self.assertEqual(n_intersections, 1)
        self.assertEqual(relative_amplitude_2, np.matrix([[3]]))
        self.assertEqual(relative_amplitude_1, np.matrix([[2]]))
        self.assertEqual(
            percentage_error_relative_amplitude_1, np.matrix([[4]]))
        self.assertEqual(
            percentage_error_relative_amplitude_2, np.matrix([[3]]))
        self.assertTrue(
            (a1_relative_amplitude_ratio == a_relative_amplitudetest[1, :]).all())
        self.assertTrue(
            (a2_relative_amplitude_ratio == a_relative_amplitudetest[0, :]).all())
