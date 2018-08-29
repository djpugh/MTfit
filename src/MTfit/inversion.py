"""
inversion
=========
Module containing the main inversion class, MTfit.inversion.Inversion class.

"""

# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.


import sys
import os
import glob
import operator
import time
import gc
import warnings
import multiprocessing
import traceback
import logging
import copy
try:
    import cPickle as pickle
except ImportError:
    import pickle
from math import ceil


import numpy as np

from .utilities.multiprocessing_helper import JobPool
from .utilities.file_io import parse_hyp
from .utilities.file_io import parse_csv
from .utilities.file_io import full_pdf_output_dicts
from .utilities.file_io import hyp_output_dicts
from .utilities.file_io import MATLAB_output
from .utilities.file_io import pickle_output
from .utilities.file_io import hyp_output
from .utilities.file_io import read_binary_output
from .utilities.file_io import read_sf_output
from .sampling import ln_bayesian_evidence
from .algorithms import BaseAlgorithm
from .algorithms import IterationSample
from .algorithms import TimeSample
from .algorithms import MarkovChainMonteCarloAlgorithmCreator
from .probability import polarity_ln_pdf
from .probability import LnPDF
from .probability import polarity_probability_ln_pdf
from .probability import amplitude_ratio_ln_pdf
from .probability import relative_amplitude_ratio_ln_pdf
from .probability import ln_marginalise
from .probability import dkl_estimate
from .utilities.extensions import get_extensions
from .extensions.scatangle import parse_scatangle


logger = logging.getLogger('MTfit.inversion')

try:
    from .probability import cprobability
except ImportError:
    cprobability = False
except Exception:
    logger.exception('Error importing c extension')
    cprobability = False

_MEMTEST = False
_DEBUG = False
_VERBOSITY = 0
_CYTHON_TESTS = False
_COMBINED_TESTS = False


def memory_profile_test(memtest=_MEMTEST):
    """
    Decorator for running memory profiler (memory_profiler module) to test memory requirements and bottlenecks.
    """
    def decorator(function):
        """Memory Profile Test Actual Decorator"""
        if memtest:
            try:
                from memory_profiler import profile
                return profile(function)
            except Exception:
                return function
        return function
    return decorator


#
# Task Objects
#


class ForwardTask(object):
    """
    Forward modelling task

    Task which carries out forward modelling and returns results dictionary containing PDF, MTs and Number of forward modelled Samples
    Callable object.
    """
    def __init__(self, mt, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio,
                 percentage_error2_amplitude_ratio, a_polarity_prob, polarity_prob, location_sample_multipliers=False, incorrect_polarity_prob=0, return_zero=False,
                 reuse=False, marginalise=True, generate_samples=100000, cutoff=100000000, dc=False, extension_data={}):
        """
        ForwardTask initialisation

        Args
            mt: numpy matrix object containing moment tensor 6 vectors.
            a_polarity: Polarity observations station-ray 6 vector.
            error_polarity: Polarity observations error.
            a1_amplitude_ratio: Amplitude ratio observations numerator station-ray 6 vector.
            a2_amplitude_ratio: Amplitude ratio observations denominator station-ray 6 vector.
            amplitude_ratio: Observed amplitude ratio.
            percentage_error1_amplitude_ratio: Amplitude ratio observations percentage error for the numerator.
            percentage_error2_amplitude_ratio: Amplitude ratio observations percentage error for the denominator.
            a_polarity_prob: Polarity PDF observations station-ray 6 vector.
            polarity_prob: Polarity PDF probability
            return_zero:[False] Boolean flag to return zero probability samples.
            reuse:[False] Boolean flag as to whether task is reused (mt changed and re-run...)
            marginalise:[True] Boolean flag as to whether pdf is marginalised over location/model uncertainty or not.
            generate_samples:[100000] Number of samples to generate when generating samples.
            cutoff:[100000000] Max number of samples to try when generating samples.
            dc:[False] DC or MT when generating samples.
            extension_data:[{}] A dictionary of processed data for use by an MTfit.data_types extension.
        """
        self.mt = mt
        self.a_polarity = a_polarity
        self.error_polarity = error_polarity
        self.a1_amplitude_ratio = a1_amplitude_ratio
        self.a2_amplitude_ratio = a2_amplitude_ratio
        self.amplitude_ratio = amplitude_ratio
        self.percentage_error1_amplitude_ratio = percentage_error1_amplitude_ratio
        self.percentage_error2_amplitude_ratio = percentage_error2_amplitude_ratio
        self.a_polarity_prob = a_polarity_prob
        self.polarity_prob = polarity_prob
        self.incorrect_polarity_prob = incorrect_polarity_prob
        self.location_sample_multipliers = location_sample_multipliers
        self._return_zero = return_zero
        self._reuse = reuse
        self.marginalise = marginalise
        self.dc = dc
        self.cutoff = cutoff
        self.generate_samples = 0
        self.extension_data = extension_data
        if isinstance(self.mt, bool) and cprobability is not False:
            self.generate_samples = generate_samples

    @memory_profile_test(_MEMTEST)
    def __call__(self):
        """
        Runs the ForwardTask and returns the result as a dictionary

        Returns
            Dictionary of results as {'moment_tensors':MTs,'ln_pdf':prob,'n':numberOfInitialSamples}
        """
        global _VERBOSITY
        global _DEBUG
        try:
            if _VERBOSITY >= 3:
                try:
                    import memory_profiler
                    logger.info('Memory Usage - Initial: {}'.format(memory_profiler.memory_usage()))
                except ImportError:
                    memory_profiler = False
            if self.generate_samples or len(self.mt):
                _return = False  # set _return flag to False before starting
                if cprobability and not _CYTHON_TESTS and not _COMBINED_TESTS:
                    try:
                        # Get number of samples if not generating them
                        if not self.generate_samples:
                            N = self.mt.shape[1]
                        # Check if there are location PDF samples.
                        location_samples = False
                        n_location_samples = 1
                        if isinstance(self.a_polarity, np.ndarray) and len(self.a_polarity.shape) > 2 and self.a_polarity.shape[1] > 1:
                            location_samples = True
                            n_location_samples = self.a_polarity.shape[1]
                        elif isinstance(self.a1_amplitude_ratio, np.ndarray) and len(self.a1_amplitude_ratio.shape) > 2 and self.a1_amplitude_ratio.shape[1] > 1:
                            location_samples = True
                            n_location_samples = self.a1_amplitude_ratio.shape[1]
                        elif isinstance(self.a_polarity_prob, np.ndarray) and len(self.a_polarity_prob.shape) > 2 and self.a_polarity_prob.shape[1] > 1:
                            location_samples = True
                            n_location_samples = self.a_polarity_prob.shape[1]
                        # Check location sample probabilities (easier on memory and computation to have 2 samples in the same place have double the probability)
                        if self.location_sample_multipliers:
                            ln_location_sample_multipliers = np.log(np.array(self.location_sample_multipliers).astype(np.float64, copy=False))
                        else:
                            ln_location_sample_multipliers = np.zeros(n_location_samples)
                        # Try to use cprobabilities combined PDF code:
                        if self.generate_samples:
                            ln_p_total, mt, N = cprobability.combined_ln_pdf(self.mt, self.a_polarity, self.error_polarity, self.a1_amplitude_ratio,
                                                                             self.a2_amplitude_ratio, self.amplitude_ratio, self.percentage_error1_amplitude_ratio,
                                                                             self.percentage_error2_amplitude_ratio, self.a_polarity_prob, self.polarity_prob,
                                                                             self.incorrect_polarity_prob, generate_samples=self.generate_samples, dc=self.dc,
                                                                             cutoff=self.cutoff, marginalised=int(self.marginalise), location_samples_multipliers=ln_location_sample_multipliers)
                            self.mt = np.asarray(mt)
                        else:
                            ln_p_total = cprobability.combined_ln_pdf(self.mt, self.a_polarity, self.error_polarity, self.a1_amplitude_ratio, self.a2_amplitude_ratio, self.amplitude_ratio,
                                                                      self.percentage_error1_amplitude_ratio, self.percentage_error2_amplitude_ratio, self.a_polarity_prob, self.polarity_prob,
                                                                      self.incorrect_polarity_prob, generate_samples=self.generate_samples, dc=self.dc, cutoff=self.cutoff,
                                                                      marginalised=int(self.marginalise), location_samples_multipliers=ln_location_sample_multipliers)
                        # Handle extensions
                        if len(self.extension_data):
                            extension_names, extensions = get_extensions('MTfit.data_types')
                            for key in self.extension_data.keys():
                                try:
                                    if key in extension_names and 'relative' not in key:
                                        ln_p_ext = extensions[key](self.mt, **self.extension_data[key])
                                        ln_p_total = cprobability.ln_combine(ln_p_total, ln_p_ext)
                                    else:
                                        raise KeyError('Extension {} function not found.'.format(key))
                                except Exception:
                                    logger.exception('Exception for extension: {}'.format(key))
                        # Set nans to inf
                        ln_p_total[np.isnan(ln_p_total)] = -np.inf
                        # Check if  any probabilities are non zero otherwise return
                        if not np.prod(ln_p_total.shape) or not ln_p_total.max() > -np.inf:
                            if not self._return_zero:
                                return {'moment_tensors': np.matrix([]), 'ln_pdf': LnPDF(np.matrix([])), 'n': N}
                            return {'moment_tensors': self.mt, 'ln_pdf': LnPDF(-np.inf*np.ones(self.mt.shape[1])), 'n': N}
                        # Otherwise probability evaluation is complete so set return flag to True
                        _return = True
                    except Exception:
                        # Exception in cprobabilities combined PDF - print error message
                        logger.exception('Cython Combined PDFs failed')
                # Check if the combined PDF has been evaluated, otherwise evaluate individual PDFs
                if not _return:
                    # Check if generating samples (MT is bool rather than np.array)
                    if isinstance(self.mt, bool):
                        if self.dc:
                            self.mt = cprobability.random_dc(self.generate_samples)
                        else:
                            self.mt = cprobability.random_mt(self.generate_samples)
                    _return = False
                    location_samples = False
                    N = self.mt.shape[1]
                    # Initialise ln_p_total and check location_samples
                    if len(self.mt.shape) > 2:
                        ln_p_total = np.zeros((1, self.mt.shape[1], self.mt.shape[2]))
                    elif isinstance(self.a_polarity, np.ndarray) and len(self.a_polarity.shape) > 2 and self.a_polarity.shape[1] > 1:
                        location_samples = True
                        ln_p_total = np.zeros((self.a_polarity.shape[1], self.mt.shape[1]))
                    elif isinstance(self.a1_amplitude_ratio, np.ndarray) and len(self.a1_amplitude_ratio.shape) > 2 and self.a1_amplitude_ratio.shape[1] > 1:
                        location_samples = True
                        ln_p_total = np.zeros((self.a1_amplitude_ratio.shape[1], self.mt.shape[1]))
                    elif isinstance(self.a_polarity_prob, np.ndarray) and len(self.a_polarity_prob.shape) > 2 and self.a_polarity_prob.shape[1] > 1:
                        location_samples = True
                        ln_p_total = np.zeros((self.a_polarity_prob.shape[1], self.mt.shape[1]))
                    else:
                        ln_p_total = np.zeros((1, self.mt.shape[1]))
                    # Set manual polarities flag
                    manual_polarities = False
                    try:
                        # Ignore np.divide errors
                        with warnings.catch_warnings() and np.errstate(divide='ignore'):
                            warnings.simplefilter("ignore")
                            # polarity ln_pdf (will try to use cython again, and only fall back to python if necessary)
                            ln_p_total = polarity_ln_pdf(self.a_polarity, self.mt, self.error_polarity, self.incorrect_polarity_prob)
                        # Set manual_polarities (don't use auto-polarities) and return flags (have evaluated a PDF)
                        manual_polarities = True
                        _return = True
                        # Print number of non zero samples (Verbosity>=3)
                        logger.debug('Polarity non-zero samples = {}'.format(sum(ln_p_total > -np.inf)))
                    except Exception:
                        # Exception - check if data exists and print output, otherwise ignore
                        if not isinstance(self.error_polarity, np.ndarray) and self.error_polarity:
                            logger.exception('Polarity Exception')
                    # Force garbage collect to tidy memory
                    gc.collect()
                    # If manual polarity data didn't work or doesn't exist, try automated polarities
                    if not manual_polarities:
                        try:
                            # Ignore np.divide errors
                            with warnings.catch_warnings() and np.errstate(divide='ignore'):
                                warnings.simplefilter("ignore")
                                # Polarity probability ln_pdf (will try to use cython again, and only fall back to python if necessary)
                                ln_p_total = polarity_probability_ln_pdf(np.tensordot(self.a_polarity_prob, self.mt, 1), self.polarity_prob[0], self.polarity_prob[1], self.incorrect_polarity_prob)
                            _return = True
                            # Print number of non zero samples (Verbosity>=3)
                            logger.debug('Polarity probability non-zero samples = {}'.format(sum(ln_p_total > -np.inf)))
                        except Exception:
                            # Exception  - check if data exists and print output, otherwise ignore
                            if not isinstance(self.a_polarity_prob, np.ndarray) and self.a_polarity_prob:
                                logging.exception('Polarity PDF Exception')
                        # Force garbage collect to tidy memory
                        gc.collect()
                    # Check if  any probabilities are non zero otherwise return
                    if not ln_p_total.max() > -np.inf:
                        if not self._return_zero:
                            return {'moment_tensors': np.matrix([]), 'ln_pdf': LnPDF(np.matrix([])), 'n': N}
                        if location_samples and self.marginalise:
                            # Return marginalised zero PDF
                            return {'moment_tensors': self.mt, 'ln_pdf': LnPDF(-np.inf*np.ones(self.mt.shape[1])), 'n': N}
                        return {'moment_tensors': self.mt, 'ln_pdf': ln_p_total, 'n': N}
                    try:
                        # Ignore np.divide errors
                        with warnings.catch_warnings() and np.errstate(divide='ignore'):
                            warnings.simplefilter("ignore")
                            # amplitude ratio ln_pdf (will try to use cython again, and only fall back to python if necessary)
                            ln_p_amp_rat = amplitude_ratio_ln_pdf(self.amplitude_ratio, self.mt, self.a1_amplitude_ratio, self.a2_amplitude_ratio, self.percentage_error1_amplitude_ratio,
                                                                  self.percentage_error2_amplitude_ratio)
                        if _return:
                            # if Cython and polarity data exists, combine using Cython
                            if cprobability and not _CYTHON_TESTS:
                                ln_p_total = cprobability.ln_combine(ln_p_total, ln_p_amp_rat)
                            else:
                                ln_p_total = ln_p_total+ln_p_amp_rat
                        else:
                            ln_p_total = ln_p_amp_rat
                        # free ln_p_amp_rat memory
                        del ln_p_amp_rat
                        _return = True
                        # Print number of non zero samples (Verbosity>=3)
                        if _VERBOSITY >= 3:
                            logger.info('Amplitude Ratio non-zero samples:{}. Total Non zero: {}'.format(sum(ln_p_amp_rat > -np.inf), sum(ln_p_total > -np.inf)))
                    except Exception:
                        if (not isinstance(self.percentage_error1_amplitude_ratio, np.ndarray) and self.percentage_error1_amplitude_ratio) or (not isinstance(self.percentage_error2_amplitude_ratio, np.ndarray) and self.percentage_error2_amplitude_ratio):
                            logger.exception('Amplitude Ratio Exception')
                    # Force garbage collection
                    # Handle extensions
                    if len(self.extension_data):
                        extension_names, extensions = get_extensions('MTfit.data_types')
                        for key in self.extension_data.keys():
                            try:
                                if key in extension_names and 'relative' not in key:
                                    ln_p_ext = extensions[key](self.mt, **self.extension_data[key])
                                    if cprobability and not _CYTHON_TESTS:
                                        ln_p_total = cprobability.ln_combine(ln_p_total, ln_p_ext)
                                    else:
                                        ln_p_total = ln_p_total+ln_p_ext
                                else:
                                    raise KeyError('Extension {} function not found.'.format(key))
                            except Exception:
                                logging.exception('Exception for extension: {}'.format(key))
                    # Clear the memory
                    gc.collect()
                    if not _return:
                        # No data has been used, so print exceptions
                        logging.error('No data used in polarity, amplitude ratio or polarity PDF')
                    # If location PDF samples exist and sample PDF multipliers exist (increased Prob) - For Oct -tree sampling  sample density correspond to PDF values (more samples in higher prob). Grid sampling needs location multipliers set (can be done using Scat2Angle with 'grid' option in command line)
                    if location_samples and self.location_sample_multipliers and sum(self.location_sample_multipliers) != len(self.location_sample_multipliers):
                        if cprobability:
                            location_sample_multipliers = np.array(self.location_sample_multipliers)
                            location_sample_multipliers = location_sample_multipliers.astype(np.float64, copy=False)
                            ln_p_total = cprobability.ln_multipliers(ln_p_total, location_sample_multipliers)
                        else:
                            ln_p_total += np.log(np.matrix(self.location_sample_multipliers).T)  # Conversion to matrix and transpose to correct for list order to ln_p_total dimensions (0 is sample dimension)
                    # If there are location samples and the marginalise flag is set, then marginalise, trying to use Cython
                    if location_samples and self.marginalise:
                        ln_p_total = ln_marginalise(ln_p_total)
                # Delete arrays to free memory if not reusing
                if not self._reuse:
                    del self.a_polarity
                    del self.a_polarity_prob
                    del self.a1_amplitude_ratio
                    del self.a2_amplitude_ratio
                    del self.amplitude_ratio
                    del self.polarity_prob
                    del self.percentage_error1_amplitude_ratio
                    del self.percentage_error2_amplitude_ratio
                    del self.error_polarity
                    del self.incorrect_polarity_prob
                    del self.location_sample_multipliers
                    gc.collect()
                # Print end memory usage (Verbosity >=3)
                if _VERBOSITY >= 3 and memory_profiler:
                    print('end usage {}'.format(memory_profiler.memory_usage()))
                # Check results
                if _return:
                    # Flag for nan errors if _DEBUG is set
                    if np.isnan(ln_p_total).any() and _DEBUG:
                        raise ValueError('NaN result')
                    ln_p_total = LnPDF(ln_p_total)
                    # remove zeros if _return_zero flag not set.
                    if not self._return_zero:
                        self.mt = self.mt[:, ln_p_total.nonzero()]
                        ln_p_total = ln_p_total[:, ln_p_total.nonzero()]
                    return {'moment_tensors': self.mt, 'ln_pdf': LnPDF(ln_p_total), 'n': N}
                else:
                    return {'moment_tensors': np.matrix([]), 'ln_pdf': LnPDF(np.matrix([])), 'n': N}
            return False
        except Exception as e:
            # Overall error catch
            warnings.warn('Error in forward task:{}\n\nReturning no result and continuing [no action required].'.format(e), RuntimeWarning)
            traceback.print_exc()
            return {'moment_tensors': np.matrix([]), 'ln_pdf': LnPDF(np.matrix([])), 'n': 0}


class McMCForwardTask(object):
    """
    Markov chain Monte Carlo Forward modelling task

    Task which carries out Markov chain Monte Carlo forward modelling and returns results dictionary containing PDF, MTs and Number of forward modelled Samples
    Runs for whole Markov chain. Callable object.
    """

    def __init__(self, algorithm_kwargs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio,
                 percentage_error2_amplitude_ratio, a_polarity_prob, polarity_prob, incorrect_polarity_prob=0, fid=False, event_data=False, location_samples=False,
                 location_sample_multipliers=False, marginalise=True, normalise=True, convert=False, extension_data={}):
        """
        McMCForwardTask initialisation

        Args
            algorithm_kwargs: Algorithm Keyword Arguments for running Markov chain Monte Carlo inversion with.
            a_polarity: Polarity observations station-ray 6 vector.
            error_polarity: Polarity observations error.
            a1_amplitude_ratio: Amplitude ratio observations numerator station-ray 6 vector.
            a2_amplitude_ratio: Amplitude ratio observations denominator station-ray 6 vector.
            amplitude_ratio: Observed amplitude ratio.
            percentage_error1_amplitude_ratio: Amplitude ratio observations percentage error for the numerator.
            percentage_error2_amplitude_ratio: Amplitude ratio observations percentage error for the denominator.
            a_polarity_prob: Polarity PDF observations station-ray 6 vector.
            polarity_prob: Polarity PDF probability
            fid:[False] File name for output.
            event_data:[False] event_data data for output.
            location_samples:[False] Station angle scatter samples for output.
            location_sample_multipliers:[False] Station angle probability samples for output.
            marginalise:[True] Boolean flag as to whether pdf is marginalised over location/model uncertainty.
            normalise:[True] Normalise output (doesn't matter for McMC main output, but will affect probability output)
            convert:[False] Convert output data to tape, hudson and fault plane coordinates.
            extension_data:[{}] A dictionary of processed data for use by an MTfit.data_types extension.

        """
        self.algorithm_kwargs = algorithm_kwargs
        self.a_polarity = a_polarity
        self.error_polarity = error_polarity
        self.a1_amplitude_ratio = a1_amplitude_ratio
        self.a2_amplitude_ratio = a2_amplitude_ratio
        self.amplitude_ratio = amplitude_ratio
        self.percentage_error1_amplitude_ratio = percentage_error1_amplitude_ratio
        self.percentage_error2_amplitude_ratio = percentage_error2_amplitude_ratio
        self.a_polarity_prob = a_polarity_prob
        self.polarity_prob = polarity_prob
        self.incorrect_polarity_prob = incorrect_polarity_prob
        self.fid = fid
        self.event_data = event_data
        self.location_samples = location_samples
        self.location_sample_multipliers = location_sample_multipliers
        self.marginalise = marginalise
        self.normalise = normalise
        self.convert = convert
        self.extension_data = extension_data

    def __call__(self):
        """
        Runs the McMCForwardTask and returns the result as a dictionary

        Returns
            Dictionary of results as {'algorithm_output_data':algorithm_data,'event_data':event_data,'location_samples':location_samples,'location_sample_multipliers':location_sample_multipliers}
        """
        # Set algorithm
        self.algorithm = MarkovChainMonteCarloAlgorithmCreator(**self.algorithm_kwargs)
        # Initialise algorithm
        mts, end = self.algorithm.initialise()
        # Run forward tasks for MT samples
        forward = ForwardTask(mts, self.a_polarity, self.error_polarity, self.a1_amplitude_ratio, self.a2_amplitude_ratio, self.amplitude_ratio, self.percentage_error1_amplitude_ratio,
                              self.percentage_error2_amplitude_ratio, self.a_polarity_prob, self.polarity_prob, self.location_sample_multipliers, self.incorrect_polarity_prob, return_zero=True,
                              reuse=True, marginalise=self.marginalise, extension_data=self.extension_data)
        while not end:
            forward.mt = mts
            result = forward()
            mts, end = self.algorithm.iterate(result)
        # Close logger
        # Get output data and print results
        output_data, output_string = self.algorithm.output(self.normalise, self.convert, 0)
        return {'algorithm_output_data': output_data, 'event_data': self.event_data, 'location_samples': self.location_samples,
                'location_sample_multipliers': self.location_sample_multipliers}


class MultipleEventsForwardTask(object):
    """
    Multiple Events Forward modelling task

    Task which carries out forward modelling with relative amplitudes and returns dictionary containing PDF, MTs and Number of forward modelled Samples
    Callable object.
    """
    def __init__(self, mts, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio,
                 a_polarity_prob, polarity_prob, a_relative_amplitude, relative_amplitude, percentage_error_relative_amplitude, relative_amplitude_stations, location_sample_multipliers=False,
                 incorrect_polarity_prob=0, minimum_number_intersections=2, return_zero=False, reuse=False, relative=False, location_sample_size=1, marginalise_relative=False,
                 combine=True, extension_data=[]):
        """
        Initialisation of MultipleEventsForwardTask

        Used to calculate forward model parameters for multiple event samples (e.g. using relative amplitudes).

        Args
            mts: List of numpy matrix objects containing moment tensor 6 vectors for each event.
            a_polarity: List of Polarity observations station-ray 6 vector for each event.
            error_polarity: List of Polarity observations error for each event.
            a1_amplitude_ratio: List of Amplitude ratio observations numerator station-ray 6 vector for each event.
            a2_amplitude_ratio: List of Amplitude ratio observations denominator station-ray 6 vector for each event.
            amplitude_ratio: List of Observed amplitude ratios for each event.
            percentage_error1_amplitude_ratio: List of Amplitude ratio observations percentage_error for the numerator for each event.
            percentage_error2_amplitude_ratio: List of Amplitude ratio observations percentage_error for the denominator for each event.
            a_polarity_prob: List of Polarity PDF observations station-ray 6 vector for each event.
            polarity_prob: List of Polarity PDF probability for each event.
            a_relative_amplitude: List of numpy matrix objects for station-ray 6 vector for each event
            relative_amplitude: List of Observed  amplitude values for relative amplitude calculation for each event
            percentage_error_relative_amplitude: List of Amplitude ratio observations percentage_error for the relative amplitude calculation for each event
            relative_amplitude_stations: List of relative amplitude stations allowing for intersections when evaluating the relative amplitudes
            return_zero:[False] Boolean flag to return zero probability samples.
            reuse:[False] Boolean flag as to whether task is reused (mt changed and re-run...)
            relative:[False] Use relative amplitude on multiple
            location_samples:[1]  Integer number of station samples, (default is 1)
            marginalise_relative:[False] Boolean flag to marginalise the location uncertainty between absolute and relative data.
            combine:[False] Boolean flag to combine the probabilities between multiple events.
            extension_data:[] A list of dictionaries of processed data for use by an MTfit.data_types extension.
        """
        self.mts = mts
        self.a_polarity = a_polarity
        self.error_polarity = error_polarity
        self.a1_amplitude_ratio = a1_amplitude_ratio
        self.a2_amplitude_ratio = a2_amplitude_ratio
        self.amplitude_ratio = amplitude_ratio
        self.percentage_error1_amplitude_ratio = percentage_error1_amplitude_ratio
        self.percentage_error2_amplitude_ratio = percentage_error2_amplitude_ratio
        self.a_polarity_prob = a_polarity_prob
        self.polarity_prob = polarity_prob
        self.incorrect_polarity_prob = incorrect_polarity_prob
        self.a_relative_amplitude = a_relative_amplitude
        self.relative_amplitude_stations = relative_amplitude_stations
        self.relative_amplitude = relative_amplitude
        self.percentage_error_relative_amplitude = percentage_error_relative_amplitude
        self.location_sample_multipliers = location_sample_multipliers
        self._return_zero = return_zero
        self._reuse = reuse
        self._relative = relative
        self.forward_tasks = []
        self.minimum_number_intersections = minimum_number_intersections
        self.location_sample_size = location_sample_size
        self._marginalise_relative = marginalise_relative
        self._combine = combine
        # Need to implement marginalisation of scale factor for self._marginalise_relative and location_samples
        if self._relative and self._marginalise_relative and self.location_sample_size > 1:
            raise NotImplementedError('Marginalisation of location samples on relative data not implemented yet - we may well expect location uncertainty between the co-located events to be zero')
            # https://github.com/djpugh/MTfit/issues/11
            # Scale factor is location/amplitude specific
            # We would expect amplitude ratio location uncertainty to be minimal between co-located events
            # So probably need to remove any location uncertainty for the relative amplitude terms
        self.extension_data = extension_data
        if self._relative and not self._combine and self._return_zero:
            self._combine = True

    @memory_profile_test(_MEMTEST)
    def __call__(self):
        """
        Runs the MultipleEventsForwardTask and returns the result as a dictionary

        Returns
            Dictionary of results as {'mt1':mt1,'mt2':mt2,'PDF':pdf}
        """
        # If combining outputs make a single ln_pdf else make a list of them
        if self._combine:
            # Assuming that location uncertainty matches for each event (i.e if samples used, event is still co-located...)
            ln_p_total = np.matrix(np.zeros((self.location_sample_size, self.mts[0].shape[1])))
        else:
            ln_p_total = []
        # Set non-zero array
        non_zeros = np.ones((self.mts[0].shape[1]), dtype=bool)
        # Set scale factors and scale factor uncertainties
        scale_factor = np.array(np.zeros((self.location_sample_size, self.mts[0].shape[1], len(self.mts), len(self.mts))))
        scale_factor_uncertainty = np.array(np.zeros((self.location_sample_size, self.mts[0].shape[1], len(self.mts), len(self.mts))))
        N = self.mts[0].shape[1]
        # Loop over MTs
        extension_scale = {}
        for i, mt in enumerate(self.mts):
            # Set diagonal scale_factors and scale_factor_uncertainties
            scale_factor[:, :, i, i] = 1
            scale_factor_uncertainty[:, :, i, i] = 0
            # Get station coefficients
            if isinstance(self.a_polarity, list):
                a_polarity = self.a_polarity[i]
            else:
                a_polarity = self.a_polarity
            if isinstance(self.a1_amplitude_ratio, list):
                a1_amplitude_ratio = self.a1_amplitude_ratio[i]
            else:
                a1_amplitude_ratio = self.a1_amplitude_ratio
            if isinstance(self.a2_amplitude_ratio, list):
                a2_amplitude_ratio = self.a2_amplitude_ratio[i]
            else:
                a2_amplitude_ratio = self.a2_amplitude_ratio
            if isinstance(self.a_polarity_prob, list):
                a_polarity_prob = self.a_polarity_prob[i]
            else:
                a_polarity_prob = self.a_polarity_prob
            if isinstance(self.incorrect_polarity_prob, list):
                incorrect_polarity_prob = self.incorrect_polarity_prob[i]
            else:
                incorrect_polarity_prob = self.incorrect_polarity_prob
            if len(self.extension_data) and len(self.extension_data) >= i:
                extension_data = self.extension_data[i]
            else:
                extension_data = {}
            # Update mt with non-zeros if combining
            if not self._return_zero and self._combine:
                self.mts[i] = np.ascontiguousarray(self.mts[i][:, non_zeros])
                mt = self.mts[i]
            # If forward tasks are being re-used and one doesn't exist or no tasks are re-used
            if self._reuse and len(self.forward_tasks) < len(self.mts) or not self._reuse:
                forward_task = ForwardTask(mt, a_polarity, self.error_polarity[i], a1_amplitude_ratio, a2_amplitude_ratio, self.amplitude_ratio[i],
                                           self.percentage_error1_amplitude_ratio[i], self.percentage_error2_amplitude_ratio[i], a_polarity_prob,
                                           self.polarity_prob[i], self.location_sample_multipliers, incorrect_polarity_prob, return_zero=True,
                                           reuse=True, marginalise=self._marginalise_relative, extension_data=extension_data)
                if self._reuse:
                    self.forward_tasks.append(forward_task)
            elif self._reuse:
                self.forward_tasks[i].mt = mt
                forward_task = self.forward_tasks[i]
            # Run forward task
            result = forward_task()
            # Process results amd combine
            ln_pdf = result['ln_pdf']
            if isinstance(result['ln_pdf'], LnPDF):
                ln_pdf = result['ln_pdf']._ln_pdf
            # If not combining add to ln_pdf list
            if not self._combine:
                if not (ln_pdf > -np.inf).any() and self._relative and ((isinstance(self.a_relative_amplitude, list) and len(self.a_relative_amplitude)) or not isinstance(self.a_relative_amplitude, (bool, list))):
                    # in relative loop  and all zero prob
                    return {'moment_tensors': np.matrix([]), 'ln_pdf': LnPDF(np.matrix([])), 'n': N}
                ln_p_total.append(ln_pdf)
                continue
            # all zero so breaking
            if ln_pdf.shape[1] == 0:
                ln_p_total = 0*ln_p_total-np.inf
                break
            ln_p_total = ln_p_total+ln_pdf
            # return if ln_p_total is all zero
            if not (ln_p_total > -np.inf).any():
                # return zeros
                if not self._return_zero:
                    return {'moment_tensors': np.matrix([]), 'ln_pdf': LnPDF(np.matrix([])), 'n': N}
                return {'moment_tensors': self.mts, 'ln_pdf': ln_p_total, 'n': N, 'scale_factor': scale_factor}
            # Update mts
            if not self._return_zero:
                # Update ln_pdfs etc. for non-zeros
                if sum(non_zeros) != sum(np.array(np.sum(ln_p_total, 0) > -np.inf).flatten()):
                    non_zeros[non_zeros] = np.array(np.sum(ln_p_total, 0) > -np.inf).flatten()  # update non zeros
                    for j in range(i+1):
                        self.mts[j] = np.ascontiguousarray(self.mts[j][:, np.array(np.sum(ln_p_total, 0) > -np.inf). flatten()])
                    scale_factor = np.ascontiguousarray(scale_factor[:, np.array(np.sum(ln_p_total, 0) > -np.inf).flatten(), :, :])
                    scale_factor_uncertainty = np.ascontiguousarray(scale_factor_uncertainty[:, np.array(np.sum(ln_p_total, 0) > -np.inf).flatten(), :, :])
                    ln_p_total = ln_p_total[:, np.array(np.sum(ln_p_total, 0) > -np.inf).flatten()]
                    mt = self.mts[i]
            #####
            if self._relative and ((isinstance(self.a_relative_amplitude, list) and len(self.a_relative_amplitude)) or not isinstance(self.a_relative_amplitude, (bool, list))) and not self._combine:
                # Do relative amplitude calculations if not combining PDFs - Relative loop over non-zero samples, introduces some bias but not much, and should reduce computation time
                # Set non zeros
                non_zero = []
                for i, ln_p in enumerate(ln_p_total):
                    nonzero_p = LnPDF(ln_p).nonzero()
                    self.mts[i] = self.mts[i][:, nonzero_p]
                    ln_p_total[i] = LnPDF(ln_p)[:, nonzero_p]
                    non_zero.append(len(nonzero_p))
                # Get fewest number of mts and set all non-zero mts to that
                min_mts = min(non_zero)
                if min_mts == 0:
                    return {'moment_tensors': np.matrix([]), 'ln_pdf': LnPDF(np.matrix([])), 'n': N}
                for i, ln_p in enumerate(ln_p_total):
                    if i == 0:
                        ln_p_t = ln_p[:, 0: min_mts]
                    else:
                        ln_p_t += ln_p[:, 0: min_mts]
                    self.mts[i] = np.ascontiguousarray(self.mts[i][:, 0: min_mts])
                # Set up ln_p_total to run amp ratios through
                ln_p_total = LnPDF(ln_p_t)
                if not self._return_zero:
                    nonzero_p = ln_p_total.nonzero()
                    self.mts = [mts[:, nonzero_p] for mts in self.mts]
                ln_p_total = ln_p_total._ln_pdf
            if self._relative and ((isinstance(self.a_relative_amplitude, list) and len(self.a_relative_amplitude)) or not isinstance(self.a_relative_amplitude, (bool, list))):
                for j in range(i):
                    # Go backwards to ease calculation as there are  fewer events at start when most non-zero mts
                    try:
                        # Get amplitude ratio station coefficients
                        if isinstance(self.a_relative_amplitude, list) and len(self.a_relative_amplitude):
                            a1_relative_amplitude_ratio = self.a_relative_amplitude[i]
                            a2_relative_amplitude_ratio = self.a_relative_amplitude[j]
                        else:
                            a1_relative_amplitude_ratio = self.a_relative_amplitude
                            a2_relative_amplitude_ratio = self.a_relative_amplitude
                        # Intersect stations
                        intersected_data = _intersect_stations(self.relative_amplitude_stations[i], self.relative_amplitude_stations[j], a1_relative_amplitude_ratio,
                                                               a2_relative_amplitude_ratio, self.relative_amplitude[i], self.relative_amplitude[j],
                                                               self.percentage_error_relative_amplitude[i], self.percentage_error_relative_amplitude[j])
                        a1_relative_amplitude_ratio = intersected_data[0]
                        a2_relative_amplitude_ratio = intersected_data[1]
                        relative_amplitude_i = intersected_data[2]
                        relative_amplitude_j = intersected_data[3]
                        percentage_error_relative_amplitude_i = intersected_data[4]
                        percentage_error_relative_amplitude_j = intersected_data[5]
                        n_intersections = intersected_data[6]
                        # Check number of intersections
                        if n_intersections >= self.minimum_number_intersections:
                            # Evaluate relative amplitude ratio pdf
                            ln_p_amp_rat, scale, scale_uncertainty = relative_amplitude_ratio_ln_pdf(relative_amplitude_i, relative_amplitude_j, np.ascontiguousarray(mt),
                                                                                                     np.ascontiguousarray(self.mts[j]), a1_relative_amplitude_ratio,
                                                                                                     a2_relative_amplitude_ratio, percentage_error_relative_amplitude_i,
                                                                                                     percentage_error_relative_amplitude_j)
                            # Set scale_factor and scale_factor_uncertainties
                            scale_factor[:, :, i, j] = scale
                            scale_factor[:, :, j, i] = scale
                            scale_factor_uncertainty[:, :, j, i] = scale_uncertainty
                            scale_factor_uncertainty[:, :, i, j] = scale_uncertainty
                            # Handle location samples PDFs
                            if self.location_sample_multipliers and sum(self.location_sample_multipliers) != len(self.location_sample_multipliers):
                                if cprobability:
                                    location_sample_multipliers = np.array(self.location_sample_multipliers)
                                    location_sample_multipliers = location_sample_multipliers.astype(np.float64, copy=False)
                                    ln_p_amp_rat = cprobability.ln_multipliers(ln_p_amp_rat, location_sample_multipliers)
                                else:
                                    ln_p_amp_rat += np.log(np.matrix(self.location_sample_multipliers).T)
                            # Marginalise if marginalise_relative flag is True
                            if self.location_sample_size > 1 and self._marginalise_relative:
                                ln_p_amp_rat = ln_marginalise(ln_p_amp_rat, _cython=cprobability is not False)
                            # Combine ln_p_total and amp rat
                            if cprobability and not _CYTHON_TESTS:
                                ln_p_total = cprobability.ln_combine(np.ascontiguousarray(ln_p_total), ln_p_amp_rat)
                            else:
                                ln_p_total = ln_p_total+ln_p_amp_rat
                            # return zeros
                            if not (ln_p_total > -np.inf).any():
                                if not self._return_zero:
                                    return {'moment_tensors': np.matrix([]), 'ln_pdf': LnPDF(np.matrix([])), 'n': N}
                                return {'moment_tensors': self.mts, 'ln_pdf': ln_p_total, 'n': N}
                            # Update mts for non-zeros
                            if not self._return_zero:
                                non_zero_indices = np.array(np.sum(ln_p_total, 0) > -np.inf).flatten()
                                if sum(non_zeros) != sum(non_zero_indices):
                                    non_zeros[non_zeros] = non_zero_indices
                                    for j in range(i+1):
                                        self.mts[j] = np.ascontiguousarray(self.mts[j][:, non_zero_indices])
                                    ln_p_total = np.ascontiguousarray(ln_p_total[:, non_zero_indices])
                                    scale_factor = np.ascontiguousarray(scale_factor[:, non_zero_indices, :, :])
                                    scale_factor_uncertainty = np.ascontiguousarray(scale_factor_uncertainty[:, non_zero_indices, :, :])
                    except Exception as e:
                        print('Relative Amplitude Ratio Exception {}'.format(e))
                        # Exception in cprobabilities combined PDF - print error message
                        if _VERBOSITY >= 4:
                            traceback.print_exc()
            # Handle relative inversion extensions
            if self._relative and len(self.extension_data):
                extension_names, extensions = get_extensions('MTfit.data_types')
                extension_scale = {}
                extension_scale_uncertainty = {}
                for key in self.extension_data[i].keys():
                    extension_scale[key] = np.array(np.zeros((self.location_sample_size, self.mts[0].shape[1], len(self.mts), len(self.mts))))
                    extension_scale_uncertainty[key] = np.array(np.zeros((self.location_sample_size, self.mts[0].shape[1], len(self.mts), len(self.mts))))
                    extension_scale[key][:, :, i, i] = 1
                    extension_scale_uncertainty[key][:, :, i, i] = 0
                for j in range(i):
                    for key in self.extension_data[i].keys():
                        if key in self.extension_data[j].keys():
                            try:
                                if key in extension_names and 'relative' in key:
                                    extension_data_1, extension_data_2, n_intersections = _intersect_stations_extension_data(self.extension_data[i][key].copy(),
                                                                                                                             self.extension_data[j][key].copy())
                                    if not n_intersections:
                                        continue
                                    ln_p_ext, scale, scale_uncertainty = extensions[key](self.mt, extension_data_1, extension_data_2)
                                    extension_scale[key][:, :, i, j] = scale
                                    extension_scale[key][:, :, j, i] = scale
                                    extension_scale_uncertainty[key][:, :, j, i] = scale_uncertainty
                                    extension_scale_uncertainty[key][:, :, i, j] = scale_uncertainty
                                    if self.location_sample_multipliers and sum(self.location_sample_multipliers) != len(self.location_sample_multipliers):
                                        if cprobability:
                                            location_sample_multipliers = np.array(self.location_sample_multipliers)
                                            location_sample_multipliers = location_sample_multipliers.astype(np.float64, copy=False)
                                            ln_p_extensions = cprobability.ln_multipliers(ln_p_ext, location_sample_multipliers)
                                        else:
                                            ln_p_extensions += np.log(np.matrix(self.location_sample_multipliers).T)
                                    # Marginalise if marginalise_relative flag is True
                                    if self.location_sample_size > 1 and self._marginalise_relative:
                                        ln_p_extensions = ln_marginalise(ln_p_extensions, _cython=cprobability is not False)
                                    if cprobability and not _CYTHON_TESTS:
                                        ln_p_total = cprobability.ln_combine(ln_p_total, ln_p_extensions)
                                    else:
                                        ln_p_total = ln_p_total+ln_p_extensions

                                    if not (ln_p_total > -np.inf).any():
                                        if not self._return_zero:
                                            return {'moment_tensors': np.matrix([]), 'ln_pdf': LnPDF(np.matrix([])), 'n': N}
                                        return {'moment_tensors': self.mts, 'ln_pdf': ln_p_total, 'n': N}
                                    # Update mts for non-zeros
                                    if not self._return_zero:
                                        non_zero_indices = np.array(np.sum(ln_p_total, 0) > -np.inf).flatten()
                                        if sum(non_zeros) != sum(non_zero_indices):
                                            non_zeros[non_zeros] = non_zero_indices
                                            for j in range(i+1):
                                                self.mts[j] = np.ascontiguousarray(self.mts[j][:, non_zero_indices])
                                            ln_p_total = np.ascontiguousarray(ln_p_total[:, non_zero_indices])
                                            for key in extension_scale.keys():
                                                extension_scale[key] = np.ascontiguousarray(extension_scale[key][:, non_zero_indices, :, :])
                                                extension_scale_uncertainty[key] = np.ascontiguousarray(extension_scale_uncertainty[key][:, non_zero_indices, :, :])
                                else:
                                    raise KeyError('Extension {} function not found.'.format(key))
                            except Exception as e:
                                print('Exception for relative extension: {} - {}'.format(key, e))

        # Delete attributes if not reusing and force garbage collection
        if not self._reuse:
            del self.a_polarity
            del self.a_polarity_prob
            del self.a1_amplitude_ratio
            del self.a2_amplitude_ratio
            del self.amplitude_ratio
            del self.polarity_prob
            del self.percentage_error1_amplitude_ratio
            del self.percentage_error2_amplitude_ratio
            del self.error_polarity
            del self.incorrect_polarity_prob
            del self.extension_data
            gc.collect()
        # Take non-zeros here?
        if np.isnan(ln_p_total).any() and _DEBUG:
            raise ValueError('NaN result')
        # Set up results dict
        if self.location_sample_size > 1 and not self._marginalise_relative:
            # Relative not marginalised already
            ln_scale = 0
            if -ln_p_total.max() > 0 and ln_p_total.max() > -np.inf:
                ln_scale = -ln_p_total.max()
            scale_factor = np.array([{'mu': scale_factor[:, i], 'ln_p': ln_p_total[:, i], 'sigma': scale_factor_uncertainty[:, i]} for i in range(scale_factor.shape[1])])
            try:
                for key in extension_scale.keys():
                    extension_scale[key] = np.array([{'mu': extension_scale[key][:, i], 'ln_p': ln_p_total[:, i], 'sigma': extension_scale_uncertainty[key][:, i]} for i in range(extension_scale[key].shape[1])])
            except Exception:
                extension_scale = {}
            p_total = np.exp(ln_p_total+ln_scale)
            # Marginalise
            p_total = np.matrix(np.sum(p_total, 0))
            ln_p_total = np.log(p_total)-ln_scale
        elif self._relative:  # marginalise_relative is true
            if scale_factor.shape[0] == 1:
                scale_factor = scale_factor.squeeze(0)
                scale_factor_uncertainty = scale_factor_uncertainty.squeeze(0)
            elif scale_factor.shape[1] == 1:
                scale_factor = scale_factor.squeeze(1)
                scale_factor_uncertainty = scale_factor_uncertainty.squeeze(1)
            scale_factor = np.array([{'mu': scale_factor[i, :, :], 'sigma': scale_factor_uncertainty[i, :, :]} for i in range(scale_factor.shape[0])])
            try:
                for key in extension_scale.keys():
                    extension_scale[key] = extension_scale[key].squeeze(0)
                    extension_scale_uncertainty[key] = extension_scale_uncertainty[key].squeeze(0)
                    extension_scale[key] = np.array([{'mu': extension_scale[key][i, :, :], 'ln_p': ln_p_total[:, i], 'sigma': extension_scale_uncertainty[key][i, :, :]} for i in range(extension_scale[key].shape[0])])
            except Exception:
                extension_scale = {}
        else:
            scale_factor = False
        ln_p_total = LnPDF(ln_p_total)
        if not self._return_zero:
            nonzero_p = ln_p_total.nonzero()
            self.mts = [mts[:, nonzero_p] for mts in self.mts]
            ln_p_total = ln_p_total[:, nonzero_p]
            if not isinstance(scale_factor, bool):
                scale_factor = scale_factor[nonzero_p]
        return {'moment_tensors': self.mts, 'ln_pdf': ln_p_total, 'n': N, 'scale_factor': scale_factor, 'extensions_scale_factor': extension_scale}


class MultipleEventsMcMCForwardTask(McMCForwardTask):
    """
    Multiple Events McMC Forward modelling task

    Task which carries out Markov chain Monte Carlo forward modelling which carries out forward modelling for multiple events and returns dictionary containing PDF, MTs and Number of forward modelled Samples
    Callable object.
    """
    # Carry out McMC on multiple events using relative amplitudes to link them together.

    def __init__(self, algorithm_kwargs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio,
                 a_polarity_prob, polarity_prob, a_relative_amplitude, relative_amplitude, percentage_error_relative_amplitude, relative_amplitude_stations, incorrect_polarity_prob=0, relative=True,
                 minimum_number_intersections=2, fid=False, event_data=False, location_samples=False, location_sample_multipliers=False, marginalise_relative=True, normalise=True, convert=False, extension_data=[]):
        """
        Initialisation of MultipleEventsMcMCForwardTask

        Args
            algorithm_kwargs: Algorithm Keyword Arguments for running Markov chain Monte Carlo inversion with.
            a_polarity: List or numpy matrix of polarity observations station-ray 6 vector.
            error_polarity: List of numpy matrices of polarity observations error.
            a1_amplitude_ratio: List or numpy matrix of amplitude ratio observations numerator station-ray 6 vector.
            a2_amplitude_ratio: List or numpy matrix of amplitude ratio observations denominator station-ray 6 vector.
            amplitude_ratio:  List of numpy matrices of observed amplitude ratio.
            percentage_error1_amplitude_ratio: List of numpy matrices amplitude ratio observations percentage error for the numerator.
            percentage_error2_amplitude_ratio: List of numpy matrices amplitude ratio observations percentage error for the denominator.
            a_polarity_prob: List or numpy matrix of polarity PDF observations station-ray 6 vector.
            polarity_prob: List of numpy matrices of polarity PDF probability.
            a_relative_amplitude: List of numpy matrix objects for station-ray 6 vector for each event
            relative_amplitude: List of Observed  amplitude values for relative amplitude calculation for each event
            percentage_error_relative_amplitude: List of Amplitude ratio observations percentage_error for the relative amplitude calculation for each event
            relative_amplitude_stations: List of relative amplitude stations allowing for intersections when evaluating the relative amplitudes
            relative:[True] Carry out relative amplitude step in inversion.
            minimum_number_intersections:[2] Integer - minimum number of station intersections required to include the relative amplitude information.
            fid:[False] File name for output.
            event_data:[False] event_data data for output.
            location_samples:[False] Station angle scatter samples for output.
            location_sample_multipliers:[False] Station angle probability samples for output.
            marginalise_relative:[False] Boolean flag as to whether pdf is marginalised over location/model uncertainty - IGNORED.
            normalise:[True] Normalise output (doesn't matter for McMC main output, but will affect probability output)
            convert:[False] Convert output data to tape, hudson and fault plane coordinates.
            extension_data:[] A list of dictionaries of processed data for use by an MTfit.data_types extension.

        """
        self.algorithm_kwargs = algorithm_kwargs
        self.a_polarity = a_polarity
        self.error_polarity = error_polarity
        self.a1_amplitude_ratio = a1_amplitude_ratio
        self.a2_amplitude_ratio = a2_amplitude_ratio
        self.amplitude_ratio = amplitude_ratio
        self.percentage_error1_amplitude_ratio = percentage_error1_amplitude_ratio
        self.percentage_error2_amplitude_ratio = percentage_error2_amplitude_ratio
        self.a_polarity_prob = a_polarity_prob
        self.polarity_prob = polarity_prob
        self.incorrect_polarity_prob = incorrect_polarity_prob
        self.a_relative_amplitude = a_relative_amplitude
        self.relative_amplitude = relative_amplitude
        self.percentage_error_relative_amplitude = percentage_error_relative_amplitude
        self.relative_amplitude_stations = relative_amplitude_stations
        self._relative = relative
        self.fid = fid
        self.event_data = event_data
        self.location_samples = location_samples
        self.location_sample_multipliers = location_sample_multipliers
        self.minimum_number_intersections = minimum_number_intersections
        self._marginalise_relative = True  # Setting this to False causes issues at the moment marginalise_relative
        # TODO fix thie marginalise relative handling in the McMC
        self.normalise = normalise
        self.convert = convert
        self.extension_data = extension_data
        # Set logger

    def __call__(self):
        """
        Runs the multiple events McMC forward task

        Runs the MultipleEventsMcMCForwardTask and returns the result as a dictionary

        Returns
            Dictionary of results as {'algorithm_output_data':algorithm_data,'event_data':event_data,'location_samples':location_samples,'location_sample_multipliers':location_sample_multipliers}

        """
        location_sample_size = 1
        if self.location_samples:
            location_sample_size = len(self.location_samples)
        # Set algorithm
        self.algorithm = MarkovChainMonteCarloAlgorithmCreator(**self.algorithm_kwargs)
        # Initialise algorithm
        mts, end = self.algorithm.initialise()
        iteration = 1
        # Initialise algorithms separately (no relative data)
        while self.algorithm._initialising and self.algorithm._initialiser:
            results = []
            # Sort station angle coefficients (list or array)
            for i, mt in enumerate(mts):
                if isinstance(self.a_polarity, list):
                    a_polarity = self.a_polarity[i]
                else:
                    a_polarity = self.a_polarity
                if isinstance(self.error_polarity, list):
                    error_polarity = self.error_polarity[i]
                else:
                    error_polarity = self.error_polarity
                if isinstance(self.a1_amplitude_ratio, list):
                    a1_amplitude_ratio = self.a1_amplitude_ratio[i]
                else:
                    a1_amplitude_ratio = self.a1_amplitude_ratio
                if isinstance(self.a2_amplitude_ratio, list):
                    a2_amplitude_ratio = self.a2_amplitude_ratio[i]
                else:
                    a2_amplitude_ratio = self.a2_amplitude_ratio
                if isinstance(self.a_polarity_prob, list):
                    a_polarity_prob = self.a_polarity_prob[i]
                else:
                    a_polarity_prob = self.a_polarity_prob
                if isinstance(self.incorrect_polarity_prob, list):
                    incorrect_polarity_prob = self.incorrect_polarity_prob[i]
                else:
                    incorrect_polarity_prob = self.incorrect_polarity_prob
                if len(self.extension_data) and len(self.extension_data) >= i:
                    extension_data = self.extension_data[i]
                else:
                    extension_data = {}
                if isinstance(self.polarity_prob, list) and len(self.polarity_prob):
                    polarity_prob = self.polarity_prob[i]
                else:
                    polarity_prob = None
                if isinstance(self.amplitude_ratio, list):
                    amplitude_ratio = self.amplitude_ratio[i]
                else:
                    amplitude_ratio = self.amplitude_ratio
                if isinstance(self.percentage_error1_amplitude_ratio, list):
                    percentage_error1_amplitude_ratio = self.percentage_error1_amplitude_ratio[i]
                else:
                    percentage_error1_amplitude_ratio = self.percentage_error1_amplitude_ratio
                if isinstance(self.percentage_error2_amplitude_ratio, list):
                    percentage_error2_amplitude_ratio = self.percentage_error2_amplitude_ratio[i]
                else:
                    percentage_error2_amplitude_ratio = self.percentage_error2_amplitude_ratio
                # Run forward task
                forward_task = ForwardTask(mt, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio,
                                           amplitude_ratio, percentage_error1_amplitude_ratio,
                                           percentage_error2_amplitude_ratio, a_polarity_prob,
                                           polarity_prob, self.location_sample_multipliers, incorrect_polarity_prob,
                                           return_zero=False, reuse=False, extension_data=extension_data)
                results.append(forward_task())
            mts, end = self.algorithm.iterate(results)
            iteration += 1
        # Run Relative amplitude data multiple events
        multiple_events_forward_task = MultipleEventsForwardTask(mts, self.a_polarity, self.error_polarity, self.a1_amplitude_ratio,
                                                                 self.a2_amplitude_ratio, self.amplitude_ratio, self.percentage_error1_amplitude_ratio,
                                                                 self.percentage_error2_amplitude_ratio, self.a_polarity_prob, self.polarity_prob,
                                                                 self.a_relative_amplitude, self.relative_amplitude,
                                                                 self.percentage_error_relative_amplitude, self.relative_amplitude_stations,
                                                                 self.location_sample_multipliers, self.incorrect_polarity_prob,
                                                                 self.minimum_number_intersections, return_zero=True, reuse=True,
                                                                 relative=self._relative, location_sample_size=location_sample_size,
                                                                 marginalise_relative=self._marginalise_relative,
                                                                 extension_data=self.extension_data)
        while not end:
            multiple_events_forward_task.mts = mts
            result = multiple_events_forward_task()
            mts, end = self.algorithm.iterate(result)
        # Close logger
        # Output data
        output_data, output_string = self.algorithm.output(self.normalise, self.convert, 0)
        # Print output string
        print(output_string)
        return {'algorithm_output_data': output_data, 'event_data': self.event_data, 'location_samples': self.location_samples,
                'location_sample_multipliers': self.location_sample_multipliers}


class CombineMpiOutputTask(object):
    """
    Task for combining MPI output

    Combines and  saves mpi (hyp) output (including scale factors from relative inversion)

    Initialisation

        Args
            uid: UID for combining.
            format: ['matlab'] Output format
            results_format: ['full_pdf'] Set result format

    """
    def __init__(self, uid, format='matlab', results_format='full_pdf', parallel=False, binary_file_version=2):
        """
        Initialisation of CombineMpiOutputTask

        Args
            uid: UID for combining.
            format: Output format
        """
        self.uid = uid
        self.format = format
        self.results_format = results_format
        self.parallel = parallel
        self.binary_file_version = binary_file_version

    def __call__(self):
        """
        Runs the mpi output combining task

        Runs the CombineMpiOutputTask and returns a result code.

        Returns
            resultCode: 10 if successful.

        """
        print('\n\tCombining {}'.format(self.uid))
        # Get hyp, mt and sf files
        hypfiles = glob.glob(self.uid+'.*.hyp')
        fids = glob.glob(self.uid+'.*.mt')
        scale_factor_fids = glob.glob(self.uid+'.*.sf')
        inv_fid = glob.glob(self.uid.replace('MT', '').replace('DC', '')+'.inv')
        # Try to use the pkl file as input
        try:
            data = pickle.load(open(self.uid+'.0.pkl'))
            # Check fids exist (haven't moved the files)
            if all([os.path.exists(fid) for fid in data['fids']]):
                fids = data['fids']
            data = data['event_data']
            if len(inv_fid):
                inversion = Inversion(data_file=inv_fid[0], parallel=self.parallel)
            else:
                inversion = Inversion(data=data, parallel=self.parallel)
        # Otherwise use the datafile or hypfile
        except Exception:
            data = False
            if len(inv_fid):
                inversion = Inversion(data_file=inv_fid[0], parallel=self.parallel)
            elif len(hypfiles):
                inversion = Inversion(data_file=hypfiles[0], parallel=self.parallel)
        # Combine MPI output
        try:
            inversion._combine_mpi_output(inversion.data[0], fids, scale_factor_fids, self.uid+'.mat', format=self.format, results_format=self.results_format, parallel=False, mpi=False, binary_file_version=self.binary_file_version)
        except Exception:
            traceback.print_exc()
        return 10


#
# Inversion object
#


class Inversion(object):
    """
    Main Inversion object

    Runs the MT inversion, follwing a parameterisation set on initialisation. Parameters can be set for the algorithm to use in the kwargs.

    Algorithm options are:

        * Time - Monte Carlo random sampling - runs until time limit reached.
        * Iterate - Monte Carlo random sampling - runs until sample limit reached.
        * McMC - Markov chain Monte Carlo sampling.
        * TransDMcMC - Markov chain Monte Carlo sampling.

    These are discussed in more detail in the MTfit.algorithms documentation.

    The inversion is run by calling the forward function (MTfit.inversion.Inversion.forward)

    **Data Format**

    The inversion expects a python dictionary of the data in the format::

        >>data={'PPolarity':{'Measured':numpy.matrix([[-1],[-1]...]),
                             'Error':numpy.matrix([[0.01],[0.02],...]),
                             'Stations':{'Name':[Station1,Station2,...]
                                         'Azimuth':numpy.matrix([[248.0],[122.3]...]),
                                         'TakeOffAngle':numpy.matrix([[24.5],[2.8]...])
                                        }
                         },
                'PSHAmplitudeRatio':{...},
                ...
                'UID':'Event1'
                }

    The initial key arguments correspond to the data types that can be used in the inversion. The inversion uses polarity observations and amplitude ratios, and can also use relative amplitudes.
    This means that the useable data types are:

        **Polarity**

            * PPolarity
            * SHPolarity
            * SVPolarity

            Polarity observations made manually or automatically. The corresponding data dictionary for polarities needs the following keys:

                * *Measured*: numpy matrix of polarity observations
                * *Error*: numpy matrix of fractional uncertainty in the polarity observations.
                * *Stations*: dictionary of station information with keys:

                    * *Name*: list of station names
                    * *Azimuth*: numpy matrix of azimuth values in degrees
                    * *TakeOffAngle*: numpy matrix of take off angle values in degrees - 0 down (NED coordinate system).

            As with all of these dictionaries, the indexes of the observations and errors must correspond to the stations, i.e. `data['Measured'][0,0]` -> from `data['Stations']['Name'][0]` with `error data['Error'][0,:]` etc.

            If polarity probabilities are being used, the keys are:

                * PPolarityProbability
                * SHPolarityProbability
                * SVPolarityProbability

            With a similar structure to the Polarity data, except the measured matrix has an additional dimension, i.e. `data['Measured'][0,0]` is the positive polarity probability and `data['Measured'][0,1]` is the negative polarity probability.

        **Amplitude ratios**

            * P/SHRMSAmplitudeRatio
            * P/SVRMSAmplitudeRatio
            * SH/SVRMSAmplitudeRatio
            * P/SHQRMSAmplitudeRatio
            * P/SVQRMSAmplitudeRatio
            * SH/SVQRMSAmplitudeRatio
            * P/SHAmplitudeRatio
            * P/SVAmplitudeRatio
            * SH/SVAmplitudeRatio
            * P/SHQAmplitudeRatio
            * P/SVQAmplitudeRatio
            * SH/SVQAmplitudeRatio

            Amplitude ratio observations made manually or automatically. The Q is not necessary but is useful to label the amplitudes with a Q correction.
            The corresponding data dictionary for amplitude ratios needs the following keys:

                * *Measured*: numpy matrix of corrected numerator and denominator amplitude ratio observations, needs to have two columns, one for the numerator and one for the denominator.
                * *Error*: numpy matrix of uncertainty (standard deviation) in the amplitude ratio observations, needs to have two columns, one for the numerator and one for the denominator.
                * *Stations*: dictionary of station information with keys:

                    * *Name*: list of station names
                    * *Azimuth*: numpy matrix of azimuth values in degrees
                    * *TakeOffAngle*: numpy matrix of take off angle values in degrees - 0 down (NED coordinate system).

            As with all of these dictionaries, the indexes of the observations and errors must correspond to the stations, i.e. `data['Measured'][0,0]` -> from `data['Stations']['Name'][0]` with `error data['Error'][0,:]` etc.

        **Relative Amplitude Ratios**

            * PAmplitude
            * SHAmplitude
            * SVAmplitude
            * PQAmplitude
            * SHQAmplitude
            * SVQAmplitude
            * PRMSAmplitude
            * SHRMSAmplitude
            * SVRMSAmplitude
            * PQRMSAmplitude
            * SHQRMSAmplitude
            * SVQRMSAmplitude

            Relative Amplitude ratios use amplitude observations for different events made manually or automatically. The Q is not necessary but is useful to label the amplitudes with a Q correction.
            The corresponding data dictionary for amplitude ratios needs the following keys:

                * *Measured*: numpy matrix of amplitude observations for the event.
                * *Error*: numpy matrix of uncertainty (standard deviation) in the amplitude observations.
                * *Stations*: dictionary of station information with keys:

                    * *Name*: list of station names
                    * *Azimuth*: numpy matrix of azimuth values in degrees
                    * *TakeOffAngle*: numpy matrix of take off angle values in degrees - 0 down (NED coordinate system).

            As with all of these dictionaries, the indexes of the observations and errors must correspond to the stations, i.e. `data['Measured'][0,0]` -> from `data['Stations']['Name'][0]` with `error data['Error'][0,:]` etc.



    **Angle Scatter Format**
    The angle scatter files can be generated using a utility Scat2Angle based on the NonLinLoc angle code. The angle scatter file is a text file with samples separated by blank lines.
    The expected format is for samples from the location PDF that have been converted into take off and azimuth angles (in degrees) for the stations, along with a probability value. It is important that the samples are drawn
    from the location PDF as a Monte Carlo based integration approach is used for marginalising over the uncertainty.

    It is possible to use XYZ2Angle to samples drawn from a location PDF using the NonLinLoc angle approach.

    Expected format is::

        Probability
        StationName Azimuth TakeOffAngle
        StationName Azimuth TakeOffAngle

        Probability
        .
        .
        .

    With TakeOffAngle as 0 down (NED coordinate system).
    e.g.::
        504.7
        S0271   231.1   154.7
        S0649   42.9    109.7
        S0484   21.2    145.4
        S0263   256.4   122.7
        S0142   197.4   137.6
        S0244   229.7   148.1
        S0415   75.6    122.8
        S0065   187.5   126.1
        S0362   85.3    128.2
        S0450   307.5   137.7
        S0534   355.8   138.2
        S0641   14.7    120.2
        S0155   123.5   117
        S0162   231.8   127.5
        S0650   45.9    108.2
        S0195   193.8   147.3
        S0517   53.7    124.2
        S0004   218.4   109.8
        S0588   12.9    128.6
        S0377   325.5   165.3
        S0618   29.4    120.5
        S0347   278.9   149.5
        S0529   326.1   131.7
        S0083   223.7   118.2
        S0595   42.6    117.8
        S0236   253.6   118.6

        502.7
        S0271   233.1   152.7
        S0649   45.9    101.7
        S0484   25.2    141.4
        S0263   258.4   120.7
        .
        .
        .

    """

    @memory_profile_test(_MEMTEST)
    def __init__(self, data={}, data_file=False, location_pdf_file_path=False, algorithm='iterate', parallel=True, n=0, phy_mem=8, dc=False, **kwargs):
        """
        Initialisation of inversion object

        Args:

            data (dict/list): Dictionary or list of dictionaries containing data for inversion. Can be ignored if a data_file is passed as an argument (for data format, see below).
            data_file (str): Path or list of file paths containing (binary) pickled data dictionaries.
            location_pdf_file_path (str): Path or list of file paths to angle scatter files (for file format, see above - other format extensions can be added using setuptools entry points).
            algorithm (str):['iterate'] algorithm selector
            parallel (bool):[True] Run the inversion in parallel using multiprocessing
            n (int):[0] Number of workers, default is to use as many as returned by multiprocessing.cpu_count
            phy_mem (int):[8] Estimated physical memory to use (used for determining array sizes, it is likely that more memory will be used, and if so no errors are forced).
                 *On python versions <2.7.4 there is a bug (http://bugs.python.org/issue13555) with pickle that limits the total number of samples when running in parallel, so large (>20GB) phy_mem allocations per process are ignored.*
            dc (bool):[False] Boolean flag as to run inversion constrained to double-couple or allowed to explore the full moment tensor space.

        Keyword Arguments:

            number_stations (int):[0] Used for estimating sample sizes in the Monte Carlo random sampling algorithms (Time,Iterate) if set.
            number_location_samples (int):[0] Used for estimating sample sizes in the Monte Carlo random sampling algorithms (Time,Iterate) if set.
            path (str): File path for output. Default is current working dir (interactive) or PBS workdir.
            fid (str): File name root for output, default is to use MTfitOutput or the event UID if set in the data dictionary.
            inversion_options (list): List of data types to be used in the iversion, if not set, the inversion uses all the data types in the data dictionary, irrespective of independence.
            diagnostic_output (bool): [False] Boolean flag to output diagnostic information in MATLAB save file.
            marginalise_relative (bool): [False] Boolean flag to marginalise over location/model uncertainty during relative amplitude inversion.
            mpi (bool): [False] Boolean flag to run using MPI from mpi4py (useful on cluster).
            multiple_events (bool): [False] Boolean flag to run all the events in the inversion in one single joint inversion.
            relative_amplitude (bool): [False] Boolean flag to run multiple_events including relative amplitude inversion.
            output_format (str): ['matlab'] str format style.
            minimum_number_intersections (int): [2] Integer minimum number of station intersections between events in relative amplitude inversion.
            quality_check (bool): [False] Boolean flag/Float value for maximum non-zero percentage check (stops inversion if event quality is poor)
            recover (bool): [False] Tries to recover pre-existing inversions, and does not re-run if an output file exists and is readable.
            file_sample (bool): [False] Saves samples to a mat file (allowing for easier recovery and reduced memory requirements but can slow the inversion down as requires disk writing every non-zero iteration. Only works well for recovery with random sampling methods)
            results_format (str): ['full_pdf'] Data format for the output
            marginalise_relative (bool): [False] Boolean flag to marginalise the location uncertainty between absolute and relative data.
            file_sample (bool): [False] save samples to file rather than keeping in memory (not necessary in normal use).
            normalise (bool): [True] normalise output Probability (doesn't affect ln_pdf output).
            convert (bool):  Convert output moment tensors to Tape parameters, Hudson u&v coordinates and strike-dip-rake triples.
            discard (bool): [False] Probability cut-off for discarding samples Discarding samples - samples less than 1/(discard*n_samples) of the maximum likelihood value are discarded as negligeable. False means no samples are discarded.
            c_generate (bool): [False] Generate samples in the probability calculation when using Cython.
            generate_cutoff (int): Set number of samples to cut-off at when using c_generate (Default is the value of max_samples)
            relative_loop (bool): [False] Loop over non-zero samples when using relative amplitudes.
            bin_angle_coefficient_samples (int): [0] Bin size in degrees when binning angle coefficients (All station angle differences must be within this range for samples to fall in the same bin)
            no_station_distribution (bool): [True] Boolean flag to output station distribution or not.
            max_samples (int): [6000000] Max number of samples when using the iterate algorithm.
            max_time (int): [600] Max time when using the time algorithm.
            verbosity (int): [0] Set verbosity level (0-4) high numbers mean more logging output and verbosity==4 means the debugger will be called on errors.
            debug (bool): [False] Sets debug on or off (True means verbosity set to 4).

        Other kwargs are passed to the algorithm - see MTfit.algorithms documentation for help on those.
        """
        data_file_path = data_file
        self.dc = dc
        self._marginalise_relative = kwargs.get('marginalise_relative', False)
        self.file_sample = kwargs.get('file_sample', False)
        # If data and data_file doesn't exist, try to read sys.argv
        if not len(data) and not data_file:
            try:
                data_file = sys.argv[1]
            except Exception:
                pass
        # Set MPI parameters
        self._MPI = kwargs.get('mpi', False)
        self.comm = False
        if self._MPI:
            try:
                from mpi4py import MPI
                self.comm = MPI.COMM_WORLD
                self._print('Running MTfit using MPI')
            except Exception:
                self._MPI = False
        # Set DC parameters
        if self.dc:
            self._print('\nDouble Couple Constrained Inversion\n')
        else:
            self._print('\nFull Moment Tensor Inversion\n')
        # Parse data_files to data_dict
        if data_file:
            if isinstance(data_file, list):
                self._print('\n*********\ndata files: '+','.join(data_file))
                data = []
                for filename in data_file:
                    file_data = self._load(filename)
                    if isinstance(file_data, list):
                        data.extend(file_data)
                    else:
                        data.append(file_data)
            else:
                self._print('\n*********\ndata file: '+data_file)
                data = self._load(data_file)
        # Raise error if no data
        if not data or not len(data):
            raise ValueError('Inversion requires an input data_file or data dictionary')
        # Set data parameters
        if isinstance(data, list):
            number_events = len(data)
            self.data = data
        else:
            self.data = [data.copy()]
            number_events = 1
        # Check multiple events flag
        self.multiple_events = kwargs.get('multiple_events', False)
        # Check number of events vs. multiple events
        if self.multiple_events and number_events == 1:
            self.multiple_events = False
            kwargs['multiple_events'] = False
        if not self.multiple_events:
            number_events = 1
        # Set more parameters from kwargs
        self.number_events = number_events
        kwargs['number_events'] = number_events
        self._relative = kwargs.get('relative_amplitude', False)
        self.normalise = kwargs.get('normalise', True)
        self.convert = kwargs.get('convert', False)
        self.discard = kwargs.get('discard', False)
        self.c_generate = kwargs.get('c_generate', False)
        self._relative_loop = kwargs.get('relative_loop', False)
        self.bin_angle_coefficient_samples = kwargs.get('bin_angle_coefficient_samples', 0)
        number_stations = 40
        self.location_pdf_files = False
        self.number_location_samples = 0
        # Get Debug and verbositu arguments
        global _DEBUG
        global _VERBOSITY
        _VERBOSITY = kwargs.get('verbosity', 0)
        _DEBUG = kwargs.get('debug', False)
        if _DEBUG:
            _VERBOSITY = 4
        try:
            kwargs.pop('debug')
        except Exception:
            pass
        # Handle location PDFs
        if location_pdf_file_path:
            # Get number of location samples
            number_location_samples = int(kwargs.get('number_location_samples', 0))
            self.number_location_samples = number_location_samples
            # Get number of stations
            number_stations = int(kwargs.get('number_stations', 0))
            if isinstance(location_pdf_file_path, list):
                self.location_pdf_files = location_pdf_file_path
            else:
                # Search in location_pdf_file_path
                self.location_pdf_files = glob.glob(location_pdf_file_path)
            # Get number of stations and number of location samples
            if not number_stations or not number_location_samples:
                location_samples, location_sample_multipliers = self._read_location([u for u in self.location_pdf_files if len(u)][0])
                if not number_location_samples:
                    number_location_samples = len(location_samples)
                number_stations = len(location_samples[0])
                # Clear results and garbage collect
                del location_samples
                del location_sample_multipliers
                gc.collect()
        else:
            # No location PDFs
            number_location_samples = 1
        self.number_stations = number_stations
        self.number_events = number_events
        # Print out location PDF files
        if self.location_pdf_files:
            self._print('Angle Scatter files: '+','.join(self.location_pdf_files))
        # Set run parameters
        self._print('\n*********\n\nRun parameters\n')
        self._algorithm_name = algorithm
        self.output_format = kwargs.get('output_format', 'matlab')
        self.results_format = kwargs.get('results_format', 'full_pdf')
        self._output_station_distribution = not kwargs.get('no_station_distribution', False)
        self.max_samples = kwargs.get('max_samples', 6000000)
        self.max_time = kwargs.get('max_time', 600)
        # Set worker parameters
        self._worker_params(parallel, n, phy_mem)
        # Check parallel parameters
        if self.parallel and (self.pool or self._MPI):
            if self.pool:
                number_workers = len(self.pool)
            else:
                number_workers = self._number_workers
            if self._MPI:
                self._print('MPI: '+str(number_workers)+' workers')
            else:
                self._print('Multi-threaded: '+str(number_workers)+' workers')
        else:
            self._print('Single-threaded')
        self.minimum_number_intersections = kwargs.get('minimum_number_intersections', 2)
        self._quality_check = kwargs.get('quality_check', False)
        self.kwargs = kwargs
        self.kwargs['number_samples'] = self.number_samples
        self.generate_samples = 0
        self.generate_cutoff = kwargs.get('generate_cutoff', self.max_samples)
        self.file_sample = self.kwargs.get('file_sample', False)
        self._set_algorithm(**self.kwargs)
        self.inversion_options = self.kwargs.get('inversion_options', False)
        self.fid = self.kwargs.get('fid', False)
        self._recover = self.kwargs.get('recover', False)
        self.kwargs['file_sample'] = self.file_sample
        if isinstance(data_file_path, list):
            data_file_path = data_file_path[0]
        # Check if the shell is interactive and get local path,otherwise get workdir path
        if ('PBS_ENVIRONMENT' in os.environ.keys() and 'interactive' in os.environ['PBS_ENVIRONMENT'].lower()) or len([env for env in os.environ.keys() if 'PBS' in env and env != 'PBS_DEFAULT']) < 0:
            default_path = os.getcwd()
        else:
            default_path = os.environ.get('PBS_O_WORKDIR', os.getcwd())
        # If path set in kwargs, get that path instead
        if 'path' in kwargs:
            self._path = kwargs.get('path', default_path)
        elif data_file_path and os.path.isdir(data_file_path):
            self._path = data_file_path
        elif data_file_path and os.path.isdir(os.path.split(data_file_path)[0]):
            self._path = os.path.split(data_file_path)[0]
        else:
            self._path = default_path
        self._diagnostic_output = kwargs.get('diagnostic_output', False)
        self.mpi_output = kwargs.get('mpi_output', True)
        self._print('Algorithm: '+str(self._algorithm_name))
        self._print('Algorithm Options: \n\t'+'\n\t'.join([str(u)+'\t'+str(v) for u, v in self.kwargs.items()]))
        self._print('\n*********\n')

    def _print(self, string, force=False):
        """MPI safe print (rank 0 only unless forced)"""
        # If MPI, only print if rank 0
        if not(self._MPI) or (self._MPI and self.comm.Get_rank() == 0 and not force):
            print(string)

    def _load(self, filename):
        """
        Function to load data

        Args
            filename: filename of binary pickled dictionary or csv file to load

        Returns
            data: Loaded data from filename or False if exception thrown.
        """
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        except Exception:
            try:
                with open(filename, 'r') as f:
                    data = pickle.load(f)
            except Exception:
                data = False
                # Parser plug-in extensions.
                parser_names, parsers = get_extensions('MTfit.parsers', {'.csv': parse_csv, '.hyp': parse_hyp})
                try:
                    try:
                        ext = os.path.splitext(filename)[1]
                        data = parsers[ext](filename)
                    # Else try all the parsers
                    except Exception:
                        for parser in parsers.values():
                            try:
                                data = parser(filename)
                            except Exception:
                                pass
                except Exception:
                    # If errors then try to parse csv and then parse hyp
                    try:
                        data = parse_csv(filename)
                    except Exception:
                        try:
                            data = parse_hyp(filename)
                        except Exception:
                            print('Parsers available are: {}'.format(','.join(list(parsers.keys()))))
                            traceback.print_exc()
                if isinstance(data, bool):
                    print("No data available using available parsers:\n\t"+'\n\t'.join(parser_names))
        return data

    def _read_location(self, filename):
        """
        Reads the location PDF file

        Function allows for extension to other location format types (default is scatangle using parse_scatangle)

        Expected plugin type is to match a file extension and return the sample_records,sample_probability where sample records is a list of the sample records::

        sample_records=[{'Name':['S001','S002',...],'Azimuth':np.matrix([[121.],[37.],...]),'TakeOffAngle':np.matrix([[88.],[12.],...])},
         {'Name':['S001','S002',...],'Azimuth':np.matrix([[120.],[36.],...]),'TakeOffAngle':np.matrix([[87.],[11.],...])}]
        sample_probability=[0.8,1.2,...]


        """
        parser_names, parsers = get_extensions('MTfit.location_pdf_parsers', {'.scatangle': parse_scatangle})
        try:
            # Try to call the plugin for the correct extension
            try:
                ext = os.path.splitext(filename)[1]
                return parsers[ext](filename)
            # Else try with all the parsers
            except Exception:
                for parser in parsers.values():
                    try:
                        return parser(filename)
                    except Exception:
                        pass
                return False
        except Exception:
            if len(parser_names) == 0:  # If not built correctly, can load scatangle file
                return parse_scatangle(filename)
            traceback.print_exc()
            return False

    def _set_logger(self, fid):
        """Sets file loggers up"""
        # now = datetime.datetime.now()
        # If mpi, only set loggers if rank 0
        if (self._MPI and self.comm.Get_rank() == 0) or not self._MPI:
            self.output_logger = None
            self.error_logger = None

    def _close_logger(self):
        """Closes file loggers"""
        # If mpi, only close loggers if rank 0 (no other loggers opened])
        if (self._MPI and self.comm.Get_rank() == 0) or not self._MPI:
            self.output_logger.close()
            self.error_logger.close()

    def _set_algorithm(self, single=False, **kwargs):
        """
        Sets up algorithm for inversion

        Keyword Arguments
            keyword arguments for algorithm configuration

        For information on the kwargs, see the MTfit.Algorithm docstrings.

        """
        self.McMC = False
        # If no algorithm is set - use BaseAlgorithm (Will not behave as expected, but used for testing)
        if not self._algorithm_name:
            self.algorithm = BaseAlgorithm(number_samples=self.number_samples, dc=self.dc, quality_check=self._quality_check,
                                           number_events=self.number_events, file_sample=self.file_sample,
                                           fname=self.kwargs.get('fid', 'MTfit_run'), file_safe=not self.kwargs.get('no_file_safe', False),
                                           sampling=self.kwargs.get('sampling', False), sampling_prior=self.kwargs.get('sampling_prior', False),
                                           sample_distribution=self.kwargs.get('sample_distribution', False))
        # Iterate algorithm
        elif self._algorithm_name.lower() in ['iterate']:
            max_samples = self.max_samples
            number_samples = self.number_samples
            if self._MPI and single:
                # set max samples divided by n_workers
                max_samples = self.max_samples/self._number_workers
                number_samples = min(self.number_samples, max_samples)
                self._print('Iterate algorithm chosen, maximum samples: '+str(self.max_samples)+' across '+str(self._number_workers)+' processes - '+str(max_samples)+' per process')
            else:
                self._print('Iterate algorithm chosen, maximum samples: '+str(self.max_samples))
            self.algorithm = IterationSample(number_samples=number_samples, dc=self.dc, max_samples=max_samples,
                                             quality_check=self._quality_check, number_events=self.number_events,
                                             file_sample=self.file_sample, fname=self.kwargs.get('fid', 'MTfit_run'),
                                             file_safe=not self.kwargs.get('no_file_safe', False),
                                             generate=single and self.c_generate, sampling=self.kwargs.get('sampling', False),
                                             sampling_prior=self.kwargs.get('sampling_prior', False),
                                             sample_distribution=self.kwargs.get('sample_distribution', False))
            if single and self.c_generate:
                # Set generate cutoff size
                self.generate_samples = self.number_samples/5
                self.generate_cutoff = max_samples
        # Time algorithm
        elif self._algorithm_name.lower() in ['time']:
            self._print('Time algorithm chosen, maximum time: '+str(self.max_time))
            self.algorithm = TimeSample(number_samples=self.number_samples, dc=self.dc, max_time=self.max_time,
                                        quality_check=self._quality_check, number_events=self.number_events,
                                        file_sample=self.file_sample, fname=self.kwargs.get('fid', 'MTfit_run'),
                                        file_safe=not self.kwargs.get('no_file_safe', False),
                                        generate=single and self.c_generate, sampling=self.kwargs.get('sampling', False),
                                        sampling_prior=self.kwargs.get('sampling_prior', False),
                                        sample_distribution=self.kwargs.get('sample_distribution', False))
            if single and self.c_generate:
                # Set generate sample size, cutoff is default or set in initialisation
                self.generate_samples = self.number_samples/5
        # Algorithm extensions
        elif self._algorithm_name.lower() in get_extensions('MTfit.parallel_algorithms')[0]:
            extension_algorithms = get_extensions('MTfit.parallel_algorithms')[0]
            kwargs['number_samples'] = self.number_samples
            self.kwargs['number_samples'] = self.number_samples
            self.algorithm = extension_algorithms[self._algorithm_name.lower()](**kwargs)
        # McMC algorithm
        elif 'mcmc' in self._algorithm_name.lower() or self._algorithm_name.lower() in get_extensions('MTfit.directed_algorithms')[0]:
            if 'transd' in self._algorithm_name.lower():
                kwargs['trans_dimensional'] = True
                self.kwargs['trans_dimensional'] = True
                kwargs['number_samples'] = self.number_samples
                self.kwargs['number_samples'] = self.number_samples
            if self._algorithm_name.lower() in get_extensions('MTfit.directed_algorithms')[0]:
                kwargs['mode'] = self._algorithm_name.lower()
            self.McMC = True
            self.algorithm = MarkovChainMonteCarloAlgorithmCreator(**kwargs)

    def _update_samples(self, show=True, _cython=cprobability is not False):
        """
        Update number of samples based on available memory etc.

        Args
            show:[True] Print result to stdout
            _cython: If cython code is being used - default is set by the parameter cprobability which is """+str(cprobability is not False)+'\n\n'
        # Get number of floats from available memory
        nfloats = self._floatmem/8.
        # Get number of location PDF samples
        try:
            number_location_samples = len(self.location_sample_multipliers)
        except Exception:
            number_location_samples = self.number_location_samples
        if not number_location_samples:
            number_location_samples = 1
        A = number_location_samples*6*self.number_events*self._max_data_size
        D = 5*self._max_data_size
        # Calculation allows for a bit of overflow...
        # Calculation depends on Cython or not.
        if _cython:
            if self._relative:
                number_samples = 0.8*(nfloats-(self._number_workers+1)*(4*A+6*D))/((self._number_workers+1)*6*self.number_events)
            else:
                number_samples = 0.8*(nfloats-(self._number_workers+1)*(4*A+6*D))/((self._number_workers+1)*6*self.number_events)
        else:
            if self._relative:
                number_samples = 0.01*(nfloats-(self._number_workers+1)*(4*A+6*D))/((self._number_workers+1)*6*self.number_events+4*(self._number_workers+1)*self._max_data_size*number_location_samples*self.number_events+4*(self._number_workers+1)*number_location_samples*self.number_events*self.number_events)
            else:
                number_samples = 0.8*(nfloats-(self._number_workers+1)*(4*A+6*D))/((self._number_workers+1)*6*self.number_events+4*(self._number_workers+1)*self._max_data_size*number_location_samples*self.number_events)
        if self._MPI:
            number_samples /= 2
        if self.parallel and not self._MPI and sys.version_info[:2] <= (2, 7, 4) and (number_samples*6*self.number_events*8*8 < 2**30):
            # Check for pickle bug #13555 http://bugs.python.org/issue13555 which seems linked to multiprocessing issue #17560 http://bugs.python.org/issue17560
            # cannot pickle files longer than 2**31 (32 bit encoding used for pickle length)
            # Possible fix https://stackoverflow.com/questions/15118344/system-error-while-running-subprocesses-using-multiprocessing
            number_samples = min([((2**30)/(6*8*8*self.number_events)), number_samples])
        # Bodge to prevent memory usage above limit
        number_samples /= 20
        # If estiamted total number of samples <1 raise an error.
        if number_samples < 1:
            self._print('Warning - possible memory error, number of samples less than 1')
            number_samples = 1
        # If iterate, check against max samples set (don't want to evaluate more than that in one sample).
        if hasattr(self, '_algorithm_name') and isinstance(self._algorithm_name, str) and self._algorithm_name.lower() in ['iterate']:
            number_samples = min(number_samples, self.max_samples)
        if show:
            self._print('Number of Random Samples: '+str(int(number_samples))+' with '+str(number_location_samples)+' location samples and '+str(self._max_data_size)+' stations')
        self.number_samples = int(number_samples)
        try:
            self.kwargs['number_samples'] = self.number_samples
        except Exception:
            pass

    def _worker_params(self, parallel=True, n=0, phy_mem=0):
        """Determines worker parameters for inversion

        Determines the sample sizes and worker parameters for the inversion.

        Args
            parallel:[True] Boolean flag for running in parallel
            n:[0] Number of workers to generate, if 0 uses the number returned by multiprocessing.cpu_count()
            phy_mem:[0] Estimated physical memory to use (used for determining array sizes, it is likely that more memory will be used, and if so no errors are forced).
            number_location_samples:[1] Number of station samples (angle scatter samples) used in the inversion - again used for determing array sizes.
            number_events:[1] Number of events - used for determing sample sizes
            number_stations:[40] Number of stations - used for determing sample sizes
        """
        # Check if on cluster or otherwise.
        if len([u for u in os.environ.keys() if 'PBS_' in u]):
            self.PBS = True
        else:
            self.PBS = False
        # Check for pool
        if hasattr(self, 'pool'):
            self._close_pool()
        # Get number of processors available
        if self.PBS:
            # If PBS get number of nodes
            try:
                n = int(os.environ['PBS_NUM_PPN'])*int(os.environ['PBS_NUM_NODES'])
            except KeyError:
                n = 1
        if not parallel:
            n = 1
        # Check parallel - close pool if MPI or only on processor
        if not parallel or n == 1 or self._MPI:
            self._close_pool()
            self.pool = False
        elif n:
            # Set pool for bumber of workers
            self.pool = JobPool(n, task=ForwardTask)
        else:
            # Get number of workers from multiprocessing
            n = multiprocessing.cpu_count() or 1  # Check CPU count if errors use 1
            self.pool = JobPool(n, task=ForwardTask)
        self._number_workers = n
        # Amount of memory available for calculations (Can cause problems for very large non-zero percentages)
        mem_scale = 0.9
        # Set available float memory
        if not phy_mem:
            try:
                import psutil
                self._floatmem = mem_scale*psutil.virtual_memory().available/n
            except Exception:
                phy_mem = 8
                self._floatmem = mem_scale*phy_mem*(1024*1024*1024.)
        else:
            self._floatmem = mem_scale*phy_mem*(1024*1024*1024.)
        self.parallel = parallel
        # calculate data size
        max_data = 0
        max_data_size = 0
        for data in self.data:
            if isinstance(data, dict):
                for key in data:
                    if key not in ['UID', 'hyp_file'] and len(data[key]):
                        max_data_size = max([max_data, len(data[key]['Stations']['Name'])])
        self._max_data_size = max_data_size
        # Calculate Sample Size
        self._update_samples(show=False)

    def __del__(self):
        """Closes pool and exits, allowing object to be deleted."""
        try:
            self._close_pool()
        except Exception:
            traceback.print_exc()

        gc.collect()

    def _close_pool(self):
        """Closes and removes JobPool

        """
        try:
            if hasattr(self, 'pool') and self.pool:
                self.pool.close()
                del self.pool
            self.pool = False
        except Exception:
            traceback.print_exc()

        gc.collect()

    def _combine_mpi_output(self, event_data, fids, scale_factor_fids=[], fid='MTfitOutput.mat', output_data=False, location_samples=False,
                            location_sample_multipliers=False, format='matlab', binary_file_version=2, *args, **kwargs):
        """
        Combine mpi output binary files - Rank 0 only

        Args:
        event_data: Event data dictionary
        fids: storage fids
        scale_factor_fids:[[]] Scale_factor fids if relative amplitudes are used.
        fid=[MTfitOutput.mat] Output filename.
        output_data:[False] Output data (Stations etc.)
        location_samples:[False] Location PDF samples
        location_sample_multipliers:[False] Location PDF sample probabilities (should be one for Oct-tree samples)
        format=[matlab] Output format.


        """
        # Get old results format
        old_results_format = self.results_format
        # Check kwargs for update format
        self.results_format = kwargs.get('results_format', old_results_format)
        print('---------Combining MPI Output-----------')
        output = {}
        # Loop over fids
        for i, mpi_fid in enumerate(fids):
            # Get mt files
            mt_fid = os.path.splitext(mpi_fid)[0]+'.mt'
            # Read mt output in.
            if not len(output):
                # No files read
                try:
                    output = read_binary_output(mt_fid, version=binary_file_version)
                except Exception:
                    continue
                if isinstance(output, list) and len(output) == 1:
                    output = output[0]
            else:
                try:
                    binary_output = read_binary_output(mt_fid)
                except Exception:
                    continue
                if isinstance(binary_output, list) and len(binary_output) == 1:
                    binary_output = binary_output[0]
                if isinstance(binary_output, list) and len(binary_output) > 1:
                    for i, event_output in enumerate(binary_output):
                        for key in event_output.keys():
                            if key in output[i].keys():
                                if key == 'total_number_samples':
                                    output[i][key] += event_output[key]
                                elif key in ['ln_bayesian_evidence', 'dkl']:
                                    output[i][key] = 0  # Updated later
                                else:
                                    output[i][key] = np.append(output[i][key], event_output[key], 1)
                else:
                    for key in binary_output.keys():
                        if key in output.keys():
                            if key == 'total_number_samples':
                                output[key] += binary_output[key]
                            elif key in ['ln_bayesian_evidence', 'dkl']:
                                output[key] = 0  # Updated later
                            else:
                                output[key] = np.append(output[key], binary_output[key], 1)
        # Handle Scale factors if they exist.
        if len(scale_factor_fids):
            for i, fid in enumerate(scale_factor_fids):
                fid = os.path.splitext(fid)[0]+'.sf'
                # Read scale_factors
                if not len(output):
                    try:
                        output = read_sf_output(fid)
                    except Exception:
                        continue
                    if isinstance(output, list) and len(output) == 1:
                        output = output[0]
                else:
                    try:
                        scale_factors_output = read_sf_output(fid)
                    except Exception:
                        continue
                    if isinstance(scale_factors_output, list) and len(scale_factors_output) == 1:
                        scale_factors_output = scale_factors_output[0]
                    if isinstance(scale_factors_output, list) and len(scale_factors_output) > 1:
                        for i, event_scale_factors in enumerate(scale_factors_output):
                            for key in event_scale_factors.keys():
                                if key in output[i].keys():
                                    if key == 'total_number_samples':
                                        output[i][key] += event_scale_factors[key]
                                    elif key in ['ln_bayesian_evidence', 'dkl']:
                                        output[i][key] = 0  # Updated later
                                    else:
                                        output[i][key] = np.append(output[i][key], event_scale_factors[key], 1)
                    else:
                        for key in scale_factors_output.keys():
                            if key in output.keys():
                                if key == 'total_number_samples':
                                    output[key] += scale_factors_output[key]
                                elif key in ['ln_bayesian_evidence', 'dkl']:
                                    output[key] = 0  # Updated later
                                else:
                                    output[key] = np.append(output[key], scale_factors_output[key], 1)
        # Output - handle special cases
        if isinstance(output, list):
            for i, out in enumerate(output):
                if len(output):
                    print(output[i]['total_number_samples'])
                    output[i]['ln_bayesian_evidence'] = ln_bayesian_evidence(output[i], output[i]['total_number_samples'])
                    try:
                        if np.max(output[i]['g'])-np.min(output[i]['g']) < 0.000001 and np.abs(np.mean(output[i]['g'])) < 0.000001 and np.max(output[i]['d'])-np.min(output[i]['d']) < 0.000001 and np.abs(np.mean(output[i]['d'])) < 0.000001:
                            V = (2*np.pi*np.pi)
                        else:
                            V = (np.pi*np.pi*np.pi)
                        output[i]['dkl'] = dkl_estimate(output[i]['ln_pdf'], V, output[i]['total_number_samples'])
                    except Exception:
                        pass
                    output[i]['dV'] = 1
        else:
            if not len(output):
                return
            output['dV'] = 1
            output['ln_bayesian_evidence'] = ln_bayesian_evidence(output, output['total_number_samples'])
            try:
                if np.max(output['g'])-np.min(output['g']) < 0.000001 and np.abs(np.mean(output['g'])) < 0.000001 and np.max(output['d'])-np.min(output['d']) < 0.000001 and np.abs(np.mean(output['d'])) < 0.000001:
                    V = (2*np.pi*np.pi)
                else:
                    V = (np.pi*np.pi*np.pi)
                output['dkl'] = dkl_estimate(output['ln_pdf'], V, output['total_number_samples'])
            except Exception:
                pass
        if os.path.splitext(fid)[0].split('.')[-1] == str(0):
            fid = '.'.join(os.path.splitext(fid)[0].split('.')[:-1])+os.path.splitext(fid)[1]
        # format specifics - output for MATLAB only is first
        if isinstance(output, list):
            for i, event_output in enumerate(output):
                fid_i = os.path.splitext(fid)[0]+'_'+str(i)+os.path.splitext(fid)[1]
                self.output(event_data, fid_i, event_output, location_samples, location_sample_multipliers, format, *args, **kwargs)
        else:
            self.output(event_data, fid, output, location_samples, location_sample_multipliers, format, *args, **kwargs)
        # Reset output format
        self.results_format = old_results_format

    def output(self, event_data, fid='MTfitOutput.mat', output_data=False, location_samples=False, location_sample_multipliers=False,
               output_format='matlab', *args, **kwargs):
        """
        Outputs event_data results to fid

        Default output format is matlab.

        Args
            event_data: data dictionary for the event_data.
            fid:['MTfitOutput.mat'] Filename for output.
            Algorithm:[False] Algorith for output - only needed if multiple event_datas/algorithms used
            location_samples:[False] Station angle scatter samples.
            location_sample_multipliers:[False] Station angle scatter probabilities.
            output_format:['matlab'] output format.

        Keyword Arguments
            station_only:[False] Only output station data

        """
        self._print('Saving')
        # Check location samples
        if location_samples and location_sample_multipliers and self._output_station_distribution:
            pass
        elif not hasattr(self, 'location_samples') or not hasattr(self, 'location_sample_multipliers'):
            location_samples = False
            location_sample_multipliers = False
        elif self.location_samples and self.location_sample_multipliers_original and self._output_station_distribution:
            location_samples = self.location_samples
            location_sample_multipliers = self.location_sample_multipliers_original
        elif not self._output_station_distribution:
            location_samples = False
            location_sample_multipliers = False
        # Check if output data exists (or just outputting data)
        if not output_data and not kwargs.get('station_only', False):
            output_data, output_string = self.algorithm.output(self.normalise, self.convert, self.discard)
            if not self._MPI or (self._MPI and (self.mpi_output or(not isinstance(self.comm, bool) and self.comm.Get_rank()))) == 0:
                if self._MPI and self.mpi_output:
                    output_string.replace('-MTfit Forward Model Output-', '-MTfit Forward Model Output Process '+str(self.comm.Get_rank())+'-')
                print(output_string)
        try:
            if (not isinstance(output_data, dict) or not len(output_data['probability'])) and not kwargs.get('station_only', False):
                return
        except TypeError:
            return
        # get results format
        output_data_names, output_data_formats = get_extensions('MTfit.output_data_formats', {'full_pdf': full_pdf_output_dicts, 'hyp': hyp_output_dicts})
        try:
            # Try to call the plugin for the correct extension'
            if output_format and output_format.lower() == 'hyp':
                out_data = output_data_formats[output_format.lower()](event_data, self.inversion_options, output_data, location_samples, location_sample_multipliers, self.multiple_events, self._diagnostic_output, *args, **kwargs)
            else:
                out_data = output_data_formats[self.results_format](event_data, self.inversion_options, output_data, location_samples, location_sample_multipliers, self.multiple_events, self._diagnostic_output, *args, **kwargs)
            # Else try with full pdf output
        except Exception:
            traceback.print_exc()
            out_data = full_pdf_output_dicts(event_data, self.inversion_options, output_data, location_samples, location_sample_multipliers, self.multiple_events, self._diagnostic_output, *args, **kwargs)

        if output_format and output_format.lower() == 'matlab' and self.results_format == 'full_pdf':
            try:
                if out_data[0] and out_data[0]['Events']['Probability'].shape[-1]*len([key for key in out_data[0]['Events'].keys() if 'MTSpace' in key]) < 20000000:
                    kwargs['version'] = '7'
            except Exception:
                out_data[0] = False
        # output data
        output_data_names, output_data_formats = get_extensions('MTfit.output_formats', {'matlab': MATLAB_output, 'pickle': pickle_output, 'hyp': hyp_output})
        try:
            try:
                # Try to call the plugin for the correct extension
                output_string, fid = output_data_formats[output_format.lower()](out_data, fid, self.pool, *args, **kwargs)
                # Else try with full pdf output
            except Exception:
                traceback.print_exc()
                if output_format and output_format.lower() == 'matlab':
                    version = kwargs.get('version', '7.3')
                    kwargs['version'] = version
                    kwargs.pop('version')
                    output_string, fid = MATLAB_output(out_data, fid, self.pool, version=version, *args, **kwargs)
                if output_format and output_format.lower() in ['pickle']:
                    output_string, fid = pickle_output(out_data, fid, self.pool, *args, **kwargs)
            self._print(output_string)
        except Exception:
            self._print(output_format+' output')
            traceback.print_exc()

    def _station_angles(self, event, i):
        """Convert event station angles to GF six vector

        Converts the station azimuth and take-off angle to the six vector corresponding to the homogeneous greens functions, as presented in ######################

        Args
            event: Event data dictionary.
            i: Event index for scatter files.

        Returns
            a_polarity,error_polarity,a1_amplitude_ratio,a2_amplitude_ratio,amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio,a_polarity_prob,polarity_prob

            a_polarity: Polarity station angles matrix.
            error_polarity: Polarity error matrix.
            a1_amplitude_ratio: Amplitude ratio numerator station angles matrix.
            a2_amplitude_ratio: Amplitude ratio denominator station angles matrix.
            amplitude_ratio: Amplitude Ratio.
            percentage_error1_amplitude_ratio: Amplitude ratio numerator percentage uncertainty matrix.
            percentage_error2_amplitude_ratio: Amplitude ratio denominator percentage uncertainty matrix.
            a_polarity_prob: Polarity PDF station angles matrix.
            polarity_prob: Polarity PDF probabilities
        """
        location_samples = False
        location_sample_multipliers = False
        # Check location PDF samples and set if they exist
        if self.location_pdf_files and len(self.location_pdf_files[i]):
            location_samples, location_sample_multipliers = self._read_location(self.location_pdf_files[i])
            self.location_samples = location_samples
            self.location_sample_multipliers = location_sample_multipliers
            self.location_sample_multipliers_original = self.location_sample_multipliers[:]
        elif self.location_pdf_files and self._relative and not self._marginalise_relative:
            location_pdf_file = [location_pdf_file for location_pdf_file in self.location_pdf_files if len(location_pdf_file)][0]
            location_samples, location_sample_multipliers = self._read_location(location_pdf_file)
            self.location_samples = location_samples
            self.location_sample_multipliers = location_sample_multipliers
            self.location_sample_multipliers_original = self.location_sample_multipliers[:]
        else:
            self.location_samples = False
            self.location_sample_multipliers = False
            self.location_sample_multipliers_original = False
        # Get station angles and measurements from data
        a_polarity, error_polarity, incorrect_polarity_probability = polarity_matrix(event, location_samples)
        a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio = amplitude_ratio_matrix(event, location_samples)
        a_polarity_probability, polarity_probability, incorrect_polarity_probability_2 = polarity_probability_matrix(event, location_samples)
        # Extensions:
        extension_data = {}
        extension_names, extensions = get_extensions('MTfit.process_data_types')
        for ext in extension_names:
            extension_data[ext] = extensions[ext](event)
        # Bin station location PDF samples
        if self.bin_angle_coefficient_samples > 0 and location_samples:
            (a_polarity, a1_amplitude_ratio, a2_amplitude_ratio, a_polarity_probability, extension_data,
             self.location_sample_multipliers) = bin_angle_coefficient_samples(a_polarity, a1_amplitude_ratio, a2_amplitude_ratio, a_polarity_probability, self.location_sample_multipliers,
                                                                               self.bin_angle_coefficient_samples, extension_data)
        if isinstance(a_polarity, bool):
            incorrect_polarity_probability = incorrect_polarity_probability_2
        return (a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio,
                a_polarity_probability, polarity_probability, incorrect_polarity_probability, extension_data)

    def _fid(self, event, i, source='MT', single=False):
        """Generates event fid

        Args
            event: event data dictionary
            i: event index
            source: 'MT' or 'dc'

        Returns
            fid string
        """
        # Try to make fid from attribute
        if self.fid:
            fid = self.fid.split('.mat')[0]+str(i)+source+'.mat'
        # Otherwise try to use UID or fall back to MTfitOutput as default
        else:
            try:
                fid = self._path+os.path.sep+str(event['UID'])+source+'.mat'
            except Exception:
                fid = self._path+os.path.sep+'MTfitOutput'+source+'.mat'
        # Add rank to MPI file output (so they don't overwrite)
        if self._MPI and single and self.mpi_output:
            fid = os.path.splitext(fid)[0]+'.'+str(self.comm.Get_rank())+os.path.splitext(fid)[1]
        return fid

    def _recover_test(self, fid):
        """
        Tests if the output file exists in the correct format

        Args:
            fid: Filename to check for.

        Returns
            Boolean: True if exists.
        """
        recover = False
        # If recover option set
        if self._recover:
            # Check MPI and rank
            if not self._MPI or self.comm.Get_rank() == 0:
                # Either Rank 0 or no MPI
                # Check MPI output
                if self._MPI and self.mpi_output:
                    if len(glob.glob(fid.split('.mat')[0]+'.pkl')) or len(glob.glob(fid.split('.mat')[0]+'.0.pkl')):
                        recover = True
                # Check if MATLAB format
                if not recover and self.output_format.lower() == 'matlab':
                    try:
                        try:
                            import h5py
                            if h5py.is_hdf5(fid.split('.mat')[0]+'.mat'):
                                from hdf5storage import loadmat
                            else:
                                from scipy.io import loadmat
                        except Exception as e:
                            self._print(e)
                            from scipy.io import loadmat
                        loadmat(fid.split('.mat')[0]+'.mat')
                        gc.collect()
                        self._print('Recover option enabled and output file exists: '+fid)
                        recover = True
                    except Exception as e:
                        self._print('Recover option enabled and output file does not exist or is corrupted')
                        self._print(e)
                        recover = False
        # Broadcast recover if MPI
        if self._MPI:
            recover = self.comm.bcast(recover, root=0)
        return recover

    def _file_sample_test(self, fid):
        """
        Recover test if file sample option enabled

        Args:
            fid: Filename to check for.

        Returns
            Boolean: True if exists.
        """
        if self._recover and self.file_sample:
            try:
                from hdf5storage import loadmat
                loadmat(fid.split('.mat')[0]+'_in_progress.mat')
                gc.collect()
                self._print('Recover option enabled and in progress file exists, returning to this point')  # How to handle mcmc?
                return True
            except Exception:
                return False
        elif not self._recover:
            try:
                os.remove(fid.split('.mat')[0]+'_in_progress.mat')
            except Exception:
                pass
            return False

    @memory_profile_test(_MEMTEST)
    def _mcmc_forward(self, source_type='MT', **kwargs):
        """
        Markov chain Monte Carlo event forward function

        Runs event forward model using Markov chain Monte Carlo approach. Runs multiple events in parallel.

        Args
            source_type:['MT'] 'MT' or 'dc' for fid

        """
        # Loop over events in self.data list
        for i, event in enumerate(self.data):
            # Set output fid
            fid = self._fid(event, i, source_type)
            self._print('\n\nEvent '+str(i+1)+'\n--------\n')
            try:
                self._print('UID: '+str(event['UID'])+'\n')
            except Exception:
                self._print('No UID\n')
            # Do recovery test
            if self._recover_test(fid):
                continue
            try:
                # Check data
                if not event:
                    self._print('No Data')
                    continue
                try:
                    event = self._trim_data(event)
                except ValueError:
                    self._print('No Data')
                    continue
                del self.algorithm
                gc.collect()
                self.kwargs['fid'] = fid
                # Get station angle coefficients and data
                (a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio, a_polarity_probability,
                 polarity_probability, incorrect_polarity_probability, extension_data) = self._station_angles(event, i)
                # Update sample numbers
                self._update_samples()
                # Set algorithms
                self._set_algorithm(**self.kwargs)
                # Check parallel etc. and run McMCForwardTask
                if self.pool:
                    if i > len(self.pool):
                        result = self.pool.result()
                        self._print('Inversion completed,  '+str(result['algorithm_output_data']['total_number_samples'])+' samples evaluated: '+str((result['algorithm_output_data']['accepted'])) +
                                    ' accepted samples: '+str((float(result['algorithm_output_data']['accepted'])/float(result['algorithm_output_data']['total_number_samples']))*100)[:4]+'%')
                        try:
                            self.output(result['event_data'], result['fid'], result['algorithm_output_data'], result['location_samples'], result['location_sample_multipliers'], output_format=self.output_format)
                        except Exception:
                            self._print('Output Error')
                            traceback.print_exc()

                    self.pool.custom_task(McMCForwardTask, self.kwargs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio,
                                          percentage_error2_amplitude_ratio, a_polarity_probability, polarity_probability, incorrect_polarity_probability, fid, event, self.location_samples,
                                          self.location_sample_multipliers, self.normalise, self.convert, extension_data)
                else:
                    result = McMCForwardTask(self.kwargs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio,
                                             percentage_error2_amplitude_ratio, a_polarity_probability, polarity_probability, incorrect_polarity_probability, fid, normalise=self.normalise,
                                             convert=self.convert, extension_data=extension_data)()
                    try:
                        accepted = (float(result['algorithm_output_data']['accepted'])/float(result['algorithm_output_data']['total_number_samples']))*100
                    except ZeroDivisionError:
                        accepted = 0
                    self._print('Inversion completed,  '+str(result['algorithm_output_data']['total_number_samples'])+' samples evaluated: '+str((result['algorithm_output_data']['accepted'])) +
                                ' accepted samples: '+str(accepted)[:4]+'%')
                    try:
                        self.output(event, fid, result['algorithm_output_data'], a_polarity=a_polarity, error_polarity=error_polarity, a1_amplitude_ratio=a1_amplitude_ratio,
                                    a2_amplitude_ratio=a2_amplitude_ratio, amplitude_ratio=amplitude_ratio, percentage_error1_amplitude_ratio=percentage_error1_amplitude_ratio,
                                    percentage_error2_amplitude_ratio=percentage_error2_amplitude_ratio, output_format=self.output_format)
                    except Exception:
                        self._print('Output Error')
                        traceback.print_exc()
            except NotImplementedError as e:
                raise e
            except Exception:
                traceback.print_exc()
        # Get pool results
        if self.pool:
            results = self.pool.all_results()
            for result in results:
                try:
                    accepted = (float(result['algorithm_output_data']['accepted'])/float(result['algorithm_output_data']['total_number_samples']))*100
                except ZeroDivisionError:
                    accepted = 0
                self._print('Inversion completed,  '+str(result['algorithm_output_data']['total_number_samples'])+' samples evaluated: '+str((result['algorithm_output_data']['accepted'])) +
                            ' accepted samples: '+str(accepted)[:4]+'%')
                try:
                    self.output(result['event_data'], result['fid'], result['algorithm_output_data'], result['location_samples'], result['location_sample_multipliers'], output_format=self.output_format)
                except Exception:
                    self._print('Output Error')
                    traceback.print_exc()

    @memory_profile_test(_MEMTEST)
    def _mcmc_multiple_forward(self, source_type='MT', **kwargs):
        """
        Markov chain Monte Carlo event forward function for multiple event joint pdf

        Runs event forward model using Markov chain Monte Carlo approach for multiple events.

        Args
            source_type:['MT'] 'MT' or 'dc' for fid

        """
        # Get fid
        fid = self._fid({}, '', '_joint_inversion'+source_type)
        # Recover test
        if self._recover_test(fid):
            return
        # Sort station angles and data for multiple event
        a_polarity = []
        error_polarity = []
        a1_amplitude_ratio = []
        a2_amplitude_ratio = []
        amplitude_ratio = []
        percentage_error1_amplitude_ratio = []
        percentage_error2_amplitude_ratio = []
        a_polarity_probability = []
        polarity_probability = []
        a_relative_amplitude = []
        relative_amplitude = []
        percentage_error_relative_amplitude = []
        relative_amplitude_stations = []
        incorrect_polarity_probability = []
        extension_data = []
        for i, event in enumerate(self.data):
            try:
                # Check event data
                if not event:
                    self._print('No Data')
                    continue
                try:
                    event = self._trim_data(event)
                except ValueError:
                    self._print('No Data')
                    continue
                # Get event station angles and data etc.  for event and add to list
                (a_polarity_i, error_polarity_i, a1_amplitude_ratio_i, a2_amplitude_ratio_i, amplitude_ratio_i, percentage_error1_amplitude_ratio_i,
                 percentage_error2_amplitude_ratio_i, a_polarity_probability_i, polarity_probability_i, incorrect_polarity_probability_i, extension_data_i) = self._station_angles(event, i)
                a_polarity.append(a_polarity_i)
                error_polarity.append(error_polarity_i)
                a1_amplitude_ratio.append(a1_amplitude_ratio_i)
                a2_amplitude_ratio.append(a2_amplitude_ratio_i)
                amplitude_ratio.append(amplitude_ratio_i)
                percentage_error1_amplitude_ratio.append(percentage_error1_amplitude_ratio_i)
                percentage_error2_amplitude_ratio.append(percentage_error2_amplitude_ratio_i)
                a_polarity_probability.append(a_polarity_probability_i)
                polarity_probability.append(polarity_probability_i)
                incorrect_polarity_probability.append(incorrect_polarity_probability_i)
                extension_data.append(extension_data_i)
                # If running relative get relative data
                if self._relative:
                    a_relative_amplitude_i, relative_amplitude_i, percentage_error_relative_amplitude_i, relative_amplitude_stations_i = relative_amplitude_ratio_matrix(event, self.location_samples)
                    a_relative_amplitude.append(a_relative_amplitude_i)
                    relative_amplitude.append(relative_amplitude_i)
                    percentage_error_relative_amplitude.append(percentage_error_relative_amplitude_i)
                    relative_amplitude_stations.append(relative_amplitude_stations_i)
            except NotImplementedError as e:
                raise e
            except Exception:
                traceback.print_exc()
        self._print('\n\nJoint Inversion: '+str(self.number_events)+' Events\n--------\n')
        try:
            # Set algorithm
            del self.algorithm
            gc.collect()
            self._update_samples()
            self._set_algorithm(**self.kwargs)
            # Run forward model using MultipleEventsMcMCForwardTask
            if self.pool:
                self.pool.custom_task(MultipleEventsMcMCForwardTask, self.kwargs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio,
                                      amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio, a_polarity_probability,
                                      polarity_probability, a_relative_amplitude, relative_amplitude, percentage_error_relative_amplitude,
                                      relative_amplitude_stations, incorrect_polarity_probability, self._relative, self.minimum_number_intersections,
                                      fid, event, self.location_samples, self.location_sample_multipliers, self._marginalise_relative, self.normalise,
                                      self.convert, extension_data)

                result = self.pool.result()
                try:
                    accepted = (float(result['algorithm_output_data']['accepted'])/float(result['algorithm_output_data']['total_number_samples']))*100
                except ZeroDivisionError:
                    accepted = 0
                self._print('Inversion completed, '+str(result['algorithm_output_data']['total_number_samples'])+' samples evaluated: '+str((result['algorithm_output_data']['accepted'])) +
                            ' accepted samples: '+str(accepted)[:4]+'%')
                try:
                    self.output(result['event_data'], result['fid'], result['algorithm_output_data'], result['location_samples'], result['location_sample_multipliers'],
                                output_format=self.output_format)
                except NotImplementedError as e:
                    raise e
                except Exception:
                    self._print('Output Error')
                    traceback.print_exc()

            else:
                result = MultipleEventsMcMCForwardTask(self.kwargs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio,
                                                       amplitude_ratio, percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio, a_polarity_probability, polarity_probability,
                                                       a_relative_amplitude, relative_amplitude, percentage_error_relative_amplitude, relative_amplitude_stations, incorrect_polarity_probability,
                                                       relative=self._relative, minimum_number_intersections=self.minimum_number_intersections, fid=fid, location_samples=self.location_samples,
                                                       marginalise_relative=self._marginalise_relative, normalise=self.normalise, convert=self.convert, extension_data=extension_data)()
                self._print('Inversion completed, '+str(result['algorithm_output_data']['total_number_samples'])+' samples evaluated: '+str((result['algorithm_output_data']['accepted'])) +
                            ' accepted samples: '+str((float(result['algorithm_output_data']['accepted'])/float(result['algorithm_output_data']['total_number_samples']))*100)[:4]+'%')
                try:
                    self.output(self.data, fid, result['algorithm_output_data'], a_polarity=a_polarity, error_polarity=error_polarity, a1_amplitude_ratio=a1_amplitude_ratio,
                                a2_amplitude_ratio=a2_amplitude_ratio, amplitude_ratio=amplitude_ratio, percentage_error1_amplitude_ratio=percentage_error1_amplitude_ratio,
                                percentage_error2_amplitude_ratio=percentage_error2_amplitude_ratio, output_format=self.output_format)
                except NotImplementedError as e:
                    raise e
                except Exception:
                    self._print('Output Error')
                    traceback.print_exc()
        except NotImplementedError as e:
            raise e
        except Exception:
            traceback.print_exc()

    @memory_profile_test(_MEMTEST)
    def _random_sampling_forward(self, source_type='MT', return_zero=False):
        """
        Monte Carlo random sampling event forward function

        Runs event forward model using Monte Carlo random sampling approach - ie parallel for multiple samples,  single event.

        Args
            source_type:['MT'] 'MT' or 'DC' for fid
            return_zero:[True] Return zero probability samples in task result.

        """
        # Loop over events
        for i, event in enumerate(self.data):
            # Get fid
            fid = self._fid(event, i, source_type, single=True)
            # Set logger
            self._set_logger(fid)
            self._print('\n\nEvent '+str(i+1)+'\n--------\n')
            try:
                self._print('UID: '+str(event['UID'])+'\n')
            except Exception:
                self._print('No UID\n')
            # Recover test
            if self._recover_test(fid):
                continue
            # File sample recover test
            if self._file_sample_test(fid):
                self._print('Continuing from previous sampling')
            self.kwargs['fid'] = fid
            # Check event data
            if not event:
                self._print('No Data')
                continue
            try:
                try:
                    event = self._trim_data(event)
                except ValueError:
                    self._print('No Data')
                    continue
                # Get station angle coefficients and data etc.
                (a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio,
                 percentage_error2_amplitude_ratio, a_polarity_probability, polarity_probability, incorrect_polarity_probability, extension_data) = self._station_angles(event, i)
                # Update sample size
                self._update_samples()
                # Set algorithm
                del self.algorithm
                gc.collect()
                self._set_algorithm(single=True, **self.kwargs)
                self._print('\nInitialisation Complete\n\nBeginning Inversion\n')
                # Run ForwardTasks
                if self.pool:
                    # initialise all tasks
                    for i in range(self.pool.number_workers):
                        MTs, end = self._parse_job_result(False)
                        self.pool.task(MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio,
                                       percentage_error2_amplitude_ratio, a_polarity_probability, polarity_probability, self.location_sample_multipliers,
                                       incorrect_polarity_probability, return_zero, False, True, self.generate_samples, self.generate_cutoff, self.dc, extension_data)
                elif self._MPI:
                    end = False
                    # Carried out in each worker
                    MTs, ignored_end = self._parse_job_result(False)
                    result = ForwardTask(MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio,
                                         percentage_error2_amplitude_ratio, a_polarity_probability, polarity_probability, self.location_sample_multipliers,
                                         incorrect_polarity_probability, return_zero, generate_samples=self.generate_samples, cutoff=self.generate_cutoff,
                                         dc=self.dc, extension_data=extension_data)()
                    # Return to initiator algorithm
                else:
                    MTs, end = self._parse_job_result(False)
                # Continue until max samples/time reached (end =  = True)
                while not end:
                    if self.pool:
                        result = self.pool.result()
                        MTs, end = self._parse_job_result(result)
                        self.pool.task(MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio,
                                       percentage_error2_amplitude_ratio, a_polarity_probability, polarity_probability, self.location_sample_multipliers,
                                       incorrect_polarity_probability, return_zero, False, True, self.generate_samples, self.generate_cutoff, self.dc, extension_data)
                    elif self._MPI:
                        if not self.mpi_output:
                            # Handle result split and together using gather
                            mts = self.comm.gather(result['moment_tensors'], 0)
                            Ps = self.comm.gather(result['ln_pdf'], 0)
                            Ns = self.comm.gather(result['n'], 0)
                            if self.comm.Get_rank() == 0:
                                end = False
                                for i, mt in enumerate(mts):
                                    result = {'moment_tensors': mts[i], 'ln_pdf': Ps[i], 'n': Ns[i]}
                                    ignoredMTs, end = self._parse_job_result(result)
                                if end:
                                    end = True  # Process all results before ending
                            else:
                                end = None
                            end = self.comm.bcast(end, root=0)  # Broadcast end to all mpis
                            _iteration = self.algorithm.iteration
                            _start_time = False
                            try:
                                _start_time = self.algorithm.start_time
                            except Exception:
                                pass
                            MTs, ignored_end = self._parse_job_result(False)  # Get new random MTs
                            self.algorithm.iteration = _iteration
                            if _start_time:
                                self.algorithm.start_time = _start_time
                        else:
                            # Just run in parallel
                            MTs, end = self._parse_job_result(result)
                            end = self.comm.bcast(end, root=0)
                        result = ForwardTask(MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio,
                                             percentage_error2_amplitude_ratio, a_polarity_probability, polarity_probability, self.location_sample_multipliers,
                                             incorrect_polarity_probability, return_zero, generate_samples=self.generate_samples, cutoff=self.generate_cutoff, dc=self.dc,
                                             extension_data=extension_data)()
                    else:
                        result = ForwardTask(MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio,
                                             percentage_error2_amplitude_ratio, a_polarity_probability, polarity_probability, self.location_sample_multipliers,
                                             incorrect_polarity_probability, return_zero, generate_samples=self.generate_samples, cutoff=self.generate_cutoff, dc=self.dc,
                                             extension_data=extension_data)()
                        MTs, end = self._parse_job_result(result)
                # Get left over pool results
                if self.pool:
                    results = self.pool.all_results()
                    for result in results:
                        if result:
                            MTs, end = self._parse_job_result(result)
                try:
                    self._print('Inversion completed\n\t'+'Elapsed time: '+str(time.time()-self.algorithm.start_time).split('.')[0]+' seconds\n\t'+str(self.algorithm.pdf_sample.n) +
                                ' samples evaluated\n\t'+str(len(self.algorithm.pdf_sample.nonzero())) +
                                ' non-zero samples\n\t'+'{:f}'.format((float(len(self.algorithm.pdf_sample.nonzero()))/float(self.algorithm.pdf_sample.n))*100)+'%')
                except ZeroDivisionError:
                    self._print('Inversion completed\n\t'+'Elapsed time: '+str(time.time()-self.algorithm.start_time).split('.')[0]+' seconds\n\t'+str(self.algorithm.pdf_sample.n) +
                                ' samples evaluated\n\t'+str(len(self.algorithm.pdf_sample.nonzero()))+' non-zero samples\n\t'+'{:f}'.format(0)+'%')
                self._print('Algorithm max value: '+self.algorithm.max_value())
                output_format = self.output_format
                if self._MPI and self.mpi_output:
                    # MPI output (hyp format)
                    output_format = 'hyp'
                    results_format = self.results_format
                    self.results_format = 'hyp'
                    normalise = self.normalise
                    self.normalise = False
                # Output results
                if (self._MPI and not self.mpi_output and self.comm.Get_rank() == 0) or (self._MPI and self.mpi_output) or not self._MPI:
                    try:
                        self.output(event, fid, a_polarity=a_polarity, error_polarity=error_polarity, a1_amplitude_ratio=a1_amplitude_ratio, a2_amplitude_ratio=a2_amplitude_ratio,
                                    amplitude_ratio=amplitude_ratio, percentage_error1_amplitude_ratio=percentage_error1_amplitude_ratio,
                                    percentage_error2_amplitude_ratio=percentage_error2_amplitude_ratio, output_format=output_format)
                    except Exception:
                        self._print('Output Error')
                        traceback.print_exc()

                if self._MPI and self.mpi_output:
                    # Output pkl file with data
                    self.normalise = normalise
                    self.results_format = results_format
                    fids = self.comm.gather(fid, 0)
                    if isinstance(fids, list):
                        self._print('All moment tensor samples outputted: '+str(len(fids))+' files')
                    else:
                        self._print('Error with fids '+str(fids)+' type:'+str(type(fids)))
                    if self.comm.Get_rank() == 0:
                        with open(os.path.splitext(fid)[0]+'.pkl', 'wb') as f:
                            pickle.dump({'fids': fids, 'event_data': event}, f)
            except Exception:
                traceback.print_exc()

    @memory_profile_test(_MEMTEST)
    def _random_sampling_multiple_forward(self, source_type='MT', **kwargs):
        """
        Monte Carlo event forward function for multiple event joint pdf

        Runs event forward model using Markov chain Monte Carlo approach for multiple events.

        Args
            source_type:['MT'] 'MT' or 'dc' for fid

        """
        # Set fid
        fid = self._fid({}, '', '_joint_inversion'+source_type, single=True)
        # Set logger
        self._set_logger(fid)
        # Recovery test
        if self._recover_test(fid):
            return
        # File sample recovery test
        if self._file_sample_test(fid):
            self._print('Continuing from previous sampling')
        self.kwargs['fid'] = fid
        # Get station angle coefficients, data etc. for multiple events
        a_polarity = []
        error_polarity = []
        a1_amplitude_ratio = []
        a2_amplitude_ratio = []
        amplitude_ratio = []
        percentage_error1_amplitude_ratio = []
        percentage_error2_amplitude_ratio = []
        a_polarity_probability = []
        polarity_probability = []
        a_relative_amplitude = []
        relative_amplitude = []
        percentage_error_relative_amplitude = []
        relative_amplitude_stations = []
        incorrect_polarity_probability = []
        extension_data = []
        for i, event in enumerate(self.data):
            try:
                # Check event data
                if not event:
                    self._print('No Data')
                    continue
                try:
                    event = self._trim_data(event)
                except ValueError:
                    self._print('No Data')
                    continue
                # Get event station angles and data etc.  for event and add to list
                (a_polarity_i, error_polarity_i, a1_amplitude_ratio_i, a2_amplitude_ratio_i, amplitude_ratio_i, percentage_error1_amplitude_ratio_i,
                 percentage_error2_amplitude_ratio_i, a_polarity_probability_i, polarity_probability_i, incorrect_polarity_probability_i, extension_data_i) = self._station_angles(event, i)
                a_polarity.append(a_polarity_i)
                error_polarity.append(error_polarity_i)
                a1_amplitude_ratio.append(a1_amplitude_ratio_i)
                a2_amplitude_ratio.append(a2_amplitude_ratio_i)
                amplitude_ratio.append(amplitude_ratio_i)
                percentage_error1_amplitude_ratio.append(percentage_error1_amplitude_ratio_i)
                percentage_error2_amplitude_ratio.append(percentage_error2_amplitude_ratio_i)
                a_polarity_probability.append(a_polarity_probability_i)
                polarity_probability.append(polarity_probability_i)
                incorrect_polarity_probability.append(incorrect_polarity_probability_i)
                extension_data.append(extension_data_i)
                # If running relative get relative data
                if self._relative:
                    a_relative_amplitude_i, relative_amplitude_i, percentage_error_relative_amplitude_i, relative_amplitude_stations_i = relative_amplitude_ratio_matrix(event, self.location_samples)
                    a_relative_amplitude.append(a_relative_amplitude_i)
                    relative_amplitude.append(relative_amplitude_i)
                    percentage_error_relative_amplitude.append(percentage_error_relative_amplitude_i)
                    relative_amplitude_stations.append(relative_amplitude_stations_i)
            except Exception:
                traceback.print_exc()
        self._print('\n\nJoint Inversion: '+str(self.number_events)+' Events\n--------\n')
        self._update_samples()
        self._set_algorithm(single=True, **self.kwargs)
        location_sample_size = 1
        if self.location_samples:
            location_sample_size = len(self.location_samples)
        try:
            # Run forward model using MultipleEventsForwardTask
            if self.pool:
                # initialise all tasks
                for i in range(self.pool.number_workers):
                    MTs, end = self._parse_job_result(False)
                    self.pool.custom_task(MultipleEventsForwardTask, MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio,
                                          percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio, a_polarity_probability, polarity_probability, a_relative_amplitude,
                                          relative_amplitude, percentage_error_relative_amplitude, relative_amplitude_stations, self.location_sample_multipliers, incorrect_polarity_probability, self.minimum_number_intersections,
                                          False, False, self._relative, location_sample_size, self._marginalise_relative, not self._relative_loop, extension_data)
            elif self._MPI:
                end = False
                # Carried out in each worker
                MTs, ignored_end = self._parse_job_result(False)
                result = MultipleEventsForwardTask(MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio,
                                                   percentage_error2_amplitude_ratio, a_polarity_probability, polarity_probability, a_relative_amplitude, relative_amplitude,
                                                   percentage_error_relative_amplitude, relative_amplitude_stations, self.location_sample_multipliers, incorrect_polarity_probability,
                                                   self.minimum_number_intersections, relative=self._relative, location_sample_size=location_sample_size,
                                                   marginalise_relative=self._marginalise_relative, combine=not self._relative_loop, extension_data=extension_data)()
            else:
                MTs, end = self._parse_job_result(False)
            while not end:
                if self.pool:
                    result = self.pool.result()
                    MTs, end = self._parse_job_result(result)
                    self.pool.custom_task(MultipleEventsForwardTask, MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio,
                                          percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio, a_polarity_probability, polarity_probability,
                                          a_relative_amplitude, relative_amplitude, percentage_error_relative_amplitude, relative_amplitude_stations, self.location_sample_multipliers, incorrect_polarity_probability,
                                          self.minimum_number_intersections, False, False, self._relative, location_sample_size, self._marginalise_relative, not self._relative_loop, extension_data)
                elif self._MPI:
                    if not self.mpi_output:
                        items = self.comm.Gather(result, 0)
                        if self.comm.Get_rank() == 0:
                            end = False
                            for result in items:
                                MTs, end = self._parse_job_result(result)
                                if end:
                                    end = True
                        else:
                            end = None
                        # Broadcast end to all mpis
                        end = self.comm.bcast(end, root=0)
                        _iteration = self.algorithm.iteration
                        _start_time = False
                        try:
                            _start_time = self.algorithm.start_time
                        except Exception:
                            pass
                        # Get new random MTs
                        MTs, ignored_end = self._parse_job_result(False)
                        self.algorithm.iteration = _iteration
                        if _start_time:
                            self.algorithm.start_time = _start_time
                    else:
                        MTs, end = self._parse_job_result(result)
                        end = self.comm.bcast(end, root=0)
                    result = MultipleEventsForwardTask(MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio,
                                                       percentage_error2_amplitude_ratio, a_polarity_probability, polarity_probability, a_relative_amplitude, relative_amplitude,
                                                       percentage_error_relative_amplitude, relative_amplitude_stations, self.location_sample_multipliers, incorrect_polarity_probability,
                                                       self.minimum_number_intersections, relative=self._relative, location_sample_size=location_sample_size,
                                                       marginalise_relative=self._marginalise_relative, combine=not self._relative_loop, extension_data=extension_data)()
                else:
                    result = MultipleEventsForwardTask(MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio,
                                                       percentage_error2_amplitude_ratio, a_polarity_probability, polarity_probability, a_relative_amplitude, relative_amplitude,
                                                       percentage_error_relative_amplitude, relative_amplitude_stations, self.location_sample_multipliers, incorrect_polarity_probability,
                                                       self.minimum_number_intersections, relative=self._relative, location_sample_size=location_sample_size,
                                                       marginalise_relative=self._marginalise_relative, combine=not self._relative_loop, extension_data=extension_data)()
                    MTs, end = self._parse_job_result(result)
            # Get all pool results
            if self.pool:
                results = self.pool.all_results()
                for result in results:
                    if result:
                        MTs, end = self._parse_job_result(result)
            try:
                self._print('Inversion completed\n\t'+'Elapsed time: '+str(time.time()-self.algorithm.start_time).split('.')[0]+' seconds\n\t'+str(self.algorithm.pdf_sample.n)+' samples evaluated\n\t'+str(len(self.algorithm.pdf_sample.nonzero()))+' non-zero samples\n\t'+'{:f}'.format((float(len(self.algorithm.pdf_sample.nonzero()))/float(self.algorithm.pdf_sample.n))*100)+'%')
            except ZeroDivisionError:
                self._print('Inversion completed\n\t'+'Elapsed time: '+str(time.time()-self.algorithm.start_time).split('.')[0]+' seconds\n\t'+str(self.algorithm.pdf_sample.n)+' samples evaluated\n\t'+str(len(self.algorithm.pdf_sample.nonzero()))+' non-zero samples\n\t'+'{:f}'.format(0)+'%')
            self._print('Algorithm max value: '+self.algorithm.max_value())
            # Handle mpi output
            output_format = self.output_format
            if self._MPI and self.mpi_output:
                output_format = 'hyp'
                results_format = self.results_format
                self.results_format = 'hyp'
                normalise = self.normalise
                self.normalise = False
            # Output results
            if (self._MPI and not self.mpi_output and self.comm.Get_rank() == 0) or (self._MPI and self.mpi_output) or not self._MPI:
                try:
                    print(fid)
                    self.output(self.data, fid, a_polarity=a_polarity, error_polarity=error_polarity, a1_amplitude_ratio=a1_amplitude_ratio, a2_amplitude_ratio=a2_amplitude_ratio, amplitude_ratio=amplitude_ratio, percentage_error1_amplitude_ratio=percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio=percentage_error2_amplitude_ratio, output_format=output_format)
                except Exception:
                    self._print('Output Error')
                    traceback.print_exc()

            # Output mpi output pkl if self.mpi_output
            if self._MPI and self.mpi_output:
                self.normalise = normalise
                self.results_format = results_format
                fids = self.comm.gather(fid, 0)
                if isinstance(fids, list):
                    self._print('All moment tensor samples outputted: '+str(len(fids))+' files')
                else:
                    self._print('Error with fids '+str(fids)+' type:'+str(type(fids)))
                if self.comm.Get_rank() == 0:
                    with open(os.path.splitext(fid)[0]+'.pkl', 'wb') as f:
                        pickle.dump({'fids': fids, 'event_data': event}, f)
        except Exception:
            traceback.print_exc()

    def forward(self):
        """
        Runs event forward model using the arguments when the Inversion object was initialised.

        Depending on the algorithm selection uses either random sampling or Markov chain Monte Carlo sampling - for more information on the different algorithms see MTfit.algorithms documentation.
        """
        source_type = 'MT'
        if self.dc:
            source_type = 'DC'
        t0 = time.time()
        # Choose type of forward model to run (McMC or Random sampling, single event or joint events)
        if self.McMC:
            if self.multiple_events:
                self._mcmc_multiple_forward(source_type)
            else:
                self._mcmc_forward(source_type)
        else:
            if self.multiple_events:
                self._random_sampling_multiple_forward(source_type)
            else:
                self._random_sampling_forward(source_type)
        # Print run time
        t1 = time.time()
        self._print('\n--------\n{} inversion complete, elapsed time: {}'.format(source_type, t1-t0))
        self._close_pool()

    def _parse_job_result(self, job_result=False):
        """
        Parse job results

        Checks and parses the job results and either iterates or re-initialises.

        Args
            job_result:[False] result from foward task.
        """
        # Check job result with algorithm
        if job_result and not isinstance(job_result, Exception):
            # in loop so process and return new results
            # Process result
            return self.algorithm.iterate(job_result)
        else:
            return self.algorithm.initialise()

    def _trim_data(self, data):
        """
        Remove unrequired data

        Checks and removes un-required data (not in inversion_options if set) from the event data dictionary.

        Args
            data: Event data dictionary.

        Returns
            data: Trimmed event data dictionary.

        Raises
            ValueError: No remaining data for selected inversion options.

        """
        # Check inverion options
        if self.inversion_options:
            for key in list(data.keys()):
                    # Remove data_type not in inversion options
                    if key not in self.inversion_options and key not in ['UID']:
                        data.pop(key)
            if not len(data) or list(data.keys()) == ['UID']:
                raise ValueError('No data remaining for the selected inversion options:\n\t'+'\n\t'.join(self.inversion_options))
            else:
                return data
        else:
            # use all options
            return data


#
# Station angle coefficient functions and helpers
#


def polarity_matrix(data, location_samples=False):
    """
    Generates the polarity observation matrices from the data.

    Args:
        data: Event data dictionary
        location_samples:[False] Station angle scatter samples.

    Returns
        a,error
        a: numpy matrix of station angle MT coefficients multiplied by the polarity.
        error: numpy matrix of fractional error.
    """
    a = False
    _a = False
    error = False
    incorrect_polarity_prob = 0
    # Check location samples
    if location_samples:
        original_samples = [u.copy() for u in location_samples]
    # Get polarity data
    for key in sorted([u for u in data.keys() if 'polarity' in u.lower() and 'prob' not in u.lower()]):
        mode = key.lower().split('polarity')[0]
        if location_samples:
            location_samples = [u.copy() for u in original_samples]
            selected_stations = sorted(list(set(location_samples[0]['Name']) & set(data[key]['Stations']['Name'])))
            n_stations = len(selected_stations)
            indices = [location_samples[0]['Name'].index(u) for u in selected_stations]
            angles = np.zeros((n_stations, len(location_samples), 6))
            for i, sample in enumerate(location_samples):
                sample['Name'] = operator.itemgetter(*indices)(sample['Name'])
                sample['TakeOffAngle'] = sample['TakeOffAngle'][indices]
                sample['Azimuth'] = sample['Azimuth'][indices]
                angles[:, i, :] = station_angles(sample, mode)
            angles = np.array(angles)
            # Fix for station order change with location samples
            indices = [data[key]['Stations']['Name'].index(u) for u in selected_stations]
            # Get measured and errors from location sample indices
            measured = data[key]['Measured'][indices]
            _error = np.array(data[key]['Error'][indices]).flatten()
            # Try to get incorrect polarity probability
            if data[key].__contains__('IncorrectPolarityProbability'):
                _incorrect_polarity_prob = np.array(data[key]['IncorrectPolarityProbability'][indices]).flatten()
            else:
                _incorrect_polarity_prob = np.array([0])
        else:
            # Otherwise get angles and measured directly
            n_stations = np.prod(data[key]['Stations']['TakeOffAngle'].shape)
            angles = np.zeros((n_stations, 1, 6))
            angles[:, 0, :] = station_angles(data[key]['Stations'], mode)
            measured = data[key]['Measured']
            _error = np.array(data[key]['Error']).flatten()
            if data[key].__contains__('IncorrectPolarityProbability'):
                _incorrect_polarity_prob = np.array(data[key]['IncorrectPolarityProbability']).flatten()
            else:
                _incorrect_polarity_prob = np.array([0])
        # If angles have multiple dimensions (location samples),  expand the measurements
        if angles.ndim > 2:
            measured = np.array(measured)
            measured = np.expand_dims(measured, 1)
        # Set or append to outputs
        if not _a:
            _a = True
            a = np.multiply(angles, measured)
            error = _error
            incorrect_polarity_prob = _incorrect_polarity_prob
        else:
            a = np.append(a, np.multiply(angles, measured), 0)
            error = np.append(error, _error, 0)
            if len(_incorrect_polarity_prob) != n_stations:
                # Make sure this is the same length as the data as we are appending it
                _incorrect_polarity_prob = np.kron(_incorrect_polarity_prob, np.ones(n_stations))
            incorrect_polarity_prob = np.append(incorrect_polarity_prob, _incorrect_polarity_prob, 0)
    # Set incorrect polarity prob if default
    if np.sum(np.abs(incorrect_polarity_prob)) == 0:
        incorrect_polarity_prob = 0
    return a, error, incorrect_polarity_prob


def polarity_probability_matrix(data, location_samples=False):
    """
    Generates the polarity PDF observation matrices from the data.

    Args:
        data: Event data dictionary
        location_samples:[False] Station angle scatter samples.

    Returns
        a,probability,error
        a: numpy matrix of station angle MT coefficients.
        probability: numpy matrix of polarity probabilities.
        error: numpy matrix of fractional error.
    """
    a = False
    positive_probability = False
    negative_probability = False
    _a = False
    incorrect_polarity_prob = 0
    # Check location samples and get sample
    if location_samples:
        original_samples = [u.copy() for u in location_samples]
    # Loop over polarity prob data
    for key in sorted([u for u in data.keys() if 'polarity' in u.lower() and 'prob' in u.lower()]):
        mode = key.lower().split('polarity')[0]
        # If location samples, then select stations appropriately
        if location_samples:
            location_samples = [u.copy() for u in original_samples]
            selected_stations = sorted(list(set(location_samples[0]['Name']) & set(data[key]['Stations']['Name'])))
            indices = [location_samples[0]['Name'].index(u) for u in selected_stations]
            angles = np.zeros((len(selected_stations), len(location_samples), 6))
            for i, sample in enumerate(location_samples):
                sample['Name'] = operator.itemgetter(*indices)(sample['Name'])
                sample['TakeOffAngle'] = sample['TakeOffAngle'][indices]
                sample['Azimuth'] = sample['Azimuth'][indices]
                angles[:, i, :] = station_angles(sample, mode)
            angles = np.array(angles)
            indices = [data[key]['Stations']['Name'].index(u) for u in selected_stations]
            measured = data[key]['Measured'][indices]
            if data[key].__contains__('IncorrectPolarityProbability'):
                _incorrect_polarity_prob = np.array(data[key]['IncorrectPolarityProbability'][indices]).flatten()
            else:
                _incorrect_polarity_prob = np.array([0])
        # Otherwise get angles and measeured
        else:
            angles = np.zeros((np.prod(data[key]['Stations']['TakeOffAngle'].shape), 1, 6))
            angles[:, 0, :] = station_angles(data[key]['Stations'], mode)
            measured = data[key]['Measured']
            if data[key].__contains__('IncorrectPolarityProbability'):
                _incorrect_polarity_prob = np.array(data[key]['IncorrectPolarityProbability']).flatten()
            else:
                _incorrect_polarity_prob = np.array([0])
        # Set or append to outputs
        if not _a:
            _a = True
            a = angles
            positive_probability = np.array(measured[:, 0]).flatten()
            negative_probability = np.array(measured[:, 1]).flatten()
            incorrect_polarity_prob = _incorrect_polarity_prob
        else:
            a = np.append(a, angles, 0)
            positive_probability = np.append(positive_probability, np.array(measured[:, 0]).flatten())
            negative_probability = np.append(positive_probability, np.array(measured[:, 1]).flatten())
            incorrect_polarity_prob = np.append(incorrect_polarity_prob, _incorrect_polarity_prob, 0)
    # Set incorrect polarity prob if default
    if np.sum(np.abs(incorrect_polarity_prob)) == 0:
        incorrect_polarity_prob = 0
    return a, (positive_probability, negative_probability), incorrect_polarity_prob


def amplitude_ratio_matrix(data, location_samples=False):
    """
    Generates the amplitude ratio observation matrices from the data.

    Args:
        data: Event data dictionary
        location_samples:[False] Station angle scatter samples.

    Returns
        a1,a2,amplitude_ratio,percentage_error1,percentage_error2
        a1: numpy matrix of station angle MT coefficients for ratio numerator.
        a2: numpy matrix of station angle MT coefficients for ratio denominator.
        amplitude_ratio: numpy matrix of observed ratios.
        percentage_error1: numpy matrix of ratio numerator percentage error.
        percentage_error2: numpy matrix of ratio denominator percentage error.
    """
    a1 = False
    a2 = False
    a = False
    percentage_error1 = False
    percentage_error2 = False
    amplitude_ratio = False
    if location_samples:
        original_samples = [u.copy() for u in location_samples]
    # Loop over data and get amplitude ratio data
    for key in sorted([u for u in data.keys() if 'amplituderatio' in u.lower() or 'amplitude_ratio' in u.lower()]):
        phase = key.replace('_', '').lower().split('amplituderatio')[0]
        phase = phase.rstrip('rms')
        phase = phase.rstrip('q')
        phase.replace('_', '')
        # If location samples, get intersection stations and data
        if location_samples:
            location_samples = [u.copy() for u in original_samples]
            selected_stations = sorted(list(set(location_samples[0]['Name']) & set(data[key]['Stations']['Name'])))
            indices = [location_samples[0]['Name'].index(u) for u in selected_stations]
            angles1 = np.zeros((len(selected_stations), len(location_samples), 6))
            angles2 = np.zeros((len(selected_stations), len(location_samples), 6))
            for i, sample in enumerate(location_samples):
                sample['Name'] = operator.itemgetter(*indices)(sample['Name'])
                sample['TakeOffAngle'] = sample['TakeOffAngle'][indices]
                sample['Azimuth'] = sample['Azimuth'][indices]
                angles1[:, i, :] = station_angles(sample, phase.split('/')[0])
                angles2[:, i, :] = station_angles(sample, phase.split('/')[1])
            angles1 = np.array(angles1)
            angles2 = np.array(angles2)
            angles = [angles1, angles2]
            indices = [data[key]['Stations']['Name'].index(u) for u in selected_stations]
            _measured = data[key]['Measured'][indices]
            error = data[key]['Error'][indices]
        # Otherwise get angles and measurements
        else:
            angles1 = np.zeros((np.prod(data[key]['Stations']['TakeOffAngle'].shape), 1, 6))
            angles2 = np.zeros((np.prod(data[key]['Stations']['TakeOffAngle'].shape), 1, 6))
            angles1[:, 0, :] = station_angles(data[key]['Stations'], phase.split('/')[0])
            angles2[:, 0, :] = station_angles(data[key]['Stations'], phase.split('/')[1])
            angles1 = np.array(angles1)
            angles2 = np.array(angles2)
            angles = [angles1, angles2]
            _measured = data[key]['Measured']
            error = data[key]['Error']
        # Set or append to outputs
        if not a:
            a = True
            a1 = angles[0]
            a2 = angles[1]
            amplitude_ratio = np.array(np.abs(np.divide(_measured[:, 0], _measured[:, 1]))).flatten()
            percentage_error1 = np.array(np.divide(error[:, 0], np.abs(_measured[:, 0]))).flatten()
            percentage_error2 = np.array(np.divide(error[:, 1], np.abs(_measured[:, 1]))).flatten()
        else:
            a1 = np.append(a1, angles[0], 0)
            a2 = np.append(a2, angles[1], 0)
            percentage_error1 = np.append(percentage_error1, np.array(np.divide(error[:, 0], np.abs(_measured[:, 0]))).flatten(), 0)
            percentage_error2 = np.append(percentage_error2, np.array(np.divide(error[:, 1], np.abs(_measured[:, 1]))).flatten(), 0)
            amplitude_ratio = np.append(amplitude_ratio, np.array(np.abs(np.divide(_measured[:, 0], _measured[:, 1]))).flatten(), 0)
    return a1, a2, amplitude_ratio, percentage_error1, percentage_error2


def relative_amplitude_ratio_matrix(data, location_samples=False):
    """
    Generates the relative amplitude ratio observation matrices from the data.

    Args:
        data: Event data dictionary
        location_samples:[False] Station angle scatter samples.

    Returns
        a_relative_amplitude,relative_amplitude,percentage_error
        a_relative_amplitude: numpy matrix of station angle MT coefficients for ratio numerator.
        relative_amplitude: numpy matrix of observations.
        percentage_error: numpy matrix of relative amplitude percentage error.
    """
    a_relative_amplitude = False
    a = False
    percentage_error = False
    relative_amplitude_stations = []
    relative_amplitude = False
    if location_samples:
        original_samples = [u.copy() for u in location_samples]
    # Loop over data and get amplitude data
    for key in sorted([u for u in data.keys() if 'amplitude' in u.lower() and 'ratio' not in u.lower()]):
        phase = key.replace('_', '').lower().split('amplitude')[0]
        phase = phase.rstrip('q')
        phase = phase.rstrip('rms')
        phase = phase.rstrip('q')
        # If location samples, get intersection stations and data
        if location_samples:
            location_samples = [u.copy() for u in original_samples]
            selected_stations = sorted(list(set(location_samples[0]['Name']) & set(data[key]['Stations']['Name'])))
            indices = [location_samples[0]['Name'].index(u) for u in selected_stations]
            angles = np.zeros((len(selected_stations), len(location_samples), 6))
            for i, sample in enumerate(location_samples):
                sample['Name'] = operator.itemgetter(*indices)(sample['Name'])
                sample['TakeOffAngle'] = sample['TakeOffAngle'][indices]
                sample['Azimuth'] = sample['Azimuth'][indices]
                angles[:, i, :] = station_angles(sample, phase)
            angles = np.array(angles)
            relative_amplitude_stations.extend(selected_stations)
            indices = [data[key]['Stations']['Name'].index(u) for u in selected_stations]
            measured = data[key]['Measured'][indices]
            error = data[key]['Error'][indices]
        # Otherwise get angles and measurements
        else:
            angles = np.zeros((np.prod(data[key]['Stations']['TakeOffAngle'].shape), 1, 6))
            angles[:, 0, :] = station_angles(data[key]['Stations'], phase)
            relative_amplitude_stations.extend(data[key]['Stations']['Name'])
            measured = np.array(data[key]['Measured']).flatten()
            error = np.array(data[key]['Error']).flatten()
        # Set or append to outputs
        if not a:
            a = True
            a_relative_amplitude = angles
            relative_amplitude = np.array(np.abs(measured)).flatten()
            percentage_error = np.array(np.divide(error, np.abs(measured))).flatten()
        else:
            a_relative_amplitude = np.append(a_relative_amplitude, angles, 0)
            relative_amplitude = np.array(np.abs(np.append(relative_amplitude, measured, 0))).flatten()
            percentage_error = np.array(np.abs(np.append(percentage_error, np.divide(error, measured), 0))).flatten()
    if not len(relative_amplitude_stations):
        relative_amplitude_stations = False
    return a_relative_amplitude, relative_amplitude, percentage_error, relative_amplitude_stations


def _intersect_stations(relative_amplitude_stations_i, relative_amplitude_stations_j, a_relative_amplitude_ratio_i, a_relative_amplitude_ratio_j,
                        relative_amplitude_i, relative_amplitude_j, percentage_error_relative_amplitude_i, percentage_error_relative_amplitude_j):
    """
    Determine the intersection of the two lists of stations, and then returns the input angles and amplitudes etc for the intersection.

    Args
        relative_amplitude_stations_i: List of stations corresponding to the data from event_i
        relative_amplitude_stations_j: List of stations corresponding to the data from event_j
        a_relative_amplitude_ratio_i: numpy matrix of station GFs from event_i
        a_relative_amplitude_ratio_j: numpy matrix of station GFs from event_j
        relative_amplitude_i: numpy matrix of amplitude observations from event_i
        relative_amplitude_j: numpy matrix of amplitude observations from event_j
        percentage_error_relative_amplitude_i: numpy matrix of percentage errors from event_i
        percentage_error_relative_amplitude_j: numpy matrix of percentage errors from event_j

    Returns
        a_relative_amplitude_ratio_i: numpy matrix of station GFs for event_i and event_j intersection corresponding to event_i
        a_relative_amplitude_ratio_j: numpy matrix of station GFs for event_i and event_j intersection corresponding to event_j
        relative_amplitude_i: numpy matrix of amplitude observations for event_i and event_j intersection corresponding to event_i
        relative_amplitude_j: numpy matrix of amplitude observations for event_i and event_j intersection corresponding to event_j
        percentage_error_relative_amplitude_i: numpy matrix of percentage errors for event_i and event_j intersection corresponding to event_i
        percentage_error_relative_amplitude_j: numpy matrix of percentage errors for event_i and event_j intersection corresponding to event_j
        number_intersections: integer number of intersections between event_i and event_j.

    """
    # Get intersection of stations between events
    selected_stations = list(set(relative_amplitude_stations_i) & set(relative_amplitude_stations_j))
    # Get indices for each event
    indices_i = [i for i, u in enumerate(relative_amplitude_stations_i) if u in selected_stations]
    indices_j = [i for i, u in enumerate(relative_amplitude_stations_j) if u in selected_stations]
    # Get parameters for event i
    a_relative_amplitude_ratio_i = a_relative_amplitude_ratio_i[indices_i]
    relative_amplitude_i = relative_amplitude_i[indices_i]
    percentage_error_relative_amplitude_i = percentage_error_relative_amplitude_i[indices_i]
    # Get parameters for event j
    a_relative_amplitude_ratio_j = a_relative_amplitude_ratio_j[indices_j]
    relative_amplitude_j = relative_amplitude_j[indices_j]
    percentage_error_relative_amplitude_j = percentage_error_relative_amplitude_j[indices_j]
    return (np.ascontiguousarray(a_relative_amplitude_ratio_i), np.ascontiguousarray(a_relative_amplitude_ratio_j),
            np.ascontiguousarray(relative_amplitude_i), np.ascontiguousarray(relative_amplitude_j),
            np.ascontiguousarray(percentage_error_relative_amplitude_i), np.ascontiguousarray(percentage_error_relative_amplitude_j),
            len(indices_i))


def _intersect_stations_extension_data(extension_data_i, extension_data_j):
    """
    Determines the intersection of the two lists of stations, and then returns the input angles and amplitudes etc for the intersection.

    Args
        extension_data_i: extension data  from event_i
        extension_data_j: extension data  from event_j

    Returns
        extension_data_i: extension data for event_i and event_j intersection corresponding to event_i
        extension_data_j: extension data  for event_i and event_j intersection corresponding to event_j
        number_intersections: integer number of intersections between event_i and event_j.

    """
    # Get intersection of stations between events
    stations_i = extension_data_i['stations']
    stations_j = extension_data_j['stations']
    selected_stations = list(set(stations_i) & set(stations_j))
    # Get indices for each event
    indices_i = [i for i, u in enumerate(stations_i) if u in selected_stations]
    indices_j = [i for i, u in enumerate(stations_j) if u in selected_stations]
    # Get parameters for event i
    for key in extension_data_i:
        extension_data_i[key] = np.ascontiguousarray(extension_data_i[key][indices_i])
    for key in extension_data_j:
        extension_data_j[key] = np.ascontiguousarray(extension_data_j[key][indices_j])
    # Get parameters for event j
    return extension_data_i, extension_data_j, len(indices_i)


def station_angles(stations, phase, radians=False):
    """
    Calculates the station MT coefficients from the station angles. ###############
    TakeOffAngle 0 down (as this is positive z-axis in NED system.

    Args
        stations: station dictionary (format is the station component of the data dictionary (MTfit.inversion.Inversion docstrings))
        phase: 'P','SH','SV' - the component for which to calculate the station angles, can be a ratio separated with a \.
        radians:[False] Boolean flag to set radians true or false

    Returns
        station angles MT coefficients.
    """
    # Get azimuth and takeoff angles
    try:
        azimuth = np.matrix(stations['Azimuth'])
        takeoff_angle = np.matrix(stations['TakeOffAngle'])
    except Exception:
        azimuth = stations[0]  # stations is angle tuple
        takeoff_angle = stations[1]
    # Transpose to correct orientation if necessary
    if azimuth.shape[0] < azimuth.shape[1]:
        azimuth = azimuth.T
    if takeoff_angle.shape[0] < takeoff_angle.shape[1]:
        takeoff_angle = takeoff_angle.T
    # Radian conversion if not radians
    if not radians:
        azimuth = azimuth*np.pi/180
    if not radians:
        takeoff_angle = takeoff_angle*np.pi/180
    # Set arrays
    azimuth = np.array(azimuth)
    takeoff_angle = np.array(takeoff_angle)
    # Get phase
    phase = phase.lower().rstrip('q')
    # Calculate angles
    if phase.lower() == 'p':
        return np.matrix(np.array([np.cos(azimuth)*np.cos(azimuth)*np.sin(takeoff_angle)*np.sin(takeoff_angle),
                         np.sin(azimuth)*np.sin(azimuth)*np.sin(takeoff_angle)*np.sin(takeoff_angle),
                         np.cos(takeoff_angle)*np.cos(takeoff_angle),
                         (np.sqrt(2))*np.sin(azimuth)*np.cos(azimuth)*np.sin(takeoff_angle)*np.sin(takeoff_angle),
                         (np.sqrt(2))*np.cos(azimuth)*np.cos(takeoff_angle)*np.sin(takeoff_angle),
                         (np.sqrt(2))*np.sin(azimuth)*np.cos(takeoff_angle)*np.sin(takeoff_angle)]).T)
    elif phase.lower() == 'sh':
        return np.matrix(np.array([-np.sin(azimuth)*np.cos(azimuth)*np.sin(takeoff_angle),
                         np.sin(azimuth)*np.cos(azimuth)*np.sin(takeoff_angle),
                         0*azimuth,
                         (1/np.sqrt(2))*np.cos(2*azimuth)*np.sin(takeoff_angle),
                         -(1/np.sqrt(2))*np.sin(azimuth)*np.cos(takeoff_angle),
                         (1/np.sqrt(2))*np.cos(azimuth)*np.cos(takeoff_angle)]).T)
    elif phase.lower() == 'sv':
        return np.matrix(np.array([np.cos(azimuth)*np.cos(azimuth)*np.sin(takeoff_angle)*np.cos(takeoff_angle),
                         np.sin(azimuth)*np.sin(azimuth)*np.sin(takeoff_angle)*np.cos(takeoff_angle),
                         -np.sin(takeoff_angle)*np.cos(takeoff_angle),
                         np.sqrt(2)*np.cos(azimuth)*np.sin(azimuth)*np.sin(takeoff_angle)*np.cos(takeoff_angle),
                         (1/np.sqrt(2))*np.cos(azimuth)*np.cos(2*takeoff_angle),
                         (1/np.sqrt(2))*np.sin(azimuth)*np.cos(2*takeoff_angle)]).T)
    elif len(phase.split('/')) == 2:
        numerator = station_angles(stations, phase.split('/')[0], radians)
        denominator = station_angles(stations, phase.split('/')[1], radians)
        return (numerator, denominator)
    else:
        raise ValueError('{} phase not recognised.'.format(phase))


#
# Solution misfit and probability check functions
#


def _polarity_misfit_check(polarity, azimuth, takeoffangle, phase, mt):
    """
    Check polarity  misfit for a given MT (private function)

    Args
        polarity: np.array of polarity observations.
        azimuth: np.matrix of azimuths.
        takeoffangle: np.matrix of takeoffangles.
        phase: Phase of polarity measurements (P, SH, SV).
        mt: np.matrix moment tensor six vector Mxx Myy Mzz sqrt(2)*Mxy sqrt(2)*Mxz sqrt(2)*Myz.

    Returns
        indices of differing model and observed polarities or boolean false if no polarities.
    """
    # If polarity exists then check model polarities vs observed polarities
    if polarity != 0:
        a_station = station_angles({'Azimuth': [azimuth], 'TakeOffAngle': [takeoffangle]}, phase)
        model_polarity = float(a_station*mt)
        return model_polarity*polarity <= 0
    return False


def polarity_misfit_check(mt, data={}, data_file=False, inversion_options=['PPolarity']):
    """
    Checks polarity misfit for a given MT and data dictionary

    Args
        mt: np.matrix moment tensor six vector Mxx Myy Mzz sqrt(2)*Mxy sqrt(2)*Mxz sqrt(2)*Myz.
        data: data dictionary or list of dictionaries.
        data_file:  data input file (data dict or data_file must be provided).
        inversion_options: polarity inversion_options list .

    Returns
        list of names of misfitting stations
    """
    # Copy data
    copied_data = copy.copy(data)
    # Set up inverson
    inversion = Inversion(copied_data, data_file, False, parallel=False, inversion_options=inversion_options)
    # Loop over events
    for i, event in enumerate(inversion.data):
        # Check data
        if not event:
            print('No Data')
            continue
        # Set algorithm
        inversion._set_algorithm()
        try:
            event = inversion._trim_data(event)
        except ValueError:
            print('No Data')
            continue
        # Get observed polarities/inversion_options in data
        observed_polarity = False
        stations = []
        for i, key in enumerate(inversion_options):
            if key in event:
                if isinstance(observed_polarity, bool):
                    observed_polarity = event[key]['Measured'].copy()
                else:
                    observed_polarity = np.append(observed_polarity, event[key]['Measured'].copy(), 0)
                event[key]['Measured'] /= event[key]['Measured']
                stations.extend(event[key]['Stations']['Name'])
        # Create model polarities
        observed_polarity = np.array(observed_polarity).flatten()
        (a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio,
         percentage_error2_amplitude_ratio, a_polarity_probability, polarity_probability, incorrect_polarity_probability) = inversion._station_angles(event, i)
        model_polarity = np.sign(np.tensordot(a_polarity, mt, 1)).flatten()
        # Compare with observed polarities
        return sorted(list(np.array(stations)[(model_polarity*observed_polarity) < 0]))
    return []


def probability_mt_check(mt, data={}, data_file=False, location_pdf_file_path=False, inversion_options=['PPolarity', 'P/SHAmplitudeRatio', 'P/SVAmplitudeRatio'], RMS=False, Q=False):
    """
    Evaluate results for a given mt and data

    Args
        mt: np.matrix moment tensor six vector Mxx Myy Mzz sqrt(2)*Mxy sqrt(2)*Mxz sqrt(2)*Myz.
        data: data dictionary or list of dictionaries.
        data_file:  data input file (data dict or data_file must be provided).
        location_pdf_file_path: location PDF file path.
        inversion_options:['PPolarity','P/SHAmplitudeRatio','P/SVAmplitudeRatio'] polarity inversion_options list.
        RMS:[False] Use RMS amplitudes (if not set in inversion options).
        Q:[False] Use Q (if not set in inversion options).

    Returns
        results list
    """
    # Copy data
    copied_data = copy.copy(data)
    # Get amplitude ratio options
    for i, u in enumerate(inversion_options):
        if 'AmplitudeRatio' in u:
            if RMS and 'RMS' not in u:
                inversion_options[i] = u.split('AmplitudeRatio')[0]+'RMSAmplitudeRatio'
            if Q and 'Q' not in u:
                inversion_options[i] = u.split('AmplitudeRatio')[0]+'QAmplitudeRatio'
    inversion = Inversion(copied_data, data_file, location_pdf_file_path, parallel=False, inversion_options=inversion_options)
    results = []
    for i, event in enumerate(inversion.data):
        if not event:
            print('No Data')
            continue
        inversion._set_algorithm()
        try:
                event = inversion._trim_data(event)
        except ValueError:
            print('No Data')
            continue
        (a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio,
         percentage_error2_amplitude_ratio, a_polarity_probability, polarity_probability, incorrect_polarity_probability) = inversion._station_angles(event, i)
        results.append(ForwardTask(mt, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio,
                                   percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio, a_polarity_probability,
                                   polarity_probability, inversion.location_sample_multipliers, incorrect_polarity_probability, True)())
    return results


def polarity_mt_check(mt, data={}, data_file=False, location_pdf_file_path=False, inversion_options=['PPolarity', 'SHPolarity', 'SVPolarity']):
    """
    Evaluate polarity results for a given mt and data

    Args
        mt: np.matrix moment tensor six vector Mxx Myy Mzz sqrt(2)*Mxy sqrt(2)*Mxz sqrt(2)*Myz
        data: data dictionary or list of dictionaries
        data_file:  data input file (data dict or data_file must be provided)
        location_pdf_file_path: location PDF file path
        inversion_options:['PPolarity','P/SHAmplitudeRatio','P/SVAmplitudeRatio'] inversion_options list

    Returns
        results list
    """
    # Copy data
    copied_data = copy.copy(data)
    # Get polarity inversion options
    processed_inversion_options = []
    for i, u in enumerate(inversion_options):
        if 'polarity' in u.lower() and 'prob' not in u:
            processed_inversion_options.append(u)
    # Create inversion object
    inversion = Inversion(copied_data, data_file, location_pdf_file_path, parallel=False, inversion_options=processed_inversion_options)
    results = []
    for i, event in enumerate(inversion.data):
        if not event:
            print('No Data')
            continue
        # Set algorithms
        inversion._set_algorithm()
        try:
            event = inversion._trim_data(event)
        except ValueError:
            print('No Data')
            continue
        # Get angles and data
        (a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio,
         percentage_error2_amplitude_ratio, a_polarity_probability, polarity_probability, incorrect_polarity_probability) = inversion._station_angles(event, i)
        # Get results
        results.append(ForwardTask(mt, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio,
                                   percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio, a_polarity_probability,
                                   polarity_probability, inversion.location_sample_multipliers, incorrect_polarity_probability, True)())
    return results


def polarity_probability_mt_check(mt, data={}, data_file=False, location_pdf_file_path=False, inversion_options=['PPolarityProbability', 'SHPolarityProbability']):
    """
    Evaluate Polarity probability results for a given mt and data

    Args
        mt: np.matrix moment tensor six vector Mxx Myy Mzz sqrt(2)*Mxy sqrt(2)*Mxz sqrt(2)*Myz
        data: data dictionary or list of dictionaries
        data_file:  data input file (data dict or data_file must be provided)
        location_pdf_file_path: location PDF file path
        inversion_options:['PPolarityProbability','SHPolarityProbability'] polarity inversion_options list


    Returns
        results list
    """
    # Copy data
    copied_data = copy.copy(data)
    # Get polarity inversion options
    processed_inversion_options = []
    for i, u in enumerate(inversion_options):
        if 'polarity' in u.lower() and 'prob' in u:
            processed_inversion_options.append(u)
    # Create inversion object
    inversion = Inversion(copied_data, data_file, location_pdf_file_path, parallel=False, inversion_options=processed_inversion_options)
    results = []
    for i, event in enumerate(inversion.data):
        if not event:
            print('No Data')
            continue
        # Set algorithms
        inversion._set_algorithm()
        try:
            event = inversion._trim_data(event)
        except ValueError:
            print('No Data')
            continue
        # Get angles and data
        (a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio, percentage_error1_amplitude_ratio,
         percentage_error2_amplitude_ratio, a_polarity_probability, polarity_probability, incorrect_polarity_probability) = inversion._station_angles(event, i)
        # Get results
        results.append(ForwardTask(mt, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio,
                                   percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio, a_polarity_probability,
                                   polarity_probability, inversion.location_sample_multipliers, incorrect_polarity_probability, True)())
    return results


def amplitude_ratio_probability_mt_check(mt, data={}, data_file=False, location_pdf_file_path=False, inversion_options=['P/SHAmplitudeRatio', 'P/SVAmplitudeRatio'], RMS=False, Q=False):
    """
    Evaluate amplitude ratio results for a given mt and data

    Args
        mt: np.matrix moment tensor six vector Mxx Myy Mzz sqrt(2)*Mxy sqrt(2)*Mxz sqrt(2)*Myz.
        data: data dictionary or list of dictionaries.
        data_file:  data input file (data dict or data_file must be provided).
        location_pdf_file_path: location PDF file path.
        inversion_options:['P/SHAmplitudeRatio','P/SVAmplitudeRatio'] inversion_options list.
        RMS:[False] Use RMS amplitudes (if not set in inversion options).
        Q:[False] Use Q (if not set in inversion options).

    Returns
        results list
    """
    # Copy data
    copied_data = copy.copy(data)
    # Get polarity inversion options
    processed_inversion_options = []
    for i, u in enumerate(inversion_options):
        if 'amplituderatio' in u.lower():
            if RMS and 'rms' not in u.lower():
                u = u.split('AmplitudeRatio')[0]+'RMSAmplitudeRatio'
            if Q and 'q' not in u.lower():
                u = u.split('AmplitudeRatio')[0]+'QAmplitudeRatio'
            processed_inversion_options.append(u)
    # Create inversion object
    inversion = Inversion(copied_data, data_file, location_pdf_file_path, parallel=False, inversion_options=processed_inversion_options)
    results = []
    for i, event in enumerate(inversion.data):
        if not event:
            print('No Data')
            continue
        # Set algorithms
        inversion._set_algorithm()
        try:
                event = inversion._trim_data(event)
        except ValueError:
            print('No Data')
            continue
        # Get angles and data
        (a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio,
         percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio, a_polarity_probability,
         polarity_probability, incorrect_polarity_probability) = inversion._station_angles(event, i)
        # Get results
        results.append(ForwardTask(mt, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio,
                                   percentage_error1_amplitude_ratio, percentage_error2_amplitude_ratio, a_polarity_probability,
                                   polarity_probability, inversion.location_sample_multipliers, incorrect_polarity_probability, True)())
    return results


def _pop_station(data, key, station):
    """Remove station from data type (key)"""
    # Get index
    index = data[key]['Stations']['Name'].index(station)
    # Delete indices
    data[key]['Measured'] = np.delete(data[key]['Measured'], index, 0)
    data[key]['Error'] = np.delete(data[key]['Error'], index, 0)
    data[key]['Stations']['Azimuth'] = np.delete(data[key]['Stations']['Azimuth'], index, 0)
    data[key]['Stations']['TakeOffAngle'] = np.delete(data[key]['Stations']['TakeOffAngle'], index, 0)
    data[key]['Stations']['Name'].pop(index)
    return data


#
# Combine hyp event output from MPI
#


def combine_mpi_output(filepath='', output_format='matlab', parallel=False, mpi=False, binary_file_version=2, **kwargs):
    """
    Combine MPI output into single output

    MPI output is hyp and mt files for each mpi process, this function combines them into a single output.

    Args:
        filepath:[.] Input filepath.
        output_format:[matlab] Output format.
        parallel:[False] Boolean flag to run in parallel using job pool (Not MPI).
        mpi:[False] Boolean MPI flag.
    """
    # Check MPI
    if mpi:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        except Exception:
            mpi = False
    # Set filepath to folder
    if os.path.isdir(filepath):
        filepath += os.path.sep
    # Get hyp files
    hypfiles = glob.glob(filepath+'*.hyp')
    # Get UIDs
    UIDs = sorted(list(set(['.'.join(os.path.splitext(u)[0].split('.')[:-1]) for u in hypfiles if len('.'.join(os.path.splitext(u)[0].split('.')[:-1]))])))
    print('---------Combining Files------------')
    print('\t{}\n\n'.format('\n\t'.join(UIDs)))
    # Run combination
    # Run in parallel
    if parallel:
        job_pool = JobPool(task=CombineMpiOutputTask)
        number_workers = len(job_pool)
        if number_workers == 1:
            job_pool.close()
            parallel = False
    # Run using MPI
    if mpi:
        # split by workers
        sample = len(UIDs)/float(comm.Get_size())
        # Get the UIDS for this worker
        UIDs = UIDs[int(comm.Get_rank()*ceil(sample)):min(int((comm.Get_rank()+1)*ceil(sample)), len(UIDs)-1)]
    # Loop over UIDs and run combine task
    for uid in UIDs:
        if parallel:
            job_pool.task(uid, output_format, kwargs.get('results_format', 'full_pdf'), binary_file_version=binary_file_version)
        else:
            CombineMpiOutputTask(uid, output_format, kwargs.get('results_format', 'full_pdf'), binary_file_version=binary_file_version)()
    if parallel:
        job_pool.close()


#
# Location pdf parsing and binning
#


def bin_angle_coefficient_samples(a_polarity, a1_amplitude_ratio, a2_amplitude_ratio, a_polarity_prob, location_sample_multipliers, extension_data, epsilon=0):
    """
    Carry out the binning over the station ray path coefficent samples

    Unlike the bin_scatangle function, this accounts for variations in angle dependencies.

    Args
         a_polarity: np.array of polarity station coefficent data.
         a1_amplitude_ratio: np.array of amplitude ratio numerator station coefficent data.
         a2_amplitude_ratio: np.array of amplitude ratio denominator station coefficient data.
         a_polarity_prob: np.array of polarity prob stationcoefficient data.
         location_sample_multipliers: location sample probabilities.
         epsilon:[0] allowed difference for the samples to be binned together.

    Returns
        a_polarity,a1_amplitude_ratio,a2_amplitude_ratio,a_polarity_prob,multipliers
        np.arrays of station coefficents for each data type and the location sample multipliers.
    """
    if epsilon > 0:
        import time
        if cprobability:
            print('C')
            t0 = time.time()
            a_polarity, a1_amplitude_ratio, a2_amplitude_ratio, a_polarity_prob, extension_data, multipliers = cprobability.bin_angle_coefficient_samples(a_polarity, a1_amplitude_ratio,
                                                                                                                                                          a2_amplitude_ratio, a_polarity_prob,
                                                                                                                                                          location_sample_multipliers, extension_data,
                                                                                                                                                          epsilon)
            print('Ctime = {}'.format(time.time()-t0))
        else:
            print('Py')
            t0 = time.time()
            multipliers = []
            no_pop = np.ones((len(location_sample_multipliers),), dtype=bool)
            for i in range(len(location_sample_multipliers)):
                multiplier = location_sample_multipliers[i]
                for j in range(i, len(location_sample_multipliers)):
                    if no_pop[j]:
                        diff_polarity = 0
                        if not isinstance(a_polarity, bool):
                            diff_polarity = a_polarity[:, i, :]-a_polarity[:, j, :]
                        diff_polarity = 0
                        if not isinstance(a1_amplitude_ratio, bool):
                            diff_a1_amplitude_ratio = a1_amplitude_ratio[:, i, :]-a1_amplitude_ratio[:, j, :]
                        diff_polarity = 0
                        if not isinstance(a2_amplitude_ratio, bool):
                            diff_a2_amplitude_ratio = a2_amplitude_ratio[:, i, :]-a2_amplitude_ratio[:, j, :]
                        diff_polarity = 0
                        if not isinstance(a_polarity_prob, bool):
                            diff_polarity_prob = a_polarity_prob[:, i, :]-a_polarity_prob[:, j, :]
                        if np.max([np.max(np.abs(diff_polarity)), np.max(np.abs(diff_a1_amplitude_ratio)), np.max(np.abs(diff_a2_amplitude_ratio)), np.max(np.abs(diff_polarity_prob))]) < epsilon:
                            multiplier += location_sample_multipliers[j]
                            no_pop[j] = 0
                multipliers.append(multiplier)
            print('Pytime = {}'.format(time.time()-t0))
            if not isinstance(a_polarity, bool):
                a_polarity = a_polarity[:, no_pop, :]
            if not isinstance(a1_amplitude_ratio, bool):
                a1_amplitude_ratio = a1_amplitude_ratio[:, no_pop, :]
            if not isinstance(a2_amplitude_ratio, bool):
                a2_amplitude_ratio = a2_amplitude_ratio[:, no_pop, :]
            if not isinstance(a_polarity_prob, bool):
                a_polarity_prob = a_polarity_prob[:, no_pop, :]
            for key in extension_data.keys():
                for k in extension_data[key].keys():
                    if k[:2] == 'a_' or k[0] == 'a' and k[2] == '_':
                        extension_data[key][k] = extension_data[key][k][:, no_pop, :]
        print('Epsilon = {} reduced {} samples to {} records.'.format(epsilon, len(location_sample_multipliers), len(multipliers)))
    return a_polarity, a1_amplitude_ratio, a2_amplitude_ratio, a_polarity_prob, extension_data, multipliers
