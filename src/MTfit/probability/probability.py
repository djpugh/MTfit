"""
probability.py
**************

Probability module for MTfit, handles all probability calculations and
contains the base LnPDF class for acting on probabilities.

LnPDF object acts as a wrapper around a numpy matrix, allowing some
additional pdf specific operations.
"""


# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.


import warnings
import gc
import sys
import operator
import logging

import numpy as np
from scipy.stats import norm as gaussian
from scipy.stats import beta
from scipy.special import erf

from ..utilities import C_EXTENSION_FALLBACK_LOG_MSG

logger = logging.getLogger('MTfit.probability')


try:
    from . import cprobability
except ImportError:
    cprobability = None
except Exception:
    logger.exception('Error importing c extension')
    cprobability = None

# Set to prevent numpy warnings
np.seterr(divide='print', invalid='print')

# Set flags for running with/without the c library

# Flag for testing Cython based C library functions even if they would be skipped
# e.g. polarity_ln_pdf on windows due to missing functions in math.h
_C_LIB_TESTS = False

# Value of a very small non-zero number for handling zero error values
_SMALL_NUMBER = 0.000000000000000000000001

# TODO - tidy and refactor code to avoid duplication


def _6sphere_prior(g, d):
    """
    6-sphere prior

    Returns 1 as samples are generated directly from the prior

    Other priors can be used either by non-uniform sampling in the prior
    and using the prior as the correction term in the bayesian evidence
    calculation, or by leaving the prior as 1 and sampling from the prior
    """
    return 1.


def polarity_ln_pdf(a, mt, sigma, incorrect_polarity_probability=0.0, _use_c=None):
    """
    Calculate the probability of a positive polarity

    Calculates the probability of a positive polarity observation given
    a theoretical amplitude X and a fractional uncertainty sigma.

    The derivation of this pdf for the polarity observation can be seen in:
        Pugh, D. J., White, R. S., & Christie, P. A. F., 2016.
        A Bayesian method for microseismic source inversion,
        Geophysical Journal International , 206(2), 1009-1038.

    Args:
        a: np.array - 3 dimensional numpy array of station coefficients
        mt: np.array - 2 dimensional numpy array of moment tensor 6
                vector samples
        sigma: np.array - 1 dimensional array of fractional uncertainties.

    Optional Args:
        incorrect_polarity_probability: float (default=0) - probability
                of a receiver orientation error, flipping the polarity dist.

    Returns:
        np.array - probabilities for each moment tensor sample
    """
    # Check inputs and expcected dimensions
    if not isinstance(sigma, np.ndarray) or sigma.ndim != 1:
        raise TypeError('Variable: sigma is expected to be a one-dimensional numpy array')
    if not isinstance(a, np.ndarray) or a.ndim != 3:
        raise TypeError('Variable: a is expected to be a three-dimensional numpy array')
    if not isinstance(mt, np.ndarray) or mt.ndim != 2:
        raise TypeError('Variable: mt is expected to be a two-dimensional numpy array of moment tensor six vectors')
    # For fractional uncertainties that are zero make them really small
    # to avoid zero divison errors
    sigma[sigma == 0] = _SMALL_NUMBER
    completed = False
    if cprobability and (_use_c is None or _use_c):
        try:
            if sys.platform.startswith('win') and not _use_c:
                # Raise an exception due to windows C_LIB using VS2008 VC math.h
                # which has no erf
                raise Exception('Windows')
            # Run using C library
            # Handle incorrect polarity probability
            if isinstance(incorrect_polarity_probability, (float, int)):
                incorrect_polarity_probability = np.ones(sigma.shape)*incorrect_polarity_probability
            incorrect_polarity_probability = np.squeeze(incorrect_polarity_probability)
            if incorrect_polarity_probability.ndim != 1:
                raise TypeError('Variable: incorrect_polarity_probability is expected to be a one-dimensional numpy array or a float')
            # Check types and convert to the correct types in place
            if a.dtype != np.float64:
                a = a.astype(np.float64, copy=False)
            if mt.dtype != np.float64:
                mt = mt.astype(np.float64, copy=False)
            if sigma.dtype != np.float64:
                sigma = sigma.astype(np.float64, copy=False)
            if incorrect_polarity_probability.dtype != np.float64:
                incorrect_polarity_probability = incorrect_polarity_probability.astype(np.float64,
                                                                                       copy=False)
            # Make sure moment tensors are C contiguous
            if not mt.flags['C_CONTIGUOUS']:
                mt = np.ascontiguousarray(mt)
            # Calculate log of pdf
            ln_p = cprobability.polarity_ln_pdf(a, mt, sigma, incorrect_polarity_probability)
            completed = True
        except Exception as e:
            # Run using python
            # Testing C code
            logger.exception('Error running cython code')
            if _C_LIB_TESTS:
                raise e
    else:
        logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
    if not completed:
        # Check moment tensor shape is correct
        if mt.shape[0] != a.shape[-1]:
            mt = mt.transpose()
        # Calcuate Theoretical Amplitude for moment tensors
        X = np.tensordot(a, mt, 1)
        # Expand errors and incorrect_polarity_probability dimensions to be
        # expected
        while sigma.ndim < 3:
            sigma = np.expand_dims(sigma, 1)
        if not isinstance(incorrect_polarity_probability, (float, int)):
            incorrect_polarity_probability = np.array(incorrect_polarity_probability)
        if isinstance(incorrect_polarity_probability, np.ndarray):
            while incorrect_polarity_probability.ndim < 3:
                incorrect_polarity_probability = np.expand_dims(
                    incorrect_polarity_probability, 1)
        # Calculate probability
        ln_p = np.log((0.5 * (1 + erf(X / (np.sqrt(2) * sigma)))) *
                      (1 - incorrect_polarity_probability) +
                      (0.5 * (1 + erf(-X / (np.sqrt(2) * sigma)))) *
                      incorrect_polarity_probability)
        # Set NaNs to zero probability
        if isinstance(ln_p, np.ndarray):
            ln_p[np.isnan(ln_p)] = -np.inf
        # Combine probabilities
        try:
            ln_p = cprobability.ln_prod(ln_p)
        except Exception:
            if cprobability:
                logger.exception('Error running cython code')
            ln_p = np.sum(ln_p, 0)
    if isinstance(ln_p, np.ndarray):
        ln_p[np.isnan(ln_p)] = -np.inf
    return ln_p


def polarity_probability_ln_pdf(a, mt, positive_probability, negative_probability, incorrect_polarity_probability=0.0, _use_c=None):
    """
    Calculate the probability of a given theoretical amplitude giving an
    observed polarity probability.

    Calculates the probability of a polarity probability observation
    given a theoretical amplitude X.

    The derivation of this pdf for the polarity probability observation
    can be seen in:
        Pugh, D. J., White, R. S., & Christie, P. A. F., 2016.
        Automatic Bayesian polarity determination,
        Geophysical Journal International , 206(1), 275-291.

    Args
        a: np.array - 3 dimensional numpy array of station coefficients
        mt: np.array - 2 dimensional numpy array of moment tensor 6 vector
                samples
        positive_probability: float/np.array - the probability of a positive polarity
                observation. Needs to be same length as mt, or a scalar.
        negative_probability: float/np.array - the probability of a negative polarity
                observation. Needs to be same length as mt, or a scalar.

    Optional Args:
        incorrect_polarity_probability: float (default=0) - probability of a
                receiver orientation error, flipping the polarity dist.

    Returns
        float/np.array -  probabilities for each moment tensor sample
    """
    # Check inputs and expcected dimensions
    if isinstance(positive_probability, (float, int)):
        positive_probability = np.array(positive_probability)
    if isinstance(negative_probability, (float, int)):
        negative_probability = np.array(negative_probability)
    if not isinstance(positive_probability, np.ndarray) or positive_probability.ndim != 1:
        raise TypeError(
            'Variable: positive_probability is expected to be a one-dimensional numpy array')
    if not isinstance(negative_probability, np.ndarray) or negative_probability.ndim != 1:
        raise TypeError(
            'Variable: negative_probability is expected to be a one-dimensional numpy array')
    if not isinstance(a, np.ndarray) or a.ndim != 3:
        raise TypeError('Variable: a is expected to be a three-dimensional numpy array')
    if not isinstance(mt, np.ndarray) or mt.ndim != 2:
        raise TypeError('Variable: mt is expected to be a two-dimensional numpy array of moment tensor six vectors')
    completed = False
    if cprobability and (_use_c is None or _use_c):
        try:
            # Run using C library
            # Handle incorrect polarity probability
            if isinstance(incorrect_polarity_probability, (float, int)):
                incorrect_polarity_probability = np.ones(positive_probability.shape)*incorrect_polarity_probability
            incorrect_polarity_probability = np.squeeze(incorrect_polarity_probability)
            if incorrect_polarity_probability.ndim != 1:
                raise TypeError('Variable: incorrect_polarity_probability is expected to be a one-dimensional numpy array or a float')
            # Check types and convert to the correct types in place
            if a.dtype != np.float64:
                a = a.astype(np.float64, copy=False)
            if mt.dtype != np.float64:
                mt = mt.astype(np.float64, copy=False)
            if incorrect_polarity_probability.dtype != np.float64:
                incorrect_polarity_probability = incorrect_polarity_probability.astype(np.float64,
                                                                                       copy=False)
            if positive_probability.dtype != np.float64:
                positive_probability = positive_probability.astype(np.float64, copy=False)
            if negative_probability.dtype != np.float64:
                negative_probability = negative_probability.astype(np.float64, copy=False)
            # Make sure moment tensors are C contiguous
            if not mt.flags['C_CONTIGUOUS']:
                mt = np.ascontiguousarray(mt)
            # Calculate log of pdf
            ln_p = cprobability.polarity_probability_ln_pdf(a, mt, positive_probability, negative_probability,
                                                            incorrect_polarity_probability)
            completed = True
        except Exception as e:
            # Run using python
            # Testing C code
            logger.exception('Error running cython code')
            if _C_LIB_TESTS:
                raise e
    else:
        logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
    if not completed:
        if mt.shape[0] != a.shape[-1]:
            mt = mt.transpose()
        # Calculate theoretical amplitude
        X = np.tensordot(a, mt, 1)
        # Expand errors and incorrect_polarity_probability dimensions to be
        # expected
        while positive_probability.ndim < 3:
            positive_probability = np.expand_dims(positive_probability, 1)
        while negative_probability.ndim < 3:
            negative_probability = np.expand_dims(negative_probability, 1)
        if not isinstance(incorrect_polarity_probability, (float, int)):
            incorrect_polarity_probability = np.array(
                incorrect_polarity_probability)
        if isinstance(incorrect_polarity_probability, np.ndarray):
            while incorrect_polarity_probability.ndim < 3:
                incorrect_polarity_probability = np.expand_dims(
                    incorrect_polarity_probability, 1)
        ln_p = np.log(
            ((heaviside(X) * positive_probability) + (heaviside(-X) * negative_probability)) *
            (1-incorrect_polarity_probability) +
            ((heaviside(X) * negative_probability) + (heaviside(-X) * positive_probability)) *
            incorrect_polarity_probability)
        # Set NaNs to zero probability
        if isinstance(ln_p, np.ndarray):
            ln_p[np.isnan(ln_p)] = -np.inf
        # Combine probabilities
        try:
            ln_p = cprobability.ln_prod(ln_p)
        except Exception:
            ln_p = np.sum(ln_p, 0)
    if isinstance(ln_p, np.ndarray):
        ln_p[np.isnan(ln_p)] = -np.inf
    return ln_p


def amplitude_ratio_ln_pdf(ratio, mt, a_x, a_y, percentage_error_x, percentage_error_y, _use_c=None):
    """
    Calculate the probability of a given theoretical amplitude ratio giving
    an observed ratio

    Calculates the probability of an amplitude ratio observation given a
    theoretical amplitude ratio and uncertainties on the numerator and
    denominator

    The derivation of this pdf for the amplitude ratio observation can be
    seen in:
        Pugh, D. J., White, R. S., & Christie, P. A. F., 2016.
        A Bayesian method for microseismic source inversion,
        Geophysical Journal International , 206(2), 1009-1038.

    Args
        ratio: np.array - 1 dimensional numpy array of observed ratio
                values
        mt: np.array - 2 dimensional numpy array of moment tensor 6
                vector samples
        a_x: np.array - 3 dimensional numpy array of station coefficients
                for the numerator
        a_y: np.array - 3 dimensional numpy array of station coefficients
                for the denominator
        percentage_error_x: np.array - 1 dimensional numpy array of
                percentage errors for the numerator
        percentage_error_y: np.array - 1 dimensional numpy array of
                percentage errors for the denominator

    Returns
        float/np.array -  probabilities for each moment tensor sample
    """
    # Check inputs and expcected dimensions
    if not isinstance(a_x, np.ndarray) or a_x.ndim != 3:
        raise TypeError('Variable a_x is expected to be a three-dimensional numpy array')
    if not isinstance(a_y, np.ndarray) or a_y.ndim != 3:
        raise TypeError('Variable: a_y is expected to be a three-dimensional numpy array')
    if not isinstance(mt, np.ndarray) or mt.ndim != 2:
        raise TypeError(
            'Variable: mt is expected to be a two-dimensional numpy array of moment tensor six vectors')
    if not isinstance(ratio, np.ndarray) or ratio.ndim != 1:
        raise TypeError(
            'Variable: ratio is expected to be a one-dimensional numpy array')
    if not isinstance(percentage_error_x, np.ndarray) or percentage_error_x.ndim != 1:
        raise TypeError(
            'Variable: percentage_error_x is expected to be a one-dimensional numpy array')
    if not isinstance(percentage_error_y, np.ndarray) or percentage_error_y.ndim != 1:
        raise TypeError(
            'Variable: percentage_error_y is expected to be a one-dimensional numpy array')
    # Handle zero errors by making them small to avoid zero division errors
    percentage_error_x[percentage_error_x == 0] = _SMALL_NUMBER
    percentage_error_y[percentage_error_y == 0] = _SMALL_NUMBER
    # Make sure the errors are positive
    percentage_error_x = np.abs(percentage_error_x)
    percentage_error_y = np.abs(percentage_error_y)
    completed = False
    if cprobability and (_use_c is None or _use_c):
        try:
            ln_p = cprobability.amplitude_ratio_ln_pdf(ratio, mt, a_x, a_y, percentage_error_x,
                                                       percentage_error_y)
            completed = True
        except Exception as e:
            # Run using python
            # Testing C code
            logger.exception('Error running cython code')
            if _C_LIB_TESTS:
                raise e
    else:
        logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
    if not completed:
        # Calculate probability using Python code
        # Calculate means for numerator and denominator
        mu_x = np.tensordot(a_x, mt, 1)
        mu_y = np.tensordot(a_y, mt, 1)
        # Make sure they are positive
        mu_x = np.abs(mu_x)
        mu_y = np.abs(mu_y)
        # Expand errors and ratio values to be 3 dimensional
        while percentage_error_x.ndim < 3:
            percentage_error_x = np.expand_dims(percentage_error_x, 1)
        while percentage_error_y.ndim < 3:
            percentage_error_y = np.expand_dims(percentage_error_y, 1)
        while ratio.ndim < 3:
            ratio = np.expand_dims(ratio, 1)
        # Get error on numerator and denominator
        numerator_error = np.multiply(percentage_error_x, mu_x)
        denominator_error = np.multiply(percentage_error_y, mu_y)
        # Calculate log of pdf
        ln_p = np.log(ratio_pdf(ratio, mu_x, mu_y, numerator_error, denominator_error) +
                      ratio_pdf(-ratio, mu_x, mu_y, numerator_error, denominator_error))
        # Set NaNs to zero probability
        if isinstance(ln_p, np.ndarray):
            ln_p[np.isnan(ln_p)] = -np.inf
        # Combine probabilities
        try:
            ln_p = cprobability.ln_prod(ln_p)
        except Exception:
            if cprobability:
                logger.exception('Error running cython code')
            ln_p = np.sum(ln_p, 0)
    if isinstance(ln_p, np.ndarray):
        ln_p[np.isnan(ln_p)] = -np.inf
    return ln_p


def relative_amplitude_ratio_ln_pdf(x_1, x_2, mt_1, mt_2, a_1, a_2, percentage_error_1, percentage_error_2, _use_c=None):
    """
    Calculate the probability of a given theoretical relative amplitude
    ratio giving an observed ratio.

    Calculates the probability of an amplitude ratio observation given a
    theoretical amplitude ratio and uncertainties on the numerator and
    denominator.

    The derivation of this pdf for the relative amplitude observation
    can be seen in:
        Pugh, D. J., 2015.
        Bayesian Source Inversion of Microseismic Events,
        Ph.D. thesis, University of Cambridge.

    Args
        x: np.array - 1 dimensional numpy array of observed numerator
                values
        y: np.array - 1 dimensional numpy array of observed denominator
                values
        mt_1: np.array - 2 dimensional numpy array of moment tensor 6
                vector samples for numerator
        mt_2: np.array - 2 dimensional numpy array of moment tensor 6
                vector samples for denominator
        a_1: np.array - 3 dimensional numpy array of station coefficients
                for the numerator
        a_2: np.array - 3 dimensional numpy array of station coefficients
                for the denominator
        percentage_error_1: np.array - 1 dimensional numpy array of
            percentage errors for the numerator
        percentage_error_2: np.array - 1 dimensional numpy array of
            percentage errors for the denominator

    Returns
        (np.array, np.array. np.array) -  tuple of probabilities for
                each joint moment tensor sample, scale factors for the
                event and the uncertainties in the scale factor.
    """
    # Check inputs and expcected dimensions
    if not isinstance(a_1, np.ndarray) or a_1.ndim != 3:
        raise TypeError('Variable: a_1 is expected to be a three-dimensional numpy array')
    if not isinstance(a_2, np.ndarray) or a_2.ndim != 3:
        raise TypeError('a_2 is expected to be a three-dimensional numpy array')
    if not isinstance(mt_1, np.ndarray) or mt_1.ndim != 2:
        raise TypeError('mt_1 is expected to be a two-dimensional numpy array')
    if not isinstance(mt_2, np.ndarray) or mt_2.ndim != 2:
        raise TypeError('mt_2 is expected to be a two-dimensional numpy array')
    if not isinstance(x_1, np.ndarray) or x_1.ndim != 1:
        raise TypeError('x is expected to be a one-dimensional numpy array')
    if not isinstance(x_2, np.ndarray) or x_2.ndim != 1:
        raise TypeError('y is expected to be a one-dimensional numpy array')
    if not isinstance(percentage_error_1, np.ndarray) or percentage_error_1.ndim != 1:
        raise TypeError('percentage_error_1 is expected to be a one-dimensional numpy array')
    if not isinstance(percentage_error_2, np.ndarray) or percentage_error_2.ndim != 1:
        raise TypeError('percentage_error_2 is expected to be a one-dimensional numpy array')
    # For fractional uncertainties that are zero make them really small
    # to avoid zero divison errors
    percentage_error_1[percentage_error_1 == 0] = _SMALL_NUMBER
    percentage_error_2[percentage_error_2 == 0] = _SMALL_NUMBER
    # Make sure the errors are positive
    percentage_error_1 = np.abs(percentage_error_1)
    percentage_error_2 = np.abs(percentage_error_2)
    completed = False
    if cprobability and (_use_c is None or _use_c):
        try:
            # raise ValueError('C code returning incorrect result for probabilities')
            ln_p, scale, scale_uncertainty = cprobability.relative_amplitude_ratio_ln_pdf(np.ascontiguousarray(x_1),
                                                                                          np.ascontiguousarray(x_2),
                                                                                          np.ascontiguousarray(mt_1),
                                                                                          np.ascontiguousarray(mt_2),
                                                                                          np.ascontiguousarray(a_1),
                                                                                          np.ascontiguousarray(a_2),
                                                                                          np.ascontiguousarray(percentage_error_1),
                                                                                          np.ascontiguousarray(percentage_error_2))
            completed = True
        except Exception as e:
            # Run using python
            # Testing C code
            logger.exception('Error running cython code')
            if _C_LIB_TESTS:
                raise e
    else:
        logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
    if not completed:
        # Calculate probability using Python code
        # Calculate the ratio
        ratio = np.divide(x_1, x_2)
        # Calculate means for numerator and denominator
        mu_x = np.tensordot(a_1, mt_1, 1)
        mu_y = np.tensordot(a_2, mt_2, 1)
        # Make sure they are positive
        mu_x = np.abs(mu_x)
        mu_y = np.abs(mu_y)
        # Expand errors and ratio values to be 3 dimensional
        while percentage_error_1.ndim < 3:
            percentage_error_1 = np.expand_dims(percentage_error_1, 1)
        while percentage_error_2.ndim < 3:
            percentage_error_2 = np.expand_dims(percentage_error_2, 1)
        while ratio.ndim < 3:
            ratio = np.expand_dims(ratio, 1)
        # Estimate scale factor
        scale, scale_uncertainty = scale_estimator(ratio, mu_x, mu_y,
                                                   percentage_error_1, percentage_error_2)
        # Multiply in the scale factor
        mu_x = np.multiply(scale, mu_x)
        # Calculate the numerator and denominator errors
        numerator_error = np.multiply(percentage_error_1, mu_x)
        denominator_error = np.multiply(percentage_error_2, mu_y)
        # Calculate the log pdf
        ln_p = np.log(ratio_pdf(ratio, mu_x, mu_y, numerator_error, denominator_error) +
                      ratio_pdf(-ratio, mu_x, mu_y, numerator_error, denominator_error))
        # Set NaNs to zero probability
        if isinstance(ln_p, np.ndarray):
            ln_p[np.isnan(ln_p)] = -np.inf
        # Combine probabilities
    try:
        ln_p = cprobability.ln_prod(ln_p)
    except Exception:
        ln_p = np.sum(ln_p, 0)
    if isinstance(ln_p, np.ndarray):
        ln_p[np.isnan(ln_p)] = -np.inf
    return ln_p, scale, scale_uncertainty


def scale_estimator(observed_ratio, mu_x, mu_y, percentage_error_x, percentage_error_y):
    """
    Estimate the scale factor between the two events given the observed_ratio
    and theoretical ratios.

    The derivation of this calculation can be found in:
        Pugh, D. J., 2015.
        Bayesian Source Inversion of Microseismic Events,
        Ph.D. thesis, University of Cambridge.

    Args
        observed_ratio: np.array - numpy array of observed_ratio ratio values
        mu_x: np.array - numpy array of mean values for the numerator
        mu_y: np.array - numpy array of station coefficients for the
            denominator
        percentage_error_x: np.array - numpy array of percentage errors
            for the numerator
        percentage_error_y: np.array - numpy array of percentage errors
            for the denominator

    Returns
        (np.array, np.array) -  tuple of means and standard deviations
                of the scale factor for each sample
    """
    # Expand arrays to be correct dimensions
    if isinstance(mu_x, np.ndarray) and mu_x.ndim == 3:
        while observed_ratio.ndim < 3:
            observed_ratio = np.expand_dims(observed_ratio, 1)
        while percentage_error_x.ndim < 3:
            percentage_error_x = np.expand_dims(percentage_error_x, 1)
        while percentage_error_y.ndim < 3:
            percentage_error_y = np.expand_dims(percentage_error_y, 1)
    # Calculate standard deviations for the numerator and denominator
    sx = np.multiply(np.array(percentage_error_x), mu_x)
    sy = np.multiply(np.array(percentage_error_y), mu_y)
    # Convert the observed_ratio values to an array
    observed_ratio = np.array(observed_ratio)
    # Calculate PDF coefficients
    mu_x_2 = np.multiply(mu_x, mu_x)
    mu_x_3 = np.multiply(mu_x_2, mu_x)
    sx_2 = np.multiply(sx, sx)
    sy_2 = np.multiply(sy, sy)
    mu1 = np.divide(np.multiply(mu_y, observed_ratio), mu_x)
    s1 = np.sqrt(
        np.divide(np.multiply(np.multiply(sy_2, observed_ratio), observed_ratio) + sx_2, mu_x_2))
    s1_2 = np.multiply(s1, s1)
    mu1_2 = np.multiply(mu1, mu1)
    A = np.divide(np.multiply(np.multiply(mu_x, observed_ratio), sy_2), mu_x_3)
    B = np.divide(np.multiply(mu_y, sx_2), mu_x_3)
    C = np.sqrt(2./np.pi) * np.divide(
        np.multiply(np.multiply(sx_2, sy),
                    np.exp(-0.5*np.divide(np.multiply(mu_y, mu_y),
                                          sy_2))
                    ),
        np.multiply(mu_x_3, s1_2))

    N = np.multiply(A, mu1) + B + C
    # Calculate mean and standard deviations
    mu = np.divide(np.multiply(A, (s1_2 + mu1_2)) + np.multiply(B, mu1), N)
    s = np.sqrt(np.divide(np.multiply(A, (3*np.multiply(s1_2, mu1)+np.multiply(mu1_2, mu1))) +
                          np.multiply(B, (s1_2 + mu1_2)) +
                          np.multiply(C, np.divide(sx_2, mu_x_2)),
                          N) - np.multiply(mu, mu))
    # Combine mean over stations
    return combine_mu(mu, s)


def combine_mu(mu, s):
    """
    Combine the mean and standard deviations over stations

    Args
        mu: np.array - 2 dimensional numpy array of mean values
        s: np.array - 2 dimensional numpy array of standard deviation values

    Returns
        (np.array, np.array) - tuple of combined mean and standard deviation
            values
    """
    # Loop over each sample
    for i in range(mu.shape[0]):
        if i == 0:
            # First sample so initialise the mean and standard debiation
            combined_mu = mu[i, :]
            combined_s = s[i, :]
        else:
            # Combine with existing values
            si_2 = np.multiply(s[i, :], s[i, :])
            s_2 = np.multiply(combined_s, combined_s)
            combined_mu = (
                np.multiply(combined_mu, si_2)+np.multiply(mu[i, :], s_2)) / (s_2+si_2)
            combined_s = np.multiply(combined_s, s[i, :]) / np.sqrt(s_2+si_2)
    return combined_mu, combined_s


# Supporting PDF functions


def gaussian_pdf(x, mu, sigma):
    """
    Calculate the Gaussian probability

    Calculates the Gaussian probability of x given a mean mu and
    standard deviation sigma.

    Args
        x: Number or np.array of theoretical amplitudes
        mu: Number or list or np.array of means.
            Needs to be same dimensions as x, or a scalar.
        sigma: Number or list or np.array of standard deviations.
            Needs to be same dimensions as x, or a scalar.

    Returns
        float/np.array of probabilities.

    """
    # Avoid nan overflows for 0 errors, make the error very small to get a large number
    if not isinstance(sigma, (float, int, np.float64)):
        sigma[sigma == 0] = _SMALL_NUMBER
    elif sigma == 0:
        sigma = _SMALL_NUMBER
    if isinstance(x, np.ndarray) and x.ndim > 2:
        mu = np.kron(
            np.array([np.kron(mu, np.ones(x.shape[1])).T]).T, np.ones(x.shape[2]))
        sigma = np.kron(
            np.array([np.kron(sigma, np.ones(x.shape[1])).T]).T, np.ones(x.shape[2]))
    return gaussian.pdf(x, loc=mu, scale=sigma)


def gaussian_cdf(x, mu, sigma):
    """
    Calculate the Gaussian Cumulative Distribution Function

    Calculates the Gaussian CDF of x given a mean mu and standard
    deviation sigma.

    Args
        x: Number or np.array of theoretical amplitudes
        mu: Number or list or np.array of means. Needs to be same
                dimensions as x, or a scalar.
        sigma: Number or list or np.array of standard deviations. Needs
                to be same dimensions as x, or a scalar.

    Returns
        float/np.array of CDF values.

    """
    # Avoid nan overflows for 0 errors, make the error very small to get a large number
    if not isinstance(sigma, (float, int, np.float64)):
        sigma[sigma == 0] = _SMALL_NUMBER
    elif sigma == 0:
        sigma = _SMALL_NUMBER
    return gaussian.cdf(x, loc=mu, scale=sigma)


def ratio_pdf(z, mu_x, mu_y, sigma_x, sigma_y, corr=0):
    """
    Calculate the Ratio Probability

    Calculates the Ratio pdf (D. Hinkley, On the ratio of two correlated
    normal random variables, 1969, Biometrika vol 56 pp 635-639).

    Given Z = X/Y and means mu_x, mu_y and standard deviation sigma_x and
    sigma_y. The pdf is normalised.

    Args
        z: Number or numpy array of theoretical amplitudes
        mu_x: Number or list or numpy array of means. Needs to be same
                dimensions as z, or a scalar.
        mu_y: Number or list or numpy array of means. Needs to be same
                dimensions as z, or a scalar.
        sigma_x: Number or list or numpy array of standard deviations.
                Needs to be same dimensions as z, or a scalar.
        sigma_y: Number or list or numpy array of standard deviations.
                Needs to be same dimensions as z, or a scalar.

    Returns
        float/np.array of probabilities

    """
    # Set to prevent numpy warnings
    np.seterr(divide='ignore', invalid='ignore')
    # Expand parameters
    if isinstance(mu_x, np.ndarray) and mu_x.ndim == 3:
        if isinstance(mu_y, np.ndarray) and mu_y.ndim == 2:
            mu_y = np.expand_dims(mu_y, 1)
    if isinstance(mu_y, np.ndarray) and mu_y.ndim == 3:
        if isinstance(mu_x, np.ndarray) and mu_x.ndim == 2:
            mu_x = np.expand_dims(mu_x, 1)
        if isinstance(z, np.ndarray) and z.ndim == 2:
            z = np.expand_dims(z, 1)
        if isinstance(sigma_x, np.ndarray) and sigma_x.ndim == 2:
            sigma_x = np.expand_dims(sigma_x, 1)
        if isinstance(sigma_y, np.ndarray) and sigma_y.ndim == 2:
            sigma_y = np.expand_dims(sigma_y, 1)
    # Calculate some coefficients
    z_2 = np.multiply(z, z)
    sigma_x_2 = np.multiply(sigma_x, sigma_x)
    sigma_xy = np.multiply(sigma_x, sigma_y)
    sigma_y_2 = np.multiply(sigma_y, sigma_y)
    mu_x_2 = np.multiply(mu_x, mu_x)
    if corr > 0:
        a = np.sqrt(np.divide(z_2, sigma_x_2) - 2*corr*np.divide(z, sigma_xy) + 1/sigma_y_2)
    else:
        a = np.sqrt(np.divide(z_2, sigma_x_2) + 1/sigma_y_2)
    a_2 = np.multiply(a, a)
    b = np.divide(np.multiply(mu_x, z), sigma_x_2) - (corr*np.divide(sigma_x_2, sigma_xy)) + \
        np.divide(mu_y, sigma_y_2)
    c = np.divide(mu_x_2, sigma_x_2) + \
        np.divide(np.multiply(mu_y, mu_y), sigma_y_2)
    if corr > 0:
        c -= (2*corr*np.divide(np.multiply(mu_x, mu_y), sigma_xy))
    d = np.exp(np.divide((np.multiply(b, b)-np.multiply(c, a_2)),
                         (2 * (1-corr*corr) * a_2)))
    p = np.divide(np.multiply(b, d),
                  (np.sqrt(2*np.pi) * np.multiply(sigma_xy, np.multiply(a, a_2))))
    p = np.multiply(p,
                    (gaussian_cdf(np.divide(b, (np.sqrt(1-corr*corr) * a)), 0, 1) -
                     gaussian_cdf(np.divide(-b, (np.sqrt(1-corr*corr) * a)), 0, 1))
                    )
    p += np.multiply(np.sqrt(1-corr*corr) / (np.pi*np.multiply(sigma_xy, a_2)),
                     np.exp(-c / (2*(1-corr*corr))))
    if isinstance(p, np.ndarray):
        p[np.isnan(p)] = 0
    # Clear arrays and force a garbage collection to free up memory
    del z_2
    del sigma_x_2
    del sigma_xy
    del sigma_y_2
    del mu_x_2
    del a
    del b
    del c
    del d
    gc.collect()
    return p


def beta_pdf(x, a, b):
    """
    Calculate the Beta Probability

    Calculates the Beta pdf for X given shape parameters a and b.
    Uses scipy.stats.beta.pdf docstring.

    Args
        x: Number or numpy object.
        a: Number or list or numpy object of first shape parameter.
        b: Number or list or numpy object of second shape parameter.

    Returns
        np.array of probabilities

    """
    return beta.pdf(x, a, b)


def heaviside(x):
    """
    Calculate the heaviside step function.

    The heaviside step function is defined as:
        H(x) = 0   x < 0
        H(x) = 0.5 x = 0
        H(x) = 1   x > 0

    Args:
        x: float/np.array - array or float of x values

    Returns:
        float/np.array - array of heaviside values for x values
    """
    return 0.5*(np.sign(x) + 1)


def dkl(ln_probability_p, ln_probability_q, dV=1.0):
    """
    Calculate the Kullback-Liebler divergence for two distributions, p
    and q

    Args
        ln_probability_p: LnPDF - First distribution (p)
        ln_probability_q: LnPDF - Second distribution (q)

    Keyword Args
        dV: float value for the Monte Carlo integration volume element
            (V/N)

    Returns
        float - Kullback-Liebler divergence estimate
    """
    if isinstance(ln_probability_p, LnPDF):
        ln_probability_p = np.ascontiguousarray(
            np.array(ln_probability_p._ln_pdf).flatten())
    if isinstance(ln_probability_q, LnPDF):
        ln_probability_q = np.ascontiguousarray(
            np.array(ln_probability_q._ln_pdf).flatten())
    if cprobability:
        try:
            return cprobability.dkl(ln_probability_p.copy(), ln_probability_q.copy(), dV)
        except Exception:
            logger.exception('Error running cython code')
    else:
        logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
    ind = ln_probability_p > -np.inf
    ln_probability_p = ln_probability_p.copy() - ln_probability_p.max()
    ln_probability_q = ln_probability_q.copy() - ln_probability_q.max()
    probability_p = np.exp(ln_probability_p)
    # Normalise PDF p
    n = np.sum(probability_p)*dV
    ln_probability_p -= np.log(n)
    probability_p /= n
    # Normalise PDF q
    probability_q = np.exp(ln_probability_q)
    n = np.sum(probability_q)*dV
    ln_probability_q -= np.log(n)
    return np.sum(ln_probability_p[ind]*probability_p[ind] -
                  ln_probability_q[ind]*probability_p[ind]) * dV


def ln_marginalise(ln_pdf, axis=0, dV=1.0):
    """
    Marginalise the pdf from the log pdf input

    Marginalises the pdf given a dV, without the dV the normalisation is
    only proportional to the normalised marginalised distribution.

    Args
        pdf: np.array/LnPDF - array or object that can be multiplied by dV


    Optional Args
        axis: integer - axis over which to marginalise [default = 0]
        dV: float - volume element in marginalisation, default value is
                 1.0 giving a proportional marginalised distribution.

    Returns
        np.array - marginalised distribtion over axis
    """
    if cprobability and axis == 0:
        try:
            if isinstance(ln_pdf, LnPDF):
                return cprobability.ln_marginalise(ln_pdf._ln_pdf.astype(np.float64))
            return cprobability.ln_marginalise(ln_pdf.astype(np.float64))
        except Exception:
            logger.exception('Error running cython code')
    else:
        logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
    # scale and then marginalise:
    ln_scale = 0
    if ln_pdf.shape[axis] == 1:
        return np.array(ln_pdf).squeeze(axis)
    elif axis == 0 and len(ln_pdf.shape) == 1:
        # Probably shouldn't be marginalising over the last parameter
        return ln_pdf
    if np.prod(ln_pdf.shape) and ln_pdf.max() < 0 and ln_pdf.max() > -np.inf:
        ln_scale = -ln_pdf.max()
    with warnings.catch_warnings() and np.errstate(divide='ignore'):
        warnings.simplefilter("ignore")
        # if axis == 0:
        #     # Get consistency with c code
        #     ln_pdf = np.array(ln_pdf)
        result = np.log(np.sum(np.exp(ln_pdf+ln_scale)*dV, axis=axis)) - ln_scale
    if axis == 0:
        result = np.array(result).flatten()
    return result


def ln_normalise(ln_pdf, dV=1):
    """
    Normalise the pdf the pdf from the log pdf input.

    Normalises the pdf given a dV without the dV the normalisation is
    only proportional to the normalised distribution.

    Args
        pdf: Object that can be multiplied by dV (numpy array/matrix or PDF object)

    Optional Args
        dV: float - volume element in normalisation, default value is
                 1.0 giving a proportional normalised distribution.
    Returns
        np.array - normalised distribtion

    """
    if ln_pdf.ndim == 1 and ln_pdf.shape[0] == 1:
        return np.array([0])
    if cprobability:
        try:
            if ln_pdf.ndim != 1 and ln_pdf.shape[0] != 1:
                raise Exception('Incorrect shape')
            if ln_pdf.ndim == 2:
                normalised_ln_pdf = cprobability.ln_normalise(np.asarray(ln_pdf).flatten())
            else:
                normalised_ln_pdf = cprobability.ln_normalise(ln_pdf)
            return normalised_ln_pdf
        except Exception:
            logger.exception('Error running cython code')
    else:
        logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
    # scale and then marginalise:
    ln_scale = 0
    if -ln_pdf.max() > 0 and ln_pdf.max() > -np.inf:
        ln_scale = -ln_pdf.max()
    with warnings.catch_warnings() and np.errstate(divide='ignore'):
        warnings.simplefilter("ignore")
        ln_n = np.log(np.sum(np.exp(ln_pdf+ln_scale) * dV, axis=None)) - ln_scale
    if ln_pdf.ndim == 2 and 1 in list(ln_pdf.shape):
        ln_pdf = np.squeeze(np.asarray(ln_pdf))
        if not ln_pdf.shape:
            ln_pdf = np.array([ln_pdf])
    # ln_probability_scale_factor is automatically included in normalisation
    return ln_pdf - ln_n


def model_probabilities(*args):
    """
    Calculate the model probabilities for a discrete set of models using the
    ln_bayesian_evidences, provided as args.

    e.g. to compare between DC and MT:

        pDC,pMT=model_probabilities(dc_ln_bayesian_evidence,mt_ln_bayesian_evidence)

    Args
        floats - ln_bayesian_evidence for each model type

    Returns
        tuple: Tuple of the normalised model probabilities for the corresponding
            ln_bayesian_evidence inputs
    """
    max_LnBE = -np.inf
    output_model_probabilities = []
    for ln_bayesian_evidence in args:
        max_LnBE = max([max_LnBE, ln_bayesian_evidence])
    norm = 0
    for ln_bayesian_evidence in args:
        p = np.exp(ln_bayesian_evidence-max_LnBE)
        output_model_probabilities.append(p)
        norm += p
    for i, model_probability in enumerate(output_model_probabilities):
        output_model_probabilities[i] = model_probability/norm
    return tuple(output_model_probabilities)


def dkl_estimate(ln_pdf, V, N):
    """
    Calculate the Kullback-Leibeler divergence for the ln_pdf from the
    prior distribution.

    Assumes that the moment tensor prior is uniform in the sampling
    and that is the form that the PDFs have been evaluated in.

    Args
        ln_pdf: input LnPDF object
        V: float value for the Volume of the sample space for Monte
            Carlo delta V and prior estimates
        N: integer number of tried samples

    """
    dV = V/float(N)
    # Doesnt use dkl function as can be simplified to reduce calculation
    if isinstance(ln_pdf, LnPDF):
        ln_pdf = np.ascontiguousarray(np.array(ln_pdf._ln_pdf).flatten())
    if cprobability:
        try:
            return cprobability.dkl_uniform(ln_pdf.copy(), V, dV)
        except Exception:
            logger.exception('Error running cython code')
    else:
        logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
    ind = ln_pdf > -np.inf
    ln_pdf = ln_pdf[ind].copy()-ln_pdf.max()
    pdf = np.exp(ln_pdf)
    # Normalise PDF
    n = np.sum(pdf)*dV
    ln_pdf -= np.log(n)
    pdf /= n
    return np.sum(np.array(ln_pdf)*np.array(pdf) + np.array(pdf)*np.log(V), -1)*dV


class LnPDF(object):

    """
    LnPDF object, to allow arithmetic operations on the pdf

    A simple object containing a pdf and several methods for acting on the pdf.
    _ln_pdf is a numpy matrix containing the natural logarithm of the pdf
    """

    def __init__(self, ln_pdf=None, pdf=None, dV=1, *args, **kwargs):
        """LnPDF Initialisation

        Initialises the LnPDF object.

        Optional Args
            ln_pdf:[None] numpy array/matrix or list containing log pdf samples
            pdf:[None] numpy array/matrix or list containing pdf samples
            dV:[1] default volume element for normalisation
        """
        # Possibly good to add initialisation size arg for LnPDF,
        # initSize=(1,1)
        self._ln_pdf = np.matrix([])
        if ln_pdf is not None:
            self._set_ln_pdf(ln_pdf)
        elif pdf is not None:
            with warnings.catch_warnings() and np.errstate(divide='ignore'):
                warnings.simplefilter("ignore")
                self._set_ln_pdf(np.log(pdf))
        if dV:
            self._set_dv(dV)
        else:
            self._set_dv(1)

    def __getstate__(self):
        return self._ln_pdf, self.dV

    def __setstate__(self, ln_pdf, dV):
        self._set_ln_pdf(ln_pdf)
        self._set_dv(dV)

    def __getattr__(self, key):
        """x.__getattr__(y) <==> x.y"""
        # Handles cases not picked up by __getattribute__
        if key == 'shape':
            return self._ln_pdf.shape

    def __len__(self):
        """
        Return the length of ln_pdf

        Returns
            int - length of ln_pdf (axis=-1)

        """
        if isinstance(self._ln_pdf, np.ndarray):
            return self.shape[-1]
        return 1

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        return self._ln_pdf.__repr__()

    def __getitem__(self, index):
        """x.__getitem__(index) <==> x[index]"""
        return self._ln_pdf.__getitem__(index)

    def __getslice__(self, i, j):
        """x.__getslice__(i, j) <==> x[i:j]"""
        return self._ln_pdf.__getslice__(i, j)

    def __setitem__(self, index, val):
        """x.__setitem__(index, val) <==> x[index] = val"""
        return self._ln_pdf.__setitem__(index, val)

    def __setslice__(self, i, j, val):
        """x.__setslice__(i, j, val) <==> x[i, j] = val"""
        return self._ln_pdf.__setslice__(i, j, val)

    def _cmp(self, other, operator_func):
        """
        Base comparison function.

        Args
            other: object - object for comparison
            operator_func: func - function from the operator module to compare using

        Returns:
            boolean: result of the comparison
        """
        if isinstance(other, self.__class__):
            return operator_func(self._ln_pdf, other._ln_pdf)
        else:
            return operator_func(self._ln_pdf, other)

    def __lt__(self, other):
        """x.__lt__(y) <==> x < y"""
        return self._cmp(other, operator.__lt__)

    def __gt__(self, other):
        """x.__gt__(y) <==> x > y"""
        return self._cmp(other, operator.__gt__)

    def __le__(self, other):
        """x.__le__(y) <==> x <= y"""
        return self._cmp(other, operator.__le__)

    def __ge__(self, other):
        """x.__ge__(y) <==> x >= y"""
        return self._cmp(other, operator.__ge__)

    def __eq__(self, other):
        """x.__eq__(y) <==> x == y"""
        return self._cmp(other, operator.__eq__)

    def __ne__(self, other):
        """x.__ne__(y) <==> x != y"""
        return self._cmp(other, operator.__ne__)

    def _arithmetic(self, other, arithmetic_func):
        """
        Base comparison function.

        Args
            other: object - object for comparison
            operator_func: func - function from the operator module to compare using

        Returns:
            boolean: result of the comparison
        """
        if isinstance(other, self.__class__):
            if self.dV == other.dV:
                return self.__class__(ln_pdf=arithmetic_func(self._ln_pdf, other._ln_pdf), dV=self.dV)
            else:
                raise ValueError(
                    'Incorrect dV, should match between self and other')
        else:
            return self.__class__(ln_pdf=arithmetic_func(self._ln_pdf, other), dV=self.dV)

    def __mul__(self, other):
        """x.__mul__(y) <==> x*y"""
        return self._arithmetic(other, np.multiply)

    def __div__(self, other):
        """x.__div__(y) <==> x/y"""
        return self._arithmetic(other, np.divide)

    def __truediv__(self, other):
        """x.__div__(y) <==> x/y"""
        return self.__div__(other)

    def __add__(self, other):
        """x.__add__(y) <==> x+y"""
        return self._arithmetic(other, operator.__add__)

    def __sub__(self, other):
        """x.__sub__(y) <==> x-y"""
        return self._arithmetic(other, operator.__sub__)

    def __rsub__(self, other):
        """x.__rsub__(y) <==> y-x"""
        return (self-other)*-1

    def __radd__(self, other):
        """x.__radd__(y) <==> y+x"""
        return (self+other)

    def __rmul__(self, other):
        """x.__rmul__(y) <==> y*x"""
        return self*other

    def __rdiv__(self, other):
        """x.__rdiv__(2) <==> 2/x"""
        return LnPDF(ln_pdf=(other/self._ln_pdf), dV=self.dV)

    def __rtruediv__(self, other):
        """x.__rdiv__(2) <==> 2/x"""
        return self.__rdiv__(other)

    def __abs__(self):
        """x.__abs__() <==> abs(x)"""
        return LnPDF(ln_pdf=np.abs(self._ln_pdf), dV=self.dV)

    def __float__(self):
        """
        Convert pdf to float

        Returns
            Converted LnPDF to float (if a unit object, else error)

        Raises
            TypeError: array not length 1 so cannot be converted to a scalar.

        """
        return float(self._ln_pdf)

    def argmax(self, axis=1):
        """
        Return the index of maximum value over an axis

        Returns the indices of the maximum values from the pdf over the
        given axis. If the axis is set to -1 returns the index of the
        maximum value over the marginalised pdf.

        Optional Args
            axis: int - axis over which to find the argmax [default=1]

        Returns
            np.array - array of indices to maximum values.
        """
        axis = min(self._ln_pdf.ndim-1, axis)
        if self._ln_pdf.ndim > 1 and axis == -1:
            return self.marginalise().argmax(0).flatten()
        return self._ln_pdf.argmax(axis)

    def max(self, axis=1):
        """Return the maximum values over a given axis

        Returns the maximum values from the LnPDF over the given axis.
        If the axis is set to -1 returns the maximum value over the
        marginalised pdf.

        Optional Args
            axis: int - axis over which to find the argmax [default=1]

        Returns
            np.array - array of maximum values.

        """
        if not np.prod(self._ln_pdf.shape):
            return 0
        axis = min(self._ln_pdf.ndim-1, axis)
        if self._ln_pdf.ndim > 1 and axis == -1:
            return np.exp(self.marginalise().max()).flatten()
        return float(np.exp(self._ln_pdf.max(axis)))

    def _set_dv(self, dV):
        """Private Function to set dV value"""
        self.dV = dV

    def _set_ln_pdf(self, ln_pdf):
        """Private Function to set _ln_pdf value"""
        if isinstance(ln_pdf, self.__class__):
            ln_pdf = ln_pdf._ln_pdf
        self._ln_pdf = np.matrix(ln_pdf)

    def output(self, normalise=True):
        """
        Return the marginalised pdf for outputting.

        Optional Args
            normalise: bool - flag to normalise the PDF before
                    outputting [default = True]
        """
        if normalise:
            return self.marginalise().normalise()
        return self.marginalise()

    def exp(self):
        if cprobability:
            try:
                return cprobability.ln_exp(self._ln_pdf)
            except Exception:
                logger.exception('Error running cython code')
        else:
            logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
        return np.exp(self._ln_pdf)

    def nonzero(self, discard=100000., n_samples=0):
        """
        Return the non-zero indices of the pdf

        Returns the non zero indices of the marginalised pdf.

        Samples with probability less than n_samples*discard of the max
        probability can be discarded.

        Optional Args
            discard: float - discard scale [default = 100000.]
            n_samples: integer - number of samples generated [default = 0]
        """
        ln_pdf = np.array(self.marginalise(axis=0)._ln_pdf).flatten()
        m_val = -np.inf
        if n_samples > 0 and discard > 0:
            m_val = max(ln_pdf) - np.log(discard*n_samples)
        return np.nonzero(1*(ln_pdf > m_val))[0]

    def normalise(self, dV=False):
        """
        Normalise the pdf object

        Optional Args
            dV: float - update the object's dV value.

        Returns
            LnPDF object with normalised pdf.
        """
        if dV:
            self._set_dv(dV)
        new = self.__class__(dV=self.dV)
        new._ln_pdf = ln_normalise(self._ln_pdf, self.dV)
        return new

    def marginalise(self, axis=0, dV=False):
        """
        Marginalise the pdf object over a given axis

        Optional Args
            axis: integer - axis over which to marginalise [default = 0]
            dV: float - update the object's dV value.

        Returns
            LnPDF object with marginalised pdf.
        """
        if dV:
            self._set_dv(dV)
        new = self.__class__(dV=self.dV)
        new._ln_pdf = ln_marginalise(self._ln_pdf, axis=axis, dV=self.dV)
        return new

    def append(self, other, axis=1):
        """
        Append values to pdf

        Args
            other: LnPDF/np.array - Object to append to ln_pdf

        Optional Args
            axis: integer - axis over which to append [default = 1]
        """
        if not self.shape[1]:
            if isinstance(other, self.__class__):
                self._ln_pdf = other._ln_pdf
                self.dV = other.dV
            elif isinstance(other, np.ndarray):
                self._ln_pdf = other
        else:
            if isinstance(other, self.__class__) and self.dV == other.dV:
                self._ln_pdf = np.append(
                    self._ln_pdf, other._ln_pdf, axis=axis)
            elif isinstance(other, np.ndarray):
                self._ln_pdf = np.append(self._ln_pdf, other, axis=axis)
