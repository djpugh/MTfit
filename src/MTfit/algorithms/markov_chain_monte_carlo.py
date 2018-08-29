"""markov_chain_monte_carlo
****************************
Module containing algorithm classes for Markov chain Monte Carlo sampling.
"""


# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.


import time
import gc
import copy
import logging
import sys


import numpy as np


from .base import BaseAlgorithm
from .monte_carlo import IterationSample
from ..probability import gaussian_pdf, gaussian_cdf, beta_pdf, LnPDF
from ..sampling import Sample
from ..convert import Tape_MT33, basic_cdc_GD, MT33_MT6, MT6_Tape
from ..utilities.extensions import get_extensions
from ..utilities import C_EXTENSION_FALLBACK_LOG_MSG

logger = logging.getLogger('MTfit.algorithms')

try:
    from . import cmarkov_chain_monte_carlo
except ImportError:
    cmarkov_chain_monte_carlo = None
except Exception:
    logger.exception('Error importing c extension')
    cmarkov_chain_monte_carlo = None


_CYTHON = True
_CYTHON_TESTS = False

__all__ = ['McMCAlgorithmCreator',
           'MarginalisedMarkovChainMonteCarlo',
           'MarginalisedMetropolisHastings',
           'MarginalisedMetropolisHastingsGaussianTape',
           'IterativeMetropolisHastingsGaussianTape',
           'IterativeTransDMetropolisHastingsGaussianTape',
           'IterativeMultipleTryMetropolisHastingsGaussianTape',
           'IterativeMultipleTryTransDMetropolisHastingsGaussianTape']


# Priors
def zero_prior(x, *args):
    """Evaluates uniform prior probability for x

    Prior is uniform over surface of 6-sphere rather than the parameterisation. Should not be called

    Returns
        float: prior probability
    """
    return 0


def uniform_prior(xi, dc=None, basic_cdc=False, poisson=None, max_poisson=0, min_poisson=0):
    """Evaluates uniform prior probability for x

    Prior is uniform over surface of 6-sphere rather than the parameterisation.

    Returns
        float: prior probability
    """
    # Uniform samples from randomly distributed MT solutions
    if dc is None:
        if 'gamma' not in xi.keys():
            dc = False
        else:
            dc = xi['gamma'] == 0.0 and xi['delta'] == 0.0
    p = 1.
    normalisation_constant = 1.10452194071529090000
    if not dc and not basic_cdc and 'gamma' in xi.keys():
        p *= 1.5*np.cos(3*xi['gamma'])
        b = 5.745
        p *= beta_pdf((xi['delta']+np.pi/2)/np.pi, b, b)/np.pi
        p *= normalisation_constant
    elif basic_cdc:
        if poisson is None:
            probability_poisson = 1/(max_poisson-min_poisson)
            if probability_poisson > 0:
                p *= 1/probability_poisson
            else:
                # handle infinite range for poisson
                p *= 1
        p *= 1/np.pi
    # strike dip rake ignored as prior uniform so the ratio in the acceptance
    # cancels
    return p


def flat_prior(xi, dc=None, basic_cdc=False, poisson=None, max_poisson=0, min_poisson=0):
    """Evaluates flat prior probability for x over parameterisation.

    Prior is uniform over the parameterisation.

    Returns
        float: prior probability
    """
    # uniform flat prior over parameters
    if dc is None:
        if 'gamma' not in xi.keys():
            dc = False
        else:
            dc = xi['gamma'] == 0.0 and xi['delta'] == 0.0
    p = 1.
    if not dc and not basic_cdc and 'gamma' in xi.keys():
        p *= 3/np.pi
        p *= 1/np.pi
    elif basic_cdc:
        if poisson is None:
            probability_poisson = 1/(max_poisson-min_poisson)
            if probability_poisson > 0:
                p *= 1/probability_poisson
            else:
                # handle infinite range for poisson
                p *= 1  # WRONG BUT CAN BE USED (unnormalised but balanced)
        p *= 1/np.pi
    # strike dip rake ignored as prior uniform so the ratio in the acceptance
    # cancels
    return p


class DataError(ValueError):
    pass


class McMCAlgorithmCreator(object):

    """
    Generates the Markov chain Monte Carlo algorithm from parameters

    Creates the correct McMC algorithm object depending on the selected mode.

    """

    def __new__(self, mode='metropolis_hastings', alpha=False, trans_dimensional=False,
                learning_length=10000, parameterisation='tape', transition='gaussian',
                chain_length=1000000, min_acceptance_rate=0.3, max_acceptance_rate=0.5,
                acceptance_rate_window=1000, initial_sample='grid', **kwargs):
        """
        McMC Creator

        Returns initialised object of desired type. This is extensible using the
        MTfit.algorithms.markov_chain_monte_carlo group in pkg_resources (see extensions documentation)
        to add additional options

        Args
            mode:[MetropolisHastings] McMC mode to use options are: MetropolisHastings
            alpha:[False] Default parameters for Markov chain steps, depend on transition probabilities
            trans_dimensional:[False] Boolean flag to run with trans-dimensional sampling
            learning_length:[10000] Number of samples for learning period and to discard from chain.
            parameterisation:['Tape'] Source type parameterisation to use options are: Tape
            transition:['Gaussian'] Transition pdf to use options are: 'Gaussian'
            chain_length:[1000000] End point of the Markov chain.
            min_acceptance_rate:[0.3] Minimum targetted sample acceptance rate.
            max_acceptance_rate:[0.5] Maximum targetted sample acceptance rate.
            acceptance_rate_window:[1000] Number of samples to use in learning period for calculating and
                modifying the acceptance rate.
            initial_sample:['Grid'] Initialisation sampling mode to use options are 'Grid'.

        Returns
            McMC sampling object.
        """
        (algorithm_names, algorithms) = get_extensions('MTfit.directed_algorithms')
        # McMC algorithms included in this file
        if mode.lower() in ['metropolis_hastings'] and parameterisation.lower() == 'tape' and transition.lower() == 'gaussian':
            number_samples = kwargs.pop('number_samples', 1000)
            if not number_samples:  # or kwargs.get('multiple_events',False):
                if trans_dimensional:
                    dimension_jump_prob = kwargs.pop('dimension_jump_prob', 0.01)
                    return IterativeTransDMetropolisHastingsGaussianTape(alpha=alpha, learning_length=learning_length,
                                                                         chain_length=chain_length,
                                                                         min_acceptance_rate=min_acceptance_rate,
                                                                         max_acceptance_rate=max_acceptance_rate,
                                                                         initial_sample=initial_sample,
                                                                         acceptance_rate_window=acceptance_rate_window,
                                                                         dimension_jump_prob=dimension_jump_prob, **kwargs)
                else:
                    # if kwargs.get('multiple_events', False):
                    #     return IterativeMultipleEventJointMetropolisHastingsGaussianTape(alpha=alpha, learning_length=learning_length,
                    #                                                                      chain_length=chain_length,
                    #                                                                      min_acceptance_rate=min_acceptance_rate,
                    #                                                                      max_acceptance_rate=max_acceptance_rate,
                    #                                                                      initial_sample=initial_sample,
                    #                                                                      acceptance_rate_window=acceptance_rate_window,
                    #                                                                      **kwargs)
                    return IterativeMetropolisHastingsGaussianTape(alpha=alpha, learning_length=learning_length, chain_length=chain_length,
                                                                   min_acceptance_rate=min_acceptance_rate, max_acceptance_rate=max_acceptance_rate,
                                                                   initial_sample=initial_sample, acceptance_rate_window=acceptance_rate_window,
                                                                   **kwargs)
            else:
                if trans_dimensional:
                    dimension_jump_prob = kwargs.pop('dimension_jump_prob', 0.01)
                    return IterativeMultipleTryTransDMetropolisHastingsGaussianTape(alpha=alpha, learning_length=learning_length,
                                                                                    chain_length=chain_length,
                                                                                    min_acceptance_rate=min_acceptance_rate,
                                                                                    max_acceptance_rate=max_acceptance_rate,
                                                                                    initial_sample=initial_sample,
                                                                                    acceptance_rate_window=acceptance_rate_window,
                                                                                    dimension_jump_prob=dimension_jump_prob,
                                                                                    number_samples=number_samples, **kwargs)
                else:
                    return IterativeMultipleTryMetropolisHastingsGaussianTape(alpha=alpha, learning_length=learning_length,
                                                                              chain_length=chain_length,
                                                                              min_acceptance_rate=min_acceptance_rate,
                                                                              max_acceptance_rate=max_acceptance_rate,
                                                                              initial_sample=initial_sample,
                                                                              acceptance_rate_window=acceptance_rate_window,
                                                                              number_samples=number_samples, **kwargs)
        # Extension options - Pass all arguments through
        elif mode.lower() in algorithm_names:
            return algorithms[mode.lower()](mode=mode, alpha=alpha, trans_dimensional=trans_dimensional, learning_length=learning_length,
                                            parameterisation=parameterisation, transition=transition, chain_length=chain_length,
                                            min_acceptance_rate=min_acceptance_rate, max_acceptance_rate=max_acceptance_rate,
                                            acceptance_rate_window=acceptance_rate_window, initial_sample=initial_sample,
                                            **kwargs)
        else:
            # Defaults
            return IterativeMetropolisHastingsGaussianTape(alpha=alpha, learning_length=learning_length, chain_length=chain_length,
                                                           min_acceptance_rate=min_acceptance_rate, max_acceptance_rate=max_acceptance_rate,
                                                           acceptance_rate_window=acceptance_rate_window,
                                                           **kwargs)


class MarginalisedMarkovChainMonteCarlo(BaseAlgorithm):

    """
    Marginalised Markov chain Monte Carlo Algorithm

    Markov chain constructed from marginalised MT dist (p(M) not p(M|A), i.e. marginalised over station parameters.

    Default object with basic functions added

    This is a child class of the BaseAlgorithm

    """

    # Returns zero for testing - should be set in __init__
    default_sampling_priors = {'uniform_prior': zero_prior}

    def __init__(self, *args, **kwargs):
        """
        Initialisation of marginalised Markov chain Monte Carlo Algorithm

        Keyword Args
            alpha:[1] Default parameters for Markov chain steps, depend on transition probabilities
            learning_length:[10000] Number of samples for learning period and to discard from chain.
            min_acceptance_rate:[0.3] Minimum targetted sample acceptance rate.
            max_acceptance_rate:[0.5] Maximum targetted sample acceptance rate.
            acceptance_rate_window:[100] Number of samples to use in learning period for calculating and modifying the acceptance rate.
            max_alpha:[100] Maximum values for alpha to take.
            sampling_prior:['uniform_prior'] String to select the prior to use - the default is the uniform prior (Uniform over 6-sphere surface, not parameterisation).
            initial_sample:[None] Initial sample can be set if determined by some other method.
            diagnostic_output:[False] Set algorithm to output all information on iterations - for testing/plotting and debugging.
            min_number_initialisation_samples:[30000] Minimum number of samples for grid based iteration sampler.
            number_samples:[10000] Number of samples to use for each iteration of the grid sampler.
        """
        super(MarginalisedMarkovChainMonteCarlo, self).__init__(*args, **kwargs)
        self.mcmc = True
        self.number_samples = 1
        self._tried = 0
        self._accepted = -1
        self._learning_accepted = []
        # Number of learning samples accepted
        self._number_learning_accepted = 0
        self._min_number_initialisation_samples = kwargs.get('min_number_initialisation_samples', 30000)
        # Handle initial sampling - either random sampling or single
        # sampling/initial sample.
        if kwargs.get('initial_sample', '').lower() == 'grid':
            self._initialiser = IterationSample(number_samples=kwargs.get('number_samples', 10000))
            self._initialising = False
            self.x0 = None
            self._number_initialisation_samples = 0
        else:
            self._initialiser = False
            self._initialising = False
            self.x0 = kwargs.get('initial_sample', None)
        # Default attribute initialisation
        self.xi = None
        self.xi_1 = None
        self.alpha = kwargs.get('alpha', 1)
        self.max_alpha = kwargs.get('max_alpha', 100)
        self.learning_length = kwargs.get('learning_length', 100000)
        self.min_acceptance_rate = kwargs.get('min_acceptance_rate', 0.3)
        self._old_rate = self.min_acceptance_rate
        self.max_acceptance_rate = kwargs.get('max_acceptance_rate', 0.5)
        self.acceptance_rate_window = kwargs.get('acceptance_rate_window', 100)
        self._debug = kwargs.get('diagnostic_output', False)
        self._init_nonzero = False
        self._max_initialisation_probability = -np.inf
        self._init_max_mt = False
        self.p_dc = 0
        self.jump = False
        # Debug flag - adds learning samples and learning results etc to output
        # (large data size)
        if self._debug:
            self._debug_output = {'bis': [np.matrix(np.zeros((6, 0))), []], 'bit': [], 'bir': [], 'cs': [np.matrix(np.zeros((6, 0))), []]}
        # Sets prior function as uniform prior
        gc.collect()
        self.t0 = time.time()
        self.scale_factor_i = False
        # Multiple events
        if self.number_events > 1:
            self.pdf_sample = Sample(number_events=self.number_events)
            all_alpha = []
            dc = self.dc
            self.dc = []
            for i in range(self.number_events):
                all_alpha.append(self.alpha)
                self.dc.append(dc)
            self.p_dc = [0 for i in range(self.number_events)]
            self.alpha = all_alpha
            if self._initialiser:
                self._init_max_mt = [False for i in range(self.number_events)]
                self._max_initialisation_probability = [-np.inf for i in range(self.number_events)]
                self._init_nonzero = [False for i in range(self.number_events)]
                self._initialiser.number_events = self.number_events

    @property
    def total_number_samples(self):
        return self._tried

    def acceptance_rate(self):
        """
        Gets acceptance rate.

        Returns
            float: acceptance rate.
        """
        try:
            if self._number_learning_accepted < self.learning_length or not self._tried:
                rate = float(sum(self._learning_accepted))/len(self._learning_accepted)
            else:
                rate = self._accepted/float(self._tried)
        except ZeroDivisionError:
            rate = 0
        return rate

    def _get_acceptance_rate_modifier(self):
        """
        Gets values for modifying alpha and hence acceptance rate.

        Returns
            float: ratio of actual rate to min or max targetted acceptance rate.
        """
        rate = self.acceptance_rate()

        if rate and rate < 1:
            if rate < self.min_acceptance_rate:
                if (self._old_rate < self.min_acceptance_rate and rate < self._old_rate) or self._old_rate > self.max_acceptance_rate:
                    rate = 1
                ratio = rate/self.min_acceptance_rate
            elif rate > self.max_acceptance_rate:
                if (self._old_rate > self.max_acceptance_rate and rate > self._old_rate) or self._old_rate < self.min_acceptance_rate:
                    rate = 1
                ratio = rate/self.max_acceptance_rate
            else:
                ratio = 1
            ratio = max(ratio, 0.1)
        if rate and rate < 1:
            self._old_rate = rate
            self._old_ratio = ratio
            if type(self.alpha) == dict:
                self._old_alpha = self.alpha.copy()
            elif type(self.alpha) == list:
                self._old_alpha = self.alpha[:]
            else:
                self._old_alpha = self.alpha
        if rate >= 1:
            # revert to old alpha and change ratio
            try:
                if type(self.alpha) == dict:
                    self.alpha = self._old_alpha.copy()
                else:  # list
                    self.alpha = self._old_alpha[:]
                ratio = np.sqrt(self._old_ratio)
                self._old_ratio = ratio
            except Exception:
                ratio = 0
        elif not rate:
            # revert to old alpha and change ratio
            try:
                if type(self.alpha) == dict:
                    self.alpha = self._old_alpha.copy()
                else:  # list
                    self.alpha = self._old_alpha[:]
                ratio = self._old_ratio*self._old_ratio
                self._old_ratio = ratio
            except Exception:
                ratio = 0
        return ratio

    def output(self, *args, **kwargs):
        """
        Returns output dictionary

        Returns
            dict: Output dictionary
        """
        if len(args) < 3:
            kwargs['discard'] = 0
        else:
            args = tuple(args)
            args = args[:2]+(0,)
        output, output_string = super(MarginalisedMarkovChainMonteCarlo, self).output(*args, **kwargs)
        output.update({'acceptance_rate': self.acceptance_rate()})
        output.update({'accepted': self._accepted})
        if self._debug:
            try:
                output.update({'time': self._t1-self.t0})
            except Exception:
                pass
            output.update({'debug_output': self._debug_output})
        try:
            output.pop('ln_bayesian_evidence')
        except Exception:
            pass
        gc.collect()
        return output, output_string

    def _modify_acceptance_rate(self, non_zero_percentage=False):
        """Adjusts the acceptance rate parameters based on the targetted acceptance rate."""
        if non_zero_percentage:
            # Modifies alpha by initial non-zero percentage from initialising
            ratio = non_zero_percentage
        else:
            ratio = self._get_acceptance_rate_modifier()
        # ratio=min(ratio,2.0)#sanity check bounds
        self.alpha = self._modify_alpha(self.alpha, ratio)

    def _modify_alpha(self, alpha, ratio):
        """
        Modifies the transition PDF parameters based on the ratio argument
        The alpha argument is increased up to some max_alpha value
        """
        if isinstance(alpha, dict):
            newAlpha = {}
            for key in alpha.keys():
                if key not in ['gamma_dc', 'delta_dc', 'proposal_normalisation']:
                    newAlpha[key] = alpha[key]*ratio
                    if newAlpha[key] > self.max_alpha[key]:
                        newAlpha[key] = alpha[key]
        elif isinstance(alpha, list):
            newAlpha = []
            for i, alph in enumerate(alpha):
                newAlpha.append(self._modify_alpha(alph, ratio))
        else:
            newAlpha = alpha*ratio
            if newAlpha > self.max_alpha:
                newAlpha = alpha
        return newAlpha

    def transition_pdf(self, x, x1):
        """
        Evaluates transition probability for x and x1

        Returns
            float: transition probability
        """
        return 0

    def prior(self, x, *args, **kwargs):
        """
        Evaluates prior probability for x

        Returns
            float: prior probability
        """
        return self._prior(x, *args, **kwargs)

    def new_sample(self):
        """
        Generates new sample

        Returns
            New sample.
        """
        self.xi_1 = self.random_sample()
        return self.convert_sample(self.xi_1)

    def acceptance(self, x, ln_likelihoodx, dc_prior=0.5):
        """
        Calculate acceptance for x given ln_likelihoodx

        Args
            x: Model values
            ln_likelihoodx: Model ln_likelihood.

        Returns
            float:acceptance
        """
        return 0

    def _acceptance_check(self, xi_1, ln_pi_1, scale_factori_1, dc_prior=0.5):
        """Calculate acceptance"""
        if np.random.rand() <= self.acceptance(xi_1, ln_pi_1, dc_prior):
            return xi_1, ln_pi_1, scale_factori_1, 0
        else:
            return {}, False, False, 0

    def learning_check(self):
        """Check if still in learning period"""
        return self._number_learning_accepted < self.learning_length

    def _add_old(self, i=-1):
        """Add old result (not accepting test result)"""
        if self._debug:
            if self.learning_check():
                key = 'bis'
            else:
                key = 'cs'
            if isinstance(self.xi_1, list) and i >= 0:
                self._debug_output[key][0] = np.append(
                    self._debug_output[key][0], self.convert_sample(self.xi_1[i]), axis=1)
                self._debug_output[key][1].append(0)
                self._debug_output[key][0] = np.append(
                    self._debug_output[key][0], self.convert_sample(self.xi), axis=1)
                self._debug_output[key][1].append(0.5)
            elif isinstance(self.xi_1, list):
                self._debug_output[key][0] = np.append(
                    self._debug_output[key][0], self.convert_sample(self.xi_1[-1]), axis=1)
                self._debug_output[key][1].append(0)
                self._debug_output[key][0] = np.append(
                    self._debug_output[key][0], self.convert_sample(self.xi), axis=1)
                self._debug_output[key][1].append(0.5)
            else:
                self._debug_output[key][0] = np.append(
                    self._debug_output[key][0], self.convert_sample(self.xi_1), axis=1)
                self._debug_output[key][1].append(0)
                self._debug_output[key][0] = np.append(
                    self._debug_output[key][0], self.convert_sample(self.xi), axis=1)
                self._debug_output[key][1].append(0.5)

        self._add(self.xi, self.ln_likelihood_xi, self.scale_factor_i)
        # Not adding to accepted, as want length of unique samples for chain
        if self.learning_check():
            # add a zero to learning acceptance list
            self._learning_accepted.append(0)

    def _add_new(self, xi_1, ln_pi_1, scale_factori_1):
        """Add new result (accepting new value)"""
        if self._debug:
            if self.learning_check():
                key = 'bis'
            else:
                key = 'cs'
            self._debug_output[key][0] = np.append(
                self._debug_output[key][0], self.convert_sample(xi_1), axis=1)
            self._debug_output[key][1].append(1)
        self._add(xi_1, ln_pi_1, scale_factori_1)
        if not self.learning_check():
            self._accepted += 1
        else:
            self._learning_accepted.append(1)
            self._number_learning_accepted += 1

    def _add(self, xi_1, ln_pi_1, scale_factori_1):
        """Add result xi_1 with ln_pdf ln_pi_1"""
        self.xi = xi_1  # Set new sample to old sample
        # set old ln_likelihood to new ln_likelihood
        self.ln_likelihood_xi = float(ln_pi_1)
        # MULTIPLE_EVENTS
        if self.number_events > 1:
            for i, xi in enumerate(self.xi):
                if 'gamma' in xi and xi['gamma'] == 0 and xi['delta'] == 0:
                    if not self.learning_check():
                        self.p_dc[i] += 1
                    self.dc[i] = True
                else:
                    self.dc[i] = False
        # TODO handle warning about element wise comparison - sometimes output is a single element array
        # need to tidy output/array passing
        elif 'gamma' in self.xi and self.xi['gamma'] == 0 and self.xi['delta'] == 0:
            if not self.learning_check():
                self.p_dc += 1
            self.dc = True
        else:
            self.dc = False
        self.scale_factor_i = scale_factori_1
        if not self.learning_check():
            self.pdf_sample.append(self.convert_sample(xi_1), ln_pi_1, 1, scale_factori_1)
            self._tried += 1

    def iterate(self, result):
        """
        Iterate from result

        Args
            result: Result dictionary from forward task (e.g. MTfit.inversion.ForwardTask)

        Returns
            new_sample,End where End is a boolean flag to end the chain.
        """
        try:
            # Check if in initialisation stage
            if self._initialising:
                # Check number samples with max prob.
                if self._initialiser and self.number_events > 1:
                    maximum_initialisation_probability = all([u > -np.inf for u in self._max_initialisation_probability])
                else:
                    maximum_initialisation_probability = self._max_initialisation_probability > -np.inf
                # Multiple events
                if self.number_events > 1 and self._initialiser:
                    for i, individual_result in enumerate(result):
                        ln_pdf = individual_result['ln_pdf']
                        if isinstance(ln_pdf, (np.ndarray, float)):
                            ln_pdf = LnPDF(ln_pdf)
                        if len(ln_pdf.nonzero()):
                            self._init_nonzero[i] += len(ln_pdf.nonzero())
                            # Check if number of initialisation samples<desired
                            # number or if there are no non-zero probability
                            # samples
                            if self._number_initialisation_samples < self._min_number_initialisation_samples or not maximum_initialisation_probability:
                                # Update max_prob/mt combinations for each
                                # event
                                if float(ln_pdf[0, ln_pdf.argmax(-1)]) > self._max_initialisation_probability[i]:
                                    self._max_initialisation_probability[i] = float(ln_pdf[0, ln_pdf.argmax(-1)])
                                    self._init_max_mt[i] = individual_result['moment_tensors'][:, ln_pdf.argmax(-1)[0]]
                # Single event or full joint PDF initialisation
                else:
                    ln_pdf = result['ln_pdf']
                    if isinstance(ln_pdf, (np.ndarray)):
                        ln_pdf = LnPDF(ln_pdf)
                    if len(ln_pdf.nonzero()):
                        self._init_nonzero += len(ln_pdf.nonzero())
                        # Check if number of initialisation samples<desired
                        # number or if there are no non-zero probability
                        # samples
                        if self._number_initialisation_samples < self._min_number_initialisation_samples or not maximum_initialisation_probability:
                            if float(ln_pdf[0, ln_pdf.argmax(-1)]) > self._max_initialisation_probability:
                                # Update max_prob/mt combinations for the event
                                self._max_initialisation_probability = float(ln_pdf[0, ln_pdf.argmax(-1)])
                                if isinstance(result['moment_tensors'], list):
                                    self._init_max_mt = []
                                    for mt in result['moment_tensors']:
                                        self._init_max_mt.append(mt[:, ln_pdf.argmax(-1)[0]])
                                else:
                                    self._init_max_mt = result['moment_tensors'][:, ln_pdf.argmax(-1)[0]]
                # Do quality check and chek if init_max p>0
                if self._initialiser and self.number_events > 1:
                    maximum_initialisation_probability = all([u > -np.inf for u in self._max_initialisation_probability])
                    quality_check_ok = any([100*float(nz)/float(self._number_initialisation_samples) >
                                            self.quality_check for nz in self._init_nonzero])
                else:
                    maximum_initialisation_probability = self._max_initialisation_probability > -np.inf
                    quality_check_ok = 100 * float(self._init_nonzero)/float(self._number_initialisation_samples) > self.quality_check
                # Check if number of initialisation samples>desired number and
                # there are non-zero prob samples
                if self._number_initialisation_samples > self._min_number_initialisation_samples and maximum_initialisation_probability:
                    if (isinstance(self.quality_check, (float, int)) and not isinstance(self.quality_check, bool) or self.quality_check) and quality_check_ok:
                        raise DataError("Data Error: Non-zero sample percentage above {}%".format(self.quality_check))
                    # Have solutions - initialising finished
                    # ADD LOGIC FOR PERCENTAGE NON-ZERO
                    # Need to get and keep track of number of samples
                    # Get percentage non-zero for initialisation and modify the alpha0 values appropriately -
                    #   Large % larger alpha
                    #   Small % smaller alpha
                    #
                    # Simplest is probably to just multiply by decimal of non-zero samples if multiple events
                    # self._modify_acceptance_rate(float(self._init_nonzero)/float(self._number_initialisation_samples))
                    # ADDED DJP 1/9/14 - check values  - killed as alpha large already
                    #
                    # Convert initial sample
                    self.x0 = self.convert_sample(self._init_max_mt)
                    # Initialise samples
                    self.xi = self.x0
                    self.xi_1 = self.x0
                    # As this will always be accepted as initial sample
                    self._tried = -1
                    # Set ln_likelihood so accepted
                    self.ln_likelihood_xi = np.log(0)
                    self.scale_factor_i = False
                    self._initialising = False
                    # Print initialisation output
                    if self.number_events > 1 and self._initialiser:
                        nonzero_events = ""
                        for i, nonzero in enumerate(self._init_nonzero):
                            nonzero_events += 'Event {}: {}% -'.format(i+1, str(100*float(nonzero)/float(self._number_initialisation_samples))[:3])
                    else:
                        nonzero_events = '{}%'.format(str(100*float(self._init_nonzero)/float(self._number_initialisation_samples))[:3])
                    logger.info('Algorithm initialised (Percentage non-zero - {}) - Starting learning period\n---------'.format(nonzero_events))
                    self._initialiser = False
                    # Debug output (large data size)
                    if self._debug:
                        self._debug_output['bis'][0] = np.append(self._debug_output['bis'][0], np.matrix(self._init_max_mt).T, axis=1)
                    # Set t0 after initialisation
                    self.t0 = time.time()
                    self.tx = self.t0
                    return self.convert_sample(self.xi_1), False
                else:
                    return self.initialise()
            # Not in initialisation - main McMC part
            # Check for non-zero MT solutions
            if isinstance(result['ln_pdf'], float) or np.prod(result['ln_pdf'].shape):
                ln_pi_1 = result['ln_pdf']
                if isinstance(ln_pi_1, (float, np.ndarray)):
                    ln_pi_1 = LnPDF(ln_pi_1)
                # assume marginalised
                # Check acceptance (multiple tried events possible) returns
                # accepted sample or empty dict if none accepted
                xi_1, ln_pi_1, scale_factor_i, tried = self._acceptance_check(
                    self.xi_1, ln_pi_1, result.get('scale_factor', False))
                # returns dict not array/list of arrays
                if (self.number_events == 1 and len(xi_1)) or (self.number_events > 1 and all([len(u) for u in xi_1]) and len(xi_1)):
                    # Handling for multiple tries (agnostic)
                    for i in range(tried-1):
                        # Add old samples for tried but not accepted samples
                        self._add_old(i)
                    self._add_new(xi_1, ln_pi_1, scale_factor_i)
                else:
                    # Handling for multiple tries (agnostic)
                    for i in range(tried-1):
                        # Add old samples for tried but not accepted samples
                        self._add_old(i)
                    self._add_old()  # Add old sample as non-accepted
            # Check if need to do learning transition PDF update
            if self.learning_check() and len(self._learning_accepted) >= self.acceptance_rate_window:
                self._modify_acceptance_rate()
                logger.info('Learning {:.4f}% complete - this learning iteration: {} accepted and {}  tried - acceptance rate: {:.6f}'.format(100*self._number_learning_accepted/float(self.learning_length), sum(self._learning_accepted), len(self._learning_accepted), self.acceptance_rate()))
                if self._debug:  # Append rate info to Debug output
                    self._debug_output['bir'].append(float(
                        sum(self._learning_accepted[-self.acceptance_rate_window:]))/len(self._learning_accepted[-self.acceptance_rate_window:]))
                    self._debug_output['bit'].append(time.time()-self.tx)
                    self.tx = time.time()
                self._learning_accepted = []
            # Not in learning
            if not self.learning_check():
                if self._tried == 0:  # First sample
                    # Check and modify transition PDF parameters if within 25%
                    # of the acceptance_rate_window
                    if len(self._learning_accepted[-self.acceptance_rate_window:]) > 0.75*self.acceptance_rate_window:
                        self._learning_accepted = self._learning_accepted[
                            -self.acceptance_rate_window:]
                        self._modify_acceptance_rate()
                        logger.info('Learning complete - this learning iteration: {} accepted and {} tried - acceptance rate: {:.6f}'.format(sum(self._learning_accepted), len(self._learning_accepted), self.acceptance_rate()))
                    # Print out initialisation completion
                    t1 = time.time()
                    logger.info("\nLearning elapsed time: {}".format(t1-self.t0))
                    logger.info("\n\nStarting main inversion\n---------")
                    self.t0 = t1
                    # Add sample (added to saved chain)
                    self._add_new(self.xi, self.ln_likelihood_xi, self.scale_factor_i)
                elif not self._tried % 100:
                    logger.info('{} samples tried: {} samples accepted'.format(self._tried, self._accepted))
            gc.collect()
            return self.new_sample(), False
        # Error handling
        except MemoryError:
            logger.warning('Memory Error, forcing garbage collection and trying again')
            gc.collect()
            return [], True
        except DataError:
            logger.exception('Error with data')
            gc.collect()
            return [], True

    def initialise(self):
        """
        Initialse samples

        Initialises the chain either using an initialiser if set, otherwise using random_sample.

        Returns
            new_sample,False
        """
        # Initialiser - use initialise() function in the initialiser algorithm
        if self._initialiser:
            self._initialising = True
            self._number_initialisation_samples += self._initialiser.number_samples
            return self._initialiser.initialise()
        if not isinstance(self.x0, (np.ndarray, dict, list)):
            self.x0 = self.random_sample()
        if not isinstance(self.x0, (dict, list)):
            self.x0 = self.convert_sample(self.x0)
        # Set initialse sample parameters
        self.xi = self.x0
        self.xi_1 = self.x0
        self._tried = -1  # As this will always be accepted as initial sample
        self.ln_likelihood_xi = 0
        self.scale_factor_i = False
        gc.collect()
        return self.convert_sample(self.xi_1), False

    def _convert_sample_single(self, x):
        """
        Converts single event sample to MT form

        Args
            x: sample

        Returns
            Converted Sample

        """
        return x

    def convert_sample(self, x):
        """
        Converts sample to MT form

        Args
            x: sample

        Returns
            Converted Sample

        """
        if (isinstance(x, np.ndarray) and x.shape[0] != self.number_events*6) or (len(x) != self.number_events and type(x) == list):
            raise ValueError('shape of x '+str(x.shape)+' must match the number of events')
        if isinstance(x, np.ndarray) and self.number_events == 1:
            # dict(zip(['gamma','delta','kappa','h','sigma'],MT6_Tape(x)))
            return self._convert_sample_single(x)
        elif isinstance(x, np.ndarray) and self.number_events > 1:
            data = []
            for i in range(self.number_events):
                data.append(self._convert_sample_single(x[i*6:(i+1)*6, :]))
            return data
        elif isinstance(x, list):
            data = []
            for X in x:
                data.append(self._convert_sample_single(X))
            return data
        else:
            return self._convert_sample_single(x)


class MarginalisedMetropolisHastings(MarginalisedMarkovChainMonteCarlo):

    """
    Marginalised Metropolis Hastings Markov chain Monte Carlo Algorithm

    Markov chain constructed using the Metropolis Hastings method from the marginalised PDF (p(M) not p(M|A), i.e. marginalised over station parameters.

    This is a child class of the MarginalisedMarkovChainMonteCarlo


    """

    def acceptance(self, x, ln_likelihood_x, *args):
        """
        Calculates acceptance

        Calculates the acceptance from the Metropolis condition.

        Args
            x: Model values
            ln_likelihood_x: Model ln_likelihood.

        Returns
            float:acceptance
        """
        if isinstance(self.ln_likelihood_xi, (np.ndarray, LnPDF)):
            self.ln_likelihood_xi = float(self.ln_likelihood_xi)
        if isinstance(ln_likelihood_x, (np.ndarray, LnPDF)):
            ln_likelihood_x = float(ln_likelihood_x)
        # Handle multiple events
        if ln_likelihood_x == -np.inf:
            return 0
        if self.number_events > 1:
            alpha = self.alpha[:]
            acc = 1
            for i in range(self.number_events):
                self.alpha = alpha[i]
                if self.transition_pdf(x[i], self.xi[i], self.dc[i]) > 0 and self.prior(self.xi[i], self.dc[i]) > 0:
                    _numerator = self.transition_pdf(self.xi[i], x[i], self.dc[i])*self.prior(x[i], self.dc[i])
                    _denominator = self.transition_pdf(x[i], self.xi[i], self.dc[i])*self.prior(self.xi[i], self.dc[i])
                    acc *= _numerator/_denominator
                else:
                    self.alpha = alpha[:]
                    # 0 probability values for acceptance denominator so acc is
                    # infinite --> set to 1
                    return 1
            self.alpha = alpha[:]
            return min(1, acc*np.exp(ln_likelihood_x-self.ln_likelihood_xi))
        if self.ln_likelihood_xi > -np.inf and self.transition_pdf(x, self.xi) > 0 and self.prior(self.xi) > 0:
            # ln likelihoods
            acceptance = (self.transition_pdf(self.xi, x)*self.prior(x))/(self.transition_pdf(x, self.xi)*self.prior(self.xi))
            acceptance *= np.exp(ln_likelihood_x-self.ln_likelihood_xi)
            return min(1, acceptance)
        else:
            # 0 probability values for acceptance denominator so acc is
            # infinite --> set to 1
            return 1


class MarginalisedMetropolisHastingsGaussianTape(MarginalisedMetropolisHastings):

    """
    Marginalised Metropolis Hastings Markov chain Monte Carlo Algorithm using Gaussian transition PDF and Tape and Tape parameterisation

    Markov chain constructed using the Metropolis Hastings method from the marginalised pdf (p(M) not p(M|A), i.e. marginalised over station parameters.
    The transition pdf is gaussian and the parameterisation is from Tape and Tape (A geometric setting for moment tensors, Tape and Tape, 2012, GJI 190 pp 476-490).

    This is a child class of the MarginalisedMetropolisHastings

    """

    def __init__(self, *args, **kwargs):
        """
        Initialisation of marginalised Metropolis Hastings Markov chain Monte Carlo Algorithm using gaussian transition pdf and Tape source parameterisation.

        Keyword Args
            alpha: Default parameters for Markov chain steps, depend on transition probabilities Default parameters are:
                    {'kappa':np.pi/5,'h':0.2,'sigma':np.pi/10,'gamma':np.pi/15,
                     'delta':np.pi/10,'alpha':np.pi/10,'poisson':0.2}
            max_alpha: Maximum values for alpha to take. Default parameters are:
                    {'kappa':np.pi/2,'h':0.5,'sigma':np.pi/4,'gamma':np.pi/12,'delta':np.pi/4,
                     'alpha':np.pi/4,'poisson':0.1}
            poisson:[0.25] poisson ratio for CDC model.
            min_poisson:[-np.inf] Minimum poisson ratio for CDC model.
            max_poisson:[np.inf] Maximum poisson ratio for CDC model.

        This is a child class of the MarginalisedMetropolisHastings
        """
        self.default_sampling_priors = {'uniform_prior': uniform_prior, 'flat_prior': flat_prior}
        super(MarginalisedMetropolisHastingsGaussianTape, self).__init__(*args, **kwargs)
        self.alpha = {'kappa': np.pi/5, 'h': 0.2, 'sigma': np.pi/10, 'gamma': np.pi/15,
                      'delta': np.pi/10, 'alpha': np.pi/10, 'poisson': 0.2}
        # Set up alpha for multiple events
        if self.number_events > 1:
            all_alpha = []
            for i in range(self.number_events):
                all_alpha.append(self.alpha)
            self.alpha = all_alpha
        self.max_alpha = {'kappa': np.pi/2, 'h': 0.5, 'sigma': np.pi/4, 'gamma': np.pi/12,
                          'delta': np.pi/4, 'alpha': np.pi/4, 'poisson': 0.1}
        # Reflect PDFs off upper limit - maintains uniformity in transition PDF
        self._reflect_gamma = kwargs.get('reflect_gamma', False)
        self._reflect_delta = kwargs.get('reflect_delta', False)
        self._reflect_sigma = kwargs.get('reflect_sigma', False)
        self._reflect_dip = kwargs.get('reflect_dip', False)
        # alpha is the opening crack angle in the CDC model - it takes values
        # from -pi/2 to pi/2
        self._reflect_alpha = kwargs.get('reflect_alpha', False)
        # N.B. reflecting works because
        # Poisson parameters for CDC solution
        self.poisson = kwargs.get('poisson', 0.25)
        self._max_poisson = kwargs.get('max_poisson', np.inf)
        self._min_poisson = kwargs.get('min_poisson', -np.inf)

    def transition_pdf(self, x, x1, dc=None, basic_cdc=None):
        """
        Calculates gaussian transition pdf

        Evaluates the gaussian transition probability for x and x1

        Returns
            float: transition probability
        """
        # Gaussian makes transition PDF symmetrical but some parameters can be
        # truncated parameters (gamma,delta,h,sigma)
        if len(x) == len(x1) and len(x) > 0:
            if dc is None:
                dc = self.dc
            if basic_cdc is None:
                basic_cdc = self.basic_cdc
            p = 1.
            if not dc and not basic_cdc:
                if not self._reflect_gamma:
                    _numerator = gaussian_pdf(x['gamma'], x1['gamma'], self.alpha['gamma'])
                    _denominator = gaussian_cdf(np.pi/6, x1['gamma'], self.alpha['gamma'])-gaussian_cdf(-np.pi/6, x1['gamma'], self.alpha['gamma'])
                    p *= _numerator/_denominator
                if not self._reflect_delta:
                    _numerator = gaussian_pdf(x['delta'], x1['delta'], self.alpha['delta'])
                    _denominator = gaussian_cdf(np.pi/2, x1['delta'], self.alpha['delta'])-gaussian_cdf(-np.pi/2, x1['delta'], self.alpha['delta'])
                    p *= _numerator/_denominator
            elif self.basic_cdc and not self._reflect_alpha:
                # alpha is opening angle as parameter
                _numerator = gaussian_pdf(x['alpha'], x1['alpha'], self.alpha['alpha'])
                _denominator = gaussian_cdf(np.pi/2, x1['alpha'], self.alpha['alpha'])-gaussian_cdf(-np.pi/2, x1['alpha'], self.alpha['alpha'])
                p *= _numerator/_denominator
            if not self._reflect_dip:
                _numerator = gaussian_pdf(x['h'], x1['h'], self.alpha['h'])
                _denominator = gaussian_cdf(1, x1['h'], self.alpha['h'])-gaussian_cdf(0, x1['h'], self.alpha['h'])
                p *= _numerator/_denominator
            if not self._reflect_sigma:
                _numerator = gaussian_pdf(x['sigma'], x1['sigma'], self.alpha['sigma'])
                _denominator = gaussian_cdf(np.pi/2, x1['sigma'], self.alpha['sigma'])-gaussian_cdf(-np.pi/2, x1['sigma'], self.alpha['sigma'])
                p *= _numerator/_denominator
            p = np.array(p).flatten()
            return float(p[0])
        else:
            return 0

    def _new_sample_single(self):
        """
        Generates new sample

        Generates a new sample by drawing a new sample from the gaussian transition pdf, conditional on the previous sample.

        Returns
            New sample.
        """
        x = {}
        if self.dc:
            x['gamma'] = 0
            x['delta'] = 0
        elif self.basic_cdc:
            xi_alpha = np.acos(-np.sqrt(3)*np.tan(self.xi['gamma']))
            a = self.alpha['alpha']*np.random.randn(1)+xi_alpha
            while a > np.pi or a < 0:
                if self._reflect_alpha:
                    a = np.mod(a, np.pi)
                    if a > np.pi:
                        a = np.pi-a
                    if a < 0:
                        a = -a
                else:
                    a = self.alpha['alpha']*np.random.randn(1)+xi_alpha
            if self.poisson:
                x['gamma'], x['delta'] = basic_cdc_GD(a, self.poisson)
            else:
                tanphi = (
                    np.tan((np.pi/2)-self.xi['delta'])*np.sin(self.xi['gamma']))
                xi_poisson = (1-np.sqrt(2)*tanphi)/(2*np.sqrt(2)*tanphi)
                # handle sampling for poisson
                poisson = self.alpha['poisson']*np.random.randn(1)+xi_poisson
                while poisson < self._min_poisson or poisson > self._max_poisson:
                    poisson = self.alpha['poisson'] * \
                        np.random.randn(1)+xi_poisson
                x['gamma'], x['delta'] = basic_cdc_GD(a, poisson)
        else:
            g = self.alpha['gamma']*np.random.randn(1)+self.xi['gamma']
            while np.abs(g) > np.pi/6:
                if self._reflect_gamma:
                    g = np.sign(g)*np.pi/6+np.sign(g)*(np.pi/6-np.abs(g))
                else:
                    g = self.alpha['gamma']*np.random.randn(1)+self.xi['gamma']
            x['gamma'] = g
            d = self.alpha['delta']*np.random.randn(1)+self.xi['delta']
            while np.abs(d) > np.pi/2:
                if self._reflect_delta:
                    d = np.sign(d)*np.pi/2+np.sign(d)*(np.pi/2-np.abs(d))
                else:
                    d = self.alpha['delta']*np.random.randn(1)+self.xi['delta']
            x['delta'] = d
        # since strike 2*pi=strike 0 wrap round
        x['kappa'] = np.mod(self.alpha['kappa']*np.random.randn(1)+self.xi['kappa'], 2*np.pi)
        h = self.alpha['h']*np.random.randn(1)+self.xi['h']
        while h > 1 or h < 0:
            # wrap h round and change strike by pi
            # h<0 slip changes
            if self._reflect_dip:
                if h < 0:
                    h = np.abs(h)
                if h > 1:
                    h = 1-np.abs(h-1)
            else:
                # no wrap -redraw
                h = self.alpha['h']*np.random.randn(1)+self.xi['h']
        x['h'] = h
        s = self.alpha['sigma']*np.random.randn(1)+self.xi['sigma']
        # tape parameterisation limits s to -pi/2 to pi/2
        # This parameterisation does not wrap in this range - wraps over 2*pi
        # can however wrap sdr to other sdr pair that have rake in valid sigma range
        #
        while np.abs(s) > np.pi/2:
            if self._reflect_sigma:
                # If trying to wrap would be
                # [k,d,s]=SDR_SDR(x['kappa'],np.arccos(x['h']),s)
                # x['kappa']=k
                # x['h']=np.cos(d)
                # alternativly can reflect the PDF instead
                s = np.mod(s-np.sign(s)*0.5*np.pi, np.sign(s)*np.pi)+np.sign(s)*np.pi
                if s > np.pi/2:
                    s = np.pi-s
                if s < -np.pi/2:
                    s = -np.pi+s
            s = self.alpha['sigma']*np.random.randn(1)+self.xi['sigma']
        x['sigma'] = s
        gc.collect()
        return x

    def new_sample(self, *args, **kwargs):
        """
        Generates new sample

        Can handle multiple events
        """
        if self.number_events > 1:
            all_x = self.xi
            if not isinstance(all_x, list):
                raise TypeError('Expect self.xi to be list not {}\n self.xi = {}'.format(type(self.xi), self.xi))
            all_alpha = self.alpha
            x = []
            if not isinstance(all_x[0], dict):
                all_x = self.convert_sample(all_x)
            for i, xi in enumerate(all_x):
                self.xi = xi
                self.alpha = all_alpha[i]
                x.append(self._new_sample_single(*args, **kwargs))
            self.xi = all_x
            self.alpha = all_alpha
        else:
            x = self._new_sample_single(*args, **kwargs)
        self.xi_1 = x
        return self.convert_sample(self.xi_1)

    def is_dc(self, x):
        """Checks if a sample x is double-couple"""
        if not isinstance(x, dict):
            x = self.convert_sample(x)
        return x['gamma'] == 0.0 and x['delta'] == 0.0

    def _convert_sample_single(self, x):
        """
        Converts sample to and from tape parameterisation

        Args
            x: sample - can be a dict of Tape params -> converted to MT or an MT -> converted to a dict of Tape params

        Returns
            Converted Sample

        """
        if isinstance(x, np.ndarray):
            return dict(zip(['gamma', 'delta', 'kappa', 'h', 'sigma'], MT6_Tape(x)))
        if cmarkov_chain_monte_carlo:
            # Try c functions (quicker)
            return cmarkov_chain_monte_carlo.convert_sample(x['gamma'], x['delta'], x['kappa'], x['h'], x['sigma'])
        else:
            logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
        return MT33_MT6(Tape_MT33(**x))

    def _6sphere_random_mt(self):
        """
        Generates a random MT

        Generates a random MT using flat pdfs for each parameter.

        Returns
            Random MT in Tape parameterisation

        """
        x = {}
        x['gamma'] = (np.random.rand()*2-1)*np.pi/6
        x['delta'] = (np.random.rand()*2-1)*np.pi/2
        x['kappa'] = np.random.rand()*2*np.pi
        x['h'] = np.random.rand()
        x['sigma'] = (np.random.rand()*2-1)*np.pi/2
        return x

    def random_dc(self):
        """
        Generates a random DC

        Generates a random DC using flat pdfs for each parameter (gamma=delta=0).

        Returns
            Random MT in Tape parameterisation

        """
        x = {}
        x['gamma'] = 0
        x['delta'] = 0
        x['kappa'] = np.random.rand()*2*np.pi
        x['h'] = np.random.rand()
        x['sigma'] = (np.random.rand()*2-1)*np.pi/2
        return x

    def random_basic_cdc(self):
        """
        Generates a random C+DC

        Generates a random C+DC using flat pdfs for each parameter (gamma,delta given by alpha).

        Returns
            Random MT in Tape parameterisation

        """
        x = {}
        alpha = np.random.rand()*np.pi
        x['gamma'], x['delta'] = basic_cdc_GD(alpha, self.poisson)
        x['kappa'] = np.random.rand()*2*np.pi
        x['h'] = np.random.rand()
        x['sigma'] = (np.random.rand()*2-1)*np.pi/2
        return x

    def prior(self, x, dc=None, basic_cdc=None, *args, **kwargs):
        """
        Evaluates prior probability for x

        Returns
            float: prior probability
        """
        if basic_cdc is None:
            basic_cdc = self.basic_cdc
        max_poisson = self._max_poisson
        min_poisson = self._min_poisson
        return super(MarginalisedMetropolisHastingsGaussianTape, self).prior(x, dc, basic_cdc, max_poisson, min_poisson)


class IterativeMetropolisHastingsGaussianTape(MarginalisedMetropolisHastingsGaussianTape):

    """
    Metropolis-Hastings Markov chain Monte Carlo Algorithm using Gaussian Transition PDF and Tape parameterisation, chain length ends maximum length

    Algorithm ends when the number of samples in the chain equals the chain_length, where the chain length corresponds to the number of accepted samples.
    The parameterisation is from Tape and Tape (A geometric setting for moment tensors, Tape and Tape, 2012, GJI 190 pp 476-490).

    This is a child class of the MarginalisedMetropolisHastingsGaussianTape class

    """

    def __init__(self, *args, **kwargs):
        """
        Initialisation of Metropolis-Hastings Markov chain Monte Carlo Algorithm using Gaussian Transition PDF and Tape parameterisation, constrained by chain length

        The chain length corresponds to the number of accepted samples.

        Keyword Args
            chain_length:[1000000] Maximum length of the Markov chain.

        This is a child class of the MarginalisedMetropolisHastingsGaussianTape class
        """
        super(IterativeMetropolisHastingsGaussianTape, self).__init__(*args, **kwargs)
        self.chain_length = kwargs.get('chain_length', 1000000)

    def iterate(self, result):
        """
        Iterate from result

        Args
            result: Result dictionary from forward task (e.g. MTfit.inversion.ForwardTask)

        Returns
            new_sample,End where End is a boolean flag to end the chain if the length of accepted samples is longer than the chain length.

        """
        task, end = super(IterativeMetropolisHastingsGaussianTape, self).iterate(result)
        # Check chain length
        if self._tried >= self.chain_length:
            self._t1 = time.time()
            logger.info('\nChain complete\nChain elapsed time: {}\n'.format(self._t1-self.t0))
            return [], True
        else:
            return task, end


class IterativeTransDMetropolisHastingsGaussianTape(IterativeMetropolisHastingsGaussianTape):

    """
    Trans-Dimensional Metropolis-Hastings Markov chain Monte Carlo Algorithm using Gaussian Transition PDF and Tape parameterisation, chain length ends maximum length

    Trans Dimensional sampling, jumping between double-couple and full moment tensor models.
    Algorithm ends when the number of samples in the chain equals the chain_length, where the chain length corresponds to the number of accepted samples.
    The parameterisation is from Tape and Tape (A geometric setting for moment tensors, Tape and Tape, 2012, GJI 190 pp 476-490).

    This is a child class of the IterativeMetropolisHastingsGaussianTape

    """

    def __init__(self, *args, **kwargs):
        """
        Initialisation of Trans-Dimensional Metropolis-Hastings Markov chain Monte Carlo Algorithm using Gaussian Transition PDF and Tape parameterisation, chain length ends maximum length

        Keyword Args
            dimension_jump_prob:[0.01] Probability of making a dimension jump.
            dc_sigma_g:[0.2] alpha parameter for the balance vector gamma parameter
            dc_sigma_d:[0.2] alpha parameter for the balance vector delta parameter
            dc_prior:[0.5] prior probability for the DC model

        This is a child class of the IterativeMetropolisHastingsGaussianTape

        """
        super(IterativeTransDMetropolisHastingsGaussianTape, self).__init__(*args, **kwargs)
        # Update alpha with dc balance vector parameters
        if self.number_events > 1:
            for alpha in self.alpha:
                if 'gamma_dc' not in alpha.keys():
                    alpha['gamma_dc'] = kwargs.get('dc_sigma_g', .2)
                if 'delta_dc' not in alpha.keys():
                    alpha['delta_dc'] = kwargs.get('dc_sigma_d', .2)
        else:
            if 'gamma_dc' not in self.alpha.keys():
                self.alpha['gamma_dc'] = kwargs.get('dc_sigma_g', .2)
            if 'delta_dc' not in self.alpha.keys():
                self.alpha['delta_dc'] = kwargs.get('dc_sigma_d', .2)
        # Set DC model prior
        self.dc_prior = kwargs.get('dc_prior', .5)
        # Calculate balance vector proposal normalisation
        d = np.linspace(-np.pi/2, np.pi/2, 10000)
        pn = sum(np.cos(d)*gaussian_pdf(d, 0, kwargs.get('dc_sigma_d', .2)) *
                 np.mean(np.diff(d)))*(1-2*gaussian_cdf(-np.pi/6, 0, kwargs.get('dc_sigma_g', .2)))
        # Add proposal normalisation to alpha
        if self.number_events > 1:
            for alpha in self.alpha:
                alpha['proposal_normalisation'] = pn
        else:
            self.alpha['proposal_normalisation'] = pn
        # Set jump options
        self.dimension_jump_prob = kwargs.get('dimension_jump_prob', 0.01)
        self.jump = False
        self.gaussian_jump_params = kwargs.get('gaussian_jump_params', True)
        self._learning_dc_dc_accepted = []
        self._learning_mt_mt_accepted = []

    def _new_sample_single(self, *args):
        """
        Generates new sample

        Generates a new sample by drawing a new sample from the gaussian transition pdf, conditional on the previous sample.
        Probability if the sample jumping between models given by the dimension_jump_prob kwarg on initialisation.

        Returns
            New sample.
        """
        # Set jump check
        self.jump = np.random.rand() <= self.dimension_jump_prob
        if self.jump:
            # new sample is same parameters but with or without gamma and delta
            x = copy.copy(self.xi)
            x.pop('gamma')
            x.pop('delta')
            try:
                x.pop('g0')
                x.pop('d0')
            except Exception:
                pass
            if self.dc:  # dc to mt
                # set random gamma beta
                gamma, delta = self.jump_params()
                x['gamma'] = gamma
                x['delta'] = delta
            else:  # mt to dc (g0 and d0 set from gamma delta)
                x['g0'] = self.xi['gamma']
                x['d0'] = self.xi['delta']
                x['gamma'] = 0.0
                x['delta'] = 0.0
            self.xi_1 = x
            return self.xi_1
        else:
            # Normal new sample
            return super(IterativeTransDMetropolisHastingsGaussianTape, self)._new_sample_single()

    def jump_params(self, x=False):
        """
        Calculates dimension jump parameters

        Calculates either the probabilities of the balancing parameters if x exists, else returns the two balancing parameters.

        Args
            x:[False] Can be given as the two balancing parameters

        Returns
            if x is not  False:
                float: q - probability of balancing parameters
            else:
                float: gamma - Randomly distributed gamma value
                float: delta - Randomly distributed delta value
        """
        # If x is passed then this function returns the probability of
        # obtaining the two balance paraemters
        if x:
            if self.gaussian_jump_params:
                # corresponds to two normal pdfs about 0,0 with widths given by
                # sigma normalised over the lune
                q = gaussian_pdf(x['gamma'], 0, self.alpha['gamma_dc'])
                q *= gaussian_pdf(x['delta'], 0, self.alpha['delta_dc'])
                q /= self.alpha['proposal_normalisation']
            else:
                q = 3/(2*np.pi)  # Scaled uniform prob
            return q
        else:
            # Get jump params
            if self.gaussian_jump_params:
                gamma = -np.pi/2
                delta = -np.pi
                # Not wrapped/reflected
                while np.abs(gamma) > np.pi/6:
                    gamma = self.alpha['gamma_dc']*np.random.randn()
                while np.abs(delta) > np.pi/2:
                    delta = self.alpha['delta_dc']*np.random.randn()
            else:
                gamma = (np.random.rand()*2-1)*np.pi/6
                delta = (np.random.rand()*2-1)*np.pi/2
            return gamma, delta

    def acceptance(self, x, ln_likelihoodx, dc_prior=0.5):
        """
        Calculates acceptance

        Calculates the acceptance from the Trans-Dimensional Metropolis condition.

        Args
            x: Model values
            ln_likelihoodx: Model likelihood.

        Returns
            float:acceptance
        """
        if isinstance(self.ln_likelihood_xi, np.ndarray):
            self.ln_likelihood_xi = float(self.ln_likelihood_xi)
        # Handle jump parameters
        if self.jump and (self.ln_likelihood_xi > -np.inf):
            # No jump - may have been accidentally picked up
            if self.xi['gamma'] == x['gamma'] and self.xi['delta'] == x['delta']:
                return super(IterativeTransDMetropolisHastingsGaussianTape, self).acceptance(x, ln_likelihoodx)
            # dc to mt jump
            elif self.xi['gamma'] == 0.0 and self.xi['delta'] == 0.0:
                xi = copy.copy(self.xi)
                xi.pop('gamma')
                xi.pop('delta')
                model_prior_ratio = (1-dc_prior)/dc_prior
                # LnLikelihoods    N.B> self.jump_params(x) called as x
                # contains gamma delta generated using jump Params
                return min(1, (self.prior(x)/(self.jump_params(x)*self.prior(xi)))*model_prior_ratio*np.exp(ln_likelihoodx-self.ln_likelihood_xi))
            elif x['gamma'] == 0.0 and x['delta'] == 0.0:  # mt to dc jump
                xp = copy.copy(x)
                xp.pop('gamma')
                xp.pop('delta')
                model_prior_ratio = dc_prior/(1-dc_prior)
                # LnLikelihoods    N.B> self.jump_params(self.xi) called as
                # self.xi contains gamma delta generated using jump Params
                return min(1, (self.jump_params(self.xi)*self.prior(xp))/(self.prior(self.xi))*model_prior_ratio*np.exp(ln_likelihoodx-self.ln_likelihood_xi))
            else:  # no jump may have been accidentally picked up
                return super(IterativeTransDMetropolisHastingsGaussianTape, self).acceptance(x, ln_likelihoodx)
        else:
            return super(IterativeTransDMetropolisHastingsGaussianTape, self).acceptance(x, ln_likelihoodx)

    def iterate(self, *args, **kwargs):
        """
        Iterates new sample

        Calculates the acceptance and handles the pDC output
        """
        MTs, end = super(IterativeTransDMetropolisHastingsGaussianTape, self).iterate(*args, **kwargs)
        # Print pDC output
        if end:
            if isinstance(self.p_dc, float):
                logging.info('Probability of dc: {:.6f}'.format(float(self.p_dc)/float(len(self.pdf_sample.ln_pdf))))
            elif isinstance(self.p_dc, list):
                for i, pdc in enumerate(self.p_dc):
                    logging.info('Event - {} Probability of dc: {:.6f}'.format(i, float(pdc)/float(len(self.pdf_sample.ln_pdf))))
        if self.jump:
            # check current sample
            self.dc = (self.xi['gamma'] == 0.0 and self.xi['delta'] == 0.0)
        return MTs, end

    def output(self, *args, **kwargs):
        """
        Return output dict

        Return output dict including pDC value
        """
        output, output_string = super(IterativeTransDMetropolisHastingsGaussianTape, self).output(*args, **kwargs)
        output['pDC'] = self.p_dc
        return output, output_string


class IterativeMultipleTryMetropolisHastingsGaussianTape(IterativeMetropolisHastingsGaussianTape):

    """
    Metropolis-Hastings Markov chain Monte Carlo Algorithm using Gaussian Transition PDF and Tape parameterisation, chain length ends maximum length

    Algorithm ends when the number of samples in the chain equals the chain_length, where the chain length corresponds to the number of accepted samples.
    The parameterisation is from Tape and Tape (A geometric setting for moment tensors, Tape and Tape, 2012, GJI 190 pp 476-490).

    This is a child class of the IterativeMetropolisHastingsGaussianTape

    """

    def __init__(self, *args, **kwargs):
        """
        Initialisation of Metropolis-Hastings Markov chain Monte Carlo Algorithm using Gaussian Transition PDF and Tape parameterisation, constrained by chain length

        The chain length corresponds to the number of accepted samples.

        Keyword Args
            number_samples:[1000] Maximum number of samples to try on each iteration. (unused samples are dropped)

        This is a child class of the IterativeMetropolisHastingsGaussianTape

        """
        super(IterativeMultipleTryMetropolisHastingsGaussianTape, self).__init__(*args, **kwargs)
        self._max_number_samples = kwargs.get('number_samples', 1000)
        self._number_samples = min(int(1/self.min_acceptance_rate), self._max_number_samples)

    def new_sample(self, jump=0.0, gaussian_jump=False):
        """Get new samples including multiple samples to try"""
        if cmarkov_chain_monte_carlo:
            try:
                # Try c code
                # Multiple events
                if self.number_events > 1:
                    self.xi_1 = []
                    mt = []
                    for i in range(self.number_events):
                        xi_1, mt_i = cmarkov_chain_monte_carlo.new_samples(self.xi[i],
                                                                           self.alpha[i],
                                                                           self._number_samples,
                                                                           self.dc[i],
                                                                           jump=jump,
                                                                           gaussian_jump=gaussian_jump)
                        self.xi_1.append(xi_1)
                        mt.append(mt_i)
                # Single event
                else:
                    self.xi_1, mt = cmarkov_chain_monte_carlo.new_samples(self.xi,
                                                                          self.alpha,
                                                                          self._number_samples,
                                                                          self.dc,
                                                                          jump=jump,
                                                                          gaussian_jump=gaussian_jump)
                return mt
            except Exception:
                logging.exception('Cython error')
        else:
            logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
        # Otherwise/Fallback to use python code
        return super(IterativeMultipleTryMetropolisHastingsGaussianTape, self).new_sample()

    def _acceptance_check(self, xi_1, ln_pi_1, scale_factori_1=False):
        """Check acceptance for multiple tries"""
        # TODO tidy this code up
        if cmarkov_chain_monte_carlo:
            try:
                # Try C code
                if isinstance(ln_pi_1, LnPDF):
                    ln_pi_1 = ln_pi_1._ln_pdf
                # No non-zero samples
                if ln_pi_1.max() == -np.inf:
                    xi_1 = {}
                    index = ln_pi_1.shape[1]
                    if self.number_events > 1:
                        xi_1 = [{} for i in range(self.number_events)]
                    ln_pi_1 = False
                    # If in learning, increase the number of test samples
                    if self.learning_check():
                        self._number_samples = min([self._max_number_samples, max(
                            int(1.1*self._number_samples), self._number_samples+1)])
                    return xi_1, ln_pi_1, False, index
                else:
                    if sys.version_info.major > 2:
                        is_uniform = self._prior.__name__ == 'uniform_prior'
                    else:
                        is_uniform = self._prior.func_name == 'uniform_prior'
                    # Mutliple events
                    if self.number_events > 1:
                        if hasattr(self, 'dc_prior') and isinstance(self.dc_prior, (float, int)):
                            self.dc_prior = np.array([self.dc_prior])
                        if isinstance(ln_pi_1, np.ndarray):
                            xi_1, ln_pi_1, index = cmarkov_chain_monte_carlo.me_acceptance_check(xi_1,
                                                                                                 self.xi,
                                                                                                 self.alpha,
                                                                                                 np.asarray(ln_pi_1).flatten(),
                                                                                                 self.ln_likelihood_xi,
                                                                                                 is_uniform,
                                                                                                 gaussian_jump=getattr(self, 'gaussian_jump_params', False),
                                                                                                 dc_prior=getattr(self, 'dc_prior', np.array([0.])))  # dc prior ignored if not transd
                        else:
                            xi_1, ln_pi_1, index = cmarkov_chain_monte_carlo.me_acceptance_check(xi_1,
                                                                                                 self.xi,
                                                                                                 self.alpha,
                                                                                                 np.asarray(ln_pi_1._ln_pdf).flatten(),
                                                                                                 self.ln_likelihood_xi,
                                                                                                 is_uniform,
                                                                                                 gaussian_jump=getattr(self, 'gaussian_jump_params', False),
                                                                                                 dc_prior=getattr(self, 'dc_prior', np.array([0.])))
                    # Single events
                    else:
                        if isinstance(ln_pi_1, np.ndarray):
                            xi_1, ln_pi_1, index = cmarkov_chain_monte_carlo.acceptance_check(xi_1,
                                                                                              self.xi,
                                                                                              self.alpha,
                                                                                              np.asarray(ln_pi_1).flatten(),
                                                                                              self.ln_likelihood_xi,
                                                                                              is_uniform,
                                                                                              gaussian_jump=getattr(self, 'gaussian_jump_params', False),
                                                                                              dc_prior=getattr(self, 'dc_prior', 0.))
                        else:
                            xi_1, ln_pi_1, index = cmarkov_chain_monte_carlo.acceptance_check(xi_1,
                                                                                              self.xi,
                                                                                              self.alpha,
                                                                                              np.asarray(ln_pi_1._ln_pdf).flatten(),
                                                                                              self.ln_likelihood_xi,
                                                                                              is_uniform,
                                                                                              gaussian_jump=getattr(self, 'gaussian_jump_params', False),
                                                                                              dc_prior=getattr(self, 'dc_prior', 0.))
                # No accepted samples, so increase the number of test samples if in
                # learning period
                if isinstance(ln_pi_1, bool) and self.learning_check:
                    self._number_samples = min([self._max_number_samples, max(int(1.1*self._number_samples), self._number_samples+1)])
                elif self.learning_check() and min([120*(max(index, 1)), int(self._number_samples)]) != self._number_samples:
                    # Handle too many samples
                    # Probability of having accepted sample within index is
                    #
                    #   P(x<=j)=1-(1-r)^j
                    #
                    # where r is the probability of accepting a sample and j is the accepted sample index
                    # Solving for this given r~1/number_samples for P=0.01 gives j~0.01*number_samples
                    # Consequently, the index of the accepted sample has a probability of 0.01 of being < 0.01*number_samples (given the number_samples -> r estimate)
                    # The factor of 120 is a fudge to account for uncertainties
                    # (this can help prevent calculating too many forward models)
                    # max prevents 0 for _number_samples
                    self._number_samples = min([120*(max(index, 1)), int(self._number_samples)])
                # Accepted sample with scale_factor (relative inversion)
                if not isinstance(scale_factori_1, bool) and not isinstance(ln_pi_1, bool):
                    return xi_1, ln_pi_1, scale_factori_1[index], index
                return xi_1, ln_pi_1, False, index
            except Exception:
                logger.exception('Cython Error')
        else:
            logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
        # Otherwise use/fallback to Python code
        # Non-zero samples
        if not isinstance(ln_pi_1, bool) and np.prod(ln_pi_1.shape) > 1:
            if not isinstance(ln_pi_1, np.ndarray):
                ln_pi_1 = ln_pi_1._ln_pdf.transpose()
            # Loop over samples
            for u in range(ln_pi_1.shape[0]):
                if self.number_events > 1:
                    _xi_1 = []
                    if isinstance(xi_1[0], list):
                        for v in range(self.number_events):
                            _xi_1.append(xi_1[v][u])
                    else:
                        _xi_1 = xi_1

                else:
                    if isinstance(xi_1, list):
                        _xi_1 = xi_1[u]
                    else:
                        _xi_1 = xi_1
                # Get acceptance result
                oxi_1, oln_pi_1, a, b = super(IterativeMultipleTryMetropolisHastingsGaussianTape, self)._acceptance_check(_xi_1, np.asarray(ln_pi_1).flatten()[u],
                                                                                                                          False, dc_prior=getattr(self, 'dc_prior', False))
                if oln_pi_1:
                    # If accepted sample, try to get the sacale_factor
                    if not isinstance(scale_factori_1, bool):
                        oscale_factori_1 = scale_factori_1[u]
                    else:
                        oscale_factori_1 = False
                    return oxi_1, oln_pi_1, oscale_factori_1, u
            # No Accepted samples
            if self.number_events > 1:
                if isinstance(self.xi_1[0], dict):
                    tried = 1
                else:
                    tried = len(self.xi_1[0])
                return [{} for w in range(self.number_events)], False, False, tried

            if isinstance(self.xi_1, dict):
                tried = 1
            else:
                tried = len(self.xi_1)
            return {}, False, False, tried
        # No accepted samples
        elif isinstance(ln_pi_1, bool):
            if self.number_events > 1:
                if isinstance(self.xi_1[0], dict):
                    tried = 1
                else:
                    tried = len(self.xi_1[0])
                return [{} for w in range(self.number_events)], False, False, tried

            if isinstance(self.xi_1, dict):
                tried = 1
            else:
                tried = len(self.xi_1)
            return {}, False, False, tried
        elif ln_pi_1 == -np.inf:
            if self.number_events > 1:
                if self.number_events > 1:
                    if isinstance(self.xi_1[0], dict):
                        tried = 1
                    else:
                        tried = len(self.xi_1[0])
                return [{} for w in range(self.number_events)], False, False, tried

            if isinstance(self.xi_1, dict):
                tried = 1
            else:
                tried = len(self.xi_1)
            return {}, False, False, tried
        else:
            if self.number_events > 1:
                dc = [False for i in range(self.number_events)]
            else:
                dc = False
            return super(IterativeMultipleTryMetropolisHastingsGaussianTape, self)._acceptance_check(xi_1, ln_pi_1, scale_factori_1, dc_prior=getattr(self, 'dc_prior', dc))

    def _modify_acceptance_rate(self, non_zero_percentage=False):
        """Adjusts the acceptance rate parameters based on the targetted acceptance rate."""
        super(IterativeMultipleTryMetropolisHastingsGaussianTape, self)._modify_acceptance_rate(non_zero_percentage)
        self._number_samples = min(int(1/self.min_acceptance_rate), self._max_number_samples)


class IterativeMultipleTryTransDMetropolisHastingsGaussianTape(IterativeTransDMetropolisHastingsGaussianTape, IterativeMultipleTryMetropolisHastingsGaussianTape):

    """
    Trans-Dimensional Metropolis-Hastings Markov chain Monte Carlo Algorithm using Gaussian Transition PDF and Tape parameterisation, chain length ends maximum length

    Trans Dimensional sampling, jumping between double-couple and full moment tensor models.
    Algorithm ends when the number of samples in the chain equals the chain_length, where the chain length corresponds to the number of accepted samples.
    The parameterisation is from Tape and Tape (A geometric setting for moment tensors, Tape and Tape, 2012, GJI 190 pp 476-490).

    This is a child class of both the IterativeTransDMetropolisHastingsGaussianTape and IterativeMultipleTryMetropolisHastingsGaussianTape

    """

    def __init__(self, *args, **kwargs):
        """
        Initialisation of Trans-Dimensional Metropolis-Hastings Markov chain Monte Carlo Algorithm using Gaussian Transition PDF and Tape parameterisation, chain length ends maximum length


        This is a child class of both the IterativeTransDMetropolisHastingsGaussianTape and IterativeMultipleTryMetropolisHastingsGaussianTape

        """
        super(IterativeMultipleTryTransDMetropolisHastingsGaussianTape, self).__init__(*args, **kwargs)

    def new_sample(self):
        """
        Generates new sample

        Generates a new sample by drawing a new sample from the gaussian transition pdf, conditional on the previous sample.
        Probability if the sample jumping between models given by the dimension_jump_prob kwarg on initialisation.

        Returns
            New sample.
        """
        return IterativeMultipleTryMetropolisHastingsGaussianTape.new_sample(self, self.dimension_jump_prob, self.gaussian_jump_params)
