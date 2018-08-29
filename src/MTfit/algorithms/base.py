"""
algorithms/base.py
******************

Basic algorithm class for MTfit
"""


# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.


import logging
import types
import sys

import numpy as np

from ..sampling import Sample, FileSample, _6sphere_prior
from ..utilities.extensions import get_extensions
from ..utilities import C_EXTENSION_FALLBACK_LOG_MSG


logger = logging.getLogger('MTfit.algorithms')


try:
    from ..probability import cprobability
except ImportError:
    cprobability = None
except Exception:
    logger.exception('Error importing c extension')
    cprobability = None


# No. of iterations to print output on
PRINT_N_ITERATIONS = 20


class BaseAlgorithm(object):

    """
    Base class for algorithms including the base methods required for forward
    modelling based inversions.
    """

    # Returns zero for testing - should be set in __init__
    default_sampling_priors = {'6sphere_prior': _6sphere_prior}

    def __init__(self, number_samples=10000, dc=False, quality_check=False,
                 file_sample=False, file_safe=True, generate=False,
                 *args, **kwargs):
        """
        BaseAlgorithm initialisation

        Args
            number_samples:[10000] Number of samples to use per iteration.
            dc:[False] Boolean to select inversion constrained to double-couple
                space or over the full moment tensor space.
            quality_check:[False] Carry out quality check after 40,000 samples.
                If there are too many non-zero samples, the inversion is stopped
                for that event.
            file_sample:[False] Boolean to select whether to use file sampling
                (saving the output to file).
            file_safe:[False] Boolean to select whether to use file safe outputs
                when using file sampling.
            generate:[False] Boolean to select whether to generate the random
                moment tensors in the forward task or not.

        Keyword Arguments
            basic_cdc:[False] Boolean to select whether to use the Basic CDC
                model (inversion constrained to crack+double-couple space).
            number_events:[1] Integer giving the number of events in this
                inversion (for use with multiple event inversions).
            sampling_prior:[6sphere_prior] string selector for the full moment
                tensor sampling prior (MTfit.sampling_prior entry point)
            sampling:[6sphere] string selector for the full moment tensor
                sampling model (MTfit.sampling entry point)
            model:[False] selector for alternate models using the
                MTfit.sample_distribution entry_point
        """
        self.number_samples = number_samples
        self.dc = dc
        self.mcmc = False
        self.basic_cdc = kwargs.get('basic_cdc', False)
        self.number_events = kwargs.get('number_events', 1)
        self.generate = generate
        self.quality_check = quality_check
        self._model = kwargs.get('sample_distribution', False)
        self.get_sampling_model(kwargs, file_sample, file_safe)

    def get_sampling_model(self, kwargs, file_sample, file_safe):
        """Get the sampling model from the entry points"""
        if self._model:
            # Assume that prior is the correct form for the sampling (either
            # correcting or not so that Bayesian evidence calculation is ok
            model_names, model = get_extensions('MTfit.sample_distribution', {'clvd': self.random_clvd})
            try:
                try:
                    if sys.version_info.major > 3:
                        self.random_model = getattr(self, model[self._model].__name__)
                    else:
                        self.random_model = getattr(self, model[self._model].func_name)
                except AttributeError:
                    self.random_model = types.MethodType(model[self._model], self)
            except Exception:
                logger.exception('Error setting prior: {}\n'.format(self._model))
                self._model = False
        sampling = {'6sphere': _6sphere_random_mt}

        sampling_names, sampling = get_extensions('MTfit.sampling', sampling)
        # Check sampling distribution selection
        if not kwargs.get('sampling', '6sphere') in sampling_names:
            kwargs['sampling'] = '6sphere'
        # Set prior sampling as random_mt - check if the method already exists in self
        # and use that, otherwise use types to add method

        # Assume that prior is the correct form for the sampling (either
        # correcting or not so that Bayesian evidence calculation is ok
        sampling_prior_names, sampling_prior = get_extensions(
            'MTfit.sampling_prior', self.default_sampling_priors)

        # Check sampling_prior distribution selection
        if not kwargs.get('sampling_prior', '6sphere') in sampling_prior_names:
            kwargs['sampling_prior'] = sorted(list(self.default_sampling_priors.keys()), reverse=True)[0]
        try:
            self._prior = sampling_prior[kwargs.get('sampling_prior', list(self.default_sampling_priors.keys())[0])]
        except Exception:
            logger.exception('Error setting prior: {}, using default: {}\n'.format(
                kwargs.get('sampling_prior', list(self.default_sampling_priors.keys())[0]),
                list(self.default_sampling_priors.keys())[0]))
            self._prior = sampling_prior[list(self.default_sampling_priors.keys())[0]]
        try:
            if sys.version_info.major > 2:
                self.random_mt = getattr(self, sampling[kwargs.get('sampling', '6sphere')].__name__)
            else:
                self.random_mt = getattr(self, sampling[kwargs.get('sampling', '6sphere')].func_name)

        except AttributeError:
            self.random_mt = types.MethodType(sampling[kwargs.get('sampling', '6sphere')], self)
        if file_sample:
            self.pdf_sample = FileSample(fname=kwargs.get('fid', 'MTfit_run'),
                                         number_events=self.number_events,
                                         file_safe=file_safe, prior=self._prior)
        else:
            self.pdf_sample = Sample(number_events=self.number_events, prior=self._prior)

    def max_value(self):
        return 'BaseAlgorithm has no max_value'

    def random_sample(self):
        """
        Return random samples

        Returns:
            results from generating the random samples.
        """
        if self.generate:
            return False

        # Return random samples
        if self.dc:
            return self.random_dc()
        elif self.basic_cdc:
            return self.random_basic_cdc()
        elif self._model:
            return self.random_model(self.number_samples)
        else:
            return self.random_mt()

    def output(self, normalise=True, convert=False, discard=10000):
        """
        Return the algorithm results for output.

        Returns
            Dictionary containing output.
        """
        output_string = ''
        # Check if any PDF samples
        if len(self.pdf_sample):
            output, output_string = self.pdf_sample.output(normalise, convert,
                                                           self.total_number_samples,
                                                           discard, self.mcmc)
        else:
            output = {'probability': []}
        output.update({'total_number_samples': self.total_number_samples})
        return output, output_string

    def iterate(self, result):
        """
        Basic iteration function

        Args
            result: Result from forward Task

        Returns
            task,End
            task: New task for forward model
            End: Boolean flag for whether the inversion is finished.

        """
        # Function is a place-holder for more complex algorithms
        task = [1]
        End = True
        return task, End

    def initialise(self):
        """
        Basic initialisation function

        Return the first task

        Returns
            task, False
            task: New task for forward model

        """
        # Function is a place-holder for more complex algorithms
        task = [1]
        return task, False

    def random_dc(self, *args):
        """
        Generate random double-couple  moment tensors (size 6, number_samples)

        Generates random double-couple moment tensors with random orientations.

        Returns
            numpy matrix of random double-couple moment tensors, size 6,number_samples

        """
        # Check CYTHON code - use C code if possible
        if cprobability:
            return cprobability.random_dc(self.number_samples)
        else:
            logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
        # Use random_type for DC eigenvalues
        dc_diag = np.array([[1/np.sqrt(2)], [0], [-1/np.sqrt(2)]])
        return self.random_type(dc_diag)

    def random_clvd(self, *args):
        """
        Generate random CLVD moment tensors (size 6,number_samples)

        Generates random CLVD moment tensors with random orientations.

        Returns
            numpy matrix of random CLVD moment tensors, size 6,number_samples

        """
        np.random.seed()
        # Select CVLD type
        np.kron([[1], [1], [1]], np.sign(np.random.rand(self.number_samples)-0.5))*np.ones(
            (3, self.number_samples))*np.array([[2/np.sqrt(6)], [-1/np.sqrt(6)], [-1/np.sqrt(6)]])
        if np.random.rand() > 0.5:
            clvd_diag = np.array(
                [[2/np.sqrt(6)], [-1/np.sqrt(6)], [-1/np.sqrt(6)]])
        else:
            clvd_diag = np.array(
                [[-2/np.sqrt(6)], [1/np.sqrt(6)], [1/np.sqrt(6)]])
        # Use random_type for DC eigenvalues
        return self.random_type(clvd_diag)

    def random_type(self, diag):
        """
        Generate random  moment tensors from a given set of eigenvalues.

        Generates random moment tensors with a given set of eigenvalues and random orientations.

        Args
            diag: numpy matrix of eigenvalues.

        Returns
            numpy matrix of random moment tensors, size 6,number_samples

        """
        # Generate random eigenvectors
        [a, b, c] = self.random_orthogonal_eigenvectors()
        # Combine with eigenvalues
        return self.eigenvectors_mt_2_mt6(diag, a, b, c)

    def random_orthogonal_eigenvectors(self):
        """
        Generates random orthogonal eigenvectors.

        Returns
            list of numpy arrays of eigenvectors.

        """
        # Initialise seed
        np.random.seed()
        # Get random vector - eigenvector 1
        a = np.random.randn(3, self.number_samples)
        # Normalise eigenvector 1
        a = a/np.sqrt(np.sum(np.multiply(a, a), axis=0))
        # Get another random vector
        x = np.random.randn(3, self.number_samples)
        # Cross x with the first eigenvector to get another orthogonal
        # eigenvector 2
        b = np.cross(a.transpose(), x.transpose()).transpose()
        # Check if any of the x are parallel to a (unlikely but a possible edge
        # case), and redraw.
        while not np.sum(np.multiply(b, b), axis=0).all():
            x = np.random.randn(3, self.number_samples)
            b = np.cross(a.transpose(), x.transpose()).transpose()
        # Normalise eigenvector 2
        b = b/np.sqrt(np.sum(np.multiply(b, b), axis=0))
        # Obtain third eigenvector as orthogonal to first two (a and b)
        c = np.cross(a.transpose(), b.transpose()).transpose()
        # Normalise eigenvector 3
        c = c/np.sqrt(np.sum(np.multiply(c, c), axis=0))
        # Return list of vectors
        return [a, b, c]

    def eigenvectors_mt_2_mt6(self, diag, a, b, c):
        """
        Converts eigenvectors and eigenvalues to moment tensor 6-vector.

        Generates moment tensor 6-vector.

        Args
            diag: numpy matrix of eigenvalues.
            a: numpy array of eigenvectors corresponding to the first eigenvalue.
            b: numpy array of eigenvectors corresponding to the second eigenvalue.
            c: numpy array of eigenvectors corresponding to the third eigenvalue.

        Returns
            numpy matrix of moment tensor 6-vectors.

        """
        # Converts moment tensor from eigenvalues and eigenvectors to six
        # vector form.

        v1 = np.array([a[0, :], b[0, :], c[0, :]])
        v2 = np.array([a[1, :], b[1, :], c[1, :]])
        v3 = np.array([a[2, :], b[2, :], c[2, :]])
        M = np.matrix([np.sum(v1*v1*diag, axis=0),  # M11
                       np.sum(v2*v2*diag, axis=0),  # M22
                       np.sum(v3*v3*diag, axis=0),  # M33
                       np.sqrt(2)*np.sum(v1*v2*diag, axis=0),  # sqrt2*M12
                       np.sqrt(2)*np.sum(v1*v3*diag, axis=0),  # sqrt2*M13
                       np.sqrt(2)*np.sum(v2*v3*diag, axis=0)])  # sqrt2*M23
        return np.matrix(M/np.sqrt(np.sum(np.multiply(M, M), axis=0)))


def _6sphere_random_mt(self):
    """
    Generate random moment tensors (size 6,number_samples)

    Generates random moment tensors from the 6-Dimensional normal distribution N(0,1) and normalises onto the surface of the unit 6-sphere.

    Returns
        numpy matrix of random moment tensors, size 6,number_samples

    """
    # Check CYTHON code - use C code if possible
    if isinstance(self, int):
        ns = self
    else:
        ns = self.number_samples
    if cprobability:
        return cprobability.random_mt(ns)
    else:
        logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
    # Reseed np.random and generate new random samples.
    np.random.seed()
    M = np.random.randn(6, ns)
    return np.matrix(M/np.sqrt(np.sum(np.multiply(M, M), axis=0)))
