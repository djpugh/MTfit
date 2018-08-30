"""monte_carlo_random
******************
Module containing algorithm classes for Monte Carlo random sampling.
"""


# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.


import time
import gc
import logging

from .base import BaseAlgorithm
from .base import PRINT_N_ITERATIONS

logger = logging.getLogger('MTfit.algorithms')

__all__ = ['BaseMonteCarloRandomSample',
           'IterationSample',
           'TimeSample']


class BaseMonteCarloRandomSample(BaseAlgorithm):

    """
    Base class for Monte Carlo Random Sampling

    Contains base functions for new samples, iteration and initialisation.

    """

    def __init__(self, *args, **kwargs):
        """
        Initialisation of BaseMonteCarloRandomSample object

        Args
            number_samples:[10000] Number of samples to use per iteration.
            dc:[False] Boolean to select inversion constrained to double-couple space or over the full moment tensor space.

        Keyword Arguments
            basic_cdc:[False] Boolean to select whether to us the Basic CDC model (inversion constrained to double-couple space or over the full moment tensor space.

        """
        super(BaseMonteCarloRandomSample, self).__init__(*args, **kwargs)
        self._min_number_check_samples = kwargs.get('min_number_check_samples', 30000)
        self.iteration = 0

    def random_sample(self):
        """
        Return random sample

        Extends _BaseAlgorithm.random_sample to account for multiple events, and checks the source type.
        """
        if self.number_events > 1:
            MTs = []
            i = 0
            while i < self.number_events:
                if self.dc:
                    MTs.append(self.random_dc())
                elif self.basic_cdc:
                    MTs.append(self.random_basic_cdc())
                else:
                    MTs.append(self.random_mt())
                i += 1
            return MTs
        else:
            return super(BaseMonteCarloRandomSample, self).random_sample()

    @property
    def total_number_samples(self):
        return self.pdf_sample.n

    def check_finished(self, end):
        return end

    def iterate(self, result):
        """
        Iteration function for BaseMonteCarloRandomSample object

        Carries out iteration of random sampling from previous result.

        Args
            result: dictionary result from MTfit.inversion.ForwardTask, added to the pdf

        Returns
            new_samples,end
            new_samples: New set of random samples to test.
            end: Boolean flag as to whether inversion has finished, if a MemoryError error is thrown, ends iteration.
        """
        self.iteration += 1
        try:
            self.pdf_sample.append(**result)
            task, end = self.random_sample(), False
        except MemoryError:
            task, end = [], True
        if (isinstance(self.quality_check, (float, int)) and self.quality_check > 0) and self.pdf_sample.n > self._min_number_check_samples:
            if (100*float(len(self.pdf_sample.ln_pdf.nonzero()))/float(self.pdf_sample.n)) > self.quality_check:
                logger.error("Data Error: Non-zero sample percentage above {}".format(self.quality_check))
                return [], True

        end = self.check_finished(end)
        # Check if the number of iterations is a multiple of the
        # PRINT_N_ITERATIONS variable, and print iteration info.
        if not self.iteration % PRINT_N_ITERATIONS:
            if self.pdf_sample.n > 0:
                message = 'Iteration: {} | Elapsed Time: {:.0f} seconds | Total Samples: {} | Non-zero samples: {} | Percentage Non-Zero: {:f} %'
                message = message.format(self.iteration, time.time()-self.start_time, self.pdf_sample.n, len(self.pdf_sample.ln_pdf.nonzero()),
                                         100*float(len(self.pdf_sample.ln_pdf.nonzero()))/float(self.pdf_sample.n))
            else:
                message = 'Iteration: {} | Elapsed Time: {:.0f} seconds | Total Samples: {} | Non-zero samples: {}'
                message = message.format(self.iteration, time.time()-self.start_time, self.pdf_sample.n, len(self.pdf_sample.ln_pdf.nonzero()))
            logger.info(message)
        gc.collect()
        return task, end

    def initialise(self):
        """
        Initial task for ForwardTask

        Returns
            task,end=False
            task: the random samples starting the inversion.

        """
        self.start_time = time.time()
        self.iteration = 0
        return self.random_sample(), False


class TimeSample(BaseMonteCarloRandomSample):

    """
    Time based sampling algorithm

    Algorithm that runs until a maximum time is reached.

    Initialisation

    """

    def __init__(self, max_time=600, *args, **kwargs):
        """
        Initialisation of TimeSample

        Args
            max_time:[600] Maximum time in seconds for the iteration to run for.
            number_samples:[10000] Number of samples to use per iteration.
            dc:[False] Boolean to select inversion constrained to double-couple space or over the full moment tensor space.

        Keyword Arguments
            BasicCDC:[False] Boolean to select whether to us the Basic CDC model (inversion constrained to double-couple space or over the full moment tensor space.

        """
        super(TimeSample, self).__init__(*args, **kwargs)
        self.max_time = float(max_time)

    def max_value(self):
        return '{} seconds'.format(self.max_time)

    def check_finished(self, end):
        if time.time()-self.start_time >= self.max_time:
            end = True
        return end


class IterationSample(BaseMonteCarloRandomSample):

    """
    Iteration based sampling algorithm

    Algorithm that runs for a fixed number of samples.

    """

    def __init__(self, max_samples=600000, *args, **kwargs):
        """
        Initialisation of IterationSample

        Args
            max_samples:[600000] Maximum time in seconds for the iteration to run for.
            number_samples:[10000] Number of samples to use per iteration.
            dc:[False] Boolean to select inversion constrained to double-couple space or over the full moment tensor space.

        Keyword Arguments
            basic_cdc:[False] Boolean to select whether to us the Basic CDC model (inversion constrained to double-couple space or over the full moment tensor space.

        """
        super(IterationSample, self).__init__(*args, **kwargs)
        self.max_samples = int(max_samples)

    def max_value(self):
        return '{} samples'.format(self.max_samples)

    def check_finished(self, end):
        if self.pdf_sample.n >= self.max_samples:
            end = True
        return end
