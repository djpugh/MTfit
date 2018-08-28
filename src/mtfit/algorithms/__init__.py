"""
Algorithms
**********

This module contains the sampling approaches used. Two main approaches are used:

 * Random Monte Carlo sampling
 * Markov chain Monte Carlo sampling

However, there are also two variants of the Markov chain Monte Carlo (McMC) method:

 * Metropolis-Hastings
 * Trans-Dimensional Metropolis-Hastings (Reversible Jump)
"""


# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.


from .base import BaseAlgorithm  # noqa E401
from .monte_carlo import IterationSample, TimeSample  # noqa E401
from .markov_chain_monte_carlo import McMCAlgorithmCreator as MarkovChainMonteCarloAlgorithmCreator  # noqa E401


# TODO: refactor sampling logic in these modules and reduce function complexity
