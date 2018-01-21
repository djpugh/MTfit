Markov chain Monte Carlo Search Algorithm
===========================================

.. automodule:: MTfit.algorithms.markov_chain_monte_carlo

The McMC algorithms are initialised using the :class:`McMCAlgorithmCreator` class:

.. autoclass:: MTfit.algorithms.markov_chain_monte_carlo.McMCAlgorithmCreator
   :members: __new__

-------------------

The standard McMC algorithms used are the multiple try algorithms using the :ref:`Tape and Tape (2012)<Tape-2012>` parameterisation and a Gaussian proposal function.


.. autoclass:: MTfit.algorithms.markov_chain_monte_carlo.IterativeMultipleTryMetropolisHastingsGaussianTape
   :members:   
   :inherited-members:

-------------------

.. autoclass:: MTfit.algorithms.markov_chain_monte_carlo.IterativeMultipleTryTransDMetropolisHastingsGaussianTape
   :members:
   :inherited-members:
   
-------------------

The McMC sampling algorithms inherit from the :class:`BaseAlgorithm` class:

.. autoclass:: MTfit.algorithms.base.BaseAlgorithm
   :members:
