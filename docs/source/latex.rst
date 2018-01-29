£abstract
:mod:`MTfit` is a Bayesian forward model inversion code for moment tensor and double-couple source inversion using different data types, based on the Bayesian approach presented in :ref:`Pugh et al, 2016a <Pugh-2016a>` and  :ref:`Pugh, 2015 <Pugh-2015t>`. The code has been developed as part of a PhD project (:ref:`Pugh, 2015 <Pugh-2015t>`). The solutions are estimated using polarity and amplitude ratio data, although the code is extensible (see :doc:`extensions`) so it is possible to include other data-types in this framework. :mod:`MTfit` can incorporate uncertainty estimates both in the data (noise etc.) and the model (and location) in the resultant posterior probability density function. There are three sampling approaches that have been developed, with different advantages (:ref:`Pugh et al, 2015t <Pugh-2015c>`), and it is also possible to use the approach for relative amplitude inversion as well (:ref:`Pugh et al, 2015t <Pugh-2015t>`).

:mod:`MTfit` also works with the automated Bayesian polarity approach described in :ref:`Pugh et al, 2016b <Pugh-2016b>` as an alternative method of estimating polarity probabilities. This may be available on request as the :mod:`autopol` Python module.

£endabstract


*********************
Contents 
*********************



.. toctree::
   :maxdepth: 2
   :numbered:
   :caption: Table of Contents

   Introduction <introduction>
   Installing MTfit<setup>
   Running MTfit <run>
   Tutorial <tutorial>
   Tutorial: Real Data Examples <real-tutorial>
   Bayesian Approach <bayes>
   Probability Density Functions <probability>
   Search Algorithms <algorithms>
   Moment Tensor Conversion <mtconvert>
   Command Line Options <cli>
   MTplot <mtplot>
   MTplot Command Line Options <mtplotcli>
   Inversion Class <inversion>
   Extensions <extensions>
   References <references>

