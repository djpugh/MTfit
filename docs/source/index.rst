.. figure:: figures/mtfitbanner.png
   :figwidth: 50 %
   :width: 90%
   :align: center
   :alt: mtfit: A Bayesian approach to source inversion.  


*********************************
mtfit: Bayesian Source Inversion
*********************************

*Bayesian Moment Tensor Inversion Code by David J Pugh (Bullard Laboratories, Department of Earth Sciences, University of Cambridge)*


.. toctree::
   :maxdepth: 1
   :numbered:
   :hidden:
   
   Installing <setup>
   Running mtfit <run>
   Tutorial <tutorial>
   Tutorial: Real Data Examples <real-tutorial>
   Bayesian Approach <bayes>
   Probability Density Functions <probability>
   Search Algorithms <algorithms>
   Moment Tensor Conversion <mtconvert>
   Command Line Options <cli>
   Plotting <mtplot>
   Plot Classes <plot_classes>
   MTplot Command Line Options <mtplotcli>
   Inversion Class <inversion>
   Extensions <extensions>
   References <references>
   Glossary <glossary>
   Source Code <source>

:mod:`mtfit` is a Bayesian forward model inversion code for moment tensor and double-couple source inversion using different data types, based on the Bayesian approach presented in :ref:`Pugh et al, 2016a <Pugh-2016a>` and  :ref:`Pugh, 2015 <Pugh-2015t>`. The code has been developed as part of a PhD project (:ref:`Pugh, 2015 <Pugh-2015t>`). The solutions are estimated using polarity and amplitude ratio data, although the code is extensible (see :doc:`extensions`) so it is possible to include other data-types in this framework. :mod:`mtfit` can incorporate uncertainty estimates both in the data (noise etc.) and the model (and location) in the resultant posterior PDF. There are three sampling approaches that have been developed, with different advantages (:ref:`Pugh et al, 2015c <Pugh-2015t>`, and it is also possible to use the approach for relative amplitude inversion as well (:ref:`Pugh et al, 2015e <Pugh-2015t>`).

:mod:`mtfit` also works with the automated Bayesian polarity approach described in :ref:`Pugh et al, 2016b <Pugh-2016b>` as an alternative method of estimating polarity probabilities. This is available as the :mod:`autopol` Python module.

---------------------------------------

| :ref:`genindex` | :ref:`search` |