.. figure:: figures/MTfitbanner.png
   :figwidth: 50 %
   :width: 90%
   :align: center
   :alt: MTfit: A Bayesian approach to source inversion.  


*********************************
MTfit: Bayesian Source Inversion
*********************************

*Bayesian Moment Tensor Inversion Code by David J Pugh*


.. toctree::
   :maxdepth: 1
   :numbered:
   :hidden:
   
   Installing <setup>
   Running MTfit <run>
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
   GitHub Repository <https://github.com/djpugh/MTfit>

.. only:: html

   .. image:: https://travis-ci.org/djpugh/MTfit.svg?branch=develop


:mod:`MTfit` is a Bayesian forward model inversion code for moment tensor and double-couple source inversion using different data types, based on the Bayesian approach presented in :ref:`Pugh et al, 2016a <Pugh-2016a>` and  :ref:`Pugh, 2015 <Pugh-2015t>`. The code has been developed as part of a PhD project (:ref:`Pugh, 2015 <Pugh-2015t>`). The solutions are estimated using polarity and amplitude ratio data, although the code is extensible (see :doc:`extensions`) so it is possible to include other data-types in this framework. :mod:`MTfit` can incorporate uncertainty estimates both in the data (noise etc.) and the model (and location) in the resultant posterior PDF. There are three sampling approaches that have been developed, with different advantages (:ref:`Pugh et al, 2015c <Pugh-2015t>`, and it is also possible to use the approach for relative amplitude inversion as well (:ref:`Pugh et al, 2015e <Pugh-2015t>`).

:mod:`MTfit` also works with the automated Bayesian polarity approach described in :ref:`Pugh et al, 2016b <Pugh-2016b>` as an alternative method of estimating polarity probabilities. This is written as the :mod:`autopol` Python module, and may be available on request.

The source code is available from `GitHub <https://github.com/djpugh/MTfit>`_.

Issues are tracked at the `GitHub repository <https://github.com/djpugh/MTfit/issues>`_. Please raise a new issue or feature request `here <https://github.com/djpugh/MTfit/issues/new>`_. 

.. only:: html
   
   This documentation is available as a :download:`PDF<../pdf/MTfit.pdf>` and an :download:`epub<../epub/MTfit.epub>` file

Please note that this code is provided as-is, and no guarantee is given that this code will perform in the desired way. Additional development and support is carried out in the developer's free time.

**Restricted:  For Non-Commercial Use Only**
This code is protected intellectual property and is available solely for teaching
and non-commercially funded academic research purposes.
Applications for commercial use should be made to Schlumberger or the University of Cambridge.



---------------------------------------

| :ref:`genindex` | :ref:`search` |