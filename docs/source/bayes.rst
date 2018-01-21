**************************
Bayesian Approach
**************************

The Bayesian approach used by :mod:`MTfit` is based on the approach described in :ref:`Pugh et al, 2016a <Pugh-2016a>`

Bayes Theory
---------------------------
Inversion approaches fit model parameters to observed data, to find the best fitting parameters. In :mod:`MTfit`, the probability of the data being correct is evaluated for the possible sources. The resulting estimates of the :term:`PDF` can be combined for all the data to approximate the true :term:`PDF` for the source. This :term:`PDF` can be considered to describe the probability of the observed data for a given source, the likelihood, :math:`p\left(\mathrm{data}\,|\mathrm{\, model}\right)`. However the value of interest in such an inversion is the probability of the model given the observed data, the posterior :term:`PDF`. This can be evaluated from the likelihood using Bayes' Theory (:ref:`Bayes and Price, 1763 <Bayes-1763>` , :ref:`Laplace, 1812 <Laplace-1812>` , :ref:`Sivia, 2000<Sivia-2000>`) is given simply by: 

.. math::

    \mathrm{p}\left(\mathrm{model\,|\, data}\right)=\frac{\mathrm{p}\left(\mathrm{data}\,|\mathrm{\, model}\right)\, \mathrm{p}\left(\mathrm{model}\right)}{\mathrm{p}\left(\mathrm{data}\right)}
 

This relates the data likelihood :math:`\mathrm{p}\left(\mathrm{data}\,|\mathrm{\, model}\right)` to the posterior probability :math:`\mathrm{p}\left(\mathrm{model\,|\, data}\right)` via the prior probabilities :math:`\mathrm{p}\left(\mathrm{model}\right)` and :math:`\mathrm{p}\left(\mathrm{data}\right)`. The prior distributions incorporate the known information about the two parameters, and although the choice of prior is not trivial, it does not have a large affect on the posterior :term:`PDF` given enough data.


Uncertainties
--------------------------

Source inversion is particularly sensitive to uncertainties, and since it is usually carried out after several other required steps, the effect of the uncertainties may be difficult to understand. The uncertainties can be broken down into several different types: instrument errors, model errors and background noise.

Although the interdependencies between the uncertainties can be explored, a quantitative relationship is not known. For such a treatment to be truly rigorous, the variations in these errors must be included throughout the inversion. The Bayesian formulation allows rigorous inclusion of uncertainties in the problem using marginalisation. Marginalisation removes the dependence of the joint :term:`PDF`, :math:`P(A,B)` , on one of the variables, by integrating over the :term:`PDF` for that variable:

.. math::

    \mathrm{p}\left(A\right)=\int \mathrm{p}\left(A\,,\, B\right)dB=\int \mathrm{p}\left(A\,|\, B\right)\mathrm{p}\left(B\right)dB
 

Marginalisation can be used to incorporate the uncertainties in the inversion into the final :term:`PDF`. 

It is assumed that the noise has a mean and variance that are measurable, and therefore the most ambiguous (maximum entropy) distribution for these measurements is the Gaussian distribution (:ref:`Pugh et al, 2015<Pugh-2016a>`). For de-meaned data, the noise can be assumed also to have zero mean, and a standard deviation :math:`\sigma_{mes}`, so that the PDF :math:`\mathrm{p}\left(\Delta_{mes}\right)`, is:

.. math::

    \mathrm{p}\left(\Delta_{mes}\right)=\frac{1}{\sqrt{2\pi\sigma_{mes}^{2}}}\mathrm{e}^{-\frac{\Delta_{mes}^{2}}{2\sigma_{mes}^{2}}}

There are several different types of uncertainties however most can be simplified to independent uncertainties on each trace, and uncertainties in the underlying model.

Posterior PDF
-------------------------------

.. only:: not latex

    The likelihoods for the two different data types are described :doc:`here <probability>`.

.. only:: latex
    
    The likelihoods for the two different data types are described in chapter :latex:`\ref{probability::doc}`.

The posterior PDFs for the data used in :mod:`MTfit` are, for a known Earth model:

.. math::

    \mathrm{p}\left(\mathbf{d'}\:|\:\mathbf{M},\mathbf{t},\mathbf{\epsilon},\mathbf{k}\right)=\iint\sum_{j=1}^{M}\prod_{i=1}^{N}\left[\mathrm{p}\left(Y_{i}\:|\:\mathbf{\mathbf{A_{ij}}}=\mathbf{a}_{j}\cdot\mathbf{\tilde{M}},\mathbf{\sigma_{i}},\varpi_{i}\right)\mathrm{p}\left(\mathbf{R_{i}}\:|\:\mathbf{\mathbf{A_{i}}}=\mathbf{a}_{j}\cdot\mathbf{\tilde{M}},\mathbf{\sigma_{i}},\varpi_{i}\right)\right]\mathrm{p}\left(\mathbf{\sigma}\right)\mathrm{p}\left(\mathbf{\varpi}\right)d\mathbf{\sigma}d\mathbf{\varpi}


An unknown earth model, samples from the model distribution, :math:`\mathbf{G}_{k}`, are included using a :term:`Monte Carlo method` based marginalisation:

.. math::

    \mathrm{p}\left(\mathbf{d'}\:|\:\mathbf{M},\mathbf{t},\mathbf{\epsilon},\mathbf{k}\right)=\iint\sum_{k=1}^{Q}\sum_{j=1}^{M}\prod_{i=1}^{N}\left[\mathrm{p}\left(Y_{i}\:|\:\mathbf{\mathbf{A_{ijk}}}=\mathbf{a}_{jk}\cdot\mathbf{\tilde{M}},\mathbf{\sigma_{i}},\varpi_{i}\right)\mathrm{p}\left(\mathbf{R_{i}}\:|\:\mathbf{\mathbf{A_{ijk}}}=\mathbf{a}_{jk}\cdot\mathbf{\tilde{M}},\mathbf{\sigma_{i}},\varpi_{i}\right)\right]\mathrm{p}\left(\mathbf{\sigma}\right)\mathrm{p}\left(\mathbf{\varpi}\right)d\mathbf{\sigma}d\mathbf{\varpi}, 

where :math:`\mathbf{a}_{jk}=\mathbf{a}\left(\mathbf{x}_{j},\mathbf{G}_{k}\right)` refers to the station propagation coefficients associated with the location at :math:`\mathbf{x}_{j}` and earth model :math:`\mathbf{G}_{k}`.
 


.. only:: not latex

    A common location method is `NonLinLoc by Anthony Lomax <http://alomax.free.fr/nlloc>`_ which can produces samples from the location PDF to be used in the :term:`Monte Carlo method` described above (see :ref:`location uncertainty tutorial <location-uncertainty-tutorial-label>`).


.. only:: latex

    A common location method is `NonLinLoc by Anthony Lomax <http://alomax.free.fr/nlloc>`_ which can produces samples from the location PDF to be used in the :term:`Monte Carlo method` described above (see the location uncertainty tutorial in chapter :latex:`\ref{tutorial::doc}`).

The symbols used in these PDFs are:

+--------------------------+--------------------------------+
|   *Symbol*               |       *Meaning*                |
+==========================+================================+
|:math:`Y`                 | Polarity                       |
+--------------------------+--------------------------------+
|:math:`A`                 | Amplitude                      |
+--------------------------+--------------------------------+
|:math:`\sigma_{y}`        | Uncertainty                    |
+--------------------------+--------------------------------+
|:math:`\varpi`            | Mispick Probability            |
+--------------------------+--------------------------------+
|:math:`\mathbf{\tilde{M}}`| Moment Tensor 6-vector         |
+--------------------------+--------------------------------+
|:math:`\mathbf{M}`        | Moment Tensor                  |
+--------------------------+--------------------------------+
|:math:`\mathbf{a}`        | Receiver Ray Path Coefficients |
+--------------------------+--------------------------------+
|:math:`\sigma`            | Measurment Uncertainty         |
+--------------------------+--------------------------------+
|:math:`R`                 | Amplitude Ratio                |
+--------------------------+--------------------------------+
|:math:`A`                 | Amplitude                      |
+--------------------------+--------------------------------+
|:math:`\mathbf{\epsilon}` | Nuisance Parameters            |
+--------------------------+--------------------------------+
|:math:`\mathbf{k}`        | Known Parameters               |
+--------------------------+--------------------------------+
|:math:`\mathbf{t}`        | Arrival times                  |
+--------------------------+--------------------------------+