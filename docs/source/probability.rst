*************************
Probability
*************************

:mod:`MTfit` has :term:`PDF` s for two data types, with two different approaches to measuring the polarity.

However, it is possible to add :term:`PDF` s for other data-types (see :doc:`extensions`)

.. _polarity-pdf-label:


.. only:: not latex

    Polarity :term:`PDF`
    =================================
    
.. only:: latex

    Polarity PDF
    ============================

:ref:`Pugh et al, 2016a <Pugh-2016a>` has a derivation of the polarity :term:`PDF` used in :mod:`MTfit`. It is given by:

.. math::

    \mathrm{p}\left(Y=y\,|\, A,\sigma_{y},\varpi\right)=\frac{1}{2}\left(1+\mathrm{erf}\left(\frac{yA}{\sqrt{2}\sigma_{y}}\right)\right)\left(1-\varpi\right)+\frac{1}{2}\left(1+\mathrm{erf}\left(\frac{-yA}{\sqrt{2}\sigma_{y}}\right)\right)\varpi

The different symbols in this equation are:

+------------------+---------------------+
|   *Symbol*       |       *Meaning*     |
+==================+=====================+
|:math:`Y`         | Polarity            |
+------------------+---------------------+
|:math:`A`         | Amplitude           |
+------------------+---------------------+
|:math:`\sigma_{y}`| Uncertainty         |
+------------------+---------------------+
|:math:`\varpi`    | Mispick Probability |
+------------------+---------------------+

and :math:`\mathrm{erf}\left(x\right)` is the error function, given by :math:`\mathrm{erf}\left(x\right)=\frac{2}{\sqrt{\pi}}\int_{0}^{x}\mathrm{e}^{-t^{2}}\mathrm{d}t`.

This approach requires an estimate of the uncertainty :math:`\sigma_{y}`. This is not the noise at the arrival, since it does not scale correctly in comparison to the modelled amplitude due to the propagation effects. It could be estimated from the fractional amplitude uncertainty, but this will be greater than or equal to the true value, because the amplitude at a receiver is only ever less than or equal to the maximum theoretical amplitude (accounting for propagation effects). Consequently, this would most likely overestimate the uncertainty. It is clear that the uncertainty value should be station-specific as noise environments at different stations often vary, so the maximum estimate of the event signal-to-noise ratio (SNR) fails to account for the variation across the receivers. 

The difficulty in estimating :math:`\sigma_{y}` is increased further when polarity picking is done manually, so the uncertainty on the trace is perhaps not even known. Due to the difficulty in quantifying the uncertainty, it is best left as a user-defined parameter that reflects the confidence in the arrival polarity pick, which can be mapped to the pick quality. However, :ref:`Pugh et al, 2016b <Pugh-2016b>` proposes an alternate method for calculating polarity uncertainties that can be included in this framework (see :ref:`Polarity Probability PDF<polarity_prob_pdf>`)

.. _polarity_prob_pdf:

.. only:: not latex

    Polarity Probability :term:`PDF`
    =================================
    
.. only:: latex

    Polarity Probability PDF
    ============================

:ref:`Pugh et al, 2016b <Pugh-2016b>` introduces an alternate method for estimating the polarity, using an automated Bayesian probability estimate. This approach results in estimates of the postive and negative polarity probabilities.
:mod:`autopol` provides a Python module for calculating these values (:ref:`Pugh, 2016a <Pugh-2016a>`), and may be available on request. These observations can be included in  :mod:`MTfit`, although the data independence must be preserved. The :term:`PDF` is:

.. math::

    \mathrm{p}\left(\psi|A,\sigma,\tau,\sigma_{\tau},\varpi\right)=1-\varpi+\left(2\varpi-1\right)\left[\mathrm{H}\left(A\right)+\psi-2\mathrm{H}\left(A\right)\varpi\right]

The different symbols in this equation are:

+---------------------+---------------------+
|   *Symbol*          |       *Meaning*     |
+=====================+=====================+
|:math:`\psi`         | Polarity Probability|
+---------------------+---------------------+
|:math:`A`            | Amplitude           |
+---------------------+---------------------+
|:math:`\sigma`       | Trace Noise         |
+---------------------+---------------------+
|:math:`\tau`         | Pick Time           |
+---------------------+---------------------+
|:math:`\sigma_{\tau}`| Pick Time Noise     |
+---------------------+---------------------+
|:math:`\varpi`       | Mispick Probability |
+---------------------+---------------------+

and :math:`\mathrm{H}\left(x\right)` is the Heaviside step function, given by :math:`\mathrm{H}\left(x\right)=\int_{-\infty}^{x}\delta\left(s\right)\mathrm{d}s`.


.. _ratio-pdf-label:

.. only:: not latex

    Amplitude Ratio :term:`PDF`
    =================================
    
.. only:: latex

    Amplitude Ratio PDF
    ============================

The amplitude ratio :term:`PDF` used in :mod:`MTfit` is based on the ratio :term:`PDF` for two gaussian distributed variables (:ref:`Hinkley, 1969 <Hinkley-1969>`):

.. math::

    P\left(r\right)=\frac{b\left(r\right)d\left(r\right)}{\sigma_{x}\sigma_{y}a^{3}\left(r\right)\sqrt{2\pi}}\left[\Phi\left(\frac{b\left(r\right)}{a\left(r\right)\sqrt{1-\rho^{2}}}\right)-\Phi\left(\frac{-b\left(r\right)}{a\left(r\right)\sqrt{1-\rho^{2}}}\right)\right]\\
    +\frac{\sqrt{1-\rho^{2}}}{\pi\sigma_{x}\sigma_{y}a^{2}\left(r\right)}e^{\left(-\frac{c}{2\left(1-\rho^{2}\right)}\right)}
 

With coefficients :math:`a\left(r\right)`, :math:`b\left(r\right)`, :math:`c`  and :math:`d\left(r\right)` given by :

.. math::
    a\left(r\right) =   \sqrt{\frac{r^{2}}{\sigma_{x}^{2}}-2\rho\frac{r}{\sigma_{x}\sigma_{y}}+\frac{1}{\sigma_{y}^{2}}}\\
    b\left(r\right) =   \frac{\mu_{x}r}{\sigma_{x}^{2}}-\rho\frac{\mu_{x}+\mu_{y}r}{\sigma_{x}\sigma_{y}}+\frac{\mu_{y}}{\sigma_{y}^{2}}\\
    c   =   \frac{\mu_{x}^{2}}{\sigma_{x}^{2}}-2\rho\frac{\mu_{x}\mu_{y}}{\sigma_{x}\sigma_{y}}+\frac{\mu_{y}^{2}}{\sigma_{y}^{2}}\\
    d\left(r\right) =   e^{\left(\frac{b^{2}\left(r\right)-ca^{2}\left(r\right)}{2\left(1-\rho^{2}\right)a^{2}\left(r\right)}\right)}\\


The resultant :term:`PDF` is (unsigned amplitude ratios):

.. math::

    P\left(R=r\,|\, A_{x},A_{y},\sigma_{x},\sigma_{y}\right)=\mathcal{R_{N}}\left(r,A_{x},A_{y},\sigma_{x},\sigma_{y}\right)+\mathcal{R_{N}}\left(-r,A_{x},A_{y},\sigma_{x},\sigma_{y}\right)
 
With :math:`\mathcal{R_{N}}\left(r,\mu_{x},\mu_{y},\sigma_{x},\sigma_{y}\right)` referring to the ratio :term:`PDF` above, since :math:`\rho`, the correlation between the variables, is zero.


The different symbols in this equation are:

+---------------------+---------------------+
|   *Symbol*          |       *Meaning*     |
+=====================+=====================+
|:math:`R`            | Amplitude Ratio     |
+---------------------+---------------------+
|:math:`A`            | Amplitude           |
+---------------------+---------------------+
|:math:`\sigma`       | Amplitude Noise     |
+---------------------+---------------------+
