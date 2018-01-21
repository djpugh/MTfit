MTfit.algorithms: Search Algorithms
===================================

These algorithms and their effects are explored in :ref:`Pugh (2015)<Pugh-2015t>`, which expands in further detail on the topics covered here.


.. only:: not latex

    Table Of Contents:
    *********************************

       * :ref:`Random Monte Carlo sampling <MCsampling>`
       * :ref:`Markov chain Monte Carlo sampling <McMCsampling>`

.. automodule:: MTfit.algorithms

.. _MCsampling:

Random Monte Carlo sampling
*********************************
The simplest approach is that of random sampling over the moment tensor or double-couple space. Stochastic Monte Carlo sampling introduces no biases and provides an estimate for the true PDF, but requires a sufficient density of sampling to reduce the uncertainties in the estimate. The sampled PDF approaches the true distribution in the limit of infnite samples. However, this approach is limited both by time and memory. Some benefits can be gained by only keeping the samples with non-zero probability. 

The assumed prior distribution is a uniform distribution on the unit 6-sphere in moment tensor space. This is equivalent to unit normalisation of the moment tensor six vector:

.. math::

    \mathbf{\tilde{M}}  =   \left(\begin{array}{c}M_{11}\\M_{22}\\M_{33}\\\sqrt{2}M_{12}\\\sqrt{2}M_{13}\\\sqrt{2}M_{23}\end{array}\right).


This sampling is explored further in :ref:`Pugh et al, 2015t <Pugh-2015t>`.

.. _McMCsampling:

Markov chain Monte Carlo sampling
*********************************
An alternative approach is to use Markov chain Monte Carlo (McMC) sampling. This constructs a Markov chain (:ref:`Norris, 1998 <Norris-1998>`) of which the equilibrium distribution is a good sample of the target probability distribution. 

A Markov chain is a memoryless stochastic process of transitioning between states. The probability of the next value depends only on the current value, rather than all the previous values, which is known as the Markov property (:ref:`Markov, 1954 <Markov-1954>`):

.. math::

    \mathrm{p}\left({d_{n}|d_{n-1},d_{n-2},d_{n-3},\ldots d_{0}}\right)=\mathrm{p}\left({d_{n}|d_{n-1}}\right).

A suitable McMC method should converge on the target distribution rapidly. As an approach it is more complex than the Monte Carlo random sampling approach described above, and by taking samples close to other non-zero samples, there is moreintelligence to the sampling than in the random Monte Carlo sampling approach.

A Metropolis-Hastings approach is used here (:ref:`Metropolis, 1953 <Metropolis-1953>` and :ref:`Hastings, 1970 <Hastings-1970>`). The Metropolis-Hastings approach is a common method for McMC sampling and satisfies the detailed balance condition (Robert and Casella, 2004, eq. 6.22), which means that the probability density of the chain is stationary. New samples are drawn from a probability density :math:`\mathrm{q}\left(\mathbf{x'}|\mathbf{x}_\mathrm{t}\right)` to evaluate the target probability density :math:`\mathrm{p}\left(\mathbf x|\mathbf d\right)`. 

The Metropolis-Hastings algorithm begins with a random starting point and then iterates until this initial state is forgotten (:ref:`Algorithm <mcmc-alg>`). Each iteration evaluates whether a new proposed state is accepted or not. If :math:`\mathrm{q}\left(\mathbf{x'}|\mathbf{x}_\mathrm{t}\right)` is symmetric, then the ratio :math:`\frac{\mathrm{q}\left(\mathbf{x}_\mathrm{t}|\mathbf{x'}\right)}{\mathrm{q}\left(\mathbf{x'}|\mathbf{x}_\mathrm{t}\right)}=1`. The acceptance, \alpha, is given by    

.. math::

    \alpha=\mathrm{min}\left(1,\frac{\mathrm{p}\left(\mathbf{x'}|\mathbf d\right)}{\mathrm{p}\left(\mathbf{x}_{\mathrm{t}}|\mathbf d\right)}.\frac{\mathrm{q}\left(\mathbf{x}_\mathrm{t}|\mathbf{x'}\right)}{\mathrm{q}\left(\mathbf{x'}|\mathbf{x}_{\mathrm{t}}\right)}\right),

which can be expanded using :doc:`Bayes' Theorem<bayes>` to give the acceptance in terms of the likelihood :math:`\mathrm{p}\left(\mathbf d|\mathbf x\right)` and prior :math:`\mathrm{p}\left(\mathbf x\right)`,

.. math::

    \alpha    =   \mathrm{min}\left( 1,\frac{\mathrm{p}\left(\mathbf d|\mathbf{x'}\right)\mathrm{p}\left(\mathbf{x'}\right)}{\mathrm{p}\left(\mathbf d|\mathbf{x}_\mathrm{t}\right)\mathrm{p}\left(\mathbf{x}_\mathrm{t}\right)}.\frac{\mathrm{q}\left(\mathbf{x}_t|\mathbf{x'}\right)}{\mathrm{q}\left(\mathbf{x'}|\mathbf{x}_\mathrm{t}\right)}\right).

The acceptance is the probability that the new sample in the chain, :math:`\mathbf{x}_\mathrm{t+1}`  is the new sample, :math:`\mathbf{x'}`, otherwise the original value, :math:`\mathbf{x}_\mathrm{t}` , is added to the chain again,

.. math::

     \mathbf{x_{\mathrm{t}+1}}=\begin{cases}
    \mathbf{x'} & probability=\alpha\\
    \mathbf{x}_t & probability=1-\alpha
    \end{cases}.

The algorithm used in :mod:`MTfit` is:

.. only:: not latex


    +--------------------------------------------------------------------------------------------------------------------------------+
    |      **Metropolis-Hastings Markov chain Monte Carlo Sampling Algorithm**                                                       |
    +====+===========================================================================================================================+
    |1   |Determine initial value for :math:`\mathbf{x}_0` with non-zero likelihood                                                  |
    +----+---------------------------------------------------------------------------------------------------------------------------+
    |2   |Draw new sample :math:`\mathbf{x'}` from transition PDF :math:`\mathrm{q}\left(\mathbf{x'}|\mathbf{x}_\mathrm{t}\right)`   |
    +----+---------------------------------------------------------------------------------------------------------------------------+
    |3   |Evaluate likelihood for sample :math:`\mathbf{x'}`                                                                         |
    +----+---------------------------------------------------------------------------------------------------------------------------+
    |4   |Calculate acceptance, :math:`\alpha`, for :math:`\mathbf{x'}`.                                                             |
    +----+---------------------------------------------------------------------------------------------------------------------------+
    |5   |Determine sample :math:`\mathbf{x}_\mathrm{t+1}`.                                                                          |
    +----+---------------------------------------------------------------------------------------------------------------------------+
    |6   |If in learning period:                                                                                                     |
    |    +---+-----------------------------------------------------------------------------------------------------------------------+
    |    |(a)|If sufficient samples (> 100) have been obtained, update transition PDF parameters to target ideal acceptance rate.    |
    |    +---+-----------------------------------------------------------------------------------------------------------------------+
    |    |(b)|Return to 2 until end of learning period and discard learning samples.                                                 |
    |    +---+-----------------------------------------------------------------------------------------------------------------------+
    |    | Otherwise return to 2 until sufficient samples are drawn.                                                                 |
    +----+---------------------------------------------------------------------------------------------------------------------------+

.. only:: latex
    
    .. tabularcolumns:: |l|L|

    +--------------------------------------------------------------------------------------------------------------------------------+
    |**Metropolis-Hastings Markov chain Monte Carlo Sampling Algorithm**                                                             |
    +====+===========================================================================================================================+
    |1   |Determine initial value for :math:`\mathbf{x}_0` with non-zero likelihood                                                  |
    +----+---------------------------------------------------------------------------------------------------------------------------+
    |2   |Draw new sample :math:`\mathbf{x'}` from transition PDF :math:`\mathrm{q}\left(\mathbf{x'}|\mathbf{x}_\mathrm{t}\right)`   |
    +----+---------------------------------------------------------------------------------------------------------------------------+
    |3   |Evaluate likelihood for sample :math:`\mathbf{x'}`                                                                         |
    +----+---------------------------------------------------------------------------------------------------------------------------+
    |4   |Calculate acceptance, :math:`\alpha`, for :math:`\mathbf{x'}`.                                                             |
    +----+---------------------------------------------------------------------------------------------------------------------------+
    |5   |Determine sample :math:`\mathbf{x}_\mathrm{t+1}`.                                                                          |
    +----+---------------------------------------------------------------------------------------------------------------------------+
    |6(a)|If in learning period:                                                                                                     |
    |    +---------------------------------------------------------------------------------------------------------------------------+
    |    |  \i If sufficient samples (> 100) have been obtained, update transition PDF parameters to target ideal acceptance rate.   |
    |    +---------------------------------------------------------------------------------------------------------------------------+
    |    |  \ii Return to 2 until end of learning period and discard learning samples.                                               |
    +----+---------------------------------------------------------------------------------------------------------------------------+
    |6(b)|Otherwise return to 2 until sufficient samples are drawn.                                                                  |
    +----+---------------------------------------------------------------------------------------------------------------------------+


The source parameterisation is from :ref:`Tape and Tape (2012) <Tape-2012>`, and the algorithm uses an iterative parameterisation for the learning parameters during a learning period, then generates a Markov chain from the pdf. 


Reversible Jump Markov chain Monte Carlo Sampling
--------------------------------------------------
The Metropolis-Hastings approach does not account for variable dimension models. :ref:`Green (1995) <Green-1995>` introduced a new type of move, a *jump*, extending the approach to variable dimension problems. The jump introduces a dimension-balancing vector, so it can be evaluated like the normal Metropolis-Hastings shift.

:ref:`Green (1995) <Green-1995>` showed that the acceptance for a pair of models :math:`M_\mathrm{t}` and :math:`M'` is given by: 

.. math::

    \alpha=\min\left(1,\frac{\mathrm{p}\left(\mathbf d|\mathbf{x'},M'\right)\mathrm{p}\left(\mathbf{x'}|M'\right)\mathrm{p}\left(M'\right)}{\mathrm{p}\left(\mathbf d|\mathbf{x}_\mathrm{t},M_\mathrm{t}\right)\mathrm{p}\left(\mathbf{x}_\mathrm{t}|M_\mathrm{t}\right)\mathrm{p}\left(M_\mathrm{t}\right)}.\frac{\mathrm{q}\left(\mathbf{x}_\mathrm{t}|\mathbf{x'}\right)}{\mathrm{q}\left(\mathbf{x'}|\mathbf{x}_\mathrm{t}\right)}\right),

where :math:`\mathrm{q}\left(\mathbf{x'}|\mathbf{x}_\mathrm{t}\right)` is the probability of making the transition from parameters :math:`\mathbf{x}_\mathrm{t}` from model :math:`M_\mathrm{t}` to parameters :math:`\mathbf{x'}` from model :math:`M'`, and :math:`\mathrm{p}\left(M_\mathrm{t}\right)` is the prior for the model :math:`M_\mathrm{t}`. 

If the models :math:`M_\mathrm{t}` and :math:`M'` are the same, the reversible jump acceptance is the same as the Metropolis-Hastings acceptance, because the model priors are the same. The importance of the reversible jump approach is that it allows a transformation between different models, and even different dimensions.

The dimension balancing vector requires a bijection between the parameters of the two models, so that the transformation is not singular and a reverse jump can occur. In the case where :math:`\dim\left(M'\right)>\dim\left(M_\mathrm{t}\right)`, a vector :math:`\mathbf u` of length :math:`\dim\left(M'\right)-\dim\left(M_\mathrm{t}\right)` needs to be introduced to balance the number of parameters in :math:`\mathbf{x'}`. The values of :math:`\mathbf u` have probability density :math:`\mathrm{q}\left(\mathbf u\right)` and some bijection that maps :math:`\mathbf{x}_\mathrm{t},\mathbf u\rightarrow\mathbf{x'}`, :math:`\mathbf{x'}=\mathrm{h}\left(\mathbf{x}_\mathrm{t},\mathbf u\right)`.

If the jump has a probability :math:`\mathrm{j}\left(\mathbf x\right)` of occurring for a sample :math:`\mathbf x`, the transition probability depends on the jump parameters, :math:`\mathbf u` and the transition ratio is given by:

.. math::

    \frac{\mathrm{q}\left(\mathbf{x}_\mathrm{t},\mathbf u|\mathbf{x'}\right)}{\mathrm{q}\left(\mathbf{x'}|\mathbf{x}_\mathrm{t},\mathbf u\right)} =   \frac{\mathrm{j}\left(\mathbf{x'}\right)}{\mathrm{j}\left(\mathbf{x}_\mathrm{t}\right)\mathrm{q}\left(\mathbf u\right)}\left|\mathbf J\right|,

with the Jacobian matrix, :math:`\mathbf J=\frac{\partial\mathrm{h}\left(\mathbf{x}_\mathrm{t},\mathbf u\right)}{\partial\left(\mathbf{x}_\mathrm{t},\mathbf u\right)}`.
 

The general form of the jump acceptance involves a prior on the models, along with a prior on the parameters. The acceptance for this case is:

.. math::
    \alpha  =   \min\left(1,\frac{\mathrm{p}\left(\mathbf d|\mathbf{x'},M'\right)\mathrm{p}\left(\mathbf{x'}|M'\right)\mathrm{p}\left(M'\right)}{\mathrm{p}\left(\mathbf d|\mathbf{x}_\mathrm{t},M_\mathrm{t}\right)\mathrm{p}\left(\mathbf{x}_\mathrm{t}|M_\mathrm{t}\right)\mathrm{p}\left(M_\mathrm{t}\right)}.\frac{\mathrm{j}\left(\mathbf{x'}\right)}{\mathrm{j}\left(\mathbf{x}_\mathrm{t}\right)\mathrm{q}\left(\mathbf u\right)}\left|\mathbf J\right|\right) .
 

If the jump is from higher dimensions to lower, the bijection describing the transformation has an inverse that describes the transformation :math:`\mathbf{x'}\rightarrow\mathbf{x}_\mathrm{t},\mathbf u`, and the acceptance is given by:

.. math::

    \alpha=\min\left( 1,\frac{\mathrm{p}\left(\mathbf d|\mathbf{x}_\mathrm{t},M_\mathrm{t}\right)\mathrm{p}\left(\mathbf{x}_\mathrm{t}|M_\mathrm{t}\right)\mathrm{p}\left(M_\mathrm{t}\right)}{\mathrm{p}\left(\mathbf d|\mathbf{x'},M'\right)\mathrm{p}\left(\mathbf{x'}|M'\right)\mathrm{p}\left(M'\right)}.\frac{\mathrm{j}\left(\mathbf{x}_\mathrm{t}\right)\mathrm{q}\left(\mathbf u\right)}{\mathrm{j}\left(\mathbf{x'}\right)}\left|\mathbf J^{-1}\right|\right) .
 

A simple example used in the literature (see Green, 1995; Brooks et al., 2003) is a mapping from a one dimensional model with parameter :math:`\theta` to a two dimensional model with parameters :math:`\theta_{1},\theta_{2}`. A possible bijection is given by:

.. math::
    h\left(\theta,u\right) =   \begin{cases}
    \theta_{1} & =\theta-u\\
    \theta_{2} & =\theta+u
    \end{cases}

with the reverse bijection given by:

.. math::
    h\left(\theta_{1},\theta_{2}\right) =   \begin{cases}
    \theta & =\frac{1}{2}\left(\theta_{1}+\theta_{2}\right)\\
    u & =\frac{1}{2}\left(\theta_{1}-\theta_{2}\right)
    \end{cases}
 

Reversible Jump McMC in Source Inversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reversible jump McMC approach allows switching between different source models, such as the double-couple model and the higher dimensional model of the full moment tensor, and can be extended to other source models.

The full moment tensor model is nested around the double-couple point, leading to a simple description of the jumps by keeping the common parameters constant. The Tape parameterisation (:ref:`Tape and Tape, 2012<Tape-2012>`) allows for easy movement both in the source space and between models. The moment tensor model has five parameters: 

    * strike :math:`\left(\kappa\right)`
    * dip cosine :math:`\left(h\right)`
    * slip :math:`\left(\sigma\right)`
    * eigenvalue co-latitude :math:`\left(\delta\right)`
    * eigenvalue longitude :math:`\left(\gamma\right)`

while the double-couple model has only the three orientation parameters: :math:`\kappa`, :math:`h`, and :math:`\sigma`. Consequently, the orientation parameters are left unchanged between the two models, and the dimension balancing vector for the jump has two elements, which can be mapped to :math:`\gamma` and :math:`\delta`:


.. math::
    \mathrm{h}\left(\kappa_{\mathrm{DC}},h_{\mathrm{DC}},\sigma_{\mathrm{DC}},\mathbf u\right)\,\,\,\,\begin{cases}
    \kappa_{\mathrm{MT}} & =\kappa_{\mathrm{DC}}\\
    h_{\mathrm{MT}} & =h_{\mathrm{DC}}\\
    \sigma_{\mathrm{MT}} & =\sigma_{\mathrm{DC}}\\
    \gamma_{\mathrm{MT}} & =u_{1}\\
    \delta_{\mathrm{MT}} & =u_{2}
    \end{cases}.

The algorithm used is:

.. only:: not latex

    +--------------------------------------------------------------------------------------------------------------------------------+
    |**Reversible Jump Markov chain Monte Carlo sampling Algorithm**                                                                 |
    +====+===========================================================================================================================+
    |1   |Determine initial value for :math:`\mathbf{x}_0` with non-zero likelihood                                                  |
    +----+---------------------------------------------------------------------------------------------------------------------------+
    |2   |Determine which move type to carry out:                                                                                    |
    |    +---+-----------------------------------------------------------------------------------------------------------------------+
    |    |(a)|Carry out jump with probability :math:`p_\mathrm{jump}`: Draw new sample                                               |
    |    |   |:math:`\mathbf{x'}=\left(\mathbf{x},\mathbf{u}\right)` with dimension balancing vector :math:`\mathbf{u}` drawn from   |
    |    |   |:math:`\mathrm{q}\left(u1, u2\right)`.                                                                                 |
    |    +---+-----------------------------------------------------------------------------------------------------------------------+
    |    |(b)|Carry out shift with probability :math:`1-p_\mathrm{jump}`: Draw new sample :math:`\mathbf{x'}` from transition PDF    |
    |    |   |:math:`\mathrm{q}\left(\mathbf{x'}|\mathbf{x}_\mathrm{t}\right)`.                                                      |
    +----+---+-----------------------------------------------------------------------------------------------------------------------+
    |3   |Evaluate likelihood for sample :math:`\mathbf{x'}`                                                                         |
    +----+---------------------------------------------------------------------------------------------------------------------------+
    |4   |Calculate acceptance, :math:`\alpha`, for :math:`\mathbf{x'}`.                                                             |
    +----+---------------------------------------------------------------------------------------------------------------------------+
    |5   |Determine sample :math:`\mathbf{x}_\mathrm{t+1}`.                                                                          |
    +----+---------------------------------------------------------------------------------------------------------------------------+
    |6   |If in learning period:                                                                                                     |
    |    +---+-----------------------------------------------------------------------------------------------------------------------+
    |    |(a)|If sufficient samples (> 100) have been obtained, update transition PDF parameters to target ideal acceptance rate.    |
    |    +---+-----------------------------------------------------------------------------------------------------------------------+
    |    |(b)|Return to 2 until end of learning period and discard learning samples.                                                 |
    |    +---+-----------------------------------------------------------------------------------------------------------------------+
    |    |Otherwise return to 2 until sufficient samples are drawn.                                                                  |
    +----+---------------------------------------------------------------------------------------------------------------------------+

.. only:: latex

    .. tabularcolumns:: |l|L|

    +---------------------------------------------------------------------------------------------------------------------------------+
    |**Reversible Jump Markov chain Monte Carlo sampling Algorithm**                                                                  |
    +====+============================================================================================================================+
    |1   |Determine initial value for :math:`\mathbf{x}_0` with non-zero likelihood                                                   |
    +----+----------------------------------------------------------------------------------------------------------------------------+
    |2(a)|Carry out jump with probability :math:`p_\mathrm{jump}`:  Draw new sample                                                   |
    |    |:math:`\mathbf{x'}=\left(\mathbf{x},\mathbf{u}\right)` with dimension balancing vector :math:`\mathbf{u}` drawn from        |
    |    |:math:`\mathrm{q}\left(u1, u2\right)`.                                                                                      |
    +----+----------------------------------------------------------------------------------------------------------------------------+
    |2(b)|Carry out shift with probability :math:`1-p_\mathrm{jump}`: Draw new sample :math:`\mathbf{x'}` from transition PDF         |
    |    |:math:`\mathrm{q}\left(\mathbf{x'}|\mathbf{x}_\mathrm{t}\right)`.                                                           |
    +----+----------------------------------------------------------------------------------------------------------------------------+
    |3   |Evaluate likelihood for sample :math:`\mathbf{x'}`                                                                          |
    +----+----------------------------------------------------------------------------------------------------------------------------+
    |4   |Calculate acceptance, :math:`\alpha`, for :math:`\mathbf{x'}`.                                                              |
    +----+----------------------------------------------------------------------------------------------------------------------------+
    |5   |Determine sample :math:`\mathbf{x}_\mathrm{t+1}`.                                                                           |
    +----+----------------------------------------------------------------------------------------------------------------------------+
    |6(a)|If in learning period:                                                                                                      |
    |    +----------------------------------------------------------------------------------------------------------------------------+
    |    |  \i   If sufficient samples (> 100) have been obtained, update transition PDF parameters to target ideal acceptance rate.  |
    |    +----------------------------------------------------------------------------------------------------------------------------+
    |    |  \ii  Return to 2 until end of learning period and discard learning samples.                                               |
    +----+----------------------------------------------------------------------------------------------------------------------------+
    |6(b)|Otherwise return to 2 until sufficient samples are drawn.                                                                   |
    +----+----------------------------------------------------------------------------------------------------------------------------+

Relative Amplitude
*******************

Inverting for multiple events increases the dimensions of the source space for each event. This leads to a much reduced probability of obtaining a non-zero likelihood sample, because sampling from the n-event distribution leads to multiplying the probabilities of drawing a non-zero samples, resulting in sparser sampling of the joint source PDF.

The elapsed time for the random sampling is longer per sample than the individual sampling, and longer than the combined sampling for both events due to
evaluating the relative amplitude PDF. Moreover, increasing the number of samples 10-fold raises the required time by a factor of 10, requiring some method of reducing the running time
for the inversion, since, given current processor speeds, :math:`10^{15}` samples would take many years to calculate on a single core. As a result, more intelligent search algorithms are required for the full moment tensor case.

Markov chain approaches are less dependent on the model dimensionality. To account for the fact that the uncertainties in each parameter can differ between the events, the Markov
chain shape parameters can be scaled based on the relative non-zero percentages of the events when they are initialised. The initialisation approaches also need to be adjusted to
account for the reduced non-zero sample probability, such as by initialising the Markov chain independently for each event. The trans-dimensional McMC algorithm allows model
jumping independently for each event.

Tuning the Markov chain acceptance rate is difficult, as it is extremely sensitive to small changes in the proposal distribution widths, and with the higher dimensionality it may be
necessary to lower the targeted acceptance rate to improve sampling. Consequently, care needs to be taken when tuning the parameters to effectively implement the approaches for
relative amplitude data.


Running Time
*******************

Comparing different sample sizes shows that the McMC approaches require far fewer samples than random sampling. However, the random sampling algorithm is quick to calculate
the likelihood for a large number of samples, unlike the McMC approach, because of the extra computations in calculating the acceptance and obtaining new samples. Some optimisations
have been included in the McMC algorithms, including calculating the probability for multiple new samples at once, with sufficient samples that there is a high probability of containing
an accepted sample. This is more efficient than repeatedly updating the algorithm. 

.. only:: latex

    Despite these optimisations, the McMC approach is still much slower to reach comparable sample
    sizes, and is slower than would be expected just given the targeted acceptance rate, because of the additional computational overheads (Fig. :ref:`8.1<algorithm-run-time>).

.. only:: not latex

    Despite these optimisations, the McMC approach is still much slower to reach comparable sample
    sizes, and is slower than would be expected just given the targeted acceptance rate, because of the additional computational overheads:

.. _algorithm-run-time:


.. only:: not latex

    .. figure:: figures/algorithm_elapsed_time.png
       :width: 50 %
       :align: center
       :alt: Scatter plot of the run times in hours for the different algorithms

       *Elapsed Time for different sample sizes of the random sampling algorithm and for McMC algorithms with different number of unique samples.*

.. only:: latex

    .. figure:: figures/algorithm_elapsed_time.png
       :width: 90 %
       :align: center
       :alt: Scatter plot of the run times in hours for the different algorithms

       *Elapsed Time for different sample sizes of the random sampling algorithm and for McMC algorithms with different number of unique samples.*


.. only:: not latex

    Including location uncertainty and model uncertainty in the forward model causes a rapid reduction of the available samples for a given amount of RAM and increases the number of
    times the forward model must be evaluated, lengthening the time for sufficient sampling.

    The location uncertainty has less of an effect on the McMC algorithms, since the number of samples being
    tested at any given iteration are small. Consequently, as the random sampling approach becomes slower, the fewer
    samples required to construct the Markov chain starts to produce good samples of the source PDF at comparable
    times. However, there is an initial offset in the elapsed time for the Markov chain Monte Carlo approaches due to
    the burn in and initialisation of the algorithm:

.. only:: latex
    
    Including location uncertainty and model uncertainty in the forward model causes a rapid reduction of the available samples for a given amount of RAM and increases the number of
    times the forward model must be evaluated, lengthening the time for sufficient sampling (Fig. :ref:`8.2<algorithm-location-run-time>`).

    The location uncertainty has less of an effect on the McMC algorithms, since the number of samples being
    tested at any given iteration are small. Consequently, as the random sampling approach becomes slower, the fewer
    samples required to construct the Markov chain starts to produce good samples of the source PDF at comparable
    times. However, there is an initial offset in the elapsed time for the Markov chain Monte Carlo approaches due to
    the burn in and initialisation of the algorithm.

.. _algorithm-location-run-time:

.. only:: not latex

    .. figure:: figures/algorithm_elapsed_time_location.png
       :width: 50 %
       :align: center
       :alt: Scatter plot of the run times in hours for the different algorithms

       *Elapsed Time for different sample sizes of the random sampling algorithm and for McMC algorithms with different number of unique samples. The velocity model and location uncertainty in the source was included with a one degree binning reducing the number of location samples from 50,000 to 5, 463.*

.. only:: latex

    .. figure:: figures/algorithm_elapsed_time_location.png
       :width: 90 %
       :align: center
       :alt: Scatter plot of the run times in hours for the different algorithms

       *Elapsed Time for different sample sizes of the random sampling algorithm and for McMC algorithms with different number of unique samples. The velocity model and location uncertainty in the source was included with a one degree binning reducing the number of location samples from 50,000 to 5, 463.*

.. only:: latex

    The relative amplitude algorithms require an exponential increase in the number of samples as the number of events being sampled increases (Fig. :ref:`8.3<relative-algorithm-run-time>`). However, the McMC approaches are more intelligent and do not require the same increase in sample size, but these algorithms can prove difficult to estimate the appropriate sampling parameters for the proposal distribution.

.. only:: not latex

    The relative amplitude algorithms require an exponential increase in the number of samples as the number of events being sampled increases. However, the McMC approaches are more intelligent and do not require the same increase in sample size, but these algorithms can prove difficult to estimate the appropriate sampling parameters for the proposal distribution:

.. _relative-algorithm-run-time:

.. only:: not latex

    .. figure:: figures/relative_algorithm_elapsed_time.png
       :width: 50 %
       :align: center
       :alt: Scatter plot of the run times in hours for the different algorithms

       *Elapsed Time for different sample sizes of the random sampling algorithm and for McMC algorithms with different number of unique samples for a two event joint PDF with relative P-amplitudes.*

.. only:: latex

    .. figure:: figures/relative_algorithm_elapsed_time.png
       :width: 90 %
       :align: center
       :alt: Scatter plot of the run times in hours for the different algorithms

       *Elapsed Time for different sample sizes of the random sampling algorithm and for McMC algorithms with different number of unique samples for a two event joint PDF with relative P-amplitudes.*

       
Summary
**********
There are several different approaches to sampling for the forward model. The Monte Carlo random sampling is quick, although it requires a large number of samples to produce good samplings of the PDF. The McMC approaches produce good sampling of the source for far fewer samples, but the evaluation time can be long due to the overhead of calculating the acceptance and generating new samples. Furthermore, these approaches rely on achieving the desired acceptance rate to sufficiently sample the PDF given the chain length.
Including location uncertainty increases the random sampling time drastically, and has less of an effect on the McMC approaches.

The solutions for the different algorithms and different source types are consistent, but the McMC approach can be poor at sampling multi-modal distributions if the acceptance rate is too high.

The two different approaches for estimating the probability of the double-couple source type model being correct are consistent, apart from cases where the PDF is dominated by a few very high probability samples. For these solutions, the number of random samples needs to be increased sufficiently to estimate the probability well. The McMC approach cannot easily be used to estimate the double-couple model probabilities, unlike both the random sampling and trans-dimensional approaches.

Extending these models to relative source inversion leads to a drastically increased number of samples required for the Monte Carlo random sampling approach that it becomes infeasible with current computation approaches. However, the McMC approaches can surmount this problem due to the improved sampling approach, but care must be taken when initialising the algorithm to make sure that sufficient samples are discarded that any initial bias has been removed, as well as targeting an acceptance rate that does not require extremely large chain lengths to successfully explore the full space.

The required number of samples depends on each event, and the constraint on the source that is given by the data. Usually including more data-types can sharpen the PDF, requiring more samples to get a satisfactory sampling of the event.


