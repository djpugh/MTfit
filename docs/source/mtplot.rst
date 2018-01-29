*********************************
Plotting the Moment Tensor
*********************************

:mod:`MTfit` has a plotting submodule that can be used to represent the source. There are several different plot types, shown below, and MTplot can be used both from the command line, and from within the python interpreter.

This section describes how to use the plotting tools and shows the different plot types:

    * :ref:`Beachball<beachball>`
    * :ref:`Fault plane<faultplane>`
    * :ref:`Riedesel-Jordan<rjplot>`
    * :ref:`Radiation<radiation>`
    * :ref:`Lune<lune>`
    * :ref:`Hudson<hudson>`

These are plotted using `matplotlib <http://matplotlib.org/>`_, using a class based system. The main plotting class is the :class:`MTplot` class, which stores the figure and handles the plotting, and each axes plotted is shown using a plot class from :mod:`MTfit.plot.plot_classes`. The plotting methods are designed to enable easy plotting without much user input, but also allow more complex plots to be made.
The :ref:`examples section <mtplot-examples>` shows two examples of using the plotting submodule.

The source code is shown in :doc:`source-plot_classes`.

.. warning::
    
    :mod:`matplotlib` does not plot 3d plots very well, as each object is converted to a 2d object and plotted, and given a constant zorder for the whole plot. Consequently, the bi-axes plot (:ref:`Chapman and Leaney, 2011 <Chapman-2011>`) is not included as an option, and other 3D plots may not always work correctly.

.. only:: not latex

    For the full plot_classes documentation, see :doc:`plot_classes`.

Using MTplot from the command line
======================================

:mod:`MTplot` can be run from the command line. A script should have been installed onto the path during installation and should be callable as::

    $ MTplot

However it may be necessary to install the script manually. This is platform dependent.

Script Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Linux
-------------------------------

Add this python script to a directory on the $PATH environmental variable::

    #!/usr/bin/env python
    import MTfit
    MTfit.plot.__run__()

and make sure it is executable.

Windows
--------------------------------

Add the linux script (above) to the path or if using powershell edit the powershell profile (usually found in *Documents/WindowsPowerShell/* - if not present use ``$PROFILE|Format-List -Force`` to locate it, it may be necessary to create the profile) and add::

    function MTplot{
        $script={
            python -c "import MTfit;MTfit.plot.__run__()" $args
            }
        Invoke-Command -ScriptBlock $script -ArgumentList $args
        }

Windows Powershell does seem to have some errors with commandline arguments, if necessary these should be enclosed in quotation marks e.g. "-d=datafile.inv"

Command Line Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are several command line options available, these can be found by calling::

    $ MTplot -h

The command line defaults can be set using the same defaults file as for MTfit (see :doc:`run`). 

Using MTplot from the Python interpreter
==========================================

Although MTplot can be run from the command line, it is much more powerful to run it from within the python interpreter.
To run MTplot from the python interpreter, create the :class:`~MTfit.plot.plot_classes.MTplot` object::

    >>> from MTfit.plot import MTplot
    >>> MTplot(MTs,plot_type='beachball',stations={},plot=True,*args,**kwargs)

See :ref:`Making the MTplot class <mtplotclass>` for more information on the  :class:`~MTfit.plot.plot_classes.MTplot` object.


.. _input-data-label:

Input Data
==================================

:func:`MTfit.plot.__core__.read` can read the three default output formats (MATLAB, hyp and pickle) for MTfit results. 

Additional parsers can be installed using the `MTfit.plot_read` entry point described in :doc:`extensions`.


.. _mtplotclass:

MTplot Class
========================

The MTplot class is used to handle plotting the moment tensors. The moment tensors are stored as an :ref:`MTData<mtdataclass>` class.


.. autoclass:: MTfit.plot.plot_classes.MTplot
   :members:   

-------------------

.. _mtdataclass:

MTData Class
========================

The MTData class is used for storing and converting the moment tensors for plotting.

.. autoclass:: MTfit.plot.plot_classes.MTData
   :members:  

.. _beachball:

Beachball plot
==========================

The simplest plot is a beachball plot using the :class:`MTfit.plot.plot_classes._AmplitudePlot` class.

Using the MTplot function, it can be made with the following commands::

    >>> import MTfit
    >>> import numpy as np
    >>> MTfit.plot.MTplot(np.array([[1],[0],[-1],[0],[0],[0]]),'beachball',
            fault_plane=True)

This plots the equal area projection of the source (a double-couple).

Stations can be included as a dictionary, with the azimuths and takeoff angles in degrees, such as::

    >>> stations={'names':['S01','S02','S03','S04'],
                  'azimuth':np.array([120.,5.,250.,75.]),
                  'takeoff_angle':np.array([30.,60.,45.,10.]),
                  'polarity':np.array([1,0,1,-1])}
    >>> MTfit.plot.MTplot(np.array([[1],[0],[-1],[0],[0],[0]]),'beachball',
                stations=stations,fault_plane=True)

If the polarity probabilities have been used in the inversion, the probabilities can be plotted on the receivers, by setting the stations ``polarity`` array as an array of the larger polarity probabilities, with negative polarity probabilities corresponding to polarities in the negative direction, e.g.::

    >>> stations={'names':['S01','S02','S03','S04'],
                  'azimuth':np.array([120.,5.,250.,75.]),
                  'takeoff_angle':np.array([30.,60.,45.,10.]),
                  'polarity':np.array([0.8,0.5,0.7,-0.9])}
    >>> MTfit.plot.MTplot(np.array([[1],[0],[-1],[0],[0],[0]]),'beachball',
                stations=stations,fault_plane=True)

To tweak the plot further, the plot class can be used directly::

    >>> import MTfit
    >>> import numpy as np
    >>> X=MTfit.plot.plot_classes._AmplitudePlot(False,False,
            np.array([[1],[0],[-1],[0],[0],[0]]),'beachball',
            stations=stations,fault_plane=True)
    >>> X.plot()

The first two arguments correspond to the subplot_spec and matplotlib figure to be used - if these are False, then a new figure is created. 

It uses the :class:`MTfit.plot.plot_classes._AmplitudePlot` class:

.. autoclass:: MTfit.plot.plot_classes._AmplitudePlot
   :members:  plot

.. _faultplane:

Fault Plane plot
==========================

A similar plot to the amplitude beachball plot is the fault plane plot, made using the :class:`MTfit.plot.plot_classes._FaultPlanePlot` class.

Using the MTplot function, it can be made with the following commands::

    >>> import MTfit
    >>> import numpy as np
    >>> MTfit.plot.MTplot(np.array([[1],[0],[-1],[0],[0],[0]]),'faultplane',
            fault_plane=True)

This plots the equal area projection of the source (a double-couple).

Stations can be included as a dictionary, like with the beachball plot.

The fault plane plot also can plot the solutions for multiple moment tensors, so the input array can be longer::

    >>> import MTfit
    >>> import numpy as np
    >>> MTfit.plot.MTplot(np.array([[ 1,0.9, 1.1,0.4],
                                    [ 0,0.1,-0.1,0.6],
                                    [-1, -1,  -1, -1],
                                    [ 0,  0,   0,  0],
                                    [ 0,  0,   0,  0],
                                    [ 0,  0,   0,  0]]),
                         'faultplane',fault_plane=True)

There are additional initialisation arguments, such as ``show_max_likelihood`` and ``show_mean`` boolean flags, which shows the maximum likelihood fault planes in the color given by the default color argument, and the mean orientation in green.

Additionally, if  the probability argument is set, the fault planes are coloured by the probability, with more likely planes darker.


It uses the :class:`MTfit.plot.plot_classes._FaultPlanePlot` class:

.. autoclass:: MTfit.plot.plot_classes._FaultPlanePlot
   :members:  plot

.. _rjplot:

Riedesel-Jordan plot
==========================

The Riedesel-Jordan plot is more complicated, and is described in :ref:`Riedesel and Jordan (1989)<Riedesel-1989>`. It plots the source type on the focal sphere, in a region described by the source eigenvectors.

Using the MTplot function, it can be made with the following commands::

    >>> import MTfit
    >>> import numpy as np
    >>> MTfit.plot.MTplot(np.array([[1],[0],[-1],[0],[0],[0]]),'riedeseljordan')

This plots the equal area projection of the source (a double-couple).

Stations cannot be shown on this plot.

The Riedesel-Jordan plot cannot plot the solutions for multiple moment tensors, so the input array can only be one moment tensor.

It uses the :class:`MTfit.plot.plot_classes._RiedeselJordanPlot` class:

.. autoclass:: MTfit.plot.plot_classes._RiedeselJordanPlot
   :members:  plot


.. _radiation:

Radiation plot
==========================

The radiation plot shows the same pattern as the beachball plot, except the shape is scaled by the amplitude on the focal sphere.

Using the MTplot function, it can be made with the following commands::

    >>> import MTfit
    >>> import numpy as np
    >>> MTfit.plot.MTplot(np.array([[1],[0],[-1],[0],[0],[0]]),'radiation')

This plots the equal area projection of the source (a double-couple).

Stations cannot be shown on this plot.

The radiation plot cannot plot the solutions for multiple moment tensors, so the input array can only be one moment tensor.

It uses the :class:`MTfit.plot.plot_classes._RadiationPlot` class:

.. autoclass:: MTfit.plot.plot_classes._RadiationPlot
   :members:  plot

.. _hudson:

Hudson plot
==========================

The Hudson plot is a source type plot, described in :ref:`Hudson et al. (1989)<Hudson-1989>`. It plots the source type in a quadrilateral, depending on the chosen projection. There are two projections, the tau-k plot and the u-v plot, with the latter being more common (and the default).

Using the MTplot function, it can be made with the following commands::

    >>> import MTfit
    >>> import numpy as np
    >>> MTfit.plot.MTplot(np.array([[1],[0],[-1],[0],[0],[0]]),'hudson')

This plots the u-v plot of the source (a double-couple).

Stations cannot be shown on this plot.

The Hudson plot can plot the solutions for multiple moment tensors, so the input array can be longer. Additionally, it can also plot a histogram of the PDF, if the probability argument is set.

It uses the :class:`MTfit.plot.plot_classes._HudsonPlot` class:

.. autoclass:: MTfit.plot.plot_classes._HudsonPlot
   :members:  plot

.. _lune:

Lune plot
==========================

The Lune plot is a source type plot, described in :ref:`Tape and Tape (2012)<Tape-2012>`. It plots the source type in the fundamental eigenvalue lune, which can be projected into 2 dimensions.

Using the MTplot function, it can be made with the following commands::

    >>> import MTfit
    >>> import numpy as np
    >>> MTfit.plot.MTplot(np.array([[1],[0],[-1],[0],[0],[0]]),'lune')

Stations cannot be shown on this plot.

The Lune plot can plot the solutions for multiple moment tensors, so the input array can be longer. Additionally, it can also plot a histogram of the PDF, if the probability argument is set.

It uses the :class:`MTfit.plot.plot_classes._LunePlot` class:

.. autoclass:: MTfit.plot.plot_classes._LunePlot
   :members:  plot

.. _mtplot-examples:

Examples
===============================

This section shows a pair of simple examples and their results.

The first example is to plot the data from :ref:`Krafla P Polarity example<real-p-polarity>`::

    import MTfit
    import numpy as np
    #Load Data    
    st_dist=MTfit.plot.read('krafla_event_ppolarityDCStationDistribution.mat',
        station_distribution=True)
    DCs,DCstations=MTfit.plot.read('krafla_event_ppolarityDC.mat')
    MTs,MTstations=MTfit.plot.read('krafla_event_ppolarityMT.mat')
    #Plot
    plot=MTfit.plot.MTplot([np.array([1,0,-1,0,0,0]),DCs,MTs],
        stations=[DCstations,DCstations,MTstations],
        station_distribution=[st_dist,False,False],
        plot_type=['faultplane','faultplane','hudson'],fault_plane=[False,True,False],
        show_mean=False,show_max=True,grid_lines=True,TNP=False,text=[False,False,True])


.. only:: not latex

  This produces a :mod:`matplotlib` figure:

  .. figure:: figures/krafla_event_mtplot_example.png
    :figwidth: 60 %
    :width: 90%
    :align: center
    :alt: Beachball plot showing the fault plane orientations for the double-couple constrained inversion and the marginalised source-type PDF for the full moment tensor inversion of the krafla data.

    *Beachball plots showing the station location uncertainty, and the fault plane orientations for the double-couple constrained inversion and the marginalised source-type PDF for the faultplane moment tensor inversion of the krafla data using polarity probabilities.*

.. only:: latex

  This produces a :mod:`matplotlib` figure, shown in Fig. :ref:`11.1<krafla-event-mtplot-example-fig>`.

  .. _krafla-event-mtplot-example-fig:

  .. figure:: figures/krafla_event_mtplot_example.png
    :width: 100%
    :align: center
    :alt: Beachball plot showing the fault plane orientations for the double-couple constrained inversion and the marginalised source-type PDF for the full moment tensor inversion of the krafla data.

    *Beachball plots showing the station location uncertainty, and the fault plane orientations for the double-couple constrained inversion and the marginalised source-type PDF for the faultplane moment tensor inversion of the krafla data using polarity probabilities.*

The second example shows the different plot types::

    import MTfit
    import numpy as np
    import scipy.stats as sp
    #Generate Data    
    n=100
    DCs=MTfit.MTconvert.Tape_MT6(np.zeros(n),np.zeros(n),np.pi+0.1*np.random.randn(n),
            0.5+0.01*np.random.randn(n),0.1*np.random.randn(n))
    probDCs=np.random.rand(n)
    n=10000
    g=-np.pi/12+0.01*np.random.randn(n)
    d=np.pi/3+0.1*np.random.randn(n)
    MTs=MTfit.MTconvert.Tape_MT6(g,d,np.pi+0.1*np.random.randn(n),
            0.5+0.01*np.random.randn(n),0.1*np.random.randn(n))
    probMTs=sp.norm.pdf(g,-np.pi/12,0.01)*sp.norm.pdf(d,np.pi/3,0.1)
    plot_sources=[np.array([1,0,1,-1,0,0]),DCs,MTs,MTs,np.array([1,0,1,-1,0,0])]
    #Plot
    plot=MTfit.plot.MTplot(plot_sources,
        plot_type=['beachball','faultplane','hudson','lune','riedeseljordan'],
        probability=[False,probDCs,probMTs,probMTs,False],
        colormap=['bwr','bwr','viridis','viridis','bwr'],
        stations=[{'names':['S01','S02','S03','S04'],
                   'azimuth':np.array([120.,45.,238.,341.]),
                   'takeoff_angle':np.array([12.,56.,37.,78.]),
                   'polarity':[1,0,-1,-1]},{},{},{},{}],
        show_mean=True,show_max=True,grid_lines=True,TNP=False,fontsize=6,
        station_markersize=2,markersize=2)


.. only:: not latex

  This produces a :mod:`matplotlib` figure:

  .. figure:: figures/mtplot_example.png
    :figwidth: 60 %
    :width: 90%
    :align: center
    :alt: MTplot examples showing an equal area projection of a beachball for an example moment tensor source, fault plane distribution showing the mean orientation in green, Hudson and lune type plots of a full moment tensor PDF, and a Riedesel-Jordan type plot of an example moment tensor source.

    *MTplot examples showing an equal area projection of a beachball for an example moment tensor source, fault plane distribution showing the mean orientation in green, Hudson and lune type plots of a full moment tensor PDF, and a Riedesel-Jordan type plot of an example moment tensor source.*

.. only:: latex

  This produces a :mod:`matplotlib` figure, shown in Fig. :ref:`11.2<mtplot-example-fig>`.

  .. _mtplot-example-fig:

  .. figure:: figures/mtplot_example.png
    :width: 100%
    :align: center
    :alt: MTplot examples showing an equal area projection of a beachball for an example moment tensor source, fault plane distribution showing the mean orientation in green, Hudson and lune type plots of a full moment tensor PDF, and a Riedesel-Jordan type plot of an example moment tensor source.

    *MTplot examples showing an equal area projection of a beachball for an example moment tensor source, fault plane distribution showing the mean orientation in green, Hudson and lune type plots of a full moment tensor PDF, and a Riedesel-Jordan type plot of an example moment tensor source.*