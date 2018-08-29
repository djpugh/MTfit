*******************************
Running MTfit
*******************************

There are several ways to run :mod:`MTfit`, and these are described here.

Command Line
===============================

:mod:`MTfit` can be run from the command line. A script should have been installed onto the path during installation and should be callable as::

    $ MTfit


However it may be necessary to install the script manually. This is platform dependent.

Script Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Linux
-------------------------------

Add this python script to a directory in the $PATH environmental variable::

    #!/usr/bin/env python
    import MTfit
    MTfit.__run__()

And make sure it is executable.

Windows
--------------------------------

Add the linux script (above) to the path or if using powershell edit the powershell profile (usually found in *Documents/WindowsPowerShell/* - if not present use ``$PROFILE|Format-List -Force`` to locate it, it may be necessary to create the profile) and add::

    function MTfit{
        $script={
            python -c "import MTfit;MTfit.__run__()" $args
            }
        Invoke-Command -ScriptBlock $script -ArgumentList $args
        }

Windows Powershell does seem to have some errors with commandline arguments, if necessary these should be enclosed in quotation marks e.g. "-d=datafile.inv"

Command Line Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When running :mod:`MTfit` from the command line, there are many options available, and these can be listed using::

    $ MTfit -h


.. only:: latex

    For a description of these options see Chapter :latex:`\ref{cli::doc}`.

.. only:: not latex

    For a description of these options see :doc:`cli`.

The command line defaults can be set using a defaults file. This is recursively checked in 3 locations:

    1. ``MTFITDEFAULTSPATH`` environmental variable (could be a system level setting)
    2. ``.MTfitdefaults`` file in the users home directory
    3. ``.MTfitdefaults`` file in the current working directory

The higher number file over-writes defaults in the lower files if they conflict.

The structure of the defaults file is simply::

    key:attr

e.g.::
    
    dc:True
    algorithm:iterate


Python Interpreter
=================================

Running MTfit from the python interpreter is done as::

    >>> import MTfit
    >>> args=['-o','-d']
    >>> MTfit.__run__(args)

.. only:: latex

    Where args correspond to the command line arguments (see Chapter :latex:`\ref{cli::doc}`.

.. only:: not latex

    Where args correspond to the command line arguments (see :doc:`cli`).

It is also possible to create the :class:`~MTfit.inversion.Inversion` object::

    >>> import MTfit
    >>> inversion=MTfit.Inversion(*args,**kwargs)
    >>> inversion.forward()


.. only:: latex

    The descriptions of the :class:`~MTfit.inversion.Inversion` initialisation arguments can be found in the :class:`~MTfit.inversion.Inversion.__init__` docstrings, and :latex:`\ref{inversion::doc}`.

.. only:: not latex

    The descriptions of the :class:`~MTfit.inversion.Inversion` initialisation arguments can be found in the :class:`~MTfit.inversion.Inversion.__init__` docstrings, and :doc:`inversion`.




.. _input-data-label:

Input Data
==================================

There are several different input data types, and it is also possible to add additional parsers using the ``MTfit.parsers`` entry point.


The required data structure for running MTfit is very simple, the inversion expects a python dictionary of the data in the format::

    >>> data={'PPolarity':{'Measured':numpy.matrix([[-1],[-1]...]),
                         'Error':numpy.matrix([[0.01],[0.02],...]),
                         'Stations':{'Name':['Station1','Station2',...],
                                     'Azimuth':numpy.matrix([[248.0],[122.3]...]),
                                     'TakeOffAngle':numpy.matrix([[24.5],[22.8]...]),
                                    }
                         },
              'PSHAmplitudeRatio':{...},
              ...
              'UID':'Event1'
              }

For more information on the data keywords and how to set them up, see :class:`~MTfit.inversion.Inversion` docstrings.

The data dictionary can be passed directly to the :class:`~MTfit.inversion.Inversion` object (simple if running within python), or from a binary pickled object, these can be made by simply using pickle (or cPickle)::

    >>> pickle.dump(data,open(filename,'wb'))


The coordinate system is that the Azimuth is angle from x towards y and TakeOffAngle is the angle from positive z.

For data in different formats it is necessary to write a parser to convert the data into this dictionary format.

There is a parser for csv files with format

CSV
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is a CSV format parser which reads CSV files.
The CSV file format is to have events split by blank lines, a header line showing where the information is, UID and data-type information stored in the first column, e.g.::

    UID=123,,,,
    PPolarity,,,,
    Name,Azimuth,TakeOffAngle,Measured,Error
    S001,120,70,1,0.01
    S002,160,60,-1,0.02
    P/SHRMSAmplitudeRatio,,,,
    Name,Azimuth,TakeOffAngle,Measured,Error
    S003,110,10,1,0.05 0.04
    ,,,,
    PPolarity ,,,,
    Name,Azimuth,TakeOffAngle,Measured,Error
    S003,110,10,1,0.05

This is a CSV file with 2 events, one event ID of 123, and PPolarity data at station S001 and station S002 and P/SHRMSAmplitude data at station S003,
and a second event with no ID (will default to the event number, in this case 2) with PPolarity data at station S003.


hyp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is a hyp format parser which reads hyp files as defined by `NonLinLoc <http://alomax.free.fr/nlloc/soft6.00/formats.html#_location_hypphs_>`_, this allows output files from NonLinLoc to be directly read.


.. _MATLAB-output-label:

Output
==================================

The default output is to output a MATLAB file containing 2 structures and a cell array, although there are two other possible formats, and others can be added (see MTfit.extensions).
The ``Events`` structure has the following fieldnames: ``MTspace`` and ``Probability``.

    * ``MTspace`` - The moment tensor samples as a 6 by n vector of the form::

        Mxx
        Myy
        Mzz
        sqrt(2)*Mxy
        sqrt(2)*Mxz
        sqrt(2)*Myz

    * ``Probability`` - The corresponding probability values

The ``Other`` structure contains information about the inversion

The ``Stations`` cell array contains the station information, including, if available, the polarity:

    +-----+----------------------+---------------------------+--------------------------+
    |Name |Azimuth(angle from x) |TakeOffAngle(angle from z) |P Polarity (if available) |
    +-----+----------------------+---------------------------+--------------------------+

A log file for each event is also produced to help with debugging and understanding the results.

Pickle format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is also possible to output the data structure as a pickled file using the pickle output options, storing the output dictionary as a pickled file.

hyp format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The results can be outputted in the `NonLinLoc hyp format <http://alomax.free.fr/nlloc/soft6.00/formats.html#_location_hypphs_>`_,
with the range of solutions sampled outputted as a binary file with the following format::

    binary file version (unsigned long integer)
    total_number_samples(unsigned long integer)
    number_of_saved_samples(unsigned long integer)
    converted (bool flag)
    Ln_bayesian_evidence (double)
    Kullback-Liebeler Divergence from sampling prior (double)

Then for each moment tensor sample (up to ``number_of_saved_samples`` )::

    Probability (double)
    Ln_probability(double)
    Mnn (double)
    Mee (double)
    Mdd (double)
    Mne (double)
    Mnd (double)
    Med (double)

if Converted is true then each sample also contains::

    gamma (double)
    delta (double)
    kappa (double)
    h (double)
    sigma (double)
    u (double)
    v (double)
    strike1 (double)
    dip1 (double)
    rake1 (double)
    strike2 (double)
    dip2 (double)
    rake2 (double)

If there are multiple events saved, then the next event starts immediately after the last with the same format. The output binary file can be re-read into python using :func:`MTfit.inversion.read_binary_output`.





Running in parallel
==================================

The code is written to run in parallel using multiprocessing, it will initialise as many threads as the system reports available.
A single thread mode can be forced using:

    * -l, --singlethread, --single, --single_thread flag on the command line
    * parallel=False keyword in the MTfit.inversion.Inversion object initialisation

It is also possible to run this code on a cluster using qsub [requires pyqsub]. This can be called from the commandline using a flag:

    * -q, --qsub, --pbs

This runs using a set of default parameters, however it is also possible to adjust these parameters using commandline flags (use -h flag for help and usage).

There is a bug when using mpi and very large result sizes, giving a size error (negative integer) in :mod:`mpi4py`. If this occurs, lower the sample size and it will be ok.


.. warning::


    If running this on a server, be aware that not setting the number of workers option ``--numberworkers``, when running in parallel, means that as many processes as processors will be spawned, slowing down the machine for any other users.


