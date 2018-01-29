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



