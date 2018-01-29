**********************************
Installing MTfit
**********************************

:mod:`MTfit` is available from `PyPI` and can be installed using ::

    $ pip install MTfit


:mod:`MTfit` is available in several formats, including as a ``tar.gz`` file, a ``zip`` file, and as ``wheels`. Additionally the git repository can be cloned.

Apart from the wheels, all the other formats require installing from the source (after unpacking the compressed files e.g. ``tar.gz``).  

:mod:`MTfit` can be installed from the source by calling::

    $ python setup.py install

To see additional command line options for the ``setup.py`` file use::

    $ python setup.py --help

or::

    $ python setup.py --help-commands

Requirements
===================================

:mod:`MTfit` requires three modules

* `NumPy <http://www.numpy.org>`_
* `SciPy <http://www.scipy.org>`_
* `pyqsub <https://www.github.com/djpugh/pyqsub>`_ - a simple module to provide interfacing with qsub, and will be automatically installed.
* `matplotlib <http://matplotlib.org/>`_

Optional requirements
----------------------------------

Additionally there are several optional requirements which allow additional features in :mod:`MTfit`.

HDF5 (Matlab -v7.3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use the HDF5 MATLAB format (format -v7.3) or disk-based storage of the in progress sampling (slower but saves on memory requirements) requires:

* `h5py <http://www.h5py.org/>`_
* `hdf5storage <http://pythonhosted.org/hdf5storage/>`_

If installing from source these modules require:

* `HDF5 <http://www.hdfgroup.org/HDF5/>`_ 1.8.4 or newer, shared library version with development headers (libhdf5-dev or similar)
* Python 2.6 - 3.3 with development headers (python-dev or similar)
* NumPy 1.6 or newer
* Optionally: `Cython <http://cython.org/>`_, if you want to access features introduced after HDF5 1.8.4, or Parallel HDF5.

.. warning::
    HDF5 support is required if the output files are large (>2GB) and MATLAB output is used, beacause MATLAB cannot read older format files bigger than this.

MPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run on multiple nodes on a cluster requires :mod:`mpi4py` installed and a distribution of `MPI <http://www.mcs.anl.gov/research/projects/mpi/>`_ such as `OpenMPI <http://www.open-mpi.org/>`_ to run in parallel on multiple nodes (single node multi-processor uses :mod:`multiprocessing`)

* `mpi4py <http://mpi4py.scipy.org/>`_


NonLinLoc Location Uncertainty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`NonLinLoc <http://alomax.free.fr/nlloc>`_ scatter files can be used for the location PDF. This requires:

* :mod:`Scat2Angle` from `pyNLLoc <https://www.github.com/djpugh/pyNLLoc>`_. 

Building the Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To build this documentation from source requires:

* `sphinx <http://sphinx-doc.org>`_ 1.3.1 or newer

It can be built in the source directory::

    $ python setup.py build-docs

and after installation::
    
    >>MTfit.build_docs()

Running the Test Suite
==================================
:mod:`MTfit` comes with a complete test suite which can be run in the source directory::

    $ python setup.py build
    $ python setup.py test

and after installation from the python interpreter::
    
    >>> import MTfit
    >>> MTfit.run_tests()






