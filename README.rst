.. image:: https://travis-ci.org/djpugh/MTfit.svg?branch=develop

The documentation is available at `https://djpugh.github.io/MTfit/ <https://djpugh.github.io/MTfit/>`_ and can be built using `sphinx` from the source in MTfit/docs/, or using the `build_docs.py`.

The documentation includes tutorials and explanations of MTfit and the approaches used.

Please note that this code is provided as-is, and no guarantee is given that this code will perform in the desired way. Additional development and support is carried out in the developer's free time.

**Restricted:  For Non-Commercial Use Only**
This code is protected intellectual property and is available solely for teaching
and non-commercially funded academic research purposes.
Applications for commercial use should be made to Schlumberger or the University of Cambridge.


Installing MTfit
*********************************

MTfit is available on `PyPI` and can be installed using:

    >>pip install MTfit

Alternative this repository can be cloned and the package then installed simply by calling::
    
    >>python setup.py install

MTfit is dependent on numpy and scipy, and for MATLAB -v7.3 support also requires h5py.
Cluster support will be automatically installed via pyqsub from github
MPI support requires mpi4py built against a valid MPI distribution.

To build the C extensions when compiling from source you will need cython and associated C compilers




! Known Bug - running with MPI and very large non-zero MT results can lead to an error: mpi4py SystemError: Negative size passed to PyString_FromStringAndSize - to fix, re-run with smaller sample sizes
