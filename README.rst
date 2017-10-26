The documentation is available at `https://djpugh.github.io/MTfit/<https://djpugh.github.io/MTfit/>`_ and can be built using `sphinx` from the source in mtfit/docs/.

The documentation includes tutorials and explanations of mtfit and the approaches used.


**Restricted:  For Non-Commercial Use Only**
This code is protected intellectual property and is available solely for teaching
and non-commercially funded academic research purposes.
Applications for commercial use should be made to Schlumberger or the University of Cambridge.


Installing mtfit
*********************************

mtfit can be installed simply by calling::
    
    >>python setup.py install

mtfit is dependent on numpy and scipy, and for MATLAB -v7.3 support also requires h5py.
Cluster support will be automatically installed via pyqsub from github
MPI support requires mpi4py built against a valid MPI distribution.




! Known Bug - running with MPI and very large non-zero MT results can lead to an error: mpi4py SystemError: Negative size passed to PyString_FromStringAndSize - to fix, re-run with smaller sample sizes