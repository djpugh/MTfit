# cython: infer_types=True
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# **Restricted:  For Non-Commercial Use Only** 
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.

cimport cython
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
from cpython cimport bool
# DTYPE=np.float64
# ctypedef np.float64_t DTYPE_t
ctypedef double DTYPE_t
ctypedef long long LONG
# ctypedef long long
from libc.stdlib cimport rand, RAND_MAX
IF UNAME_SYSNAME == "Windows":
    from libc.math cimport HUGE_VAL as inf
ELSE:
    from libc.math cimport INFINITY as inf
    from libc.math cimport fmax
    from libc.math cimport fmin
    from libc.math cimport erf
from libc.math cimport sqrt
from libc.math cimport exp
from libc.math cimport log
from libc.math cimport fabs
from libc.math cimport cos
from libc.math cimport sin
from libc.math cimport M_PI as pi
from libc.math cimport M_SQRT2 as sqrt2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef DTYPE_t[::1] get_multipliers(DTYPE_t[:,:,::1] angles,DTYPE_t bin_size,DTYPE_t [::1] multipliers) nogil:
    cdef Py_ssize_t umax=angles.shape[0]
    cdef Py_ssize_t nsta=angles.shape[2]
    cdef Py_ssize_t u
    cdef Py_ssize_t w
    cdef int ok=1
    for u in range(umax):
        if multipliers[u]==-1:
            continue
        for v in range(u+1,umax):  
            if multipliers[v]>-1:
                ok=1
                for w in range(nsta):
                    ok=ok*(fabs(angles[u,0,w]-angles[v,0,w])<bin_size/2.)*(fabs(angles[u,1,w]-angles[v,1,w])<bin_size/2.)
                    if ok==0:
                        break
                if w==nsta-1 and ok>0:
                    multipliers[u]+=multipliers[v]
                    if v>u:
                        multipliers[v]=-1
    return multipliers


cpdef bin_scatangle(sample_records,multipliers,bin_size):
    cdef DTYPE_t[:,:,::1] angles=np.empty((len(sample_records),2,len(sample_records[0]['Name'])))
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    new_records=[]
    new_multipliers=[]
    for i,record in enumerate(sample_records):
        # print i,record
        for j in range(len(record['Name'])):
            angles[i,0,j]=record['TakeOffAngle'][j,0]
            angles[i,1,j]=record['Azimuth'][j,0]
    # print multipliers,bin_size,angles
    multipliers=np.asarray(get_multipliers(angles,bin_size,multipliers)).flatten()
    for i,record in enumerate(sample_records):
        if multipliers[i]>0:
            new_records.append(record)
            new_multipliers.append(multipliers[i])
    return new_records,new_multipliers