#!python
# cython: infer_types=True

# **Restricted:  For Non-Commercial Use Only** 
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.

from libc.math cimport fabs
from libc.math cimport fmod
from libc.math cimport cos
from libc.math cimport acos
from libc.math cimport atan2
from libc.math cimport sin
from libc.math cimport sqrt
from libc.math cimport M_PI as pi
from libc.math cimport M_SQRT2 as sqrt2
from libc.stdlib cimport free

import unittest

cimport cython
cimport numpy as np
from scipy.linalg import eigh
from scipy.optimize import fsolve
from scipy import __version__ as __scipy_version__
import numpy as np
from cpython cimport bool

from ..utilities.unittest_utils import TestCase


cdef DTYPE_t PI2=2*pi
cdef DTYPE_t sqrt3=sqrt(3)
cdef DTYPE_t rad_cor=180/pi
cdef int check_finite=1
if int(__scipy_version__.split('.')[0])==0 and int(__scipy_version__.split('.')[1])<13:
    check_finite=0
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t[::1] cE_tk(DTYPE_t[::1] E,DTYPE_t[::1] results) nogil:
    cdef DTYPE_t iso=(E[0]+E[1]+E[2])/3
    cdef DTYPE_t dev0=E[0]-iso
    cdef DTYPE_t dev1=E[2]-iso#Odd sorting from hudson paper E[0]>=E[2]>=E[1]
    cdef DTYPE_t dev2=E[1]-iso
    if dev2>0:
        results[5]=iso/(fabs(iso)-dev1)
        results[6]=-2*(dev2)/(dev1)
    elif dev2<0:
        results[5]=iso/(fabs(iso)+dev0)
        results[6]=2*dev2/dev0
    else:
        results[5]=iso/(fabs(iso)+dev0)
        results[6]=0
    results[6]=results[6]*(1-fabs(results[5]));
    return results
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t[::1] ctk_uv(DTYPE_t[::1] results) nogil:
    cdef DTYPE_t k=results[5]
    cdef DTYPE_t tau=results[6]
    if tau>0 and k>0:
        if tau<4*k:
            results[5]=tau/(1-(tau/2))
            results[6]=k/(1-(tau/2))
        else:
            results[5]=tau/(1-2*k)
            results[6]=k/(1-2*k)
    elif tau<0 and k<0:
        if tau>4*k:
            results[5]=tau/(1+(tau/2))
            results[6]=k/(1+(tau/2))
        else:
            results[5]=tau/(1+2*k)
            results[6]=k/(1+2*k)
    else:
        results[5]=tau
        results[6]=k
    return results
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void cE_gd(DTYPE_t[:] E,DTYPE_t*g,DTYPE_t*d) nogil:
    if E[0]==E[2]:
        if E[0]>0:
            g[0]=0
            d[0]=pi/2
        elif E[0]<0:
            g[0]=0
            d[0]=-pi/2
    else:
        g[0]=atan2(-E[0]+2*E[1]-E[2],sqrt3*(E[0]-E[2]))
        d[0]=pi/2-acos((E[0]+E[1]+E[2])/(sqrt3*sqrt((E[0]*E[0]+E[1]*E[1]+E[2]*E[2]))))
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void cN_SDR(DTYPE_t N0,DTYPE_t N1,DTYPE_t N2,DTYPE_t S0,DTYPE_t S1,DTYPE_t S2,DTYPE_t* strike,DTYPE_t*dip,DTYPE_t*rake) nogil:
    if N2 > 0:
        N0 = -N0
        N1 = -N1
        N2 = -N2
        S0 = -S0
        S1 = -S1
        S2 = -S2
    strike[0] = atan2(-N0, N1)
    dip[0] = atan2((N1**2+N0**2), sqrt((N0*N2)**2+(N1*N2)**2))
    rake[0] = atan2(-S2, S0*N1 - S1*N0)
    if dip[0] > pi/2:
        dip[0] = pi-dip[0]
        strike[0] = strike[0]+pi
        rake[0] = PI2-rake[0]
    if fabs(strike[0]) > PI2:
        strike[0] = fmod(strike[0],PI2)
    if strike[0] < 0.0:
        strike[0] += PI2
    if rake[0] > pi:
        rake[0] -= PI2
    elif rake[0] < -pi:
        rake[0] += PI2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef DTYPE_t[::1] cTP_SDR(DTYPE_t[::1] T,DTYPE_t[::1] P,DTYPE_t[::1] results) nogil:    
    cdef DTYPE_t Nt=sqrt((T[0]+P[0])*(T[0]+P[0])+(T[1]+P[1])*(T[1]+P[1])+(T[2]+P[2])*(T[2]+P[2]))
    cdef DTYPE_t St=sqrt((T[0]-P[0])*(T[0]-P[0])+(T[1]-P[1])*(T[1]-P[1])+(T[2]-P[2])*(T[2]-P[2]))
    cdef DTYPE_t N2=(T[2]+P[2])/Nt
    cdef DTYPE_t N0=(T[0]+P[0])/Nt
    cdef DTYPE_t N1=(T[1]+P[1])/Nt
    cdef DTYPE_t S2=(T[2]-P[2])/St
    cdef DTYPE_t S0=(T[0]-P[0])/St
    cdef DTYPE_t S1=(T[1]-P[1])/St
    cdef DTYPE_t strike1=0.
    cdef DTYPE_t dip1=0.
    cdef DTYPE_t rake1=0.
    cdef DTYPE_t strike2=0.
    cdef DTYPE_t dip2=0.
    cdef DTYPE_t rake2=0.
    cN_SDR(N0,N1,N2,S0,S1,S2,&strike1,&dip1,&rake1)
    cN_SDR(S0,S1,S2,N0,N1,N2,&strike2,&dip2,&rake2)
    #Switch for sdr2
    if fabs(rake1)>pi/2:   
        results[2]=strike2
        results[3]=cos(dip2)
        results[4]=rake2
    else:        
        results[2]=strike1
        results[3]=cos(dip1)
        results[4]=rake1
    results[7]=strike1*rad_cor
    results[8]=dip1*rad_cor
    results[9]=rake1*rad_cor
    results[10]=strike2*rad_cor
    results[11]=dip2*rad_cor
    results[12]=rake2*rad_cor
    return results
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef DTYPE_t[::1] cTPE_convert(DTYPE_t[::1] T,DTYPE_t[::1] P,DTYPE_t[::1]E,DTYPE_t[::1]results) nogil: 
    cE_gd(E,&results[0],&results[1])
    results=cE_tk(E,results)
    results=ctk_uv(results)
    results=cTP_SDR(T,P,results)
    return results
#Tape to Moment tensor conversion
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void cTape_MT6(DTYPE_t*M, DTYPE_t gamma,DTYPE_t delta,DTYPE_t kappa,DTYPE_t h,DTYPE_t sigma) nogil:
    #Get T N P axes
    cdef DTYPE_t ck=cos(kappa)
    cdef DTYPE_t cs=cos(sigma)
    cdef DTYPE_t sk=sin(kappa)
    cdef DTYPE_t ss=sin(sigma)
    cdef DTYPE_t cg=cos(gamma)
    cdef DTYPE_t sg=sin(gamma)
    cdef DTYPE_t cd=cos(delta)
    cdef DTYPE_t sh=sqrt(1-h*h)
    cdef DTYPE_t NT=sqrt((ck*cs+sk*h*ss-sk*sh)*(ck*cs+sk*h*ss-sk*sh)+(sk*cs-ck*h*ss+ck*sh)*(sk*cs-ck*h*ss+ck*sh)+(-sh*ss-h)*(-sh*ss-h))
    cdef DTYPE_t NP=sqrt((ck*cs+sk*h*ss+sk*sh)*(ck*cs+sk*h*ss+sk*sh)+(sk*cs-ck*h*ss-ck*sh)*(sk*cs-ck*h*ss-ck*sh)+(-sh*ss+h)*(-sh*ss+h))
    cdef DTYPE_t T1=(ck*cs+sk*h*ss-sk*sh)/NT
    cdef DTYPE_t T2=(sk*cs-ck*h*ss+ck*sh)/NT
    cdef DTYPE_t T3=(-sh*ss-h)/NT
    cdef DTYPE_t P1=(ck*cs+sk*h*ss+sk*sh)/NP
    cdef DTYPE_t P2=(sk*cs-ck*h*ss-ck*sh)/NP
    cdef DTYPE_t P3=(-sh*ss+h)/NP
    cdef DTYPE_t N1=T2*P3-P2*T3
    cdef DTYPE_t N2=-T1*P3+P1*T3
    cdef DTYPE_t N3=T1*P2-T2*P1
    #Normalised TNP Values
    cdef DTYPE_t sd=sin(delta)
    cdef DTYPE_t L1=(sqrt(3)*cg*cd-sg*cd+sqrt(2)*sd)/sqrt(6)
    cdef DTYPE_t L2=(2*sg*cd+sqrt(2)*sd)/sqrt(6)
    cdef DTYPE_t L3=(-sqrt(3)*cg*cd-sg*cd+sqrt(2)*sd)/sqrt(6)
    M[0]=L1*T1*T1+L2*N1*N1+L3*P1*P1
    M[1]=L1*T2*T2+L2*N2*N2+L3*P2*P2
    M[2]=L1*T3*T3+L2*N3*N3+L3*P3*P3
    M[3]=sqrt2*(L1*T1*T2+L2*N1*N2+L3*P1*P2)
    M[4]=sqrt2*(L1*T1*T3+L2*N1*N3+L3*P1*P3)
    M[5]=sqrt2*(L1*T2*T3+L2*N2*N3+L3*P2*P3)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void cMultipleTape_MT6(DTYPE_t*M,DTYPE_t* gamma,DTYPE_t* delta,DTYPE_t* kappa,DTYPE_t* h,DTYPE_t* sigma,Py_ssize_t n) nogil:
    for i from 0<=i<n:
        cTape_MT6(&M[i*6],gamma[i],delta[i],kappa[i],h[i],sigma[i])
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def Tape_MT6(DTYPE_t[::1] gamma,DTYPE_t[::1] delta,DTYPE_t[::1] kappa,DTYPE_t[::1] h,DTYPE_t[::1] sigma):
    cdef Py_ssize_t n=gamma.shape[0]
    if not (gamma.shape[0]==delta.shape[0] and gamma.shape[0]==kappa.shape[0] and gamma.shape[0]==h.shape[0] and gamma.shape[0]==sigma.shape[0]):
        raise ValueError('Arguments different size')
    cdef DTYPE_t[:,::1] MTs=np.empty((n,6))
    cMultipleTape_MT6(&MTs[0,0],&gamma[0],&delta[0],&kappa[0],&h[0],&sigma[0],n)
    return np.ascontiguousarray(np.asarray(MTs).T)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef MT_convert(DTYPE_t[:] MT,DTYPE_t [::1] results):
    T,N,P,E=cMT6_TNPE(MT)    
    return cTPE_convert(T,P,E,results)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def MT_output_convert(DTYPE_t[:,:] MTs):
    cdef Py_ssize_t imax=MTs.shape[1]
    cdef Py_ssize_t i
    cdef DTYPE_t [::1] g=np.empty((imax,))
    cdef DTYPE_t [::1] d=np.empty((imax,))
    cdef DTYPE_t [::1] k=np.empty((imax,))
    cdef DTYPE_t [::1] h=np.empty((imax,))
    cdef DTYPE_t [::1] s=np.empty((imax,))
    cdef DTYPE_t [::1] u=np.empty((imax,))
    cdef DTYPE_t [::1] v=np.empty((imax,))
    cdef DTYPE_t [::1] s1=np.empty((imax,))
    cdef DTYPE_t [::1] d1=np.empty((imax,))
    cdef DTYPE_t [::1] r1=np.empty((imax,))
    cdef DTYPE_t [::1] s2=np.empty((imax,))
    cdef DTYPE_t [::1] d2=np.empty((imax,))
    cdef DTYPE_t [::1] r2=np.empty((imax,))
    cdef DTYPE_t [::1] results=np.empty((13,))
    for i in range(imax):
        results=MT_convert(MTs[:,i],np.empty((13,)))
        g[i]=results[0]
        d[i]=results[1]
        k[i]=results[2]
        h[i]=results[3]
        s[i]=results[4]
        u[i]=results[5]
        v[i]=results[6]
        s1[i]=results[7]
        d1[i]=results[8]
        r1[i]=results[9]
        s2[i]=results[10]
        d2[i]=results[11]
        r2[i]=results[12]
    return {'g':np.asarray(g),'d':np.asarray(d),'k':np.asarray(k),'h':np.asarray(h),'s':np.asarray(s),'u':np.asarray(u),'v':np.asarray(v),'S1':np.asarray(s1),'D1':np.asarray(d1),'R1':np.asarray(r1),'S2':np.asarray(s2),'D2':np.asarray(d2),'R2':np.asarray(r2)}   
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef cMT6_TNPE(DTYPE_t[:] MT):
    cdef DTYPE_t [:,::1] MT33=np.empty((3,3))
    cdef DTYPE_t [::1] E=np.empty((3,))
    cdef DTYPE_t [::1] e=np.empty((3,))
    cdef DTYPE_t [::1] T=np.empty((3,))
    cdef DTYPE_t [::1] P=np.empty((3,))
    cdef DTYPE_t [::1] N=np.empty((3,))
    cdef DTYPE_t [:,:] L=np.empty((3,3))
    MT33[0,0]=MT[0]
    MT33[1,1]=MT[1]
    MT33[2,2]=MT[2]
    MT33[0,1]=MT[3]/sqrt2
    MT33[0,2]=MT[4]/sqrt2
    MT33[1,2]=MT[5]/sqrt2
    MT33[1,0]=MT[3]/sqrt2
    MT33[2,0]=MT[4]/sqrt2
    MT33[2,1]=MT[5]/sqrt2
    if check_finite>0:
        e,L=eigh(MT33,overwrite_a=True,check_finite=False)
    else:        
        e,L=eigh(MT33,overwrite_a=True)
    maxi=0
    mini=0
    cdef Py_ssize_t i
    for i in [0,1,2]:
        if e[i]>e[maxi]:
            maxi=i
        if e[i]<e[mini]:
            mini=i
    E[0]=e[maxi]
    E[1]=e[3-maxi-mini]
    E[2]=e[mini]
    T[0]=L[0,maxi]
    T[1]=L[1,maxi]
    T[2]=L[2,maxi]
    N[0]=L[0,3-maxi-mini]
    N[1]=L[1,3-maxi-mini]
    N[2]=L[2,3-maxi-mini]
    P[0]=L[0,mini]
    P[1]=L[1,mini]
    P[2]=L[2,mini]
    return T,N,P,E
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef MT6_TNPE(DTYPE_t[:,:] MTs):
    cdef Py_ssize_t imax=MTs.shape[1]
    cdef Py_ssize_t i
    cdef DTYPE_t [:,::1] T=np.empty((3,imax))
    cdef DTYPE_t [:,::1] N=np.empty((3,imax))
    cdef DTYPE_t [:,::1] P=np.empty((3,imax))
    cdef DTYPE_t [:,::1] E=np.empty((3,imax))
    for i in range(imax):
        t,n,p,e=cMT6_TNPE(MTs[:,i])
        T[0,i]=t[0]
        N[0,i]=n[0]
        P[0,i]=p[0]
        E[0,i]=e[0]
        T[1,i]=t[1]
        N[1,i]=n[1]
        P[1,i]=p[1]
        E[1,i]=e[1]
        T[2,i]=t[2]
        N[2,i]=n[2]
        P[2,i]=p[2]
        E[2,i]=e[2]
    return np.asarray(T),np.asarray(N),np.asarray(P),np.asarray(E)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef E_GD(DTYPE_t[:,:]E):
    cdef Py_ssize_t imax=E.shape[1]
    cdef Py_ssize_t i
    cdef DTYPE_t [::1] g=np.empty((imax,))
    cdef DTYPE_t [::1] d=np.empty((imax,))
    for i in range(imax):
        cE_gd(E[:,i],&g[i],&d[i])
    return np.asarray(g),np.asarray(d)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef cSingleTP_SDR(DTYPE_t [:] T,DTYPE_t[:]P):
    cdef DTYPE_t t0 = T[0]
    cdef DTYPE_t t1 = T[1]
    cdef DTYPE_t t2 = T[2]
    cdef DTYPE_t p0 = P[0]
    cdef DTYPE_t p1 = P[1]
    cdef DTYPE_t p2 = P[2]
    cdef DTYPE_t Nt = sqrt((t0+p0)*(t0+p0)+(t1+p1)*(t1+p1)+(t2+p2)*(t2+p2))
    cdef DTYPE_t St = sqrt((t0-p0)*(t0-p0)+(t1-p1)*(t1-p1)+(t2-p2)*(t2-p2))
    cdef DTYPE_t N0 = (t0+p0)/Nt
    cdef DTYPE_t N1 = (t1+p1)/Nt
    cdef DTYPE_t N2 = (t2+p2)/Nt
    cdef DTYPE_t S0 = (t0-p0)/St
    cdef DTYPE_t S1 = (t1-p1)/St
    cdef DTYPE_t S2 = (t2-p2)/St
    cdef DTYPE_t strike1
    cdef DTYPE_t dip1
    cdef DTYPE_t rake1
    cN_SDR(N0, N1, N2, S0, S1, S2, &strike1, &dip1, &rake1)
    return strike1, dip1, rake1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef TP_SDR(DTYPE_t[:,:]T,DTYPE_t[:,:]P):
    cdef Py_ssize_t imax=T.shape[1] 
    cdef Py_ssize_t i
    cdef DTYPE_t [::1] s=np.empty((imax,))
    cdef DTYPE_t [::1] d=np.empty((imax,))
    cdef DTYPE_t [::1] r=np.empty((imax,))
    for i in range(imax):
        s[i],d[i],r[i]=cSingleTP_SDR(T[:,i],P[:,i])
    return np.asarray(s),np.asarray(d),np.asarray(r)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void csingleSDR_SDR(DTYPE_t s,DTYPE_t d,DTYPE_t r,DTYPE_t*s2,DTYPE_t*d2,DTYPE_t*r2):
    cdef DTYPE_t ck=cos(s)
    cdef DTYPE_t cs=cos(r)
    cdef DTYPE_t sk=sin(s)
    cdef DTYPE_t ss=sin(r)
    cdef DTYPE_t h=cos(d)
    cdef DTYPE_t sh=sqrt(1-h*h)
    cdef DTYPE_t NT=sqrt((ck*cs+sk*h*ss-sk*sh)*(ck*cs+sk*h*ss-sk*sh)+(sk*cs-ck*h*ss+ck*sh)*(sk*cs-ck*h*ss+ck*sh)+(-sh*ss-h)*(-sh*ss-h))
    cdef DTYPE_t NP=sqrt((ck*cs+sk*h*ss+sk*sh)*(ck*cs+sk*h*ss+sk*sh)+(sk*cs-ck*h*ss-ck*sh)*(sk*cs-ck*h*ss-ck*sh)+(-sh*ss+h)*(-sh*ss+h))
    cdef DTYPE_t T1=(ck*cs+sk*h*ss-sk*sh)/NT
    cdef DTYPE_t T2=(sk*cs-ck*h*ss+ck*sh)/NT
    cdef DTYPE_t T3=(-sh*ss-h)/NT
    cdef DTYPE_t P1=(ck*cs+sk*h*ss+sk*sh)/NP
    cdef DTYPE_t P2=(sk*cs-ck*h*ss-ck*sh)/NP
    cdef DTYPE_t P3=(-sh*ss+h)/NP
    cdef DTYPE_t N0=(T1-P1)
    cdef DTYPE_t N1=(T2-P2)
    cdef DTYPE_t N2=(T3-P3)
    cdef DTYPE_t S0=(T1+P1)
    cdef DTYPE_t S1=(T2+P2)
    cdef DTYPE_t S2=(T3+P3)
    cdef DTYPE_t St=sqrt(S0*S0+S1*S1+S2*S2)
    cdef DTYPE_t Nt=sqrt(N0*N0+N1*N1+N2*N2)
    cN_SDR(S0/St,S1/St,S2/St,N0/Nt,N1/Nt,N2/Nt,s2,d2,r2)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef SDR_SDR(DTYPE_t[::1] s,DTYPE_t[::1] d,DTYPE_t[::1] r):
    cdef Py_ssize_t imax=s.shape[0]
    cdef Py_ssize_t i
    cdef DTYPE_t [::1] s2=np.empty((imax,))
    cdef DTYPE_t [::1] d2=np.empty((imax,))
    cdef DTYPE_t [::1] r2=np.empty((imax,))
    for i in range(imax):
        csingleSDR_SDR(s[i],d[i],r[i],&s2[i],&d2[i],&r2[i])
    return np.asarray(s2),np.asarray(d2),np.asarray(r2)

#Bi-axes
cpdef list isotropic_c(DTYPE_t l=1.,DTYPE_t mu=1.,list c=[]):
    if len(c)==21: #Calculate isotropic approacximation
            #Eqns 81a and 81b from chapman and leaney 2011
            mu=((c[0]+c[6]+c[11])+3*(c[15]+c[18]+c[20])-(c[1]+c[2]+c[7]))/15
            l=((c[0]+c[6]+c[11])-2*(c[15]+c[18]+c[20])+4*(c[1]+c[2]+c[7]))/15
    n=l+2*mu
    return [n,l,l,0,0,0,n,l,0,0,0,n,0,0,0,mu,0,0,mu,0,mu]

def MT6_biaxes(DTYPE_t[:]MT6,list c):
    lambda2mu=(3*(c[0]+c[6]+c[11])+4*(c[15]+c[18]+c[20])+2*(c[1]+c[2]+c[7]))/15
    mu=((c[0]+c[6]+c[11])+3*(c[15]+c[18]+c[20])-(c[1]+c[2]+c[7]))/15
    l=lambda2mu-2*mu
    T,N,P,E=cMT6_TNPE(MT6)
    isotropic=(l+mu)*E[1]/mu-l*(E[0]+E[2])/(2*mu)
    if is_isotropic_c(c):
        explosion=isotropic
    else:
        def isotropic_solve(iso):
            iso6=np.squeeze(iso)*np.array([[1],[1],[1],[0],[0],[0]])
            if iso6.shape!=MT6.shape:
                iso6=iso6.T
            T,N,P,E=cMT6_TNPE(MT6c_D6(np.squeeze(MT6-iso6),c).flatten())
            return E[1]
        explosion=fsolve(isotropic_solve,isotropic)

    explosion6=np.squeeze(explosion)*np.array([[1],[1],[1],[0],[0],[0]])
    if explosion6.shape!=MT6.shape:
        explosion6=explosion6.T
    T,N,P,E=cMT6_TNPE(MT6c_D6(np.squeeze(MT6-explosion6),c).flatten())   
    area_displacement = E[0]-E[2]
    phi=np.zeros((3,2))  
    if area_displacement!=0:      # to avoid undefined
        cphi=np.squeeze(np.sqrt(E[0]/area_displacement))
        sphi=np.squeeze(np.sqrt(-E[2]/area_displacement))
        phi[:,0]=np.array(cphi*T+sphi*P).flatten()
        phi[:,1]=np.array(cphi*T-sphi*P).flatten()
    return phi,explosion,area_displacement

cpdef MT6c_D6(mt6,list c=isotropic_c(l=1,mu=1)):
    mtvoigt=mt6[np.array([0,1,2,5,4,3])]
    mtvoigt=np.matrix(mtvoigt)
    if mtvoigt.shape[1]==6:
        mtvoigt=mtvoigt.T
    #Convert to voigt
    dvoigt=np.linalg.solve(np.matrix(c21_cvoigt(c)),mtvoigt)
    return np.asarray(dvoigt[np.array([0,1,2,5,4,3])])

cpdef bool is_isotropic_c(list c):
    cdef DTYPE_t tol = 1.e-6*c_norm(c);
    return ((fabs(c[ 3])<tol)and(fabs(c[ 4])<tol)and(fabs(c[ 5])<tol)and \
            (fabs(c[ 8])<tol)and(fabs(c[ 9])<tol)and(fabs(c[10])<tol)and \
            (fabs(c[12])<tol)and(fabs(c[13])<tol)and(fabs(c[14])<tol)and \
            (fabs(c[16])<tol)and(fabs(c[17])<tol)and(fabs(c[19])<tol)and \
            (fabs(c[ 0]-c[ 6])<tol)and(fabs(c[ 6]-c[11])<tol)and \
            (fabs(c[15]-c[18])<tol)and(fabs(c[18]-c[20])<tol)and \
            (fabs(c[ 1]-c[ 2])<tol)and(fabs(c[ 2]-c[ 7])<tol)and \
            (fabs(c[0]-c[1]-2*c[15])<tol))

cpdef DTYPE_t [:,::1] c21_cvoigt(list c):
    return np.array([[c[0],c[1],c[2],sqrt2*c[3],sqrt2*c[4],sqrt2*c[5]],
                     [c[1],c[6],c[7],sqrt2*c[8],sqrt2*c[9],sqrt2*c[10]],
                     [c[2],c[7],c[11],sqrt2*c[12],sqrt2*c[13],sqrt2*c[14]],
                     [sqrt2*c[3],sqrt2*c[8],sqrt2*c[12],2*c[15],2*c[16],2*c[17]],
                     [sqrt2*c[4],sqrt2*c[9],sqrt2*c[13],2*c[16],2*c[18],2*c[19]],
                     [sqrt2*c[5],sqrt2*c[10],sqrt2*c[14],2*c[17],2*c[19],2*c[20]]])

cpdef DTYPE_t c_norm(list c):
    return sqrt(c[0]**2+c[6]**2+c[11]**2+2*(c[1]**2+c[2]**2+c[7]**2)+
                 4*(c[3]**2+c[4]**2+c[5]**2+c[8]**2+c[9]**2+c[10]**2+
                    c[12]**2+c[13]**2+c[14]**2+c[15]**2+c[18]**2+c[20]**2)+
                 8*(c[16]**2+c[17]**2+c[19]**2));  

# Test functions - Not Documented

class cMomentTensorConvertTestCase(TestCase):
    def test_cTape_MT6(self):
        cdef DTYPE_t[::1] m=np.empty((6))
        cTape_MT6(&m[0],0.12,0.43,0.76,0.63,0.75)
        self.assertAlmostEqual(m[0],-0.3637,4)
        self.assertAlmostEqual(m[1],0.4209,4)
        self.assertAlmostEqual(m[2],0.6649,4)
        self.assertAlmostEqual(m[3],0.3533,4)
        self.assertAlmostEqual(m[4],-0.1952,4)
        self.assertAlmostEqual(m[5],-0.2924,4)
    def test_c_cE_tk(self):
        results=np.array([0.,0.,0.,0.,0.,0.,0.])
        E=np.array([1.,0.,-1.])
        results=cE_tk(E,results)
        self.assertAlmostEqual(results[5],0)
        self.assertAlmostEqual(results[6],0)
        E=np.array([1.,1.,1.])
        results=cE_tk(E,results)
        self.assertAlmostEqual(results[5],1.)
        self.assertAlmostEqual(results[6],0.)
        E=np.array([1.,-1.,-1.])
        results=cE_tk(E,results)
        self.assertAlmostEqual(results[5],-0.2)#y
        self.assertAlmostEqual(results[6],-0.8)#x
    def test_c_ctk_uv(self):
        results=np.array([0.,0.,0.,0.,0.,0.,0.])
        E=np.array([1.,0.,-1.])
        results=cE_tk(E,results)
        results=ctk_uv(results)
        self.assertAlmostEqual(results[5],0)
        self.assertAlmostEqual(results[6],0)
        E=np.array([1.,1.,1.])
        results=cE_tk(E,results)
        results=ctk_uv(results)
        self.assertAlmostEqual(results[5],0)
        self.assertAlmostEqual(results[6],1)
        E=np.array([1.,-1.,-1.])
        results=cE_tk(E,results)
        results=ctk_uv(results)
        self.assertAlmostEqual(results[5],-4./3)
        self.assertAlmostEqual(results[6],-1./3)
    def test_c_cE_gd(self):
        E=np.array([1.,0.,-1.])
        cdef DTYPE_t g
        cdef DTYPE_t d
        cE_gd(E,&g,&d)
        self.assertAlmostEqual(g,0)
        self.assertAlmostEqual(d,0)
        E=np.array([1.,1.,1.])
        cE_gd(E,&g,&d)
        self.assertAlmostEqual(g,0)
        self.assertAlmostEqual(d,pi/2)
        E=np.array([1.,-1.,-1.])
        cE_gd(E,&g,&d)
        self.assertAlmostEqual(g,-0.523598775598299)
        self.assertAlmostEqual(d,-0.339836909454122)
    def test_c_cN_sdr(self):
        E=np.array([1.,0.,-1.])
        cdef DTYPE_t s
        cdef DTYPE_t d
        cdef DTYPE_t r
        cN_SDR(0.,0.5/np.sqrt(1.25),1.0/np.sqrt(1.25),0.5/np.sqrt(1.25),0,1.0/np.sqrt(1.25),&s,&d,&r)
        self.assertAlmostEqual(s,pi)
        self.assertAlmostEqual(d,0.463647609000806)
        self.assertAlmostEqual(r,1.35080834939944)
        cN_SDR(0.5/np.sqrt(1.25),0,1.0/np.sqrt(1.25),0.,0.5/np.sqrt(1.25),1.0/np.sqrt(1.25),&s,&d,&r)
        self.assertAlmostEqual(s,pi/2)
        self.assertAlmostEqual(d,0.463647609000806)
        self.assertAlmostEqual(r,1.79078430419036)
        N0 = 0.70710678
        N1 = 0.11957316
        N2 = -0.69692343
        S0 = 0.70710678
        S1 = -0.11957316
        S2 = 0.69692343
        cN_SDR(N0, N1, N2, S0, S1, S2, &s, &d, &r)
        T = np.array([1., 0., 0.])
        P = np.array([0.,
                      0.16910198,
                      -0.98559856])
        # Test single TP - N conversion
        s2, d2, r2 = cSingleTP_SDR(T, P)
        self.assertEqual(s, s2)
        self.assertEqual(d, d2)
        self.assertEqual(r, r2)
        try:
            self.assertAlmostEqual(s, 279.5980303*pi/180)
            self.assertAlmostEqual(d, 45.81931182*pi/180)
            self.assertAlmostEqual(r, -76.36129238206001*pi/180)
        except Exception:
            self.assertAlmostEqual(s, 80.4019697*pi/180)
            self.assertAlmostEqual(d, 45.81931182*pi/180)
            self.assertAlmostEqual(r,-103.63870728*pi/180)

    def test_c_cTP_SDR(self):
        T=np.array([0.235702260395516,
         0.235702260395516,
         0.942809041582063])
        P=np.array([0.707106781186547,
        -0.707106781186547,
                         0])
        results=np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        results=cTP_SDR(T,P,results)
        self.assertAlmostEqual(results[2],3.6052402625906)
        self.assertAlmostEqual(results[3],0.666666666666667)
        self.assertAlmostEqual(results[4],1.10714871779409)
        try:
            self.assertAlmostEqual(results[7],206.565051177078)
            self.assertAlmostEqual(results[8],48.1896851042214)
            self.assertAlmostEqual(results[9],63.434948822922)
            self.assertAlmostEqual(results[10],63.434948822922)
            self.assertAlmostEqual(results[11],48.1896851042214)
            self.assertAlmostEqual(results[12],116.565051177078)
        except Exception:
            self.assertAlmostEqual(results[10],206.565051177078)
            self.assertAlmostEqual(results[11],48.1896851042214)
            self.assertAlmostEqual(results[12],63.434948822922)
            self.assertAlmostEqual(results[7],63.434948822922)
            self.assertAlmostEqual(results[8],48.1896851042214)
            self.assertAlmostEqual(results[9],116.565051177078)

    def test_c_cSingleTP_SDR(self):
        T = np.array([0.235702260395516,
                      0.235702260395516,
                      0.942809041582063])
        P = np.array([0.707106781186547,
                      -0.707106781186547,
                      0])
        s, d, r = cSingleTP_SDR(T, P)
        try:
            self.assertAlmostEqual(s, 3.6052402625906)
            self.assertAlmostEqual(d, 0.666666666666667)
            self.assertAlmostEqual(r, 1.10714871779409)
        except Exception:
            self.assertAlmostEqual(s, 63.434948822922*pi/180)
            self.assertAlmostEqual(d, 48.1896851042214*pi/180)
            self.assertAlmostEqual(r, 116.565051177078*pi/180)
        T = np.array([1., 0., 0.])
        P = np.array([0.,
                      0.16910198,
                      -0.98559856])
        s, d, r = cSingleTP_SDR(T, P)
        try:
            self.assertAlmostEqual(s, 279.5980303*pi/180)
            self.assertAlmostEqual(d, 45.81931182*pi/180)
            self.assertAlmostEqual(r, -76.36129238206001*pi/180)
        except Exception:
            self.assertAlmostEqual(s, 80.4019697*pi/180)
            self.assertAlmostEqual(d, 45.81931182*pi/180)
            self.assertAlmostEqual(r,-103.63870728*pi/180)

    def test_c_MT6_TNPE(self):
        MT=np.array([[1.,0.,-1.,0.,0.,0.],[0,2.0,-1.0,0.,1.0,0.]]).T
        T,N,P,E=MT6_TNPE(MT)
        self.assertAlmostEqual(E[0,0],1)
        self.assertAlmostEqual(E[1,0],0)
        self.assertAlmostEqual(E[2,0],-1)
        self.assertAlmostEqual(np.abs(T[0,0]),1)
        self.assertAlmostEqual(np.abs(T[1,0]),0)
        self.assertAlmostEqual(np.abs(T[2,0]),0)
        self.assertAlmostEqual(np.abs(N[0,0]),0)
        self.assertAlmostEqual(np.abs(N[1,0]),1)
        self.assertAlmostEqual(np.abs(N[2,0]),0)
        self.assertAlmostEqual(np.abs(P[0,0]),0)
        self.assertAlmostEqual(np.abs(P[1,0]),0)
        self.assertAlmostEqual(np.abs(P[2,0]),1)
        #Second Event
        self.assertAlmostEqual(E[0,1],2)
        self.assertAlmostEqual(E[1,1],0.366025403784439 )
        self.assertAlmostEqual(E[2,1],-1.36602540378444)
        self.assertAlmostEqual(np.abs(T[0,1]),0)
        self.assertAlmostEqual(np.abs(T[1,1]),1)
        self.assertAlmostEqual(np.abs(T[2,1]),0)
        self.assertAlmostEqual(np.abs(N[0,1]),0.888073833977115)
        self.assertAlmostEqual(np.abs(N[1,1]),0)
        self.assertAlmostEqual(np.abs(N[2,1]),0.459700843380983)
        self.assertAlmostEqual(np.abs(P[0,1]),0.459700843380983)
        self.assertAlmostEqual(np.abs(P[1,1]),0)
        self.assertAlmostEqual(np.abs(P[2,1]),0.888073833977115)
    def test_c_E_GD(self):
        E=np.array([[1.,0.,-1.],[1.,1.,1.],[1.,-1.,-1.],[-1.,-1.,-1.]]).T
        g,d=E_GD(E)
        self.assertAlmostEqual(g[0],np.array([0.]))
        self.assertAlmostEqual(d[0],np.array([0.]))
        self.assertAlmostEqual(g[1],np.array([0.]))
        self.assertAlmostEqual(d[1],np.array([pi/2]))
        self.assertAlmostEqual(g[2],np.array([-0.523598775598299]))
        self.assertAlmostEqual(d[2],np.array([-0.339836909454122]))
        self.assertAlmostEqual(g[3],np.array([0.]))
        self.assertAlmostEqual(d[3],np.array([-pi/2]))
    def test_c_TP_SDR(self):
        T=np.array([[0.235702260395516,0.],[0.235702260395516,1.0],[0.942809041582063,0.]])
        P=np.array([[0.707106781186547,1.],[-0.707106781186547,0],[0.,0.]])
        s,d,r=TP_SDR(T,P)
        self.assertAlmostEqual(s[0],np.array([1.1071487177940911]))
        self.assertAlmostEqual(d[0],np.array([0.84106867056793]))
        self.assertAlmostEqual(r[0],np.array([2.0344439357957032]))
        self.assertAlmostEqual(s[1],np.array([5.497787143782138]))
        self.assertAlmostEqual(d[1],np.array([pi/2]))
        self.assertAlmostEqual(r[1],np.array([-pi]))
    def test_c_SDR_SDR(self):
        s2,d2,r2=SDR_SDR(np.array([206.565051177078*pi/180]),np.array([48.1896851042214*pi/180]),np.array([63.434948822922*pi/180]))
        self.assertAlmostEqual(s2,63.434948822922*pi/180)
        self.assertAlmostEqual(d2,48.1896851042214*pi/180)
        self.assertAlmostEqual(r2,116.565051177078*pi/180)
        s2,d2,r2=SDR_SDR(np.array([63.434948822922*pi/180]),np.array([48.1896851042214*pi/180]),np.array([116.565051177078*pi/180]))
        self.assertAlmostEqual(s2,206.565051177078*pi/180)
        self.assertAlmostEqual(d2,48.1896851042214*pi/180)
        self.assertAlmostEqual(r2,63.434948822922*pi/180)

def test_suite(verbosity=2):
    suite = [unittest.TestLoader().loadTestsFromTestCase(cMomentTensorConvertTestCase), ]
    suite = unittest.TestSuite(suite)
    return suite