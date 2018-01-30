#!python
# cython: infer_types=True


# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.

cimport cython
cimport numpy as np
import numpy as np
import unittest
from cpython cimport bool
from libc.stdlib cimport rand, RAND_MAX
IF UNAME_SYSNAME == "Windows":
    from libc.math cimport HUGE_VAL as inf
ELSE:
    from libc.math cimport INFINITY as inf
    from libc.math cimport fmin
    from libc.math cimport erf
    from libc.math cimport tgamma
from libc.math cimport sqrt
from libc.math cimport exp
from libc.math cimport pow
from libc.math cimport fabs
from libc.math cimport fmod
from libc.math cimport cos
from libc.math cimport acos
from libc.math cimport sin
from libc.math cimport M_PI as pi
from libc.math cimport M_SQRT2 as sqrt2

from MTfit.convert.cmoment_tensor_conversion cimport cTape_MT6

ctypedef double DTYPE_t
cdef DTYPE_t PI2=2*pi  


ctypedef DTYPE_t (*jump_params_ptr)(DTYPE_t,DTYPE_t,DTYPE_t,DTYPE_t,DTYPE_t)
ctypedef DTYPE_t (*prior_ratio_ptr)(DTYPE_t,DTYPE_t,DTYPE_t,DTYPE_t) nogil
ctypedef DTYPE_t (*transition_ratio_ptr)(DTYPE_t,DTYPE_t,DTYPE_t,DTYPE_t,DTYPE_t,DTYPE_t,DTYPE_t,DTYPE_t,DTYPE_t,DTYPE_t,DTYPE_t,DTYPE_t) nogil

#Windows specific functions - in libc.math otherwise
IF UNAME_SYSNAME == "Windows":    
    cdef DTYPE_t ND=1902.9928503773808*1.10452194071529090000 #Calculated from python and MATLAB
    
    #Approx for erf function
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef inline double erf(double x) nogil:
        # constants
        cdef double a1 =  0.254829592
        cdef double a2 = -0.284496736
        cdef double a3 =  1.421413741
        cdef double a4 = -1.453152027
        cdef double a5 =  1.061405429
        cdef double p  =  0.3275911
        # Save the sign of x
        cdef int sign = 1
        if x < 0:
            sign = -1
        x = fabs(x)
        # A&S formula 7.1.26
        cdef double t = 1.0/(1.0 + p*x)
        cdef double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x)
        return sign*y
    
    #no fmin so acceptance different to *nix
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef DTYPE_t acceptance(transition_ratio_ptr transition_ratio_fn,prior_ratio_ptr prior_ratio_fn,jump_params_ptr jump_params_fn,DTYPE_t gamma,DTYPE_t delta,DTYPE_t h,DTYPE_t sigma,DTYPE_t g0,DTYPE_t g_s,DTYPE_t d0,DTYPE_t d_s,DTYPE_t h0,DTYPE_t h_s,DTYPE_t s0,DTYPE_t s_s,DTYPE_t ln_p,DTYPE_t ln_p0,int jump,DTYPE_t qg,DTYPE_t qd,DTYPE_t sg,DTYPE_t sd,DTYPE_t proposal_normalisation,DTYPE_t p_dc) :
        cdef DTYPE_t acc
        if jump>0:
            if gamma==0.0 and delta==0.0:
                #MT to DC
                model_prior_ratio=p_dc/(1-p_dc)
                acc=exp(ln_p-ln_p0)*prior_ratio_fn(gamma,delta,g0,d0)*jump_params_fn(qg,qd,sg,sd, proposal_normalisation)*model_prior_ratio
            elif g0==0.0 and d0==0.0 :
                #DC to MT
                model_prior_ratio=(1-p_dc)/p_dc
                acc=exp(ln_p-ln_p0)*prior_ratio_fn(gamma,delta,g0,d0)/jump_params_fn(qg,qd,sg,sd,proposal_normalisation)*model_prior_ratio            
        else:   
            acc=exp(ln_p-ln_p0)*transition_ratio_fn(gamma,delta,h,sigma,g0,g_s,d0,d_s,h0,h_s,s0,s_s)*prior_ratio_fn(gamma,delta,g0,d0)
        if acc<1:
            return acc
        return 1
ELSE:
    cdef DTYPE_t ND=tgamma(11.490)/(tgamma(5.745)*tgamma(5.745))*1.10452194071529090000

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef DTYPE_t acceptance(transition_ratio_ptr transition_ratio_fn,prior_ratio_ptr prior_ratio_fn,jump_params_ptr jump_params_fn,DTYPE_t gamma,DTYPE_t delta,DTYPE_t h,DTYPE_t sigma,DTYPE_t g0,DTYPE_t g_s,DTYPE_t d0,DTYPE_t d_s,DTYPE_t h0,DTYPE_t h_s,DTYPE_t s0,DTYPE_t s_s,DTYPE_t ln_p,DTYPE_t ln_p0,int jump,DTYPE_t qg,DTYPE_t qd,DTYPE_t sg,DTYPE_t sd,DTYPE_t proposal_normalisation,DTYPE_t p_dc) :
        if jump>0:
            if gamma==0.0 and delta==0.0:
                model_prior_ratio=p_dc/(1-p_dc)
                return fmin(1,exp(ln_p-ln_p0)*prior_ratio_fn(gamma,delta,g0,d0)*jump_params_fn(qg,qd,sg,sd, proposal_normalisation)*model_prior_ratio)
            elif g0==0.0 and d0==0.0 :
                model_prior_ratio=(1-p_dc)/p_dc
                return fmin(1,exp(ln_p-ln_p0)*prior_ratio_fn(gamma,delta,g0,d0)/jump_params_fn(qg,qd,sg,sd,proposal_normalisation)*model_prior_ratio)
        return fmin(1,exp(ln_p-ln_p0)*transition_ratio_fn(gamma,delta,h,sigma,g0,g_s,d0,d_s,h0,h_s,s0,s_s)*prior_ratio_fn(gamma,delta,g0,d0))

#Gaussian PDF
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t gaussian_pdf(DTYPE_t x,DTYPE_t mu,DTYPE_t s) nogil:
    return (1/(sqrt2*sqrt(pi)*s))*exp(-(x-mu)*(x-mu)/(2*s*s))

#Gaussian CDF
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t gaussian_cdf(DTYPE_t x,DTYPE_t mu,DTYPE_t s) nogil:
    return 0.5*(1+erf((x-mu)/(s*sqrt2)))

#Inline functions for Gaussian transition ratios - normalising for truncated gaussian PDFs
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t gaussian_transition_mt(DTYPE_t gamma,DTYPE_t delta,DTYPE_t h,DTYPE_t sigma,DTYPE_t g0,DTYPE_t g_s,DTYPE_t d0,DTYPE_t d_s,DTYPE_t h0,DTYPE_t h_s,DTYPE_t s0,DTYPE_t s_s) nogil:
    N=(gaussian_cdf(pi/6,g0,g_s)-gaussian_cdf(-pi/6,g0,g_s))*(gaussian_cdf(pi/2,d0,d_s)-gaussian_cdf(-pi/2,d0,d_s))
    return gaussian_transition_dc(h,sigma,h0,h_s,s0,s_s)/N

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t gaussian_transition_dc(DTYPE_t h,DTYPE_t sigma,DTYPE_t h0,DTYPE_t h_s,DTYPE_t s0,DTYPE_t s_s) nogil:
    N=(gaussian_cdf(1,h0,h_s)-gaussian_cdf(0,h0,h_s))*(gaussian_cdf(pi/2,s0,s_s)-gaussian_cdf(-pi/2,s0,s_s))
    return 1/N

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t gaussian_transition_ratio(DTYPE_t gamma,DTYPE_t delta,DTYPE_t h,DTYPE_t sigma,DTYPE_t g0,DTYPE_t g_s,DTYPE_t d0,DTYPE_t d_s,DTYPE_t h0,DTYPE_t h_s,DTYPE_t s0,DTYPE_t s_s) nogil:
    if gamma==0.0 and delta==0.0 and g0==0.0 and d0==0.0:
        return gaussian_transition_dc(h0,s0,h,h_s,sigma,s_s)/gaussian_transition_dc(h,sigma,h0,h_s,s0,s_s)
    return gaussian_transition_mt(g0,d0,h0,s0,gamma,g_s,delta,d_s,h,h_s,sigma,s_s)/(gaussian_transition_mt(gamma,delta,h,sigma,g0,g_s,d0,d_s,h0,h_s,s0,s_s))

#Prior distributions - distribution for delta from numerical tests.
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t uniform_delta_dist(DTYPE_t delta) nogil:
    #based on beta dist
    b=5.745
    d=(delta+pi/2)/pi
    return ND*pow(d*(1-d),b-1)/pi

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
#Prior ratios, including DC/MT jumps for relative inversion
cdef inline DTYPE_t uniform_prior_ratio(DTYPE_t gamma,DTYPE_t delta,DTYPE_t g0,DTYPE_t d0) nogil:
    if gamma==0.0 and delta==0.0 and g0==0.0 and d0==0.0:
        return 1.0
    elif gamma==0.0 and delta==0.0:
        return 1/(1.5*cos(3*g0)*uniform_delta_dist(d0))
    elif g0==0.0 and d0==0.0:
        return 1.5*cos(3*gamma)*uniform_delta_dist(delta)
    return cos(3*gamma)*uniform_delta_dist(delta)/(cos(3*g0)*uniform_delta_dist(d0))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t flat_prior_ratio(DTYPE_t gamma,DTYPE_t delta,DTYPE_t g0,DTYPE_t d0) nogil:
    return 1.0

# Functions for obtaining new Tape parameter samples etc.
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef DTYPE_t new_gamma(DTYPE_t g0, DTYPE_t g_s):
    cdef DTYPE_t gamma
    gamma=g_s*np.random.randn(1)+g0
    while fabs(gamma)>pi/6:
        gamma=g_s*np.random.randn(1)+g0
    return gamma

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef DTYPE_t new_delta(DTYPE_t d0, DTYPE_t d_s):
    cdef DTYPE_t delta
    delta=d_s*np.random.randn(1)+d0
    while fabs(delta)>pi/2:
        delta=d_s*np.random.randn(1)+d0
    return delta

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef DTYPE_t new_kappa(DTYPE_t k0, DTYPE_t k_s):
    cdef DTYPE_t kn=k_s*np.random.randn(1)+k0
    if fabs(kn)<PI2:
        kn=fmod(kn,PI2)
    if kn<0.0:
        kn+=PI2
    return kn

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef DTYPE_t new_h(DTYPE_t h0, DTYPE_t h_s):
    cdef DTYPE_t h
    h=h_s*np.random.randn(1)+h0
    while h>1 or h<0:
        h=h_s*np.random.randn(1)+h0
    return h

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef DTYPE_t new_sigma(DTYPE_t s0, DTYPE_t s_s):
    cdef DTYPE_t sigma
    sigma=s_s*np.random.randn(1)+s0
    while fabs(sigma)>pi/2:
        sigma=s_s*np.random.randn(1)+s0
    return sigma

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def convert_sample(DTYPE_t gamma,DTYPE_t  delta,DTYPE_t  kappa,DTYPE_t  h,DTYPE_t  sigma):
    cdef DTYPE_t[:,::1] M = np.empty((6,1))
    cdef DTYPE_t[::1] m = np.empty((6))
    cTape_MT6(&m[0],gamma,delta,kappa,h,sigma)
    M[0,0]=m[0]
    M[1,0]=m[1]
    M[2,0]=m[2]
    M[3,0]=m[3]
    M[4,0]=m[4]
    M[5,0]=m[5]
    return np.asarray(M)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t ranf() nogil:
    return <DTYPE_t>rand()/<DTYPE_t>RAND_MAX

#Acceptance check functions single and multiple events
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int c_acceptance_check(transition_ratio_ptr transition_ratio_fn,prior_ratio_ptr prior_ratio_fn,jump_params_ptr jump_params_fn,DTYPE_t[::1] gamma,DTYPE_t[::1]  delta,DTYPE_t[::1]  h,DTYPE_t[::1]  sigma,DTYPE_t[::1] kappa,DTYPE_t g0,DTYPE_t g_s,DTYPE_t d0,DTYPE_t d_s,DTYPE_t h0,DTYPE_t h_s,DTYPE_t s0,DTYPE_t s_s,DTYPE_t[::1] ln_p,DTYPE_t ln_p0,DTYPE_t k0,DTYPE_t[::1] qg,DTYPE_t[::1] qd,DTYPE_t sg,DTYPE_t sd,DTYPE_t proposal_normalisation,DTYPE_t p_dc):
    cdef Py_ssize_t umax=ln_p.shape[0]
    cdef Py_ssize_t u
    cdef int jump=0
    cdef DTYPE_t acc=0
    for u in range(umax):
        jump=0
        if ((g0==0.0 and d0==0.0) or (gamma[u]==0.0 and delta[u]==0.0)) and  not gamma[u]==g0 and not delta[u]==d0 and h0==h[u] and s0==sigma[u] and k0==kappa[u]:
            jump=1
        if ln_p[u]==-inf:
            continue
        if acceptance(transition_ratio_fn,prior_ratio_fn,jump_params_fn,gamma[u], delta[u], h[u], sigma[u], g0, g_s, d0, d_s, h0, h_s, s0, s_s, ln_p[u], ln_p0,jump,qg[u],qd[u],sg,sd,proposal_normalisation,p_dc)>np.random.rand():
            return <int>u
    return -1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int c_me_acceptance_check(transition_ratio_ptr transition_ratio_fn,prior_ratio_ptr prior_ratio_fn,jump_params_ptr jump_params_fn,DTYPE_t[:,::1] gamma,DTYPE_t[:,::1]  delta,DTYPE_t[:,::1]  h,DTYPE_t[:,::1]  sigma,DTYPE_t[:,::1] kappa,DTYPE_t[::1] g0,DTYPE_t[::1] g_s,DTYPE_t[::1] d0,DTYPE_t[::1] d_s,DTYPE_t[::1] h0,DTYPE_t[::1] h_s,DTYPE_t[::1] s0,DTYPE_t[::1] s_s,DTYPE_t[::1] ln_p,DTYPE_t ln_p0,DTYPE_t[::1] k0,DTYPE_t[:,::1] qg,DTYPE_t[:,::1] qd,DTYPE_t[::1] sg,DTYPE_t[::1] sd,DTYPE_t[::1] proposal_normalisation,DTYPE_t[::1] p_dc):
    cdef Py_ssize_t umax=ln_p.shape[0]
    cdef Py_ssize_t u
    cdef Py_ssize_t nevents=gamma.shape[1]
    cdef Py_ssize_t v
    cdef int jump=0
    cdef DTYPE_t acc=1
    for u in range(umax):
        if ln_p[u]==-inf:
            continue
        for v in range(nevents):
            jump=0
            if ((g0[v]==0.0 and d0[v]==0.0) or (gamma[u,v]==0.0 and delta[u,v]==0.0)) and  not gamma[u,v]==g0[v] and not delta[u,v]==d0[v] and h0[v]==h[u,v] and s0[v]==sigma[u,v] and k0[v]==kappa[u,v]:#Check for jump
                jump=1
            acc*=acceptance(transition_ratio_fn,prior_ratio_fn,jump_params_fn,gamma[u,v], delta[u,v], h[u,v], sigma[u,v], g0[v], g_s[v], d0[v], d_s[v], h0[v], h_s[v], s0[v], s_s[v], 0,0,jump,qg[u,v],qd[u,v],sg[v],sd[v],proposal_normalisation[v],p_dc[v])
            if acc==0:
                break
        if v<nevents-1 or acc==0.0:#No acceptance, so no point in continuing
            continue
        acc*=exp(ln_p[u]-ln_p0)
        if acc>np.random.rand():
            return <int>u
    return -1

#RJ Jump paramters
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline tuple flat_jump_params(DTYPE_t a,DTYPE_t b,):
    return  (np.random.rand()*2-1)*pi/6,acos(np.random.rand()*2-1)-pi/2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline tuple gaussian_jump_params(DTYPE_t sg,DTYPE_t sd):
    cdef DTYPE_t qg=-pi/2
    cdef DTYPE_t qd=-pi
    while fabs(qg)>pi/6:
        qg=sg*np.random.randn()
    while fabs(qd)>pi/2:
        qd=sd*np.random.randn()
    return  (qg,qd)

#RJ jump parameter priors
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t flat_jump_prob(DTYPE_t a,DTYPE_t b,DTYPE_t sg,DTYPE_t sd,DTYPE_t proposal_normalisation):
    #N.B. ILLOGICAL - DO NOT USE - WILL ALWAYS ACCEPT JUMP FROM DC to MT (gamma=0 delta=0), i.e. increasing complexity
    return  3/(PI2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t gaussian_jump_prob(DTYPE_t qg,DTYPE_t qd,DTYPE_t sg,DTYPE_t sd,DTYPE_t proposal_normalisation):
    return  gaussian_pdf(qg,0,sg)*gaussian_pdf(qd,0,sd)/proposal_normalisation

#Python accessible functions for acceptance checks
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def acceptance_check(list x,dict x0,dict alpha, ln_p, DTYPE_t ln_p0, bool uniform_prior=True, bool gaussian_jump=True, DTYPE_t dc_prior=0.5):
    cdef Py_ssize_t umax=ln_p.shape[0]
    cdef Py_ssize_t u
    cdef DTYPE_t[::1] gamma=np.empty((umax))
    cdef DTYPE_t[::1] delta=np.empty((umax))
    cdef DTYPE_t[::1] h=np.empty((umax))
    cdef DTYPE_t[::1] sigma=np.empty((umax))
    cdef DTYPE_t[::1] kappa=np.empty((umax))   
    cdef DTYPE_t[::1] qg=np.empty((umax))
    cdef DTYPE_t[::1] qd=np.empty((umax))          
    cdef transition_ratio_ptr transition_ratio_fn=&gaussian_transition_ratio
    cdef prior_ratio_ptr prior_ratio_fn    
    cdef jump_params_ptr jump_params_fn
    cdef DTYPE_t g0=<DTYPE_t>x0['gamma']
    cdef DTYPE_t g_s=<DTYPE_t>alpha['gamma']
    cdef DTYPE_t d0=<DTYPE_t>x0['delta']
    cdef DTYPE_t d_s=<DTYPE_t>alpha['delta']
    cdef DTYPE_t h0=<DTYPE_t>x0['h']
    cdef DTYPE_t h_s=<DTYPE_t>alpha['h']
    cdef DTYPE_t s0=<DTYPE_t>x0['sigma']
    cdef DTYPE_t s_s=<DTYPE_t>alpha['sigma']
    cdef DTYPE_t k0=<DTYPE_t>x0['kappa']
    cdef DTYPE_t proposal_normalisation=1.0
    if 'proposal_normalisation' in alpha.keys():
        proposal_normalisation=<DTYPE_t>alpha['proposal_normalisation']
    cdef DTYPE_t sg=0.1
    cdef DTYPE_t sd=0.1
    cdef int index
    if gaussian_jump:
        jump_params_fn=&gaussian_jump_prob
    else:
        jump_params_fn=&flat_jump_prob
    if uniform_prior:
        prior_ratio_fn=&uniform_prior_ratio
    else:
        prior_ratio_fn=&flat_prior_ratio
    if 'gamma_dc' in alpha.keys():
        sg=<DTYPE_t>alpha['gamma_dc']
    if 'delta_dc' in alpha.keys():
        sd=<DTYPE_t>alpha['delta_dc']
    for u in range(umax):
        gamma[u]=x[u]['gamma']
        delta[u]=x[u]['delta']
        kappa[u]=x[u]['kappa']
        sigma[u]=x[u]['sigma']
        h[u]=x[u]['h']
        if 'g0' in x[u].keys():
            qg[u]=x[u]['g0']#g0 from mt sample ==> dc sample
        else:
            qg[u]=x[u]['gamma'];#no g0  ==>  mt sample
        if 'd0' in x[u].keys():
            qd[u]=x[u]['d0']
        else:
            qd[u]=x[u]['delta'];
    index=c_acceptance_check(transition_ratio_fn,prior_ratio_fn,jump_params_fn, gamma, delta, h, sigma,kappa, g0, g_s, d0, d_s, h0, h_s, s0, s_s,ln_p,ln_p0,k0,qg,qd,sg,sd,proposal_normalisation,dc_prior)
    if index>=0:
        return x[index],ln_p[index],index
    return {},False,umax

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def me_acceptance_check(list x,list x0,list alpha, DTYPE_t[::1] ln_p,DTYPE_t ln_p0,bool uniform_prior=True,bool gaussian_jump=True,DTYPE_t[::1] dc_prior=np.array([0.5])):
    cdef Py_ssize_t umax=ln_p.shape[0]
    cdef Py_ssize_t nevents=len(alpha)
    cdef Py_ssize_t u
    cdef Py_ssize_t v
    cdef DTYPE_t[:,::1] gamma=np.empty((umax,nevents))
    cdef DTYPE_t[:,::1] delta=np.empty((umax,nevents))
    cdef DTYPE_t[:,::1] h=np.empty((umax,nevents))
    cdef DTYPE_t[:,::1] sigma=np.empty((umax,nevents))
    cdef DTYPE_t[:,::1] kappa=np.empty((umax,nevents))   
    cdef DTYPE_t[:,::1] qg=np.empty((umax,nevents))
    cdef DTYPE_t[:,::1] qd=np.empty((umax,nevents))          
    cdef transition_ratio_ptr transition_ratio_fn=&gaussian_transition_ratio
    cdef prior_ratio_ptr prior_ratio_fn    
    cdef jump_params_ptr jump_params_fn
    cdef DTYPE_t [::1]g0=np.empty((nevents))
    cdef DTYPE_t [::1]g_s=np.empty((nevents))
    cdef DTYPE_t [::1]d0=np.empty((nevents))
    cdef DTYPE_t [::1]d_s=np.empty((nevents))
    cdef DTYPE_t [::1]h0=np.empty((nevents))
    cdef DTYPE_t [::1]h_s=np.empty((nevents))
    cdef DTYPE_t [::1]s0=np.empty((nevents))
    cdef DTYPE_t [::1]s_s=np.empty((nevents))
    cdef DTYPE_t [::1]k0=np.empty((nevents))
    cdef DTYPE_t [::1]sg=np.empty((nevents))
    cdef DTYPE_t [::1]sd=np.empty((nevents))
    cdef DTYPE_t [::1]proposal_normalisation=np.ones((nevents))
    cdef DTYPE_t [::1]prior_dc=np.ones((nevents))
    if dc_prior.shape[0]==0:
        prior_dc=np.array([0.5])
    cdef int index
    if gaussian_jump:
        jump_params_fn=&gaussian_jump_prob
    else:
        jump_params_fn=&flat_jump_prob
    if uniform_prior:
        prior_ratio_fn=&uniform_prior_ratio
    else:
        prior_ratio_fn=&flat_prior_ratio
    for v in range(nevents):
        g0[v]=<DTYPE_t>x0[v]['gamma']
        g_s[v]=<DTYPE_t>alpha[v]['gamma']
        d0[v]=<DTYPE_t>x0[v]['delta']
        d_s[v]=<DTYPE_t>alpha[v]['delta']
        h0[v]=<DTYPE_t>x0[v]['h']
        h_s[v]=<DTYPE_t>alpha[v]['h']
        s0[v]=<DTYPE_t>x0[v]['sigma']
        s_s[v]=<DTYPE_t>alpha[v]['sigma']
        k0[v]=<DTYPE_t>x0[v]['kappa']
        if 'proposal_normalisation' in alpha[v].keys():
            proposal_normalisation[v]=<DTYPE_t>alpha[v]['proposal_normalisation']
        if 'gamma_dc' in alpha[v].keys():
            sg[v]=<DTYPE_t>alpha[v]['gamma_dc']
        if 'delta_dc' in alpha[v].keys():
            sd[v]=<DTYPE_t>alpha[v]['delta_dc']
        if dc_prior.shape[0]==1:
            prior_dc[v]=dc_prior[0]
        elif dc_prior.shape[0]>1:
            prior_dc[v]=dc_prior[v]
        for u in range(umax):
            gamma[u,v]=x[v][u]['gamma']
            delta[u,v]=x[v][u]['delta']
            kappa[u,v]=x[v][u]['kappa']
            sigma[u,v]=x[v][u]['sigma']
            h[u,v]=x[v][u]['h']
            if 'g0' in x[v][u].keys():
                qg[u,v]=x[v][u]['g0']#g0 from mt sample ==> dc sample
            else:
                qg[u,v]=x[v][u]['gamma'];#no g0  ==>  mt sample
            if 'd0' in x[v][u].keys():
                qd[u,v]=x[v][u]['d0']
            else:
                qd[u,v]=x[v][u]['delta'];
    index=c_me_acceptance_check(transition_ratio_fn,prior_ratio_fn,jump_params_fn, gamma, delta, h, sigma,kappa, g0, g_s, d0, d_s, h0, h_s, s0, s_s,ln_p,ln_p0,k0,qg,qd,sg,sd,proposal_normalisation,prior_dc)
    returnx=[]
    for v in range(nevents):
        if index>=0:
            returnx.append(x[v][index])
        else:
            returnx.append({})
    if index>=0:
        return returnx,ln_p[index],index
    return returnx,False,umax

#Python function for new sample
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def new_samples(dict x0,dict alpha,Py_ssize_t number_samples, bool dc=False,DTYPE_t jump=0.0,bool gaussian_jump=True):
    cdef Py_ssize_t u
    cdef list x=[]
    cdef DTYPE_t zero=0.0
    cdef dict _x={'gamma':zero,'delta':zero,'kappa':zero,'h':zero,'sigma':zero} 
    cdef DTYPE_t[:,::1] M=np.empty((6,number_samples))
    for u in range(number_samples):
        _x={'gamma':zero,'delta':zero,'kappa':zero,'h':zero,'sigma':zero} 
        if jump and np.random.rand()<jump:
            _x['kappa']=x0['kappa']
            _x['h']=x0['h']
            _x['sigma']=x0['sigma']
            if x0['gamma']==0.0 and x0['delta']==0.0:
                if gaussian_jump:
                    _x['gamma'],_x['delta']=gaussian_jump_params(alpha['gamma_dc'],alpha['delta_dc'])
                else:
                    _x['gamma'],_x['delta']=flat_jump_params(0,0)
            else:
                _x['gamma']=0.0
                _x['delta']=0.0
                _x['g0']=x0['gamma']
                _x['d0']=x0['delta']
        else:
            if dc or x0['gamma']==0.0 and x0['delta']==0.0:
                _x['gamma']=0.0
                _x['delta']=0.0
            else:
                _x['gamma']=new_gamma(<DTYPE_t>x0['gamma'],<DTYPE_t>alpha['gamma'])
                _x['delta']=new_delta(<DTYPE_t>x0['delta'],<DTYPE_t>alpha['delta'])
            _x['kappa']=new_kappa(<DTYPE_t>x0['kappa'],<DTYPE_t>alpha['kappa'])
            _x['h']=new_h(<DTYPE_t>x0['h'],<DTYPE_t>alpha['h'])
            _x['sigma']=new_sigma(<DTYPE_t>x0['sigma'],<DTYPE_t>alpha['sigma'])
        x.append(_x.copy())
        m=convert_sample( _x['gamma'], _x['delta'], _x['kappa'], _x['h'], _x['sigma'])
        M[0,u]=m[0,0]
        M[1,u]=m[1,0]
        M[2,u]=m[2,0]
        M[3,u]=m[3,0]
        M[4,u]=m[4,0]
        M[5,u]=m[5,0]
    return x,np.asarray(M)

# Test functions - Not Documented


class _cmarkov_chain_monte_carlo_TestCase(unittest.TestCase):
    def test_transition_ratios(self):
        self.assertAlmostEqual(gaussian_transition_ratio(0.2,0.1,0.3,0.2,0.2,0.5,0.05,0.7,0.3,0.2,0.1,0.1),1.0011,4)
        self.assertAlmostEqual(gaussian_transition_ratio(0.2,0.05,0.3,0.1, 0.2,0.5, 0.1,0.7, 0.3,0.2, 0.2,0.1),0.9989,4)
        self.assertAlmostEqual(gaussian_transition_ratio(0.1,-0.5,0.2,0.6, 0.05,0.1,0.02,0.1, 0.6,0.2,0.1,0.1),1.1600,4)
    def test_prior_ratios(self):
        self.assertEqual(flat_prior_ratio(0,0,0,0),1.0)
        self.assertEqual(flat_prior_ratio(0,21,0,0),1.0)
        self.assertEqual(flat_prior_ratio(0,0,0,12),1.0)
        self.assertEqual(flat_prior_ratio(0,0,2231,0),1.0)
        self.assertEqual(uniform_prior_ratio(0,0,0,0),1.0)
        self.assertAlmostEqual(uniform_prior_ratio(0,0,0.2,0.2),0.9382,3)
        self.assertAlmostEqual(uniform_prior_ratio(0.2,0.2,0,0),1.0659,3)
        self.assertAlmostEqual(uniform_prior_ratio(0,0,0.2,0.5),1.44127,3)
        self.assertAlmostEqual(uniform_prior_ratio(0.3,0.1,0.2,0.5),1.226,3)
    def test_erf(self):
        import sys
        if 'win' in sys.platform:
            self.assertAlmostEqual(erf(2),0.9953,4)
            self.assertAlmostEqual(erf(0.01),0.0113,4)
            self.assertAlmostEqual(erf(-0.01),-0.0113,4)
            self.assertAlmostEqual(erf(5),1.0000,4)
    def test_convert_sample(self):
        M=convert_sample(0.0,0.0,0.5,0.5,0.0)
        self.assertTrue(M.flags['C_CONTIGUOUS'])
        self.assertEqual(M.shape[0],6)
        self.assertEqual(M.shape[1],1)
    def test_c_acceptance_check(self):
        x=[{'gamma':0.1,'delta':0.05,'h':0.4,'kappa':0.2,'sigma':-0.01},{'gamma':0.05,'delta':0.01,'h':0.45,'kappa':0.4,'sigma':-0.01}]
        ln_p=np.array([0.,50.])
        cdef Py_ssize_t umax=ln_p.shape[0]
        cdef Py_ssize_t u
        cdef DTYPE_t[::1] gamma=np.empty((umax))
        cdef DTYPE_t[::1] delta=np.empty((umax))
        cdef DTYPE_t[::1] h=np.empty((umax))
        cdef DTYPE_t[::1] sigma=np.empty((umax))
        cdef DTYPE_t[::1] kappa=np.empty((umax))   
        cdef DTYPE_t[::1] qg=np.empty((umax))
        cdef DTYPE_t[::1] qd=np.empty((umax))          
        cdef transition_ratio_ptr transition_ratio_fn=&gaussian_transition_ratio
        cdef prior_ratio_ptr prior_ratio_fn=&uniform_prior_ratio  
        cdef jump_params_ptr jump_params_fn=&gaussian_jump_prob
        cdef DTYPE_t g0=0.1
        cdef DTYPE_t g_s=0.1
        cdef DTYPE_t d0=0.05
        cdef DTYPE_t d_s=0.1
        cdef DTYPE_t h0=0.5
        cdef DTYPE_t h_s=0.1
        cdef DTYPE_t s0=0
        cdef DTYPE_t s_s=0.1
        cdef DTYPE_t k0=0.3
        cdef DTYPE_t qg0=0.
        cdef DTYPE_t qd0=0.
        cdef DTYPE_t sg=0.1
        cdef DTYPE_t sd=0.1
        cdef int index
        for u in range(umax):
            gamma[u]=x[u]['gamma']
            delta[u]=x[u]['delta']
            kappa[u]=x[u]['kappa']
            sigma[u]=x[u]['sigma']
            h[u]=x[u]['h']
            qg[u]=0.0;
            qd[u]=0.0;

        index=c_acceptance_check(transition_ratio_fn,prior_ratio_fn,jump_params_fn, gamma, delta, h, sigma,kappa, g0, g_s, d0, d_s, h0, h_s, s0, s_s,ln_p,15.,k0,qg,qd,sg,sd,1,0.5)
        self.assertEqual(index,1)
    def test_gaussian_jump_params(self):
        [g,d]=gaussian_jump_params(0.01,0.01)
        self.assertTrue(fabs(g)<pi/6)
        self.assertTrue(fabs(d)<pi/2)
    def test_flat_jump_params(self):
        [g,d]=flat_jump_params(0.01,0.01)
        self.assertTrue(fabs(g)<pi/6)
        self.assertTrue(fabs(d)<pi/2)
    def test_flat_jump_prob(self):
        p=flat_jump_prob(0,0,1,1,1)
        self.assertEqual(p,3/(PI2))
        p=flat_jump_prob(3,1,2,4,1)
        self.assertEqual(p,3/(PI2))
    def test_gaussian_jump_prob(self):
        p=gaussian_jump_prob(0,0,1,1,1)
        self.assertEqual(p,gaussian_pdf(0,0,1)*gaussian_pdf(0,0,1))
        p=gaussian_jump_prob(0.2,0.2,0.2,0.2,2)
        self.assertEqual(p,gaussian_pdf(0.2,0,0.2)*gaussian_pdf(0.2,0,0.2)/2)
    def test_new_samples(self):
        x0={'h': 0.16037522711021507, 'kappa': 1.7045926666235598, 'sigma': 1.2202710645123613, 'gamma': 0.0, 'delta': 0.0}
        alpha={'gamma_dc': 0.10471975511965977, 'sigma_dc': 0.009085556327826677, 'kappa': 0.018171112655653354, 'delta_dc': 0.15707963267948966,
         'h': 0.005784044801253858, 'h_dc': 0.005784044801253858, 'poisson': 0.005784044801253858, 'delta': 0.009085556327826677,
          'alpha': 0.009085556327826677, 'kappa_dc': 0.018171112655653354, 'sigma': 0.009085556327826677, 'gamma': 0.00605703755188445,'proposal_normalisation':1.0}
        x,M=new_samples(x0,alpha,2)
        self.assertEqual(len(x),2)
        self.assertEqual(M.shape[0],6)
        self.assertEqual(M.shape[1],2)
        self.assertTrue(M.flags['C_CONTIGUOUS'])
        x,M=new_samples(x0,alpha,1)
        self.assertEqual(len(x),1)
        self.assertEqual(M.shape[0],6)
        self.assertEqual(M.shape[1],1)
        self.assertTrue(M.flags['C_CONTIGUOUS'])
        x,M=new_samples(x0,alpha,1,True,1)
        self.assertEqual(len(x),1)
        self.assertEqual(M.shape[0],6)
        self.assertEqual(M.shape[1],1)
        self.assertTrue('gamma' in x[0].keys())
        self.assertFalse('g0' in x[0].keys())
        x0={'h': 0.16037522711021507, 'kappa': 1.7045926666235598, 'sigma': 1.2202710645123613, 'gamma': 0.5, 'delta': 0.5}
        x,M=new_samples(x0,alpha,1,False,1)
        self.assertTrue('gamma' in x[0].keys())
        self.assertTrue('g0' in x[0].keys())
        self.assertTrue(M.flags['C_CONTIGUOUS'])
    def test_acceptance_check(self):
        x0={'h': 0.16037522711021507, 'kappa': 1.7045926666235598, 'sigma': 1.2202710645123613, 'gamma': 0.0, 'delta': 0.0}
        x1={'h': 0.16037522711021507, 'kappa': 1.7045926666235598, 'sigma': 1.2202710645123613, 'gamma': .3, 'delta': 0.5}
        alpha={'gamma_dc': 0.10471975511965977, 'sigma_dc': 0.009085556327826677, 'kappa': 0.018171112655653354, 'delta_dc': 0.15707963267948966,
         'h': 0.005784044801253858, 'h_dc': 0.005784044801253858, 'poisson': 0.005784044801253858, 'delta': 0.009085556327826677,
          'alpha': 0.009085556327826677, 'kappa_dc': 0.018171112655653354, 'sigma': 0.009085556327826677, 'gamma': 0.00605703755188445,'proposal_normalisation':1.0}
        ln_p=np.array([5.0,3.0])
        ln_p0=3.0
        x=[x0,x1]
        xi,ln_pi,i=acceptance_check(x,x0,alpha,ln_p,ln_p0,True,False)
        self.assertEqual(i,0) 
        self.assertEqual(x0,xi)
        ln_p=np.array([-np.inf,3.0])
        ln_p0=-np.inf
        x=[x0,x1]
        xi,ln_pi,i=acceptance_check(x,x0,alpha,ln_p,ln_p0,True,False)
        self.assertEqual(i,1)
        self.assertEqual(x1,xi)
        #Tests for gaussian jump
        x0={'h': 0.16037522711021507, 'kappa': 1.7045926666235598, 'sigma': 1.2202710645123613, 'gamma': 0.0, 'delta': 0.0}
        x1={'h': 0.16037522711021507, 'kappa': 1.7045926666235598, 'sigma': 1.2202710645123613, 'gamma': .1, 'delta': 0.1}
        alpha={'gamma_dc': 0.1, 'sigma_dc': 0.009085556327826677, 'kappa': 0.018171112655653354, 'delta_dc': 0.1,
         'h': 0.005784044801253858, 'h_dc': 0.005784044801253858, 'poisson': 0.005784044801253858, 'delta': 0.009085556327826677,
          'alpha': 0.009085556327826677, 'kappa_dc': 0.018171112655653354, 'sigma': 0.009085556327826677, 'gamma': 0.00605703755188445,'proposal_normalisation':0.99501231590631156}
        ln_p=np.array([0.])
        ln_p0=np.log(np.array([0.231]))
        #dc to mt
        xi,ln_pi,i=acceptance_check([x1],x0,alpha,ln_p,ln_p0,True,True)
        self.assertTrue(len(xi)>0)
        ln_p=np.array([-10000000000000000000000.])
        ln_p0=np.array([100000000000000000000000000000000000000000000])
        #dc to mt
        xi,ln_pi,i=acceptance_check([x1],x0,alpha,ln_p,ln_p0,True,True)
        self.assertEqual(len(xi),0)
        ln_p=np.log(np.array([0.2321]))
        ln_p0=np.array([0.])
        #mt to dc
        ln_p=np.array([0.2329])
        xi,ln_pi,i=acceptance_check([x0],x1,alpha,ln_p,ln_p0,True,True)
        self.assertTrue(len(xi)>0)
        ln_p=np.array([-10000000000000000000000.])
        ln_p0=np.array([100000000000000000000000000000000000000000000.])
        xi,ln_pi,i=acceptance_check([x0],x1,alpha,ln_p,ln_p0,True,True)
        self.assertEqual(len(xi),0)
    def test_c_me_acceptance_check(self):
        x=[{'gamma':0.1,'delta':0.05,'h':0.4,'kappa':0.2,'sigma':-0.01},{'gamma':0.05,'delta':0.01,'h':0.45,'kappa':0.4,'sigma':-0.01}]
        ln_p=np.array([0.,10000])
        cdef Py_ssize_t umax=ln_p.shape[0]
        cdef Py_ssize_t nevents=2
        cdef Py_ssize_t u
        cdef Py_ssize_t v
        cdef DTYPE_t[:,::1] gamma=np.empty((umax,nevents))
        cdef DTYPE_t[:,::1] delta=np.empty((umax,nevents))
        cdef DTYPE_t[:,::1] h=np.empty((umax,nevents))
        cdef DTYPE_t[:,::1] sigma=np.empty((umax,nevents))
        cdef DTYPE_t[:,::1] kappa=np.empty((umax,nevents))   
        cdef DTYPE_t[:,::1] qg=np.empty((umax,nevents))
        cdef DTYPE_t[:,::1] qd=np.empty((umax,nevents))          
        cdef transition_ratio_ptr transition_ratio_fn=&gaussian_transition_ratio
        cdef prior_ratio_ptr prior_ratio_fn=&uniform_prior_ratio  
        cdef jump_params_ptr jump_params_fn=&gaussian_jump_prob
        cdef DTYPE_t [::1]g0=np.empty((nevents))#0.1
        cdef DTYPE_t [::1]g_s=np.empty((nevents))#0.1
        cdef DTYPE_t [::1]d0=np.empty((nevents))#0.05
        cdef DTYPE_t [::1]d_s=np.empty((nevents))#0.1
        cdef DTYPE_t [::1]h0=np.empty((nevents))#0.5
        cdef DTYPE_t [::1]h_s=np.empty((nevents))#0.1
        cdef DTYPE_t [::1]s0=np.empty((nevents))#0
        cdef DTYPE_t [::1]s_s=np.empty((nevents))#0.1
        cdef DTYPE_t [::1]k0=np.empty((nevents))#0.3
        cdef DTYPE_t [::1]qg0=np.empty((nevents))#0.
        cdef DTYPE_t [::1]qd0=np.empty((nevents))#0.
        cdef DTYPE_t [::1]sg=np.empty((nevents))#0.1
        cdef DTYPE_t [::1]sd=np.empty((nevents))#0.1
        cdef DTYPE_t [::1]pn=np.empty((nevents))#0.1
        cdef DTYPE_t [::1]dc_prior=np.empty((nevents))#0.1
        cdef int index
        for v in range(nevents):
            g0[v]=0.1
            g_s[v]=0.1
            d0[v]=0.05
            d_s[v]=0.1
            h0[v]=0.5
            h_s[v]=0.1
            s0[v]=0.
            s_s[v]=0.1
            k0[v]=0.3
            qg0[v]=0.
            qd0[v]=0.
            sg[v]=0.1
            sd[v]=0.1
            pn[v]=1
            dc_prior[v]=0.5
            for u in range(umax):
                gamma[u,v]=x[u]['gamma']
                delta[u,v]=x[u]['delta']
                kappa[u,v]=x[u]['kappa']
                sigma[u,v]=x[u]['sigma']
                h[u,v]=x[u]['h']
                qg[u,v]=0.0;
                qd[u,v]=0.0;
        index=c_me_acceptance_check(transition_ratio_fn,prior_ratio_fn,jump_params_fn, gamma, delta, h, sigma,kappa, g0, g_s, d0, d_s, h0, h_s, s0, s_s,ln_p,-200.,k0,qg,qd,sg,sd,pn,dc_prior)
        self.assertEqual(index,0)
        index=c_me_acceptance_check(transition_ratio_fn,prior_ratio_fn,jump_params_fn, gamma, delta, h, sigma,kappa, g0, g_s, d0, d_s, h0, h_s, s0, s_s,ln_p,100,k0,qg,qd,sg,sd,pn,dc_prior)
        self.assertEqual(index,1)

    def test_me_acceptance_check(self):
        x0={'h': 0.16037522711021507, 'kappa': 1.7045926666235598, 'sigma': 1.2202710645123613, 'gamma': 0.0, 'delta': 0.0}
        x1={'h': 0.16037522711021507, 'kappa': 1.7045926666235598, 'sigma': 1.2202710645123613, 'gamma': .3, 'delta': 0.5}
        alpha={'gamma_dc': 0.10471975511965977, 'sigma_dc': 0.009085556327826677, 'kappa': 0.018171112655653354, 'delta_dc': 0.15707963267948966,
         'h': 0.005784044801253858, 'h_dc': 0.005784044801253858, 'poisson': 0.005784044801253858, 'delta': 0.009085556327826677,
          'alpha': 0.009085556327826677, 'kappa_dc': 0.018171112655653354, 'sigma': 0.009085556327826677, 'gamma': 0.00605703755188445,'proposal_normalisation':1.0}
        ln_p=np.array([5.0,3.0])
        ln_p0=-30
        x=[x0,x1]
        xi,ln_pi,i=me_acceptance_check([x,x,x],[x0,x0,x0],[alpha,alpha,alpha],ln_p,ln_p0,True,False)
        self.assertEqual(i,0) 
        self.assertEqual([x0,x0,x0],xi)
        ln_p=np.array([-np.inf,3.0])
        ln_p0=-np.inf
        x=[x0,x1]
        xi,ln_pi,i=me_acceptance_check([x],[x0],[alpha],ln_p,ln_p0,True,False)
        self.assertEqual(i,1)
        self.assertEqual([x1],xi)
        #Tests for gaussian jump
        x0={'h': 0.16037522711021507, 'kappa': 1.7045926666235598, 'sigma': 1.2202710645123613, 'gamma': 0.0, 'delta': 0.0}
        x1={'h': 0.16037522711021507, 'kappa': 1.7045926666235598, 'sigma': 1.2202710645123613, 'gamma': .1, 'delta': 0.1}
        alpha={'gamma_dc': 0.1, 'sigma_dc': 0.009085556327826677, 'kappa': 0.018171112655653354, 'delta_dc': 0.1,
         'h': 0.005784044801253858, 'h_dc': 0.005784044801253858, 'poisson': 0.005784044801253858, 'delta': 0.009085556327826677,
          'alpha': 0.009085556327826677, 'kappa_dc': 0.018171112655653354, 'sigma': 0.009085556327826677, 'gamma': 0.00605703755188445,'proposal_normalisation':0.99501231590631156}
        ln_p=np.array([0.])
        ln_p0=np.log(np.array([0.231]))
        #dc to mt
        xi,ln_pi,i=me_acceptance_check([[x1,x1]],[x0],[alpha],ln_p,ln_p0,True,True)
        self.assertTrue(len(xi)>0)
        ln_p=np.array([-10000000000000000000000.])
        ln_p0=np.array([100000000000000000000000000000000000000000000])
        #dc to mt
        xi,ln_pi,i=me_acceptance_check([[x1]],[x0],[alpha],ln_p,ln_p0,True,True)
        self.assertEqual(len(xi[0]),0)
        ln_p=np.log(np.array([0.2321]))
        ln_p0=np.array([0.])
        #mt to dc
        ln_p=np.array([0.2329])
        xi,ln_pi,i=me_acceptance_check([[x0]],[x1],[alpha],ln_p,ln_p0,True,True)
        self.assertTrue(len(xi[0])>0)
        ln_p=np.array([-10000000000000000000000.])
        ln_p0=np.array([100000000000000000000000000000000000000000000.])
        xi,ln_pi,i=me_acceptance_check([[x0]],[x1],[alpha],ln_p,ln_p0,True,True)
        self.assertEqual(len(xi[0]),0)


def _acceptance_test_fn(dict x, dict x0, dict alpha, DTYPE_t ln_p,DTYPE_t ln_p0,bool uniform_prior=True,bool gaussian_jump=True,DTYPE_t p_dc=0.5):
    cdef DTYPE_t qg
    cdef DTYPE_t qd
    cdef transition_ratio_ptr transition_ratio_fn=&gaussian_transition_ratio
    cdef prior_ratio_ptr prior_ratio_fn    
    cdef jump_params_ptr jump_params_fn
    cdef DTYPE_t proposal_normalisation=1.0
    if 'proposal_normalisation' in alpha.keys():
        proposal_normalisation=<DTYPE_t>alpha['proposal_normalisation']
    cdef DTYPE_t sg=0.1
    cdef DTYPE_t sd=0.1
    cdef int index
    if gaussian_jump:
        jump_params_fn=&gaussian_jump_prob
    else:
        jump_params_fn=&flat_jump_prob
    if uniform_prior:
        prior_ratio_fn=&uniform_prior_ratio
    else:
        prior_ratio_fn=&flat_prior_ratio
    if 'gamma_dc' in alpha.keys():
        sg=<DTYPE_t>alpha['gamma_dc']
    if 'delta_dc' in alpha.keys():
        sd=<DTYPE_t>alpha['delta_dc']
    if 'g0' in x.keys():
        qg=x['g0']#g0 from mt sample ==> dc sample
    else:
        qg=x['gamma'];#no g0  ==>  mt sample
    if 'd0' in x.keys():
        qd=x['d0']
    else:
        qd=x['delta'];
    cdef int jump=0
    if ((x0['gamma']==0.0 and x0['delta']==0.0) or (x['gamma']==0.0 and x['delta']==0.0)) and  not x['gamma']==x0['gamma'] and not x['delta']==x0['delta'] and x0['h']==x['h'] and x0['sigma']==x['sigma'] and x0['kappa']==x['kappa']:
        jump=1
    if ln_p==-inf:
        return 0
    return acceptance(transition_ratio_fn,prior_ratio_fn,jump_params_fn,x['gamma'], x['delta'], x['h'], x['sigma'], x0['gamma'], alpha['gamma'], x0['delta'], alpha['delta'], x0['h'], alpha['h'], x0['sigma'], alpha['sigma'], ln_p, ln_p0,jump,qg,qd,sg,sd,proposal_normalisation,p_dc)

def _gaussian_cdf_test(DTYPE_t x,DTYPE_t mu,DTYPE_t s):
    return gaussian_cdf(x,mu,s)

def _gaussian_transition_ratio_test(dict x, dict x0,dict alpha):
    return gaussian_transition_ratio(x['gamma'], x['delta'], x['h'], x['sigma'], x0['gamma'], alpha['gamma'], x0['delta'], alpha['delta'], x0['h'], alpha['h'], x0['sigma'], alpha['sigma']) 

def _test_suite():
    _c_markov_chain_monte_carlo_test_suite = unittest.TestLoader().loadTestsFromTestCase(_cmarkov_chain_monte_carlo_TestCase)
    return unittest.TestSuite([_c_markov_chain_monte_carlo_test_suite])

def run_tests():
    """Runs monteCarloMarkovChain module tests.
    """
    suite=_test_suite()
    unittest.TextTestRunner(verbosity=4).run(suite)

if  __name__=='__main__':
    run_tests()