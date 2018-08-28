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
import unittest
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
cdef DTYPE_t RAND_MAX_D=<DTYPE_t> RAND_MAX

cdef DTYPE_t cutoff=0.0 #Ratio PDF cutoff to approximate as Gaussian - improve calculation speed at the expense of some accuracy
#corresponds to the Percentage error in the denominator as this governs the deviation of the ratio PDF from the Gaussian
# 0.1 seems good by eye, little deviation over a range of percentage unc in x

#cdef bool bc=True

IF UNAME_SYSNAME == "Windows":

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef ln_marginalise(DTYPE_t[:,:] ln_p):
        #cdefs
        cdef Py_ssize_t vmax=ln_p.shape[0]
        cdef Py_ssize_t wmax=ln_p.shape[1]
        cdef DTYPE_t[:] Ln_P=np.empty((wmax)) 
        cdef Py_ssize_t v,w,
        cdef DTYPE_t scale,p
        if vmax==1:
            Ln_P=ln_p[0,:]
        else:
            for w from 0<=w<wmax:
                p=0
                scale=-inf
                for v from 0<=v<vmax:
                    if ln_p[v,w]>scale:
                        scale=ln_p[v,w]
                if scale>0:
                    scale=0
                if scale==-inf:
                    scale=0
                for v from 0<=v<vmax:
                    p+=exp(ln_p[v,w]+fabs(scale))
                Ln_P[w]=log(p)-fabs(scale)
        return np.asarray(Ln_P)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef ln_normalise(DTYPE_t[::1] ln_p, DTYPE_t dV=1):
        cdef Py_ssize_t wmax=ln_p.shape[0]
        cdef Py_ssize_t w
        cdef DTYPE_t scale=-inf, N=0
        for w from 0<=w<wmax:            
            if scale<ln_p[w]:
                scale=ln_p[w]
        if scale>0:
            scale=0
        for w from 0<=w<wmax:
            N+=exp(ln_p[w]+fabs(scale))*dV
        N=log(N)-fabs(scale)
        for w from 0<=w<wmax:
            ln_p[w]-=N
        return np.asarray(ln_p)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef inline double fmax(double x,double y) nogil:
        if x>=y:
            return x
        else:
            return y 
ELSE:

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef ln_marginalise(DTYPE_t[:,:] ln_p):
        #cdefs
        cdef Py_ssize_t vmax=ln_p.shape[0]
        cdef Py_ssize_t wmax=ln_p.shape[1]
        cdef DTYPE_t[:] Ln_P=np.empty((wmax)) 
        cdef Py_ssize_t v,w,
        cdef DTYPE_t scale,p
        if vmax==1:
            Ln_P=ln_p[0,:]
        else:
            for w from 0<=w<wmax:
                p=0
                scale=-inf
                for v from 0<=v<vmax:
                    scale=fmax(scale,ln_p[v,w])
                scale=fmin(scale,0)
                if scale==-inf:
                    scale=0
                for v from 0<=v<vmax:
                    p+=exp(ln_p[v,w]+fabs(scale))
                Ln_P[w]=log(p)-fabs(scale)
        return np.asarray(Ln_P)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef ln_normalise(DTYPE_t[::1] ln_p, DTYPE_t dV=1):
        cdef Py_ssize_t wmax=ln_p.shape[0]
        cdef Py_ssize_t w
        cdef DTYPE_t scale=-inf, N=0
        for w from 0<=w<wmax:
            scale=fmax(scale,ln_p[w])
        scale=fmin(scale,0)
        for w from 0<=w<wmax:
            N+=exp(ln_p[w]+fabs(scale))*dV
        N=log(N)-fabs(scale)
        for w from 0<=w<wmax:
            ln_p[w]-=N
        return np.asarray(ln_p)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t erf_inv_approx(DTYPE_t t) nogil:
    # // Abramowitz and Stegun formula 26.2.23.
    # // The absolute value of the error should be less than 4.5 e-4.
    return t - ((0.010328*t + 0.802853)*t + 2.515517)/(((0.001308*t + 0.189269)*t + 1.432788)*t + 1.0)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t  erf_inv(DTYPE_t p) nogil:
    if (p < 0.5):
        return -erf_inv_approx( sqrt(-2.0*log(p)) )
    else:
        return erf_inv_approx( sqrt(-2.0*log(1-p)) )

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t gaussian_pdf(DTYPE_t x,DTYPE_t mu,DTYPE_t s) nogil:
    return (1/(sqrt2*sqrt(pi)*s))*exp(-(x-mu)*(x-mu)/(2*s*s))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t gaussian_cdf(DTYPE_t x,DTYPE_t mu,DTYPE_t s) nogil:
    return 0.5*(1+erf((x-mu)/(s*sqrt2)))

#
# Basic PDFs
#

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t pol_pdf(DTYPE_t x,DTYPE_t s,DTYPE_t i) nogil:
    # print x,s,i,0.5*((1-i)*(1+erf(x/(sqrt2*s)))+i*(1+erf(-x/(sqrt2*s))))
    return 0.5*((1-i)*(1+erf(x/(sqrt2*s)))+i*(1+erf(-x/(sqrt2*s))))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t pol_prob_pdf(DTYPE_t x,DTYPE_t p,DTYPE_t n,DTYPE_t i) nogil:
    if x>0:
       return (1-i)*p+i*n
    elif x==0:
        return 0.5
    else:
        return (1-i)*n+i*p

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t ar_pdf(DTYPE_t z,DTYPE_t mux,DTYPE_t muy,DTYPE_t psx,DTYPE_t psy) nogil:
    cdef double sqrtpi=sqrt(pi)
    cdef DTYPE_t a,b1,b2,c,d1,d2,f1,f2,sx,sy
    if psy<cutoff:
        a=fabs(mux/muy)
        sx=a*sqrt((psx*psx)+(psy*psy))
        return gaussian_pdf(z,a,sx)+gaussian_pdf(-z,a,sx)
    else:
        sx=psx*fabs(mux)
        sy=psy*fabs(muy)
        a=sqrt(z*z/(sx*sx)+1/(sy*sy))
        b1=mux*z/(sx*sx)+muy/(sy*sy)
        b2=-mux*z/(sx*sx)+muy/(sy*sy)
        c=mux*mux/(sx*sx)+muy*muy/(sy*sy)
        d1=exp((b1*b1-c*a*a)/(2*a*a))
        f1=0.5*(1+erf(b1/(sqrt2*a)))
        d2=exp((b2*b2-c*a*a)/(2*a*a))
        f2=0.5*(1+erf(b2/(sqrt2*a)))
        return b1*d1/(sx*sy*a*a*a*sqrt2*sqrtpi)*(2*f1-1)+2/(pi*sx*sy*a*a)*exp(-c/2)+b2*d2/(sx*sy*a*a*a*sqrt2*sqrtpi)*(2*f2-1)

#
# Station Loops
#

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void station_polarity_ln_pdf(DTYPE_t*a,DTYPE_t*mt,DTYPE_t*ln_P,DTYPE_t*sigma,DTYPE_t*incorrect_polarity_prob,Py_ssize_t ipmax,Py_ssize_t v,Py_ssize_t umax,Py_ssize_t vmax,Py_ssize_t kmax,Py_ssize_t wmax,Py_ssize_t w,Py_ssize_t index) nogil:
    cdef DTYPE_t x=0.0
    # print '===========',ipmax,umax
    for u from 0<=u<umax:
        x=0.
        for k from 0<=k<kmax:# loop over num mt samples and make x
            x+=a[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
            # print a[u*vmax*kmax+v*kmax+k],mt[k*wmax+w],a[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
        # print 'x',x
        #First index is station, second is locaton sample, last is MT sample
        if ipmax==1:
            ln_P[index]+=log(pol_pdf(x,sigma[u],incorrect_polarity_prob[0]))
        else:
            ln_P[index]+=log(pol_pdf(x,sigma[u],incorrect_polarity_prob[u]))
        # print 'P',ln_P[index]

        if ln_P[index]==-inf:
            return 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void station_polarity_probability_ln_pdf(DTYPE_t*a,DTYPE_t*mt,DTYPE_t*ln_P,DTYPE_t*positive_probability,DTYPE_t*negative_probability,DTYPE_t*incorrect_polarity_prob,Py_ssize_t ipmax,Py_ssize_t v,Py_ssize_t umax,Py_ssize_t vmax,Py_ssize_t kmax,Py_ssize_t wmax,Py_ssize_t w,Py_ssize_t index) nogil:
    cdef DTYPE_t x=0.0
    for u from 0<=u<umax:
        x=0
        for k from 0<=k<kmax:# loop over num mt samples and make x
            x+=a[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]               #First index is station, second is locaton sample, last is MT sample
        if ipmax==1:
            ln_P[index]+=log(pol_prob_pdf(x,positive_probability[u],negative_probability[u],incorrect_polarity_prob[0]))
        else:
            ln_P[index]+=log(pol_prob_pdf(x,positive_probability[u],negative_probability[u],incorrect_polarity_prob[u]))
        if ln_P[index]==-inf:
            return 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void station_ar_ln_pdf(DTYPE_t*ax,DTYPE_t*ay,DTYPE_t*mt,DTYPE_t*ln_P,DTYPE_t*z,DTYPE_t*psx,DTYPE_t*psy,Py_ssize_t v,Py_ssize_t umax,Py_ssize_t vmax,Py_ssize_t kmax,Py_ssize_t wmax,Py_ssize_t w,Py_ssize_t index) nogil:
    cdef DTYPE_t mux=0.0
    cdef DTYPE_t muy=0.0
    # print '---------'
    for u from 0<=u<umax:
        mux=0
        muy=0
        for k from 0<=k<kmax:# loop over num mt samples and make x
            mux+=ax[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
            muy+=ay[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
        #First index is station, second is locaton sample, last is MT sample
        ln_P[index]+=log(ar_pdf(z[u],mux,muy,psx[u],psy[u]))
        # print ln_P[index]
        if ln_P[index]==-inf:
            return 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void station_combined_polarity_ar_ln_pdf(DTYPE_t*a,DTYPE_t*ax,DTYPE_t*ay,DTYPE_t*mt,DTYPE_t*ln_P,DTYPE_t*z,DTYPE_t*sigma,DTYPE_t*incorrect_polarity_prob,DTYPE_t*psx,DTYPE_t*psy,Py_ssize_t ipmax,Py_ssize_t v,Py_ssize_t umax,Py_ssize_t uarmax,Py_ssize_t vmax,Py_ssize_t kmax,Py_ssize_t wmax,Py_ssize_t w,Py_ssize_t index) nogil:
    cdef DTYPE_t x=0.0
    cdef DTYPE_t mux=0.0
    cdef DTYPE_t muy=0.0
    cdef Py_ssize_t utmin=umax
    cdef Py_ssize_t utmax=uarmax
    if umax>uarmax:
        utmax=umax
        utmin=uarmax
    # print '----------'
        # print 'P',ln_P[index]
    for u from 0<=u<utmax:
        if u<utmin:#both
            x=0.
            mux=0.
            muy=0.
            for k from 0<=k<kmax:# loop over num mt samples and make x
                x+=a[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                mux+=ax[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                muy+=ay[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
            if ipmax==1:
                ln_P[index]+=log(pol_pdf(x,sigma[u],incorrect_polarity_prob[0]))+log(ar_pdf(z[u],mux,muy,psx[u],psy[u]))
            else:
                ln_P[index]+=log(pol_pdf(x,sigma[u],incorrect_polarity_prob[u]))+log(ar_pdf(z[u],mux,muy,psx[u],psy[u]))
            # print 'b',x,mux,muy,sigma[u],z[u],psx[u],psy[u],'=',ln_P[index]
        elif u>=umax:#Only AR
            mux=0.
            muy=0.
            for k from 0<=k<kmax:# loop over num mt samples and make x
                mux+=ax[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                muy+=ay[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
            if ipmax==1:
                ln_P[index]+=log(ar_pdf(z[u],mux,muy,psx[u],psy[u]))
            else:
                ln_P[index]+=log(ar_pdf(z[u],mux,muy,psx[u],psy[u]))
            # print 'ar',mux,muy,z[u],psx[u],psy[u],'=',ln_P[index]
        else: #only Pol
            x=0.
            for k from 0<=k<kmax:# loop over num mt samples and make x
                x+=a[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
            if ipmax==1:
                ln_P[index]+=log(pol_pdf(x,sigma[u],incorrect_polarity_prob[0]))
            else:
                ln_P[index]+=log(pol_pdf(x,sigma[u],incorrect_polarity_prob[u]))
            # print 'p',x,sigma[u],'=',ln_P[index]
        # print 'P',ln_P[index]
        if ln_P[index]==-inf:
            return 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void station_combined_polarity_probability_ar_ln_pdf(DTYPE_t*a,DTYPE_t*ax,DTYPE_t*ay,DTYPE_t*mt,DTYPE_t*ln_P,DTYPE_t*z,DTYPE_t*positive_probability,DTYPE_t*negative_probability,DTYPE_t*incorrect_polarity_prob,DTYPE_t*psx,DTYPE_t*psy,Py_ssize_t ipmax,Py_ssize_t v,Py_ssize_t umax,Py_ssize_t uarmax,Py_ssize_t vmax,Py_ssize_t kmax,Py_ssize_t wmax,Py_ssize_t w,Py_ssize_t index) nogil:
    cdef DTYPE_t mux=0.0
    cdef DTYPE_t muy=0.0
    cdef DTYPE_t x=0.0    
    cdef Py_ssize_t utmin=umax
    cdef Py_ssize_t utmax=uarmax
    if umax>uarmax:
        utmax=umax
        utmin=uarmax
    for u from 0<=u<utmax:
        if u<utmin:#both
            x=0
            mux=0
            muy=0
            for k from 0<=k<kmax:# loop over num mt samples and make x
                x+=a[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                mux+=ax[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                muy+=ay[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
            #First index is station, second is locaton sample, last is MT sample
            if ipmax==1:
                ln_P[index]+=log(pol_prob_pdf(x,positive_probability[u],negative_probability[u],incorrect_polarity_prob[0]))+log(ar_pdf(z[u],mux,muy,psx[u],psy[u]))
            else:
                ln_P[index]+=log(pol_prob_pdf(x,positive_probability[u],negative_probability[u],incorrect_polarity_prob[u]))+log(ar_pdf(z[u],mux,muy,psx[u],psy[u]))
        elif u>=umax:#Only AR
            mux=0.
            muy=0.
            for k from 0<=k<kmax:# loop over num mt samples and make x
                mux+=ax[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                muy+=ay[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
            if ipmax==1:
                ln_P[index]+=log(ar_pdf(z[u],mux,muy,psx[u],psy[u]))
            else:
                ln_P[index]+=log(ar_pdf(z[u],mux,muy,psx[u],psy[u]))   
        else: #only Polprob
            x=0.
            for k from 0<=k<kmax:# loop over num mt samples and make x
                x+=a[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
            if ipmax==1:
                ln_P[index]+=log(pol_prob_pdf(x,positive_probability[u],negative_probability[u],incorrect_polarity_prob[0]))
            else:
                ln_P[index]+=log(pol_prob_pdf(x,positive_probability[u],negative_probability[u],incorrect_polarity_prob[u]))   
        if ln_P[index]==-inf:
            return 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void station_combined_all_ln_pdf(DTYPE_t*a,DTYPE_t*a_prob,DTYPE_t*ax,DTYPE_t*ay,DTYPE_t*mt,DTYPE_t*ln_P,DTYPE_t*z,DTYPE_t*positive_probability,DTYPE_t*negative_probability,DTYPE_t*incorrect_polarity_prob,DTYPE_t*psx,DTYPE_t*psy,DTYPE_t*sigma,Py_ssize_t ipmax,Py_ssize_t v,Py_ssize_t umax,Py_ssize_t uarmax,Py_ssize_t uprobmax,Py_ssize_t vmax,Py_ssize_t kmax,Py_ssize_t wmax,Py_ssize_t w,Py_ssize_t index) nogil:
    cdef DTYPE_t mux=0.0
    cdef DTYPE_t muy=0.0
    cdef DTYPE_t x=0.0 
    cdef DTYPE_t x1=0.0     
    cdef Py_ssize_t utmin=umax
    cdef Py_ssize_t utmax=uarmax
    if umax>uarmax:
        utmax=umax
        utmin=uarmax
    if uprobmax<utmin:
        utmin=uprobmax
    if uprobmax>utmax:
        utmax=uprobmax
    for u from 0<=u<utmax:
        if u<utmin:#all
            x=0
            x1=0
            mux=0
            muy=0
            for k from 0<=k<kmax:# loop over num mt samples and make x
                x+=a[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                x1+=a_prob[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                mux+=ax[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                muy+=ay[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
            #First index is station, second is locaton sample, last is MT sample
            if ipmax==1:
                ln_P[index]+=log(pol_pdf(x,sigma[u],incorrect_polarity_prob[0]))+log(pol_prob_pdf(x1,positive_probability[u],negative_probability[u],incorrect_polarity_prob[0]))+log(ar_pdf(z[u],mux,muy,psx[u],psy[u]))
            else:
                ln_P[index]+=log(pol_pdf(x,sigma[u],incorrect_polarity_prob[u]))+log(pol_prob_pdf(x1,positive_probability[u],negative_probability[u],incorrect_polarity_prob[u]))+log(ar_pdf(z[u],mux,muy,psx[u],psy[u]))
        elif u>=umax:#AR +pol?
            if u>=uprobmax:#AR Only
                mux=0.
                muy=0.
                for k from 0<=k<kmax:# loop over num mt samples and make x
                    mux+=ax[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                    muy+=ay[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                if ipmax==1:
                    ln_P[index]+=log(ar_pdf(z[u],mux,muy,psx[u],psy[u]))
                else:
                    ln_P[index]+=log(ar_pdf(z[u],mux,muy,psx[u],psy[u]))   
            else:#AR +pol prob
                x1=0
                mux=0
                muy=0
                for k from 0<=k<kmax:# loop over num mt samples and make x
                    x1+=a_prob[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                    mux+=ax[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                    muy+=ay[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                #First index is station, second is locaton sample, last is MT sample
                if ipmax==1:
                    ln_P[index]+=log(pol_prob_pdf(x1,positive_probability[u],negative_probability[u],incorrect_polarity_prob[0]))+log(ar_pdf(z[u],mux,muy,psx[u],psy[u]))
                else:
                    ln_P[index]+=log(pol_prob_pdf(x1,positive_probability[u],negative_probability[u],incorrect_polarity_prob[u]))+log(ar_pdf(z[u],mux,muy,psx[u],psy[u]))
        elif u>=uprobmax:#AR +pol?
            if u>=umax:#AR Only
                mux=0.
                muy=0.
                for k from 0<=k<kmax:# loop over num mt samples and make x
                    mux+=ax[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                    muy+=ay[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                if ipmax==1:
                    ln_P[index]+=log(ar_pdf(z[u],mux,muy,psx[u],psy[u]))
                else:
                    ln_P[index]+=log(ar_pdf(z[u],mux,muy,psx[u],psy[u]))   
            else:#AR +pol prob
                x1=0
                mux=0
                muy=0
                for k from 0<=k<kmax:# loop over num mt samples and make x
                    x+=a[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                    mux+=ax[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                    muy+=ay[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                #First index is station, second is locaton sample, last is MT sample
                if ipmax==1:
                    ln_P[index]+=log(pol_pdf(x,sigma[u],incorrect_polarity_prob[0]))+log(ar_pdf(z[u],mux,muy,psx[u],psy[u]))
                else:
                    ln_P[index]+=log(pol_pdf(x,sigma[u],incorrect_polarity_prob[u]))+log(ar_pdf(z[u],mux,muy,psx[u],psy[u]))
        else:#Pol prob +pol?
            if u>=uprobmax:#Pol Only
                x=0.
                for k from 0<=k<kmax:# loop over num mt samples and make x
                    x+=a[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                if ipmax==1:
                    ln_P[index]+=log(pol_pdf(x,sigma[u],incorrect_polarity_prob[0]))
                else:
                    ln_P[index]+=log(pol_pdf(x,sigma[u],incorrect_polarity_prob[u]))  
            else:#Pol +pol prob
                x1=0.
                x=0.
                for k from 0<=k<kmax:# loop over num mt samples and make x
                    x+=a[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                    x1+=a_prob[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                #First index is station, second is locaton sample, last is MT sample
                if ipmax==1:
                    ln_P[index]+=log(pol_prob_pdf(x1,positive_probability[u],negative_probability[u],incorrect_polarity_prob[0]))+log(pol_pdf(x,sigma[u],incorrect_polarity_prob[0]))
                else:
                    ln_P[index]+=log(pol_prob_pdf(x1,positive_probability[u],negative_probability[u],incorrect_polarity_prob[u]))+log(pol_pdf(x,sigma[u],incorrect_polarity_prob[u]))
        if ln_P[index]==-inf:
            return 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void station_combined_pol_ln_pdf(DTYPE_t*a,DTYPE_t*a_prob,DTYPE_t*mt,DTYPE_t*ln_P,DTYPE_t*positive_probability,DTYPE_t*negative_probability,DTYPE_t*incorrect_polarity_prob,DTYPE_t*sigma,Py_ssize_t ipmax,Py_ssize_t v,Py_ssize_t umax,Py_ssize_t uprobmax,Py_ssize_t vmax,Py_ssize_t kmax,Py_ssize_t wmax,Py_ssize_t w,Py_ssize_t index) nogil:
    cdef DTYPE_t mux=0.0
    cdef DTYPE_t muy=0.0
    cdef DTYPE_t x=0.0 
    cdef DTYPE_t x1=0.0     
    cdef Py_ssize_t utmin=umax
    cdef Py_ssize_t utmax=uprobmax
    if umax>uprobmax:
        utmax=umax
        utmin=uprobmax
        utmax=uprobmax
    for u from 0<=u<utmax:
        if u<utmin:#all
            x=0
            x1=0
            mux=0
            muy=0
            for k from 0<=k<kmax:# loop over num mt samples and make x
                x+=a[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
                x1+=a_prob[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
            #First index is station, second is locaton sample, last is MT sample
            if ipmax==1:
                ln_P[index]+=log(pol_pdf(x,sigma[u],incorrect_polarity_prob[0]))+log(pol_prob_pdf(x1,positive_probability[u],negative_probability[u],incorrect_polarity_prob[0]))
            else:
                ln_P[index]+=log(pol_pdf(x,sigma[u],incorrect_polarity_prob[u]))+log(pol_prob_pdf(x1,positive_probability[u],negative_probability[u],incorrect_polarity_prob[u]))
        elif u>=umax:#Pol prob
            x1=0
            for k from 0<=k<kmax:# loop over num mt samples and make x
                x1+=a_prob[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
            #First index is station, second is locaton sample, last is MT sample
            if ipmax==1:
                ln_P[index]+=log(pol_prob_pdf(x1,positive_probability[u],negative_probability[u],incorrect_polarity_prob[0]))
            else:
                ln_P[index]+=log(pol_prob_pdf(x1,positive_probability[u],negative_probability[u],incorrect_polarity_prob[u]))
        else:#Pol 
            x=0.
            for k from 0<=k<kmax:# loop over num mt samples and make x
                x+=a[u*vmax*kmax+v*kmax+k]*mt[k*wmax+w]
            if ipmax==1:
                ln_P[index]+=log(pol_pdf(x,sigma[u],incorrect_polarity_prob[0]))
            else:
                ln_P[index]+=log(pol_pdf(x,sigma[u],incorrect_polarity_prob[u]))  
        if ln_P[index]==-inf:
            return 

#
# Cython ln PDF loops
#

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void c_polarity_ln_pdf(DTYPE_t*ln_P,DTYPE_t[:,:,::1] a_arr, DTYPE_t [:,::1] mt,DTYPE_t[::1] sigma_arr,DTYPE_t[::1] incorrect_polarity_prob_arr,int marginalised,DTYPE_t*ln_P_loc_samples,DTYPE_t*location_samples_multiplier) nogil:
    """Calculates probability of a positive polarity

    Calculates the probability of a positive polarity observation given a theoretical amplitude X and a fractional uncertainty sigma. Handles zero uncertainty.
    The derivation of this pdf for the polarity observation can be seen in ###########. 

    Cython handling for heavy lifting.
    Returns product across stations
    """
    #cdefs
    cdef Py_ssize_t umax=a_arr.shape[0]#Station Sample
    cdef Py_ssize_t vmax=a_arr.shape[1]#Location Sample
    cdef Py_ssize_t wmax=mt.shape[1]#MT samples
    cdef Py_ssize_t kmax=a_arr.shape[2]#MT elementssamples
    cdef Py_ssize_t ipmax=incorrect_polarity_prob_arr.shape[0]#MT elementssamples
    cdef DTYPE_t[:,:,::1]a=a_arr
    cdef DTYPE_t[::1]sigma=sigma_arr
    cdef DTYPE_t[::1]incorrect_polarity_prob=incorrect_polarity_prob_arr
    cdef DTYPE_t max_ln_p_loc=-inf
    cdef Py_ssize_t u,v,w,k
    cdef DTYPE_t x=0.0
    for w from 0<=w<wmax:
        if marginalised>0:
            max_ln_p_loc=-inf
            #loc_samples_multiplier
            for v from 0<=v<vmax:
                ln_P_loc_samples[v]=location_samples_multiplier[v]
                station_polarity_ln_pdf(&a[0,0,0],&mt[0,0],&ln_P_loc_samples[0],&sigma[0],&incorrect_polarity_prob[0],ipmax,v,umax,vmax,kmax,wmax,w,v)
                max_ln_p_loc=fmax(max_ln_p_loc,ln_P_loc_samples[v])
            if max_ln_p_loc>-inf:
                ln_P[w]=0.0
                for v from 0<=v<vmax:
                    ln_P[w]+=exp(ln_P_loc_samples[v]-max_ln_p_loc)
                ln_P[w]=log(ln_P[w])+max_ln_p_loc
            else:
                ln_P[w]=-inf
        else:
            for v from 0<=v<vmax:
                ln_P[v*wmax+w]=location_samples_multiplier[v]
                station_polarity_ln_pdf(&a[0,0,0],&mt[0,0],&ln_P[00],&sigma[0],&incorrect_polarity_prob[0], ipmax, v, umax,vmax,kmax,wmax, w, v*wmax+w)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void c_polarity_probability_ln_pdf(DTYPE_t*ln_P,DTYPE_t[:,:,::1]  a_arr, DTYPE_t [:,::1] mt,DTYPE_t[::1]  positive_probability_arr,DTYPE_t[::1]  negative_probability_arr,DTYPE_t[::1]  incorrect_polarity_prob_arr,int marginalised,DTYPE_t*ln_P_loc_samples,DTYPE_t*location_samples_multiplier) nogil:
    """Calculates probability of a given amplitude giving an observed polarity probability

    Calculates the probability of a given polarity probability observation given a theoretical amplitude X and a fractional uncertainty sigma. Handles zero uncertainty.
    The derivation of this pdf for the polarity observation can be seen in ###########. 

    Cython handling for heavy lifting.

    """
    #cdefs
    cdef Py_ssize_t umax=a_arr.shape[0]#Station Sample
    cdef Py_ssize_t vmax=a_arr.shape[1]#Location Sample
    cdef Py_ssize_t wmax=mt.shape[1]#MT samples
    cdef Py_ssize_t kmax=a_arr.shape[2]#MT elementssamples
    cdef Py_ssize_t ipmax=incorrect_polarity_prob_arr.shape[0]#MT elementssamples
    cdef DTYPE_t[:,:,::1]a=a_arr
    cdef DTYPE_t[::1]positive_probability=positive_probability_arr
    cdef DTYPE_t[::1]negative_probability=negative_probability_arr
    cdef DTYPE_t[::1]incorrect_polarity_prob=incorrect_polarity_prob_arr
    cdef Py_ssize_t u,v,w,k
    cdef DTYPE_t max_ln_p_loc=-inf
    ## log(location_samples_multiplier) is  ln_P_loc_samples initialisation
    for w from 0<=w<wmax:
        if marginalised>0:
            max_ln_p_loc=-inf
            #loc_samples_multiplier
            for v from 0<=v<vmax:
                ln_P_loc_samples[v]=location_samples_multiplier[v]
                station_polarity_probability_ln_pdf(&a[0,0,0],&mt[0,0],&ln_P_loc_samples[0],&positive_probability[0],&negative_probability[0],&incorrect_polarity_prob[0],ipmax,v,umax,vmax,kmax,wmax,w,v)
                max_ln_p_loc=fmax(max_ln_p_loc,ln_P_loc_samples[v])
            if max_ln_p_loc>-inf:
                ln_P[w]=0.0
                for v from 0<=v<vmax:
                    ln_P[w]+=exp(ln_P_loc_samples[v]-max_ln_p_loc)
                ln_P[w]=log(ln_P[w])+max_ln_p_loc
            else:
                ln_P[w]=-inf
        else:
            for v from 0<=v<vmax:
                ln_P[v*wmax+w]=location_samples_multiplier[v]
                station_polarity_probability_ln_pdf(&a[0,0,0],&mt[0,0],&ln_P[0],&positive_probability[0],&negative_probability[0],&incorrect_polarity_prob[0],ipmax,v,umax,vmax,kmax,wmax,w,v*wmax+w)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void c_amplitude_ratio_ln_pdf(DTYPE_t*ln_P,DTYPE_t[::1]  z_arr, DTYPE_t [:,::1] mt,DTYPE_t[:,:,::1]  ax_arr,DTYPE_t[:,:,::1]  ay_arr,DTYPE_t[::1] psx_arr,DTYPE_t[::1] psy_arr,int marginalised,DTYPE_t*ln_P_loc_samples,DTYPE_t*location_samples_multiplier) nogil:
    """Calculates Amplitude Ratio Probability

    Calculates the Ratio pdf (D. Hinkley, On the ratio of two correlated normal random variables, 1969, Biometrika vol 56 pp 635-639).
    Given Z=X/Y and means mux, muy and standard deviation sigmax and sigmay. The pdf is normalised.

    Cython handling for heavy lifting.

    """
    #cdefs    
    cdef Py_ssize_t umax=ax_arr.shape[0]#Station Sample
    cdef Py_ssize_t vmax=ax_arr.shape[1]#Location Sample
    cdef Py_ssize_t wmax=mt.shape[1]#MT samples
    cdef Py_ssize_t kmax=ax_arr.shape[2]#MT elementssamples
    cdef DTYPE_t[:,:,::1]ax=ax_arr
    cdef DTYPE_t[:,:,::1]ay=ay_arr
    cdef DTYPE_t[::1]z=z_arr
    cdef DTYPE_t[::1]psx=psx_arr
    cdef DTYPE_t[::1]psy=psy_arr
    cdef Py_ssize_t u,v,w,k
    cdef DTYPE_t max_ln_p_loc=-inf
    for w from 0<=w<wmax:
        if marginalised>0:
            max_ln_p_loc=-inf
            #loc_samples_multiplier
            for v from 0<=v<vmax:
                ln_P_loc_samples[v]=location_samples_multiplier[v]
                station_ar_ln_pdf(&ax[0,0,0],&ay[0,0,0],&mt[0,0],&ln_P_loc_samples[0],&z[0],&psx[0],&psy[0],v,umax,vmax,kmax,wmax,w,v)
                max_ln_p_loc=fmax(max_ln_p_loc,ln_P_loc_samples[v])
            if max_ln_p_loc>-inf:
                ln_P[w]=0.0
                for v from 0<=v<vmax:
                    ln_P[w]+=exp(ln_P_loc_samples[v]-max_ln_p_loc)
                ln_P[w]=log(ln_P[w])+max_ln_p_loc
            else:
                ln_P[w]=-inf
        else:
            for v from 0<=v<vmax:
                ln_P[v*wmax+w]=location_samples_multiplier[v]
                station_ar_ln_pdf(&ax[0,0,0],&ay[0,0,0],&mt[0,0],&ln_P[0],&z[0],&psx[0],&psy[0],  v, umax,vmax,kmax,wmax, w, v*wmax+w)

#
# Cython Combined ln PDF loops 
#

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void c_polarity_prob_combined_ln_pdf(DTYPE_t* ln_P,DTYPE_t[:,:,::1] a_arr, DTYPE_t [:,::1] mt,DTYPE_t[::1]  positive_probability_arr,DTYPE_t[::1]  negative_probability_arr,DTYPE_t[::1]  incorrect_polarity_prob_arr,DTYPE_t[::1]  z_arr,DTYPE_t[:,:,::1]  ax_arr,DTYPE_t[:,:,::1]  ay_arr,DTYPE_t[::1] psx_arr,DTYPE_t[::1] psy_arr,int marginalised,DTYPE_t*ln_P_loc_samples,DTYPE_t*location_samples_multiplier) nogil:
    #cdefs
    cdef Py_ssize_t umax=a_arr.shape[0]#Station Sample
    cdef Py_ssize_t uarmax=ax_arr.shape[0]#Station Sample AR
    cdef Py_ssize_t vmax=a_arr.shape[1]#Location Sample
    cdef Py_ssize_t wmax=mt.shape[1]#MT samples
    cdef Py_ssize_t kmax=a_arr.shape[2]#MT elementssamples
    cdef Py_ssize_t ipmax=incorrect_polarity_prob_arr.shape[0]
    cdef DTYPE_t[:,:,::1]a=a_arr
    cdef DTYPE_t[:,:,::1]ax=ax_arr
    cdef DTYPE_t[:,:,::1]ay=ay_arr
    cdef DTYPE_t[::1]z=z_arr
    cdef DTYPE_t[::1]positive_probability=positive_probability_arr
    cdef DTYPE_t[::1]negative_probability=negative_probability_arr
    cdef DTYPE_t[::1]psx=psx_arr
    cdef DTYPE_t[::1]psy=psy_arr
    cdef DTYPE_t[::1]incorrect_polarity_prob=incorrect_polarity_prob_arr
    cdef Py_ssize_t u,v,w,k
    cdef DTYPE_t max_ln_p_loc=-inf
    for w from 0<=w<wmax:
        if marginalised>0:
            max_ln_p_loc=-inf
            #loc_samples_multiplier
            for v from 0<=v<vmax:
                ln_P_loc_samples[v]=location_samples_multiplier[v]
                station_combined_polarity_probability_ar_ln_pdf(&a[0,0,0],&ax[0,0,0],&ay[0,0,0],&mt[0,0],&ln_P_loc_samples[0],&z[0],&positive_probability[0],&negative_probability[0],&incorrect_polarity_prob[0],&psx[0],&psy[0], ipmax, v, umax,uarmax,vmax,kmax,wmax, w, v) 
                max_ln_p_loc=fmax(max_ln_p_loc,ln_P_loc_samples[v])
            if max_ln_p_loc>-inf:
                ln_P[w]=0.0
                for v from 0<=v<vmax:
                    ln_P[w]+=exp(ln_P_loc_samples[v]-max_ln_p_loc)
                ln_P[w]=log(ln_P[w])+max_ln_p_loc
            else:
                ln_P[w]=-inf
        else:
            for v from 0<=v<vmax:
                ln_P[v*wmax+w]=location_samples_multiplier[v]
                station_combined_polarity_probability_ar_ln_pdf(&a[0,0,0],&ax[0,0,0],&ay[0,0,0],&mt[0,0],&ln_P[0],&z[0],&positive_probability[0],&negative_probability[0],&incorrect_polarity_prob[0],&psx[0],&psy[0], ipmax, v, umax,uarmax,vmax,kmax,wmax, w, v*wmax+w) 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void c_polarity_ar_ln_pdf(DTYPE_t*ln_P,DTYPE_t[:,:,::1] a_arr,  DTYPE_t [:,::1]mt,DTYPE_t[::1] sigma_arr,DTYPE_t[::1] incorrect_polarity_prob_arr,DTYPE_t[::1]  z_arr, DTYPE_t[:,:,::1]  ax_arr,DTYPE_t[:,:,::1]  ay_arr,DTYPE_t[::1] psx_arr,DTYPE_t[::1] psy_arr,int marginalised,DTYPE_t*ln_P_loc_samples,DTYPE_t*location_samples_multiplier) nogil:
    cdef Py_ssize_t umax=a_arr.shape[0]#Station Sample
    cdef Py_ssize_t uarmax=ax_arr.shape[0]#Station Sample AR
    cdef Py_ssize_t vmax=a_arr.shape[1]#Location Sample
    cdef Py_ssize_t kmax=a_arr.shape[2]#MT elementssamples
    cdef Py_ssize_t wmax=mt.shape[1]#MT samples
    cdef Py_ssize_t ipmax=incorrect_polarity_prob_arr.shape[0]
    cdef Py_ssize_t u,v,w,k
    cdef DTYPE_t[:,:,::1]a=a_arr
    cdef DTYPE_t[:,:,::1]ax=ax_arr
    cdef DTYPE_t[:,:,::1]ay=ay_arr
    cdef DTYPE_t[::1]z=z_arr
    cdef DTYPE_t[::1]sigma=sigma_arr
    cdef DTYPE_t[::1]psx=psx_arr
    cdef DTYPE_t[::1]psy=psy_arr
    cdef DTYPE_t[::1]incorrect_polarity_prob=incorrect_polarity_prob_arr
    cdef DTYPE_t max_ln_p_loc=-inf
    # print umax,uarmax
    for w from 0<=w<wmax:
        # if vmax==1:
        #     ln_P[0*wmax+w]=location_samples_multiplier[0]
        #     station_combined_polarity_ar_ln_pdf(&a[0,0,0],&ax[0,0,0],&ay[0,0,0],&mt[0,0],&ln_P[0],&z[0],&sigma[0],&incorrect_polarity_prob[0],&psx[0],&psy[0], ipmax, v, umax,uarmax,vmax,kmax,wmax, w, v*wmax+w)
        # el
        if marginalised>0:
            max_ln_p_loc=-inf
            #loc_samples_multiplier
            for v from 0<=v<vmax:
                ln_P_loc_samples[v]=location_samples_multiplier[v]
                station_combined_polarity_ar_ln_pdf(&a[0,0,0],&ax[0,0,0],&ay[0,0,0],&mt[0,0],&ln_P_loc_samples[0],&z[0],&sigma[0],&incorrect_polarity_prob[0],&psx[0],&psy[0], ipmax, v, umax,uarmax,vmax,kmax,wmax, w, v)
                max_ln_p_loc=fmax(max_ln_p_loc,ln_P_loc_samples[v])
                # print max_ln_p_loc
            if max_ln_p_loc>-inf:
                ln_P[w]=0.0
                for v from 0<=v<vmax:
                    ln_P[w]+=exp(ln_P_loc_samples[v]-max_ln_p_loc)
                ln_P[w]=log(ln_P[w])+max_ln_p_loc
            else:
                ln_P[w]=-inf
        else:
            for v from 0<=v<vmax:
                ln_P[v*wmax+w]=location_samples_multiplier[v]
                station_combined_polarity_ar_ln_pdf(&a[0,0,0],&ax[0,0,0],&ay[0,0,0],&mt[0,0],&ln_P[0],&z[0],&sigma[0],&incorrect_polarity_prob[0],&psx[0],&psy[0], ipmax, v, umax,uarmax,vmax,kmax,wmax, w, v*wmax+w)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void c_all_combined_ln_pdf(DTYPE_t* ln_P,DTYPE_t[:,:,::1] a_arr, DTYPE_t [:,::1] mt,DTYPE_t[::1] sigma_arr,DTYPE_t[:,:,::1] a_prob_arr,DTYPE_t[::1]  positive_probability_arr,DTYPE_t[::1]  negative_probability_arr,DTYPE_t[::1]  incorrect_polarity_prob_arr,DTYPE_t[::1]  z_arr,DTYPE_t[:,:,::1]  ax_arr,DTYPE_t[:,:,::1]  ay_arr,DTYPE_t[::1] psx_arr,DTYPE_t[::1] psy_arr,int marginalised,DTYPE_t*ln_P_loc_samples,DTYPE_t*location_samples_multiplier) nogil:
    #cdefs
    cdef Py_ssize_t umax=a_arr.shape[0]#Station Sample
    cdef Py_ssize_t uprobmax=a_prob_arr.shape[0]#Station Sample Pol Prob
    cdef Py_ssize_t uarmax=ax_arr.shape[0]#Station Sample AR
    cdef Py_ssize_t vmax=a_arr.shape[1]#Location Sample
    cdef Py_ssize_t wmax=mt.shape[1]#MT samples
    cdef Py_ssize_t kmax=a_arr.shape[2]#MT elementssamples
    cdef Py_ssize_t ipmax=incorrect_polarity_prob_arr.shape[0]
    cdef DTYPE_t[:,:,::1]a=a_arr
    cdef DTYPE_t[:,:,::1]ax=ax_arr
    cdef DTYPE_t[:,:,::1]ay=ay_arr
    cdef DTYPE_t[:,:,::1]a_prob=a_prob_arr
    cdef DTYPE_t[::1]z=z_arr
    cdef DTYPE_t[::1]positive_probability=positive_probability_arr
    cdef DTYPE_t[::1]negative_probability=negative_probability_arr
    cdef DTYPE_t[::1]sigma=sigma_arr
    cdef DTYPE_t[::1]psx=psx_arr
    cdef DTYPE_t[::1]psy=psy_arr
    cdef DTYPE_t[::1]incorrect_polarity_prob=incorrect_polarity_prob_arr
    cdef Py_ssize_t u,v,w,k
    cdef DTYPE_t max_ln_p_loc=-inf
    for w from 0<=w<wmax:
        if marginalised>0:
            max_ln_p_loc=-inf
            #loc_samples_multiplier
            for v from 0<=v<vmax:
                ln_P_loc_samples[v]=location_samples_multiplier[v]
                station_combined_all_ln_pdf(&a[0,0,0],&a_prob[0,0,0],&ax[0,0,0],&ay[0,0,0],&mt[0,0],&ln_P_loc_samples[0],&z[0],&positive_probability[0],&negative_probability[0],&incorrect_polarity_prob[0],&psx[0],&psy[0],&sigma[0], ipmax, v, umax,uarmax,uprobmax,vmax,kmax,wmax, w, v) 
                max_ln_p_loc=fmax(max_ln_p_loc,ln_P_loc_samples[v])
            if max_ln_p_loc>-inf:
                ln_P[w]=0.0
                for v from 0<=v<vmax:
                    ln_P[w]+=exp(ln_P_loc_samples[v]-max_ln_p_loc)
                ln_P[w]=log(ln_P[w])+max_ln_p_loc
            else:
                ln_P[w]=-inf
        else:
            for v from 0<=v<vmax:
                ln_P[v*wmax+w]=location_samples_multiplier[v]
                station_combined_all_ln_pdf(&a[0,0,0],&a_prob[0,0,0],&ax[0,0,0],&ay[0,0,0],&mt[0,0],&ln_P[0],&z[0],&positive_probability[0],&negative_probability[0],&incorrect_polarity_prob[0],&psx[0],&psy[0],&sigma[0], ipmax, v, umax,uarmax,uprobmax,vmax,kmax,wmax, w, v*wmax+w) 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void c_combined_pol_ln_pdf(DTYPE_t* ln_P,DTYPE_t[:,:,::1] a_arr, DTYPE_t [:,::1] mt,DTYPE_t[::1] sigma_arr,DTYPE_t[:,:,::1] a_prob_arr,DTYPE_t[::1]  positive_probability_arr,DTYPE_t[::1]  negative_probability_arr,DTYPE_t[::1]  incorrect_polarity_prob_arr,int marginalised,DTYPE_t*ln_P_loc_samples,DTYPE_t*location_samples_multiplier) nogil:
    #cdefs
    cdef Py_ssize_t umax=a_arr.shape[0]#Station Sample
    cdef Py_ssize_t uprobmax=a_prob_arr.shape[0]#Station Sample Pol Prob
    cdef Py_ssize_t vmax=a_arr.shape[1]#Location Sample
    cdef Py_ssize_t wmax=mt.shape[1]#MT samples
    cdef Py_ssize_t kmax=a_arr.shape[2]#MT elementssamples
    cdef Py_ssize_t ipmax=incorrect_polarity_prob_arr.shape[0]
    cdef DTYPE_t[:,:,::1]a=a_arr
    cdef DTYPE_t[:,:,::1]a_prob=a_prob_arr
    cdef DTYPE_t[::1]positive_probability=positive_probability_arr
    cdef DTYPE_t[::1]negative_probability=negative_probability_arr
    cdef DTYPE_t[::1]sigma=sigma_arr
    cdef DTYPE_t[::1]incorrect_polarity_prob=incorrect_polarity_prob_arr
    cdef Py_ssize_t u,v,w,k
    cdef DTYPE_t max_ln_p_loc=-inf
    for w from 0<=w<wmax:
        if marginalised>0:
            max_ln_p_loc=-inf
            #loc_samples_multiplier
            for v from 0<=v<vmax:
                ln_P_loc_samples[v]=location_samples_multiplier[v]
                station_combined_pol_ln_pdf(&a[0,0,0],&a_prob[0,0,0],&mt[0,0],&ln_P_loc_samples[0],&positive_probability[0],&negative_probability[0],&incorrect_polarity_prob[0],&sigma[0], ipmax, v, umax,uprobmax,vmax,kmax,wmax, w, v) 
                max_ln_p_loc=fmax(max_ln_p_loc,ln_P_loc_samples[v])
            if max_ln_p_loc>-inf:
                ln_P[w]=0.0
                for v from 0<=v<vmax:
                    ln_P[w]+=exp(ln_P_loc_samples[v]-max_ln_p_loc)
                ln_P[w]=log(ln_P[w])+max_ln_p_loc
            else:
                ln_P[w]=-inf
        else:
            for v from 0<=v<vmax:
                ln_P[v*wmax+w]=location_samples_multiplier[v]
                station_combined_pol_ln_pdf(&a[0,0,0],&a_prob[0,0,0],&mt[0,0],&ln_P[0],&positive_probability[0],&negative_probability[0],&incorrect_polarity_prob[0],&sigma[0], ipmax, v, umax,uprobmax,vmax,kmax,wmax, w, v*wmax+w) 

#
# Cython ln PDF generating loops 
#

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void c_polarity_ln_pdf_gen(DTYPE_t*ln_P,DTYPE_t[:,:,::1] a_arr, DTYPE_t * mt,Py_ssize_t wmax,DTYPE_t[::1] sigma_arr,DTYPE_t[::1] incorrect_polarity_prob_arr,LONG* n_tried,LONG cutoff,LONG* cut_ind,bool dc,int marginalised,DTYPE_t*ln_P_loc_samples,DTYPE_t*location_samples_multiplier):
    """Calculates probability of a positive polarity

    Calculates the probability of a positive polarity observation given a theoretical amplitude X and a fractional uncertainty sigma. Handles zero uncertainty.
    The derivation of this pdf for the polarity observation can be seen in ###########. 

    Cython handling for heavy lifting.

    """
    #cdefs
    cdef Py_ssize_t umax=a_arr.shape[0]#Station Sample
    cdef Py_ssize_t vmax=a_arr.shape[1]#Location Sample
    # cdef Py_ssize_t wmax=mt.shape[1]#MT samples
    cdef Py_ssize_t tmax=wmax*4#MT test samples
    cdef Py_ssize_t kmax=a_arr.shape[2]#MT elementssamples
    cdef Py_ssize_t ipmax=incorrect_polarity_prob_arr.shape[0]#MT elementssamples
    cdef Py_ssize_t u,v,w,k,t
    cdef DTYPE_t x=0.0
    cdef bool ok=False
    cdef DTYPE_t[:,::1]test_mt=np.empty((6,tmax))
    cdef DTYPE_t[:,:,::1]a=a_arr
    cdef DTYPE_t[::1]sigma=sigma_arr
    cdef DTYPE_t[::1]incorrect_polarity_prob=incorrect_polarity_prob_arr
    cdef DTYPE_t max_ln_p_loc=-inf
    if dc:
        test_mt=rand_dc(tmax)
    else:
        test_mt=rand_mt(tmax)
    w=0
    t=0
    while w<wmax:
        if n_tried[0]>cutoff and w==0:
            cut_ind[0]=-1
            return
        elif n_tried[0]>cutoff:
            cut_ind[0]=w
            return
        if t>=tmax:
            if dc:
                test_mt=rand_dc(tmax)
            else:
                test_mt=rand_mt(tmax)
            t=0
        ok=False
        if marginalised>0:
            max_ln_p_loc=-inf
            for v from 0<=v<vmax:
                ln_P_loc_samples[v]=location_samples_multiplier[v]
                station_polarity_ln_pdf(&a[0,0,0],&test_mt[0,0],&ln_P_loc_samples[0],&sigma[0],&incorrect_polarity_prob[0],ipmax,v,umax,vmax,kmax,tmax,t,v)
                max_ln_p_loc=fmax(max_ln_p_loc,ln_P_loc_samples[v])
            if max_ln_p_loc>-inf:
                ln_P[w]=0.0
                for v from 0<=v<vmax:
                    ln_P[w]+=exp(ln_P_loc_samples[v]-max_ln_p_loc)
                ln_P[w]=log(ln_P[w])+max_ln_p_loc
                if ln_P[w]>-inf:
                    ok=True
        else:
            for v from 0<=v<vmax:
                ln_P[v*wmax+w]=location_samples_multiplier[v]
                station_polarity_ln_pdf(&a[0,0,0],&test_mt[0,0],&ln_P[0],&sigma[0],&incorrect_polarity_prob[0], ipmax, v, umax,vmax,kmax,tmax, t, v*wmax+w)
                if ln_P[v*wmax+w]>-inf:
                    ok=True
        if ok:
            for k from 0<=k<kmax:
                mt[k*wmax+w]=test_mt[k,t]
            w+=1
        t+=1
        n_tried[0]+=1
    cut_ind[0]=wmax

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void c_polarity_probability_ln_pdf_gen(DTYPE_t*ln_P,DTYPE_t[:,:,::1]  a_arr, DTYPE_t *mt,Py_ssize_t wmax,DTYPE_t[::1]  positive_probability_arr,DTYPE_t[::1]  negative_probability_arr,DTYPE_t[::1]  incorrect_polarity_prob_arr,LONG* n_tried,LONG cutoff,LONG*cut_ind,bool dc,int marginalised,DTYPE_t*ln_P_loc_samples,DTYPE_t*location_samples_multiplier):
    """Calculates probability of a given amplitude giving an observed polarity probability

    Calculates the probability of a given polarity probability observation given a theoretical amplitude X and a fractional uncertainty sigma. Handles zero uncertainty.
    The derivation of this pdf for the polarity observation can be seen in ###########. 

    Cython handling for heavy lifting.

    """
    #cdefs
    cdef Py_ssize_t umax=a_arr.shape[0]#Station Sample
    cdef Py_ssize_t vmax=a_arr.shape[1]#Location Sample
    cdef Py_ssize_t tmax=wmax*4#MT test samples
    cdef Py_ssize_t kmax=a_arr.shape[2]#MT elementssamples
    cdef Py_ssize_t ipmax=incorrect_polarity_prob_arr.shape[0]#MT elementssamples
    cdef Py_ssize_t u,v,w,k,t
    cdef DTYPE_t x=0.0
    cdef bool ok=False
    cdef DTYPE_t[:,::1]test_mt=np.empty((6,tmax))
    cdef DTYPE_t[:,:,::1]a=a_arr
    cdef DTYPE_t[::1]positive_probability=positive_probability_arr
    cdef DTYPE_t[::1]negative_probability=negative_probability_arr
    cdef DTYPE_t[::1]incorrect_polarity_prob=incorrect_polarity_prob_arr
    cdef DTYPE_t max_ln_p_loc=-inf
    if dc:
        test_mt=rand_dc(tmax)
    else:
        test_mt=rand_mt(tmax)
    w=0
    t=0
    while w<wmax:
        if n_tried[0]>cutoff and w==0:
            cut_ind[0]=-1
            return
        elif n_tried[0]>cutoff:
            cut_ind[0]=w
            return
        if t>=tmax:
            if dc:
                test_mt=rand_dc(tmax)
            else:
                test_mt=rand_mt(tmax)
            t=0
        ok=False
        if marginalised>0:
            max_ln_p_loc=-inf
            for v from 0<=v<vmax:
                ln_P_loc_samples[v]=location_samples_multiplier[v]
                station_polarity_probability_ln_pdf(&a[0,0,0],&test_mt[0,0],&ln_P_loc_samples[0],&positive_probability[0],&negative_probability[0],&incorrect_polarity_prob[0],ipmax,v,umax,vmax,kmax,tmax,t,v)
                max_ln_p_loc=fmax(max_ln_p_loc,ln_P_loc_samples[v])
            if max_ln_p_loc>-inf:
                ln_P[w]=0.0
                for v from 0<=v<vmax:
                    ln_P[w]+=exp(ln_P_loc_samples[v]-max_ln_p_loc)
                ln_P[w]=log(ln_P[w])+max_ln_p_loc
                if ln_P[w]>-inf:
                    ok=True
        else:
            for v from 0<=v<vmax:
                ln_P[v*wmax+w]=location_samples_multiplier[v]  
                station_polarity_probability_ln_pdf(&a[0,0,0],&test_mt[0,0],&ln_P[0],&positive_probability[0],&negative_probability[0],&incorrect_polarity_prob[0],ipmax,v,umax,vmax,kmax,tmax,t,v*wmax+w)
                if ln_P[v*wmax+w]>-inf:
                    ok=True
        if ok:
            for k from 0<=k<kmax:
                mt[k*wmax+w]=test_mt[k,t]
            w+=1
        t+=1
        n_tried[0]+=1
    cut_ind[0]=wmax

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void c_amplitude_ratio_ln_pdf_gen(DTYPE_t* ln_P,DTYPE_t[::1]  z_arr, DTYPE_t *mt,Py_ssize_t wmax,DTYPE_t[:,:,::1]  ax_arr,DTYPE_t[:,:,::1]  ay_arr,DTYPE_t[::1] psx_arr,DTYPE_t[::1] psy_arr,LONG* n_tried,LONG cutoff,LONG*cut_ind,bool dc,int marginalised,DTYPE_t*ln_P_loc_samples,DTYPE_t*location_samples_multiplier):
    """Calculates Amplitude Ratio Probability

    Calculates the Ratio pdf (D. Hinkley, On the ratio of two correlated normal random variables, 1969, Biometrika vol 56 pp 635-639).
    Given Z=X/Y and means mux, muy and standard deviation sigmax and sigmay. The pdf is normalised.

    Cython handling for heavy lifting.

    """
    #cdefs    
    cdef Py_ssize_t umax=ax_arr.shape[0]#Station Sample
    cdef Py_ssize_t vmax=ax_arr.shape[1]#Location Sample
    # cdef Py_ssize_t wmax=mt.shape[1]#MT samples
    cdef Py_ssize_t tmax=wmax*4#MT test samples
    cdef Py_ssize_t kmax=ax_arr.shape[2]#MT elementssamples
    cdef Py_ssize_t u,v,w,k,t
    cdef DTYPE_t mux=0.0
    cdef DTYPE_t muy=0.0
    cdef bool ok=False
    cdef DTYPE_t[:,::1]test_mt=np.empty((6,tmax))
    cdef DTYPE_t[:,:,::1]ax=ax_arr
    cdef DTYPE_t[:,:,::1]ay=ay_arr
    cdef DTYPE_t[::1]z=z_arr
    cdef DTYPE_t[::1]psx=psx_arr
    cdef DTYPE_t[::1]psy=psy_arr
    cdef DTYPE_t max_ln_p_loc=-inf
    if dc:
        test_mt=rand_dc(tmax)
    else:
        test_mt=rand_mt(tmax)
    w=0
    t=0
    while w<wmax:
        if n_tried[0]>cutoff and w==0:
            cut_ind[0]=-1
            return
        elif n_tried[0]>cutoff:
            cut_ind[0]=w
            return
        
        if t>=tmax:
            if dc:
                test_mt=rand_dc(tmax)
            else:
                test_mt=rand_mt(tmax)
            t=0
        ok=False
        if marginalised>0:
            max_ln_p_loc=-inf
            #loc_samples_multiplier
            for v from 0<=v<vmax:
                ln_P_loc_samples[v]=location_samples_multiplier[v]
                station_ar_ln_pdf(&ax[0,0,0],&ay[0,0,0],&test_mt[0,0],&ln_P_loc_samples[0],&z[0],&psx[0],&psy[0],v,umax,vmax,kmax,tmax,t,v)
                max_ln_p_loc=fmax(max_ln_p_loc,ln_P_loc_samples[v])
            if max_ln_p_loc>-inf:
                ln_P[w]=0.0
                for v from 0<=v<vmax:
                    ln_P[w]+=exp(ln_P_loc_samples[v]-max_ln_p_loc)
                ln_P[w]=log(ln_P[w])+max_ln_p_loc
                if ln_P[w]>-inf:
                    ok=True
        else:
            for v from 0<=v<vmax:
                ln_P[v*wmax+w]=location_samples_multiplier[v]            
                station_ar_ln_pdf(&ax[0,0,0],&ay[0,0,0],&test_mt[0,0],&ln_P[0],&z[0],&psx[0],&psy[0],  v, umax,vmax,kmax,tmax, t, v*wmax+w)   
                if  ln_P[v*wmax+w]>-inf:
                    ok=True
        if ok:
            for k from 0<=k<kmax:
                mt[k*wmax+w]=test_mt[k,t]
            w+=1
        t+=1
        n_tried[0]+=1
    cut_ind[0]=wmax

#
# Cython Combined ln PDF generating loops 
#

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void c_polarity_prob_combined_ln_pdf_gen(DTYPE_t*ln_P,DTYPE_t[:,:,::1] a_arr, DTYPE_t *mt,Py_ssize_t wmax,DTYPE_t[::1]  positive_probability_arr,DTYPE_t[::1]  negative_probability_arr,DTYPE_t[::1]  incorrect_polarity_prob_arr,DTYPE_t[::1]  z_arr,DTYPE_t[:,:,::1]  ax_arr,DTYPE_t[:,:,::1]  ay_arr,DTYPE_t[::1] psx_arr,DTYPE_t[::1] psy_arr,LONG* n_tried,LONG cutoff,LONG*cut_ind,bool dc,int marginalised,DTYPE_t*ln_P_loc_samples,DTYPE_t*location_samples_multiplier):
    #cdefs
    cdef Py_ssize_t umax=a_arr.shape[0]#Station Sample
    cdef Py_ssize_t uarmax=ax_arr.shape[0]#Station Sample AR
    cdef Py_ssize_t vmax=a_arr.shape[1]#Location Sample
    # cdef Py_ssize_t wmax=mt.shape[1]#MT samples
    cdef Py_ssize_t tmax=wmax*4#MT test samples
    cdef Py_ssize_t kmax=a_arr.shape[2]#MT elementssamples
    cdef Py_ssize_t ipmax=incorrect_polarity_prob_arr.shape[0]
    cdef Py_ssize_t u,v,w,k,t
    cdef DTYPE_t x=0.0
    cdef DTYPE_t mux=0.0
    cdef DTYPE_t muy=0.0
    cdef bool ok=False
    cdef DTYPE_t[:,::1]test_mt=np.empty((6,tmax))
    cdef DTYPE_t[:,:,::1]a=a_arr
    cdef DTYPE_t[:,:,::1]ax=ax_arr
    cdef DTYPE_t[:,:,::1]ay=ay_arr
    cdef DTYPE_t[::1]z=z_arr
    cdef DTYPE_t[::1]positive_probability=positive_probability_arr
    cdef DTYPE_t[::1]negative_probability=negative_probability_arr
    cdef DTYPE_t[::1]psx=psx_arr
    cdef DTYPE_t[::1]psy=psy_arr
    cdef DTYPE_t[::1]incorrect_polarity_prob=incorrect_polarity_prob_arr
    cdef DTYPE_t max_ln_p_loc=-inf
    if dc:
        test_mt=rand_dc(tmax)
    else:
        test_mt=rand_mt(tmax)
    w=0
    t=0
    while w<wmax:
        if n_tried[0]>cutoff and w==0:
            cut_ind[0]=-1
            return
        elif n_tried[0]>cutoff:
            cut_ind[0]=w
            return
        if t>=tmax:
            if dc:
                test_mt=rand_dc(tmax)
            else:
                test_mt=rand_mt(tmax)
            t=0
        ok=False
        if marginalised>0:
            max_ln_p_loc=-inf
            #loc_samples_multiplier
            for v from 0<=v<vmax:
                ln_P_loc_samples[v]=location_samples_multiplier[v]
                station_combined_polarity_probability_ar_ln_pdf(&a[0,0,0],&ax[0,0,0],&ay[0,0,0],&test_mt[0,0],&ln_P_loc_samples[0],&z[0],&positive_probability[0],&negative_probability[0],&incorrect_polarity_prob[0],&psx[0],&psy[0], ipmax, v, umax,uarmax,vmax,kmax,tmax, t, v) 
                max_ln_p_loc=fmax(max_ln_p_loc,ln_P_loc_samples[v])
            if max_ln_p_loc>-inf:
                ln_P[w]=0.0
                for v from 0<=v<vmax:
                    ln_P[w]+=exp(ln_P_loc_samples[v]-max_ln_p_loc)
                ln_P[w]=log(ln_P[w])+max_ln_p_loc
                if ln_P[w]>-inf:
                    ok=True

        else:
            for v from 0<=v<vmax:
                ln_P[v*wmax+w]=location_samples_multiplier[v]
                station_combined_polarity_probability_ar_ln_pdf(&a[0,0,0],&ax[0,0,0],&ay[0,0,0],&test_mt[0,0],&ln_P[0],&z[0],&positive_probability[0],&negative_probability[0],&incorrect_polarity_prob[0],&psx[0],&psy[0], ipmax, v, umax,uarmax,vmax,kmax,tmax, t, v*wmax+w) 
                if ln_P[v*wmax+w]>-inf:
                    ok=True
        if ok:
            for k from 0<=k<kmax:
                mt[k*wmax+w]=test_mt[k,t]
            w+=1
        t+=1
        n_tried[0]+=1
    cut_ind[0]=wmax

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void c_polarity_ar_ln_pdf_gen(DTYPE_t*ln_P,DTYPE_t[:,:,::1] a_arr, DTYPE_t *mt,Py_ssize_t wmax,DTYPE_t[::1] sigma_arr,DTYPE_t[::1] incorrect_polarity_prob_arr,DTYPE_t[::1]  z_arr, DTYPE_t[:,:,::1]  ax_arr,DTYPE_t[:,:,::1]  ay_arr,DTYPE_t[::1] psx_arr,DTYPE_t[::1] psy_arr,LONG* n_tried,LONG cutoff, LONG*cut_ind,bool dc,int marginalised,DTYPE_t*ln_P_loc_samples,DTYPE_t*location_samples_multiplier):
    cdef Py_ssize_t umax=a_arr.shape[0]#Station Sample
    cdef Py_ssize_t uarmax=ax_arr.shape[0]#Station Sample AR
    cdef Py_ssize_t vmax=a_arr.shape[1]#Location Sample
    cdef Py_ssize_t kmax=a_arr.shape[2]#MT elementssamples
    cdef Py_ssize_t tmax=wmax*4#MT test samples
    # cdef Py_ssize_t wmax=mt.shape[1]#MT samples
    cdef Py_ssize_t ipmax=incorrect_polarity_prob_arr.shape[0]
    cdef Py_ssize_t u,v,w,k,t
    # cdef DTYPE_t[:,:,::1]  ln_P=np.zeros((umax,vmax,wmax)) 
    cdef DTYPE_t x=0.0
    cdef DTYPE_t mux=0.0
    cdef DTYPE_t muy=0.0
    cdef bool ok=False
    cdef DTYPE_t[:,::1]test_mt=np.empty((6,tmax))
    cdef DTYPE_t[:,:,::1]a=a_arr
    cdef DTYPE_t[:,:,::1]ax=ax_arr
    cdef DTYPE_t[:,:,::1]ay=ay_arr
    cdef DTYPE_t[::1]z=z_arr
    cdef DTYPE_t[::1]sigma=sigma_arr
    cdef DTYPE_t[::1]psx=psx_arr
    cdef DTYPE_t[::1]psy=psy_arr
    cdef DTYPE_t[::1]incorrect_polarity_prob=incorrect_polarity_prob_arr

    if dc:
        test_mt=rand_dc(tmax)
    else:
        test_mt=rand_mt(tmax)
    w=0
    t=0
    while w<wmax:
        if n_tried[0]>cutoff and w==0:
            cut_ind[0]=-1
            return
        elif n_tried[0]>cutoff:
            cut_ind[0]=w
            return
        if t>=tmax:
            if dc:
                test_mt=rand_dc(tmax)
            else:
                test_mt=rand_mt(tmax)
            t=0
        ok=False
        if marginalised>0:
            max_ln_p_loc=-inf
            #loc_samples_multiplier
            for v from 0<=v<vmax:
                ln_P_loc_samples[v]=location_samples_multiplier[v]
                station_combined_polarity_ar_ln_pdf(&a[0,0,0],&ax[0,0,0],&ay[0,0,0],&test_mt[0,0],&ln_P_loc_samples[0],&z[0],&sigma[0],&incorrect_polarity_prob[0],&psx[0],&psy[0], ipmax, v, umax,uarmax,vmax,kmax,tmax, t, v)
                max_ln_p_loc=fmax(max_ln_p_loc,ln_P_loc_samples[v])
            if max_ln_p_loc>-inf:
                ln_P[w]=0.0
                for v from 0<=v<vmax:
                    ln_P[w]+=exp(ln_P_loc_samples[v]-max_ln_p_loc)
                ln_P[w]=log(ln_P[w])+max_ln_p_loc
                if ln_P[w]>-inf:
                    ok=True
        else:
            for v from 0<=v<vmax:
                ln_P[v*wmax+w]=location_samples_multiplier[v]
                station_combined_polarity_ar_ln_pdf(&a[0,0,0],&ax[0,0,0],&ay[0,0,0],&test_mt[0,0],&ln_P[0],&z[0],&sigma[0],&incorrect_polarity_prob[0],&psx[0],&psy[0], ipmax, v, umax,uarmax,vmax,kmax,tmax, t, v*wmax+w)
                if ln_P[v*wmax+w]>-inf:
                    ok=True
        if ok:
            for k from 0<=k<kmax:
                mt[k*wmax+w]=test_mt[k,t]
            w+=1
        t+=1
        n_tried[0]+=1
    cut_ind[0]=wmax

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void c_all_combined_ln_pdf_gen(DTYPE_t* ln_P,DTYPE_t[:,:,::1] a_arr,DTYPE_t *mt,Py_ssize_t wmax,DTYPE_t[::1] sigma_arr,DTYPE_t[:,:,::1] a_prob_arr,DTYPE_t[::1]  positive_probability_arr,DTYPE_t[::1]  negative_probability_arr,DTYPE_t[::1]  incorrect_polarity_prob_arr,DTYPE_t[::1]  z_arr,DTYPE_t[:,:,::1]  ax_arr,DTYPE_t[:,:,::1]  ay_arr,DTYPE_t[::1] psx_arr,DTYPE_t[::1] psy_arr,LONG* n_tried,LONG cutoff, LONG*cut_ind,bool dc,int marginalised,DTYPE_t*ln_P_loc_samples,DTYPE_t*location_samples_multiplier):
    #cdefs
    cdef Py_ssize_t umax=a_arr.shape[0]#Station Sample
    cdef Py_ssize_t uprobmax=a_prob_arr.shape[0]#Station Sample Pol Prob
    cdef Py_ssize_t uarmax=ax_arr.shape[0]#Station Sample AR
    cdef Py_ssize_t vmax=a_arr.shape[1]#Location Sample
    cdef Py_ssize_t kmax=a_arr.shape[2]#MT elementssamples
    cdef Py_ssize_t tmax=wmax*4#MT test samples
    cdef Py_ssize_t ipmax=incorrect_polarity_prob_arr.shape[0]
    cdef DTYPE_t[:,:,::1]a=a_arr
    cdef DTYPE_t[:,:,::1]ax=ax_arr
    cdef DTYPE_t[:,:,::1]ay=ay_arr
    cdef DTYPE_t[:,:,::1]a_prob=a_prob_arr
    cdef DTYPE_t[::1]z=z_arr
    cdef DTYPE_t[::1]positive_probability=positive_probability_arr
    cdef DTYPE_t[::1]negative_probability=negative_probability_arr
    cdef DTYPE_t[::1]sigma=sigma_arr
    cdef DTYPE_t[::1]psx=psx_arr
    cdef DTYPE_t[::1]psy=psy_arr
    cdef DTYPE_t[::1]incorrect_polarity_prob=incorrect_polarity_prob_arr
    cdef Py_ssize_t u,v,w,k
    cdef DTYPE_t max_ln_p_loc=-inf
    if dc:
        test_mt=rand_dc(tmax)
    else:
        test_mt=rand_mt(tmax)
    w=0
    t=0
    while w<wmax:
        if n_tried[0]>cutoff and w==0:
            cut_ind[0]=-1
            return
        elif n_tried[0]>cutoff:
            cut_ind[0]=w
            return
        if t>=tmax:
            if dc:
                test_mt=rand_dc(tmax)
            else:
                test_mt=rand_mt(tmax)
            t=0
        ok=False
        if marginalised>0:
            max_ln_p_loc=-inf
            #loc_samples_multiplier
            for v from 0<=v<vmax:
                ln_P_loc_samples[v]=location_samples_multiplier[v]
                station_combined_all_ln_pdf(&a[0,0,0],&a_prob[0,0,0],&ax[0,0,0],&ay[0,0,0],&test_mt[0,0],&ln_P_loc_samples[0],&z[0],&positive_probability[0],&negative_probability[0],&incorrect_polarity_prob[0],&psx[0],&psy[0],&sigma[0], ipmax, v, umax,uarmax,uprobmax,vmax,kmax,wmax, w, v) 
                max_ln_p_loc=fmax(max_ln_p_loc,ln_P_loc_samples[v])
            if max_ln_p_loc>-inf:
                ln_P[w]=0.0
                for v from 0<=v<vmax:
                    ln_P[w]+=exp(ln_P_loc_samples[v]-max_ln_p_loc)
                ln_P[w]=log(ln_P[w])+max_ln_p_loc
                if ln_P[w]>-inf:
                    ok=True
        else:
            for v from 0<=v<vmax:
                ln_P[v*wmax+w]=location_samples_multiplier[v]
                station_combined_all_ln_pdf(&a[0,0,0],&a_prob[0,0,0],&ax[0,0,0],&ay[0,0,0],&test_mt[0,0],&ln_P[0],&z[0],&positive_probability[0],&negative_probability[0],&incorrect_polarity_prob[0],&psx[0],&psy[0],&sigma[0], ipmax, v, umax,uarmax,uprobmax,vmax,kmax,wmax, w, v*wmax+w) 
                if ln_P[v*wmax+w]>-inf:
                    ok=True
        if ok:
            for k from 0<=k<kmax:
                mt[k*wmax+w]=test_mt[k,t]
            w+=1
        t+=1
        n_tried[0]+=1
    cut_ind[0]=wmax

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void c_combined_pol_ln_pdf_gen(DTYPE_t* ln_P,DTYPE_t[:,:,::1] a_arr,DTYPE_t *mt,Py_ssize_t wmax,DTYPE_t[::1] sigma_arr,DTYPE_t[:,:,::1] a_prob_arr,DTYPE_t[::1]  positive_probability_arr,DTYPE_t[::1]  negative_probability_arr,DTYPE_t[::1]  incorrect_polarity_prob_arr,LONG* n_tried,LONG cutoff, LONG*cut_ind,bool dc,int marginalised,DTYPE_t*ln_P_loc_samples,DTYPE_t*location_samples_multiplier):
    #cdefs
    cdef Py_ssize_t umax=a_arr.shape[0]#Station Sample
    cdef Py_ssize_t uprobmax=a_prob_arr.shape[0]#Station Sample Pol Prob
    cdef Py_ssize_t vmax=a_arr.shape[1]#Location Sample
    cdef Py_ssize_t kmax=a_arr.shape[2]#MT elementssamples
    cdef Py_ssize_t tmax=wmax*4#MT test samples
    cdef Py_ssize_t ipmax=incorrect_polarity_prob_arr.shape[0]
    cdef DTYPE_t[:,:,::1]a=a_arr
    cdef DTYPE_t[:,:,::1]a_prob=a_prob_arr
    cdef DTYPE_t[::1]positive_probability=positive_probability_arr
    cdef DTYPE_t[::1]negative_probability=negative_probability_arr
    cdef DTYPE_t[::1]sigma=sigma_arr
    cdef DTYPE_t[::1]incorrect_polarity_prob=incorrect_polarity_prob_arr
    cdef Py_ssize_t u,v,w,k
    cdef DTYPE_t max_ln_p_loc=-inf
    if dc:
        test_mt=rand_dc(tmax)
    else:
        test_mt=rand_mt(tmax)
    w=0
    t=0
    while w<wmax:
        if n_tried[0]>cutoff and w==0:
            cut_ind[0]=-1
            return
        elif n_tried[0]>cutoff:
            cut_ind[0]=w
            return
        if t>=tmax:
            if dc:
                test_mt=rand_dc(tmax)
            else:
                test_mt=rand_mt(tmax)
            t=0
        ok=False
        if marginalised>0:
            max_ln_p_loc=-inf
            #loc_samples_multiplier
            for v from 0<=v<vmax:
                ln_P_loc_samples[v]=location_samples_multiplier[v]
                station_combined_pol_ln_pdf(&a[0,0,0],&a_prob[0,0,0],&test_mt[0,0],&ln_P_loc_samples[0],&positive_probability[0],&negative_probability[0],&incorrect_polarity_prob[0],&sigma[0], ipmax, v, umax,uprobmax,vmax,kmax,wmax, w, v) 
                max_ln_p_loc=fmax(max_ln_p_loc,ln_P_loc_samples[v])
            if max_ln_p_loc>-inf:
                ln_P[w]=0.0
                for v from 0<=v<vmax:
                    ln_P[w]+=exp(ln_P_loc_samples[v]-max_ln_p_loc)
                ln_P[w]=log(ln_P[w])+max_ln_p_loc
                if ln_P[w]>-inf:
                    ok=True
        else:
            for v from 0<=v<vmax:
                ln_P[v*wmax+w]=location_samples_multiplier[v]
                station_combined_pol_ln_pdf(&a[0,0,0],&a_prob[0,0,0],&test_mt[0,0],&ln_P[0],&positive_probability[0],&negative_probability[0],&incorrect_polarity_prob[0],&sigma[0], ipmax, v, umax,uprobmax,vmax,kmax,wmax, w, v*wmax+w) 
                if ln_P[v*wmax+w]>-inf:
                    ok=True
        if ok:
            for k from 0<=k<kmax:
                mt[k*wmax+w]=test_mt[k,t]
            w+=1
        t+=1
        n_tried[0]+=1
    cut_ind[0]=wmax

#
# Python PDF calls (Polarity, Polarity Probability and Amplitude Ratios)
#

def polarity_ln_pdf(a,mt_arr,sigma,incorrect_polarity_prob=np.array([0.]),generate_samples=0,cutoff=1000000000,dc=False,int marginalised=0,location_samples_multipliers=np.array([0.])):
    if generate_samples:
        mt_arr=np.empty((a.shape[2],generate_samples))
    cdef DTYPE_t[:,::1] mt=mt_arr
    cdef Py_ssize_t wmax=mt.shape[1]    
    cdef Py_ssize_t umax=a.shape[0]#Station Sample
    cdef Py_ssize_t vmax=a.shape[1]#Location Sample
    cdef LONG n_tried=0
    cdef LONG cut_ind=0
    if location_samples_multipliers.shape[0]!=vmax:
        location_samples_multipliers=np.zeros((vmax))
    if marginalised>0:
        vmax=1
    marginalised=int(marginalised)
    cdef DTYPE_t[:,::1] ln_P=np.empty((vmax,wmax)) 
    cdef DTYPE_t[::1] location_samples_multiplier=location_samples_multipliers 
    cdef DTYPE_t[::1] ln_P_loc_samples=np.empty(location_samples_multipliers.shape)
    if generate_samples:
        c_polarity_ln_pdf_gen(&ln_P[0,0],a,&mt[0,0],wmax,sigma,incorrect_polarity_prob,&n_tried,cutoff,&cut_ind,dc,marginalised,&ln_P_loc_samples[0],&location_samples_multiplier[0])
        if cut_ind<0:
            # no non_zero samples
            return np.array([]),np.array([[],[],[],[],[],[]]),n_tried
        elif cut_ind!=wmax:
            #ended prematurely
            ln_P=ln_P[:,:cut_ind]
            mt=mt[:,:cut_ind]
        return np.asarray(ln_P),np.asarray(mt),n_tried  
    c_polarity_ln_pdf(&ln_P[0,0],a,mt,sigma,incorrect_polarity_prob,marginalised,&ln_P_loc_samples[0],&location_samples_multiplier[0])
    return np.asarray(ln_P)

def polarity_probability_ln_pdf(  a, mt_arr,positive_probability,negative_probability,incorrect_polarity_prob=np.array([0.]),generate_samples=0,cutoff=1000000000,dc=False,int marginalised=0,location_samples_multipliers=np.array([0.])):
    if generate_samples:
        mt_arr=np.empty((a.shape[2],generate_samples))
    cdef DTYPE_t[:,::1] mt=mt_arr
    cdef Py_ssize_t wmax=mt.shape[1]  
    cdef Py_ssize_t umax=a.shape[0]#Station Sample
    cdef Py_ssize_t vmax=a.shape[1]#Location Sample
    if location_samples_multipliers.shape[0]!=vmax:
        location_samples_multipliers=np.zeros((vmax))
    cdef DTYPE_t[::1] location_samples_multiplier=location_samples_multipliers
    cdef DTYPE_t[::1] ln_P_loc_samples=np.empty(location_samples_multipliers.shape)
    if marginalised>0:
        vmax=1
    cdef LONG n_tried=0
    cdef LONG cut_ind=0
    cdef DTYPE_t[:,::1] ln_P=np.empty((vmax,wmax))  
    if generate_samples:
        c_polarity_probability_ln_pdf_gen(&ln_P[0,0],a,&mt[0,0],wmax,positive_probability, negative_probability,incorrect_polarity_prob,&n_tried,cutoff,&cut_ind,dc,marginalised,&ln_P_loc_samples[0],&location_samples_multiplier[0])
        if cut_ind<0:
            # no non_zero samples
            return np.array([]),np.array([[],[],[],[],[],[]]),n_tried
        elif cut_ind!=wmax:
            #ended prematurely
            ln_P=ln_P[:,:cut_ind]
            mt=mt[:,:cut_ind]
        return np.asarray(ln_P),np.asarray(mt),n_tried  
    c_polarity_probability_ln_pdf(&ln_P[0,0],a, mt, positive_probability, negative_probability,incorrect_polarity_prob,marginalised,&ln_P_loc_samples[0],&location_samples_multiplier[0])
    return np.asarray(ln_P)

def amplitude_ratio_ln_pdf( z, mt_arr,ax,ay, psx, psy,generate_samples=0,cutoff=1000000000,dc=False,int marginalised=0,location_samples_multipliers=np.array([0.])):
    if generate_samples:
        mt_arr=np.empty((ax.shape[2],generate_samples))
    cdef DTYPE_t[:,::1] mt=mt_arr
    cdef Py_ssize_t wmax=mt.shape[1]    
    cdef Py_ssize_t umax=ax.shape[0]#Station Sample
    cdef Py_ssize_t vmax=ax.shape[1]#Location Sample
    if location_samples_multipliers.shape[0]!=vmax:
        location_samples_multipliers=np.zeros((vmax))
    cdef DTYPE_t[::1] location_samples_multiplier=location_samples_multipliers
    cdef DTYPE_t[::1] ln_P_loc_samples=np.empty(location_samples_multipliers.shape)
    if marginalised>0:
        vmax=1
    cdef LONG n_tried=0
    cdef LONG cut_ind=0
    cdef DTYPE_t[:,::1] ln_P=np.empty((vmax,wmax))  
    if generate_samples:
        c_amplitude_ratio_ln_pdf_gen(&ln_P[0,0],z,&mt[0,0],wmax,ax,ay,psx,psy,&n_tried,cutoff,&cut_ind,dc,marginalised,&ln_P_loc_samples[0],&location_samples_multiplier[0])
        if cut_ind<0:
            # no non_zero samples
            return np.array([]),np.array([[],[],[],[],[],[]]),n_tried
        elif cut_ind!=wmax:
            #ended prematurely
            ln_P=ln_P[:,:cut_ind]
            mt=mt[:,:cut_ind]
        return np.asarray(ln_P),np.asarray(mt),n_tried  
    c_amplitude_ratio_ln_pdf(&ln_P[0,0],z, mt,ax,ay,psx,psy,marginalised,&ln_P_loc_samples[0],&location_samples_multiplier[0])
    return np.asarray(ln_P)

def log0test():
    return log(0)==-inf

def combined_ln_pdf(mt_arr,a_polarity,error_polarity,a1_amplitude_ratio,a2_amplitude_ratio,amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio,a_polarity_prob,polarity_prob,incorrect_polarity_prob=0,generate_samples=0,cutoff=1000000000,dc=False,marginalised=False,location_samples_multipliers=np.array([0.])):
    if isinstance(incorrect_polarity_prob, int) and incorrect_polarity_prob == 0:
        incorrect_polarity_prob=np.array([0.])
    generate_mts=False  
    cdef Py_ssize_t umax=0#Station Sample
    cdef Py_ssize_t vmax=0#Location Sample
    cdef Py_ssize_t kmax=0
    if not isinstance(a_polarity, bool):
        kmax=a_polarity.shape[2]
        umax=a_polarity.shape[0]
        vmax=a_polarity.shape[1]
    elif not isinstance(a_polarity_prob, bool):
        kmax=a_polarity_prob.shape[2]
        umax=a_polarity_prob.shape[0]
        vmax=a_polarity_prob.shape[1]
    elif not isinstance(a1_amplitude_ratio, bool) and not isinstance(a2_amplitude_ratio, bool):
        kmax=a1_amplitude_ratio.shape[2]
        umax=a1_amplitude_ratio.shape[0]
        vmax=a1_amplitude_ratio.shape[1]
    if location_samples_multipliers.shape[0]!=vmax:
        location_samples_multipliers=np.zeros((vmax))
    if marginalised>0:
        vmax=1
    if generate_samples:
        generate_mts=True
        mt_arr=np.empty((kmax,generate_samples))
    cdef DTYPE_t[:,::1] mt=mt_arr
    cdef DTYPE_t[::1] location_samples_multiplier=location_samples_multipliers
    cdef DTYPE_t[::1] ln_P_loc_samples=np.empty(location_samples_multipliers.shape)
    cdef Py_ssize_t wmax=mt.shape[1]
    cdef LONG n_tried=0
    cdef LONG cut_ind=0
    cdef DTYPE_t[:,::1] ln_P=np.empty((vmax,wmax))  
    # data preparation
    if isinstance(a_polarity, np.ndarray) and a_polarity.dtype!=np.float64:
        a_polarity=a_polarity.astype(np.float64,copy=False)
    if isinstance(mt, np.ndarray) and mt.dtype!=np.float64:
        mt=mt.astype(np.float64,copy=False)
    if isinstance(error_polarity, np.ndarray) and error_polarity.dtype!=np.float64:
        error_polarity=error_polarity.astype(np.float64,copy=False)
    if isinstance(incorrect_polarity_prob, np.ndarray) and incorrect_polarity_prob.dtype!=np.float64:
        incorrect_polarity_prob=incorrect_polarity_prob.astype(np.float64,copy=False)
    if isinstance(a1_amplitude_ratio, np.ndarray) and a1_amplitude_ratio.dtype!=np.float64:
        a1_amplitude_ratio=a1_amplitude_ratio.astype(np.float64,copy=False)
    if isinstance(a2_amplitude_ratio, np.ndarray) and a2_amplitude_ratio.dtype!=np.float64:
        a2_amplitude_ratio=a2_amplitude_ratio.astype(np.float64,copy=False)
    if isinstance(amplitude_ratio, np.ndarray) and amplitude_ratio.dtype!=np.float64:
        amplitude_ratio=amplitude_ratio.astype(np.float64,copy=False)
    if isinstance(percentage_error1_amplitude_ratio, np.ndarray) and percentage_error1_amplitude_ratio.dtype!=np.float64:
        percentage_error1_amplitude_ratio=percentage_error1_amplitude_ratio.astype(np.float64,copy=False)
    if isinstance(percentage_error2_amplitude_ratio, np.ndarray) and percentage_error2_amplitude_ratio.dtype!=np.float64:
        percentage_error2_amplitude_ratio=percentage_error2_amplitude_ratio.astype(np.float64,copy=False)
    if isinstance(a_polarity_prob, np.ndarray) and a_polarity_prob.dtype!=np.float64:
        a_polarity_prob=a_polarity_prob.astype(np.float64,copy=False)
    if isinstance(polarity_prob, np.ndarray) and polarity_prob.dtype!=np.float64:
        polarity_prob=polarity_prob.astype(np.float64,copy=False)
    #
    #Bounds checking
    #
    #Polarities
    if not isinstance(a_polarity, bool) and a_polarity.shape[0]>incorrect_polarity_prob.shape[0] and incorrect_polarity_prob.shape[0]>1:
        raise IndexError('Bounds Exception for incorrect_polarity_prob.\nShape: '+str(incorrect_polarity_prob.shape)+'\nincompatible with number of observations:\n'+str(a_polarity.shape[0]))
    if not isinstance(a_polarity, bool) and a_polarity.shape[0]>error_polarity.shape[0]:
        raise IndexError('Bounds Exception for polarity uncertainty.\nShape: '+str(error_polarity.shape)+'\nincompatible with number of observations:\n'+str(a_polarity.shape[0]))
    # Amplitude ratios
    if not isinstance(a1_amplitude_ratio, bool) and not isinstance(a2_amplitude_ratio, bool) and (a1_amplitude_ratio.shape[0]>a2_amplitude_ratio.shape[0] or a1_amplitude_ratio.shape[1]!=a2_amplitude_ratio.shape[1]):
        raise IndexError('Bounds Exception for a2_amplitude_ratio.\nShape: '+str(a2_amplitude_ratio.shape)+'\nincompatible with number of observations and location samples:\n'+str(a1_amplitude_ratio.shape))
    if not isinstance(a1_amplitude_ratio, bool) and not isinstance(a2_amplitude_ratio, bool) and a1_amplitude_ratio.shape[0]>amplitude_ratio.shape[0]:
        raise IndexError('Bounds Exception for Amplitude ratio observations.\nShape: '+str(amplitude_ratio.shape)+'\nincompatible with number of receivers:\n'+str(a1_amplitude_ratio.shape[0]))
    if not isinstance(a1_amplitude_ratio, bool) and not isinstance(a2_amplitude_ratio, bool) and a1_amplitude_ratio.shape[0]>percentage_error1_amplitude_ratio.shape[0]:
        raise IndexError('Bounds Exception for Amplitude ratio uncertainty.\nShape: '+str(percentage_error1_amplitude_ratio.shape)+'\nincompatible with number of receivers:\n'+str(a1_amplitude_ratio.shape[0]))
    if not isinstance(a1_amplitude_ratio, bool) and not isinstance(a2_amplitude_ratio, bool) and  a1_amplitude_ratio.shape[0]>percentage_error2_amplitude_ratio.shape[0]:
        raise IndexError('Bounds Exception for Amplitude ratio uncertainty.\nShape: '+str(percentage_error2_amplitude_ratio.shape)+'\nincompatible with number of receivers:\n'+str(a1_amplitude_ratio.shape[0]))
    #Polarity Probabilities
    if not isinstance(a_polarity_prob, bool) and a_polarity_prob.shape[0]>incorrect_polarity_prob.shape[0] and incorrect_polarity_prob.shape[0]>1:
        raise IndexError('Bounds Exception for incorrect_polarity_prob.\nShape: '+str(incorrect_polarity_prob.shape)+'\nincompatible with number of receivers:\n'+str(a_polarity_prob.shape[0]))
    if not isinstance(a_polarity_prob, bool) and a_polarity_prob.shape[0]>polarity_prob[0].shape[0]:
        raise IndexError('Bounds Exception for polarity probability observations.\nShape: '+str(polarity_prob[0].shape)+'\nincompatible with number of receivers:\n'+str(a_polarity_prob.shape[0]))
    if not isinstance(a_polarity_prob, bool) and a_polarity_prob.shape[0]>polarity_prob[1].shape[0]:
        raise IndexError('Bounds Exception for polarity probability observations.\nShape: '+str(polarity_prob[1].shape)+'\nincompatible with number of receivers:\n'+str(a_polarity_prob.shape[0]))
    #
    # Call loop functions
    #
    if not isinstance(a_polarity_prob, bool) and not isinstance(a_polarity, bool) and not isinstance(a1_amplitude_ratio, bool) and not isinstance(a2_amplitude_ratio, bool):
        if generate_mts:
            c_all_combined_ln_pdf_gen(&ln_P[0,0],a_polarity,&mt[0,0],wmax,error_polarity,a_polarity_prob,polarity_prob[0],polarity_prob[1],incorrect_polarity_prob,amplitude_ratio,a1_amplitude_ratio,a2_amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio,&n_tried,cutoff,&cut_ind,dc,marginalised,&ln_P_loc_samples[0],&location_samples_multiplier[0])   
        else:
            c_all_combined_ln_pdf(&ln_P[0,0],a_polarity,mt,error_polarity,a_polarity_prob,polarity_prob[0],polarity_prob[1],incorrect_polarity_prob,amplitude_ratio,a1_amplitude_ratio,a2_amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio,marginalised,&ln_P_loc_samples[0],&location_samples_multiplier[0])
    elif not isinstance(a_polarity_prob, bool) and not isinstance(a_polarity, bool):
        if generate_mts:
            c_combined_pol_ln_pdf_gen(&ln_P[0,0],a_polarity,&mt[0,0],wmax,error_polarity,a_polarity_prob,polarity_prob[0],polarity_prob[1],incorrect_polarity_prob,&n_tried,cutoff,&cut_ind,dc,marginalised,&ln_P_loc_samples[0],&location_samples_multiplier[0])   
        else:
            c_combined_pol_ln_pdf(&ln_P[0,0],a_polarity,mt,error_polarity,a_polarity_prob,polarity_prob[0],polarity_prob[1],incorrect_polarity_prob,marginalised,&ln_P_loc_samples[0],&location_samples_multiplier[0])
    elif not isinstance(a_polarity, bool) and not isinstance(a1_amplitude_ratio, bool) and not isinstance(a2_amplitude_ratio, bool):
        if generate_mts:
            c_polarity_ar_ln_pdf_gen(&ln_P[0,0],a_polarity,&mt[0,0],wmax,error_polarity,incorrect_polarity_prob,amplitude_ratio,a1_amplitude_ratio,a2_amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio,&n_tried,cutoff,&cut_ind,dc,marginalised,&ln_P_loc_samples[0],&location_samples_multiplier[0])   
        else:
            c_polarity_ar_ln_pdf(&ln_P[0,0],a_polarity,mt,error_polarity,incorrect_polarity_prob,amplitude_ratio,a1_amplitude_ratio,a2_amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio,marginalised,&ln_P_loc_samples[0],&location_samples_multiplier[0])
    elif not isinstance(a_polarity_prob, bool) and not isinstance(a1_amplitude_ratio, bool) and not isinstance(a2_amplitude_ratio, bool):
        if generate_mts:
            c_polarity_prob_combined_ln_pdf_gen(&ln_P[0,0],a_polarity_prob,&mt[0,0],wmax,polarity_prob[0],polarity_prob[1],incorrect_polarity_prob,amplitude_ratio,a1_amplitude_ratio,a2_amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio,&n_tried,cutoff,&cut_ind,dc,marginalised,&ln_P_loc_samples[0],&location_samples_multiplier[0])
        else:
            c_polarity_prob_combined_ln_pdf(&ln_P[0,0],a_polarity_prob,mt,polarity_prob[0],polarity_prob[1],incorrect_polarity_prob,amplitude_ratio,a1_amplitude_ratio,a2_amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio,marginalised,&ln_P_loc_samples[0],&location_samples_multiplier[0])
    elif not isinstance(a_polarity, bool):
        if generate_mts:
            c_polarity_ln_pdf_gen(&ln_P[0,0],a_polarity,&mt[0,0],wmax,error_polarity,incorrect_polarity_prob,&n_tried,cutoff,&cut_ind,dc,marginalised,&ln_P_loc_samples[0],&location_samples_multiplier[0])
        else:
            c_polarity_ln_pdf(&ln_P[0,0],a_polarity,mt,error_polarity,incorrect_polarity_prob,marginalised,&ln_P_loc_samples[0],&location_samples_multiplier[0])
    elif not isinstance(a_polarity_prob, bool):
        if generate_mts:
            c_polarity_probability_ln_pdf_gen(&ln_P[0,0],a_polarity_prob,&mt[0,0],wmax,polarity_prob[0],polarity_prob[1],incorrect_polarity_prob,&n_tried,cutoff,&cut_ind,dc,marginalised,&ln_P_loc_samples[0],&location_samples_multiplier[0])
        else:
            c_polarity_probability_ln_pdf(&ln_P[0,0],a_polarity_prob,mt,polarity_prob[0],polarity_prob[1],incorrect_polarity_prob,marginalised,&ln_P_loc_samples[0],&location_samples_multiplier[0])
    elif not isinstance(a1_amplitude_ratio, bool) and not isinstance(a2_amplitude_ratio, bool):
        if generate_mts:
            c_amplitude_ratio_ln_pdf_gen(&ln_P[0,0],amplitude_ratio,&mt[0,0],wmax,a1_amplitude_ratio,a2_amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio,&n_tried,cutoff,&cut_ind,dc,marginalised,&ln_P_loc_samples[0],&location_samples_multiplier[0])
        else:
            c_amplitude_ratio_ln_pdf(&ln_P[0,0],amplitude_ratio,mt,a1_amplitude_ratio,a2_amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio,marginalised,&ln_P_loc_samples[0],&location_samples_multiplier[0])
    if generate_samples:
        if cut_ind<0:
            # no non_zero samples
            return np.array([]),np.array([[],[],[],[],[],[]]),n_tried
        if cut_ind!=wmax:
            #ended prematurely
            ln_P=ln_P[:,:cut_ind]
            mt=mt[:,:cut_ind]
        return np.asarray(ln_P),np.asarray(mt),n_tried  
    return np.asarray(ln_P)

#
# Other functions 
#

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t combine_mu(DTYPE_t mu1,DTYPE_t mu2,DTYPE_t s1, DTYPE_t s2) nogil:
    return (mu1*s2*s2+mu2*s1*s1)/(s1*s1+s2*s2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t combine_s(DTYPE_t s1, DTYPE_t s2) nogil:
    return s1*s2/sqrt(s1*s1+s2*s2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void estimate_scale_mu_s(DTYPE_t *mu, DTYPE_t *s, DTYPE_t x,DTYPE_t y,DTYPE_t mux,DTYPE_t muy,DTYPE_t psx,DTYPE_t psy) nogil:
    cdef double sqrtpi=sqrt(pi)
    cdef DTYPE_t N,sx,sy,z,mu1,mux3,mux2,sy_2,sx_2,s1_2,mu1_2,s1,A,B,C

    z=fabs(x/y)
    mux=fabs(mux)
    muy=fabs(muy)
    sx=psx*mux
    sy=psy*muy
    sy_2=sy*sy
    sx_2=sx*sx
    mux2=mux*mux
    mux3=mux*mux2
    exp_muy=exp(-muy*muy/(2*sy_2))
    s1=sqrt((sy_2*z*z+sx_2)/(mux2))
    s1_2=s1*s1
    mu1=muy*z/mux
    mu1_2=mu1*mu1
    N=((sy_2*mux*z*mu1+sx_2*muy)/(mux3)+(sqrt2*sx_2*sy*exp_muy)/(sqrtpi*mux3*s1_2))
    A=(sy_2*mux*z*mu1)/(mux3)
    B=sx_2*muy/mux3
    C=sqrt2*sx_2*sy*exp_muy/(sqrtpi*mux3*s1_2)
    #CYTHON POINTER ASSIGNMENT
    mu[0]=(sy_2*mux*z*(s1_2+mu1_2)+sx_2*mu1*muy)/(mux3*N)
    s[0]=sqrt(((sy_2*(mu1_2*mu1+3*mu1*s1_2)*mux*z+sx_2*(s1_2+mu1_2)*muy)/mux3+sqrt2*sx_2*sx_2*sy*exp_muy/(sqrtpi*mux3*mux2*s1_2))/N-mu[0]*mu[0])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef scale_estimator(DTYPE_t[::1]  x,DTYPE_t[::1] y,   DTYPE_t [:,::1] mt1,DTYPE_t[:,::1]  mt2,DTYPE_t[:,:,::1] a,DTYPE_t[::1] psx,DTYPE_t[::1] psy):
    cdef Py_ssize_t umax=a.shape[0]
    cdef Py_ssize_t vmax=a.shape[1]
    cdef Py_ssize_t wmax=mt1.shape[1]
    cdef Py_ssize_t kmax=a.shape[2]#MT elementssamples
    cdef Py_ssize_t u,v,w,k
    cdef DTYPE_t mu1,s1
    cdef DTYPE_t[:,::1]  mu=np.empty((vmax,wmax)) 
    cdef DTYPE_t[:,::1]  s=np.empty((vmax,wmax)) 
    cdef DTYPE_t[::1] mux=np.empty((umax))
    cdef DTYPE_t[::1] muy=np.empty((umax))
    for v from 0<=v<vmax:
        for w from 0<=w<wmax:
            for u from 0<=u<umax:
                mux[u]=0
                muy[u]=0
                for k from 0<=k<kmax:# loop over num mt samples and make x
                    mux[u]+=a[u,v,k]*mt1[k,w]
                    muy[u]+=a[u,v,k]*mt2[k,w]
                estimate_scale_mu_s(&mu1, &s1, x[u],y[u],mux[u],muy[u],psx[u],psy[u])
                if u==0:
                    mu[v,w]=mu1
                    s[v,w]=s1
                else:
                    mu[v,w]=combine_mu(mu[v,w],mu1,s[v,w],s1)
                    s[v,w]=combine_s(s[v,w],s1)
    return np.asarray(mu),np.asarray(s)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef relative_amplitude_ratio_ln_pdf(DTYPE_t[::1]  x,DTYPE_t[::1] y,  DTYPE_t [:,::1] mt1,DTYPE_t[:,::1]  mt2,DTYPE_t[:,:,::1] a1,DTYPE_t[:,:,::1] a2,DTYPE_t[::1] psx,DTYPE_t[::1] psy):
    """
    Calculates Amplitude Ratio Probability

    Calculates the Ratio pdf (D. Hinkley, On the ratio of two correlated normal random variables, 1969, Biometrika vol 56 pp 635-639).
    Given Z=X/Y and means mux, muy and standard deviation sigmax and sigmay. The pdf is normalised.

    Cython handling for heavy lifting.

    """
    #cdefs    
    cdef Py_ssize_t umax=a1.shape[0]
    cdef Py_ssize_t vmax=a1.shape[1]
    cdef Py_ssize_t wmax=mt1.shape[1]
    cdef Py_ssize_t kmax=a1.shape[2]#MT elementssamples
    cdef Py_ssize_t u,v,w,k
    cdef DTYPE_t[:,:,::1]  ln_P=np.empty((umax,vmax,wmax)) 
    cdef DTYPE_t[:,::1]  mu=np.empty((vmax,wmax)) 
    cdef DTYPE_t[:,::1]  s=np.empty((vmax,wmax)) 
    cdef DTYPE_t mu1
    cdef DTYPE_t s1
    cdef DTYPE_t [::1]mux=np.empty((umax))
    cdef DTYPE_t [::1]muy=np.empty((umax))
    #First index is station, second is locaton sample, last is MT sample
    for v from 0<=v<vmax:
        for w from 0<=w<wmax:
            for u from 0<=u<umax:
                mux[u]=0
                muy[u]=0
                for k from 0<=k<kmax:# loop over num mt samples and make x
                    mux[u]+=a1[u,v,k]*mt1[k,w]
                    muy[u]+=a2[u,v,k]*mt2[k,w]
                estimate_scale_mu_s(&mu1, &s1, x[u],y[u],mux[u],muy[u],psx[u],psy[u])
                # print 's',s1
                if u==0:
                    mu[v,w]=mu1
                    s[v,w]=s1
                else:
                    mu[v,w]=combine_mu(mu[v,w],mu1,s[v,w],s1)
                    s[v,w]=combine_s(s[v,w],s1)
                # print mu[v,w],s[v,w]
            for u from 0<=u<umax:
                if np.isnan(s[v,w]):
                    ln_P[u,v,w]=-inf
                else:
                    ln_P[u,v,w]=log(ar_pdf(x[u]/y[u],mu[v,w]*mux[u],muy[u],psx[u],psy[u]))
    return np.asarray(ln_P),np.asarray(mu),np.asarray(s)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef ln_prod(DTYPE_t[:,:,::1] p):
    #cdefs
    cdef Py_ssize_t umax=p.shape[0]
    cdef Py_ssize_t vmax=p.shape[1]
    cdef Py_ssize_t wmax=p.shape[2]
    cdef DTYPE_t[:,::1] Ln_P=np.empty((vmax,wmax)) 
    cdef Py_ssize_t u,v,w
    for v from 0<=v<vmax:
        for w from 0<=w<wmax:
            Ln_P[v,w]=0
            for u from 0<=u<umax:
                Ln_P[v,w]+=(p[u,v,w])
    return np.asarray(Ln_P)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef ln_combine(DTYPE_t[:,::1] ln_p_1,DTYPE_t[:,::1] ln_p_2):
    assert  ln_p_1.shape[0] ==  ln_p_2.shape[0] and ln_p_1.shape[1] == ln_p_2.shape[1]
    cdef Py_ssize_t vmax=ln_p_1.shape[0]
    cdef Py_ssize_t wmax=ln_p_1.shape[1]
    cdef Py_ssize_t v,w
    for w from 0<=w<wmax:
        for v from 0<=v<vmax:
            ln_p_1[v,w]+=ln_p_2[v,w]
    return np.asarray(ln_p_1)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef ln_multipliers(DTYPE_t[:,::1] ln_p, DTYPE_t[::1] multipliers):
    assert  ln_p.shape[0] ==  multipliers.shape[0]
    cdef Py_ssize_t vmax=ln_p.shape[0]
    cdef Py_ssize_t wmax=ln_p.shape[1]
    cdef Py_ssize_t v,w
    for w from 0<=w<wmax:
        for v from 0<=v<vmax:
            ln_p[v,w]+=log(multipliers[v])
    return np.asarray(ln_p)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef ln_non_zero(DTYPE_t[:] ln_p):
    cdef Py_ssize_t vmax=ln_p.shape[0]
    cdef Py_ssize_t v
    cdef int[::1] non_zero=np.empty((vmax))
    for v from 0<=v<vmax:
        if ln_p[v]>-np.inf:
            non_zero[v]=1
        else:
            non_zero[v]=0
    return np.asarray(non_zero)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef ln_exp(DTYPE_t[::1] ln_p, DTYPE_t dV=1):
    cdef Py_ssize_t wmax=ln_p.shape[1]
    cdef Py_ssize_t w
    cdef DTYPE_t[::1] p =np.empty(wmax)
    for w from 0<=w<wmax:
        p[w]=exp(ln_p[w])
    return np.asarray(p)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void c_ln_normalise(DTYPE_t*ln_probability_p,DTYPE_t dV,Py_ssize_t n) nogil:
    cdef DTYPE_t norm=0.0
    cdef DTYPE_t max_ln_p=-inf
    for i from 0<=i<n:
        if ln_probability_p[i]>max_ln_p: 
            max_ln_p=ln_probability_p[i]
    for i from 0<=i<n:
        norm+=exp(ln_probability_p[i]-max_ln_p)
    norm*=dV
    for i from 0<=i<n:
        ln_probability_p[i]-=log(norm)+max_ln_p

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef DTYPE_t c_dkl(DTYPE_t*ln_probability_p,DTYPE_t*ln_probability_q,DTYPE_t dV,Py_ssize_t n) nogil:
    cdef DTYPE_t dkl=0.0
    cdef DTYPE_t p
    for i from 0<=i<n:
        if ln_probability_p[i]>-inf:#Check no zeros in p 
            p=exp(ln_probability_p[i])
            dkl+=p*ln_probability_p[i]-p*ln_probability_q[i]
    return dkl*dV

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef dkl(DTYPE_t[::1] ln_probability_p,DTYPE_t[::1] ln_probability_q,DTYPE_t dV=1.0):
    cdef Py_ssize_t n=ln_probability_p.shape[0]
    if ln_probability_q.shape[0]!=ln_probability_p.shape[0]:
        raise ValueError('Input array sizes must be the same shape')
    c_ln_normalise(&ln_probability_p[0],dV,n)
    c_ln_normalise(&ln_probability_q[0],dV,n)
    return c_dkl(&ln_probability_p[0],&ln_probability_q[0],dV,n) #Tested against MATLAB code

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef DTYPE_t c_dkl_uniform(DTYPE_t*ln_probability_p,DTYPE_t V,DTYPE_t dV,Py_ssize_t n) nogil:
    cdef DTYPE_t dkl=0.0
    cdef DTYPE_t p
    cdef DTYPE_t ln_V=log(V)
    for i from 0<=i<n:
        if ln_probability_p[i]>-inf:#Check no zeros in p 
            p=exp(ln_probability_p[i])
            dkl+=p*ln_probability_p[i]+p*ln_V
    return dkl*dV

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef dkl_uniform(DTYPE_t[::1] ln_probability_p,DTYPE_t V,DTYPE_t dV=1.0):
    cdef Py_ssize_t n=ln_probability_p.shape[0]
    c_ln_normalise(&ln_probability_p[0],dV,n)
    return c_dkl_uniform(&ln_probability_p[0],V,dV,n) #Tested against MATLAB code

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline DTYPE_t ranf() nogil:
    return <DTYPE_t>(rand()/RAND_MAX_D)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef DTYPE_t randn() nogil:
    return erf_inv(ranf())

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DTYPE_t[:,::1] randomn(int x,int y):
    cdef DTYPE_t[:,::1] R=np.empty((x,y))
    cdef Py_ssize_t i,j
    for i from 0<=i<x:
        for j from 0<=j<y:
            R[i,j]=randn()
    return R

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef DTYPE_t[:,::1] rand_mt(int x):
    np.random.seed()
    cdef DTYPE_t[:,::1] R=np.random.randn(6,x)
    cdef Py_ssize_t i,j
    cdef DTYPE_t N=0
    for j from 0<=j<x:
        N=0
        for i from 0<=i<6:
            N+=R[i,j]*R[i,j]
        N=sqrt(N)
        for i from 0<=i<6:
            R[i,j]/=N
    return R

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef DTYPE_t[:,::1] rand_dc(int x):
    cdef DTYPE_t[:,::1] mt=np.empty((6,x))
    cdef Py_ssize_t sj
    cdef DTYPE_t ck
    cdef DTYPE_t cs
    cdef DTYPE_t sk
    cdef DTYPE_t s2k
    cdef DTYPE_t ss
    cdef DTYPE_t sh
    for j from 0<=j<x:
        # h=ranf()
        # kappa=2*pi*ranf()
        # sigma=pi*(ranf()-0.5)
        h=np.random.rand()
        kappa=2*pi*np.random.rand()
        sigma=pi*(np.random.rand()-0.5)
        ck=cos(kappa)
        cs=cos(sigma)
        sk=sin(kappa)
        s2k=sin(2*kappa)
        ss=sin(sigma)
        sh=sqrt(1-h*h)
        mt[0,j]=-sh*cs*s2k-2*h*sh*ss*sk*sk
        mt[1,j]=sh*cs*2*sk*ck-2*sh*h*ss*ck*ck
        mt[2,j]=2*sh*h*ss
        mt[3,j]=sqrt2*(sh*cs*(ck*ck-sk*sk)+h*sh*ss*s2k)
        mt[4,j]=sqrt2*(-h*cs*ck-(2*h*h-1)*ss*sk)
        mt[5,j]=sqrt2*(-h*cs*sk+(2*h*h-1)*ss*ck)
    return mt

cpdef random_mt(int x):
    return np.asarray(rand_mt(x))

cpdef random_dc(int x):
    return np.asarray(rand_dc(x))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef DTYPE_t[::1] get_angle_coeff_multipliers(DTYPE_t[:,:,::1] a_polarity,DTYPE_t[:,:,::1] a1_amplitude_ratio,DTYPE_t[:,:,::1] a2_amplitude_ratio,DTYPE_t[:,:,::1] a_polarity_prob,DTYPE_t[::1] multipliers,DTYPE_t epsilon) nogil:
    cdef Py_ssize_t nsamples
    cdef Py_ssize_t nsta
    cdef Py_ssize_t kmax
    cdef int polarity=0
    cdef int amplitude_ratio=0
    cdef int polarity_prob=0
    if a_polarity.shape[0]>0:
        nsta=a_polarity.shape[0]
        nsamples=a_polarity.shape[1]
        kmax=a_polarity.shape[2]
        pol=1
    if a1_amplitude_ratio.shape[0]>0:
        if pol==0:
            nsta=a1_amplitude_ratio.shape[0]
            nsamples=a1_amplitude_ratio.shape[1]
            kmax=a1_amplitude_ratio.shape[2]
        amplitude_ratio=1
    if a_polarity_prob.shape[0]>0:
        if pol==0:
            if amplitude_ratio==0:
                nsta=a_polarity_prob.shape[0]
                nsamples=a_polarity_prob.shape[1]
                kmax=a_polarity_prob.shape[2]
        polarity_prob=1
    cdef Py_ssize_t u
    cdef Py_ssize_t w
    cdef int ok=1
    for u in range(nsamples):
        if multipliers[u]==-1:
            continue
        for v in range(u,nsamples):  
            if multipliers[v]>-1:
                ok=1
                for w in range(nsta):
                    for k in range(kmax):
                        if pol>0:
                            ok*=(fabs(a_polarity[w,u,k]-a_polarity[w,v,k])<epsilon)
                        if polarity_prob>0:
                            ok*=(fabs(a_polarity_prob[w,u,k]-a_polarity_prob[w,v,k])<epsilon)
                        if amplitude_ratio>0:
                            ok*=(fabs(a1_amplitude_ratio[w,u,k]-a1_amplitude_ratio[w,v,k])<epsilon)*(fabs(a2_amplitude_ratio[w,u,k]-a2_amplitude_ratio[w,v,k])<epsilon)
                    if ok==0:
                        break
                if w==nsta-1 and ok>0:
                    multipliers[u]+=multipliers[v]
                    if v>u:
                        multipliers[v]=-1
    return multipliers

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef bin_angle_coefficient_samples(a_polarity,a1_amplitude_ratio,a2_amplitude_ratio,a_polarity_prob,location_sample_multipliers,ext_data,epsilon):
    cdef DTYPE_t[::1] multipliers=np.array(location_sample_multipliers)
    new_records=[]
    new_multipliers=[]
    polarity=False
    amplitude_ratio=False
    polarity_prob=False
    if not isinstance(a_polarity, bool):
        polarity=True
    else:
        a_polarity=np.empty((0,0,0))
    if not isinstance(a1_amplitude_ratio, bool) and not isinstance(a2_amplitude_ratio, bool):
        amplitude_ratio=True
    else:
        a1_amplitude_ratio=np.empty((0,0,0))
        a2_amplitude_ratio=np.empty((0,0,0))
    if not isinstance(a_polarity_prob, bool):
        polarity_prob=True
    else:
        a_polarity_prob=np.empty((0,0,0))
    # print multipliers,bin_size,angles
    multipliers=np.asarray(get_angle_coeff_multipliers(a_polarity,a1_amplitude_ratio,a2_amplitude_ratio,a_polarity_prob,np.array(location_sample_multipliers,dtype=np.longlong),epsilon)).flatten()
    cdef Py_ssize_t i
    for i,record in enumerate(multipliers):
        if multipliers[i]>0:
            new_multipliers.append(multipliers[i])
    if polarity:
        a_polarity=np.ascontiguousarray(a_polarity[:,np.asarray(multipliers)>0,:])
    else:
        a_polarity=False
    if amplitude_ratio:
        a1_amplitude_ratio=np.ascontiguousarray(a1_amplitude_ratio[:,np.asarray(multipliers)>0,:])
        a2_amplitude_ratio=np.ascontiguousarray(a2_amplitude_ratio[:,np.asarray(multipliers)>0,:])
    else:
        a1_amplitude_ratio=False
        a2_amplitude_ratio=False
    if  polarity_prob:
        a_polarity_prob=np.ascontiguousarray(a_polarity_prob[:,np.asarray(multipliers)>0,:])
    else:
        a_polarity_prob=False
    for key in ext_data.keys():
        for k in ext_data[key].keys():
            if k[:2]=='a_' or k[0]=='a' and k[2]=='_':
                ext_data[key][k]=np.ascontiguousarray(ext_data[key][k][:,np.asarray(multipliers)>0,:])
    return a_polarity,a1_amplitude_ratio,a2_amplitude_ratio,a_polarity_prob,ext_data,new_multipliers

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef relative_amplitude_loop(DTYPE_t[:,::1] MT1,DTYPE_t[:,::1] MT2,DTYPE_t[:,:,::1] a1,DTYPE_t[:,:,::1] a2,DTYPE_t[::1]  mu1,DTYPE_t[::1]  mu2,DTYPE_t[::1] ps1,DTYPE_t[::1] ps2):
    cdef Py_ssize_t w2max=MT1.shape[1]
    cdef Py_ssize_t wmax=MT2.shape[1]
    cdef Py_ssize_t i=0
    if wmax!=w2max:
        raise ValueError('Incorrect sizes')
    cdef Py_ssize_t umax=a1.shape[0]#Location Sample
    cdef Py_ssize_t vmax=a1.shape[1]#Location Sample
    cdef Py_ssize_t kmax=a1.shape[2]#MT elementssamples
    cdef DTYPE_t[:,::1] ln_P = np.zeros((umax,wmax))  
    cdef Py_ssize_t u,v,w,k
    cdef DTYPE_t x1=0.0
    cdef DTYPE_t x2=0.0
    cdef DTYPE_t[:,::1]  mu=np.empty((umax,wmax)) 
    cdef DTYPE_t[:,::1]  s=np.empty((umax,wmax)) 
    cdef DTYPE_t mui
    cdef DTYPE_t si
    for u from 0<=u<umax:
        for w from 0<=w<wmax:
            ln_P[u,w]=0.0
            for v from 0<=v<vmax:
                x1=0.
                x2=0.
                for k from 0<=k<kmax:# loop over num mt samples and make x
                    x1+=a1[u,v,k]*MT1[k,w]
                    x2+=a2[u,v,k]*MT2[k,w]
                estimate_scale_mu_s(&mui, &si, mu1[v],mu2[v],x1,x2,ps1[v],ps2[v])
                if u==0:
                    mu[u,w]=mui
                    s[u,w]=si
                else:
                    mu[u,w]=combine_mu(mu[u,w],mui,s[u,w],si)
                    s[u,w]=combine_s(s[u,w],si)
                #First index is station, second is locaton sample, last is MT sample
                ln_P[u,w]+=ar_pdf(mu1[v]/mu2[v],mu[u,w]*x1,x2,ps1[v],ps2[v])
            ln_P[u,w]=log(ln_P[u,w])
    return np.asarray(ln_P),np.asarray(mu),np.asarray(s)

# Test functions

class cProbabilityTestCase(unittest.TestCase):
    def test_erf(self):
        from scipy.special import erf as scipy_erf
        self.assertAlmostEqual(scipy_erf(0.4), erf(0.4), 5)
        self.assertAlmostEqual(scipy_erf(0.1), erf(0.1), 5)
        self.assertAlmostEqual(scipy_erf(0.2), erf(0.2), 5)
        self.assertAlmostEqual(scipy_erf(-0.4), erf(-0.4), 5)
        self.assertAlmostEqual(scipy_erf(0.3), erf(0.3), 5)
        self.assertAlmostEqual(scipy_erf(0.9), erf(0.9), 5)
        self.assertAlmostEqual(scipy_erf(-0.6), erf(-0.6), 5)

    def test_ar_pdf(self):
        from MTfit.probability.probability import ratio_pdf
        c_p = ar_pdf(4.1, 0.2, 2.5, 0.1, 0.5)
        py_p = ratio_pdf(4.1, 0.2, 2.5, 4.1*0.1, 2.5*0.5)
        self.assertAlmostEqual(c_p, py_p, 3)
        c_p = ar_pdf(0.4, 0.3, 0.5, 0.4, 0.5)
        py_p = ratio_pdf(0.4, 0.3, 0.5, 0.4*0.3, 0.5*0.5)
        self.assertAlmostEqual(c_p, py_p, 2)
        c_p = ar_pdf(0.4, 0.2, 0.5, 0.4, 0.5)
        py_p = ratio_pdf(0.4, 0.2, 0.5, 0.4*0.2, 0.5*0.5)
        self.assertAlmostEqual(c_p, py_p, 1)

    def test_dkl(self):
        x=np.linspace(0,10,100)
        from scipy.stats import norm as normalDist
        p=normalDist.pdf(x,3.0,0.1)
        q=normalDist.pdf(x,6.0,0.5)
        ln_p=np.log(p)
        ln_q=np.log(q)
        assert(abs(dkl(np.ascontiguousarray(ln_p),np.ascontiguousarray(ln_q),0.1)-19.129437813140147)<0.000000001)
        p=normalDist.pdf(x,3.0,0.5)
        ln_p=np.log(p)
        assert(abs(dkl(np.ascontiguousarray(ln_p),np.ascontiguousarray(ln_q),0.1)-17.99999998188251)<0.000000001)
        p=normalDist.pdf(x,3.0,0.1)
        q=np.ones(x.shape)/10.
        ln_p=np.log(p)
        ln_q=np.log(q)
        assert(abs(dkl(np.ascontiguousarray(ln_p),np.ascontiguousarray(ln_q),0.1)- 3.196281943707682)<0.000000001)
    def test_dkl_uniform(self):
        x=np.linspace(0,10,100)
        from scipy.stats import norm as normalDist
        p=normalDist.pdf(x,3.0,0.1)
        ln_p=np.log(p)
        assert(abs(dkl_uniform(np.ascontiguousarray(ln_p),10,0.1)- 3.196281943707682)<0.000000001)
        n=100000
        x=10*np.random.rand(n)
        p=normalDist.pdf(x,3.0,0.1)
        ln_p=np.log(p)
        assert(abs(dkl_uniform(np.ascontiguousarray(ln_p),10.,10./n)-3.15)<2.0)
        assert(abs(dkl_uniform(np.ascontiguousarray(ln_p),10,10./n)-dkl(ln_p,np.log(np.ones(x.shape)/10.),10./n))<0.000001)
