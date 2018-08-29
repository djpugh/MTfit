"""
moment_tensor_conversion.py
***************************

Module containing moment tensor conversion functions. Acts on the parameters of the moment tensor 3x3 form or the modified 6-vector form, dependent on the name

The function naming is OriginalVariables_NewVariables

The coordinate system is North (X), East (Y), Down (Z)
"""


# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.
import logging

import numpy as np
from scipy.optimize import fsolve

from ..utilities import C_EXTENSION_FALLBACK_LOG_MSG

logger = logging.getLogger('MTfit.convert')


try:
    from . import cmoment_tensor_conversion
except ImportError:
    cmoment_tensor_conversion = None
except Exception:
    logger.exception('Error importing c extension')
    cmoment_tensor_conversion = None


def MT33_MT6(MT33):
    """
    Convert a 3x3 matrix to six vector maintaining normalisation. 6-vector has the form::

        Mxx
        Myy
        Mzz
        sqrt(2)*Mxy
        sqrt(2)*Mxz
        sqrt(2)*Myz

    Args
        M33: 3x3 numpy matrix

    Returns
        numpy.matrix: MT 6-vector

    """
    MT6 = np.matrix([[MT33[0, 0]], [MT33[1, 1]], [MT33[2, 2]], [np.sqrt(2)*MT33[0, 1]],
                     [np.sqrt(2)*MT33[0, 2]], [np.sqrt(2)*MT33[1, 2]]])
    MT6 = np.matrix(MT6/np.sqrt(np.sum(np.multiply(MT6, MT6), axis=0)))
    return MT6


def MT6_MT33(MT6):
    """
    Convert a six vector to a 3x3 MT maintaining normalisation. 6-vector has the form::

        Mxx
        Myy
        Mzz
        sqrt(2)*Mxy
        sqrt(2)*Mxz
        sqrt(2)*Myz

    Args
        MT6: numpy matrix Moment tensor 6-vector

    Returns
        numpy.matrix: 3x3 Moment Tensor
    """
    if np.prod(MT6.shape) != 6:
        raise ValueError("Input MT must be 6 vector not {}".format(MT6.shape))
    if len(MT6.shape) == 1:
        MT6 = np.matrix(MT6)
    if len(MT6.shape) == 2 and MT6.shape[1] == 6:
        MT6 = MT6.T
    return np.matrix([[MT6[0, 0], (1/np.sqrt(2))*MT6[3, 0], (1/np.sqrt(2))*MT6[4, 0]],
                      [(1/np.sqrt(2))*MT6[3, 0], MT6[1, 0],
                       (1/np.sqrt(2))*MT6[5, 0]],
                      [(1/np.sqrt(2))*MT6[4, 0], (1/np.sqrt(2))*MT6[5, 0], MT6[2, 0]]])


def MT6_TNPE(MT6):
    """
    Convert the 6xn Moment Tensor to the T,N,P vectors and the eigenvalues.

    Args
        MT6: 6xn numpy matrix

    Returns
        (numpy.matrix, numpy.matrix, numpy.matrix, numpy.array): tuple of T, N, P
                        vectors and Eigenvalue array
    """
    if cmoment_tensor_conversion:
        try:
                if not isinstance(MT6, np.ndarray):
                    MT6 = np.array(MT6)
                if MT6.ndim < 2:
                    MT6 = np.array([MT6])
                if MT6.shape[1] == 6 and MT6.shape[0] != 6:
                    MT6 = MT6.T
                return cmoment_tensor_conversion.MT6_TNPE(MT6.astype(np.float64))
        except Exception:
            logger.exception('Error with C extension')
    else:
        logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
    try:
        n = MT6.shape[1]
    except Exception:
        MT6 = np.array([MT6]).T
        n = MT6.shape[1]
    T = np.matrix(np.empty((3, n)))
    N = np.matrix(np.empty((3, n)))
    P = np.matrix(np.empty((3, n)))
    E = np.empty((3, n))
    for i in range(n):
        T[:, i], N[:, i], P[:, i], E[:, i] = MT33_TNPE(MT6_MT33(MT6[:, i]))
    return T, N, P, E


def MT6_Tape(MT6):
    """
    Convert the moment tensor 6-vector to the Tape parameters.

    6-vector has the form::

        Mxx
        Myy
        Mzz
        sqrt(2)*Mxy
        sqrt(2)*Mxz
        sqrt(2)*Myz

    Args
        MT6: numpy matrix six-vector

    Returns
        (numpy.array, numpy.array, numpy.array, numpy.array, numpy.array): tuple of
                        gamma, delta, strike, cos(dip) and slip (angles in radians)

    """
    if len(MT6.shape) > 1 and MT6.shape[1] > 1:
        gamma = np.array(np.empty((MT6.shape[1],)))
        delta = np.array(np.empty((MT6.shape[1],)))
        kappa = np.array(np.empty((MT6.shape[1],)))
        h = np.array(np.empty((MT6.shape[1],)))
        sigma = np.array(np.empty((MT6.shape[1],)))
        for i in range(MT6.shape[1]):
            gamma[i], delta[i], kappa[i], h[i], sigma[i] = MT6_Tape(MT6[:, i])
        return gamma, delta, kappa, h, sigma
    MT33 = MT6_MT33(MT6)
    T, N, P, E = MT33_TNPE(MT33)
    gamma, delta = E_GD(E)
    kappa, dip, sigma = TNP_SDR(T, N, P)
    if np.abs(sigma) > np.pi/2:
        kappa, dip, sigma = SDR_SDR(kappa, dip, sigma)
    h = np.cos(dip)
    return (np.array(gamma).flatten(), np.array(delta).flatten(),
            np.array(kappa).flatten(), np.array(h).flatten(),
            np.array(sigma).flatten())


def MT33_TNPE(MT33):
    """
    Convert the 3x3 Moment Tensor to the T,N,P vectors and the eigenvalues.

    Args
        MT33: 3x3 numpy matrix

    Returns
        (numpy.matrix, numpy.matrix, numpy.matrix, numpy.array): tuple of T, N, P
                        vectors and Eigenvalue array
    """
    E, L = np.linalg.eig(MT33)
    idx = E.argsort()[::-1]
    E = E[idx]
    L = L[:, idx]
    T = L[:, 0]
    P = L[:, 2]
    N = L[:, 1]
    return (T, N, P, E)


def MT33_SDR(MT33):
    """
    Convert the 3x3 Moment Tensor to the strike, dip and rake.

    Args
        MT33: 3x3 numpy matrix

    Returns
        (float, float, float): tuple of strike, dip, rake angles in radians
    """
    T, N, P, E = MT33_TNPE(MT33)
    N1, N2 = TP_FP(T, P)
    return FP_SDR(N1, N2)


def MT33_GD(MT33):
    """
    Convert the 3x3 Moment Tensor to theTape parameterisation gamma and delta.

    Args
        MT33: 3x3 numpy matrix

    Returns
        (numpy.array, numpy.array): tuple of gamma, delta
    """
    E, L = np.linalg.eig(MT33)
    return E_GD(E)


def E_tk(E):
    """
    Convert the moment tensor eigenvalues to the Hudson tau, k parameters

    Args
        E: indexable list/array (e.g numpy.array) of moment tensor eigenvalues

    Returns
        (float, float): tau, k  tuple
    """
    if isinstance(E, np.ndarray):
        E = np.squeeze(np.array(E))
    if len(E.shape) > 1 and E.shape[1] > 1:
        tau = np.empty(E.shape[1])
        k = np.empty(E.shape[1])
        for i in range(E.shape[1]):
            tau[i], k[i] = E_tk(E[:, i])
    else:
        idx = [0, 2, 1]  # Odd sorting
        E = E[idx]
        iso = (E[0]+E[1]+E[2])/3
        dev0 = E[0]-iso
        dev1 = E[1]-iso
        dev2 = E[2]-iso
        if dev2 > 0:
            k = iso/(abs(iso)-dev1)
            T = -2*(dev2)/(dev1)
        elif dev2 < 0:
            k = iso/(abs(iso)+dev0)
            T = 2*dev2/dev0
        else:
            k = iso/(abs(iso)+dev0)
            T = 0
        tau = T*(1-abs(k))
    return tau, k


def tk_uv(tau, k):
    """
    Convert the Hudson tau, k parameters to the Hudson u, v parameters

    Args
        tau: float, Hudson tau parameter
        k: float, Hudson k parameter

    Returns
        (float, float): u, v tuple
    """
    try:
        u = tau.copy()
        v = k.copy()
        idx = (tau > 0)*(k > 0)*(tau < 4*k)
        u[idx] = tau[idx]/(1-(tau[idx]/2))
        v[idx] = k[idx]/(1-(tau[idx]/2))
        idx = (tau > 0)*(k > 0)*~(tau < 4*k)
        u[idx] = tau[idx]/(1-2*k[idx])
        v[idx] = k[idx]/(1-2*k[idx])
        idx = (tau < 0)*(k < 0)*(tau > 4*k)
        u[idx] = tau[idx]/(1+(tau[idx]/2))
        v[idx] = k[idx]/(1+(tau[idx]/2))
        idx = (tau < 0)*(k < 0)*~(tau > 4*k)
        u[idx] = tau[idx]/(1+2*k[idx])
        v[idx] = k[idx]/(1+2*k[idx])
    except Exception:
        if tau > 0 and k > 0:
            if tau < 4*k:
                u = tau/(1-(tau/2))
                v = k/(1-(tau/2))
            else:
                u = tau/(1-2*k)
                v = k/(1-2*k)
        elif tau < 0 and k < 0:
            if tau > 4*k:
                u = tau/(1+(tau/2))
                v = k/(1+(tau/2))
            else:
                u = tau/(1+2*k)
                v = k/(1+2*k)
        else:
            u = tau
            v = k
    return u, v


def E_uv(E):
    """
    Convert the eigenvalues to the Hudson i, v parameters

    Args
        E: indexable list/array (e.g numpy.array) of moment tensor eigenvalues

    Returns
        (float, float): u, v  tuple
    """
    return tk_uv(*E_tk(E))


def E_GD(E):
    """
    Convert the eigenvalues to the Tape parameterisation gamma and delta.

    Args
        E: array of eigenvalues

    Returns
        (numpy.array, numpy.array): tuple of gamma, delta

    """
    if cmoment_tensor_conversion:
        try:
            # Expect E to be an array of arrays
            if E.ndim < 2:
                E = np.array([E])
            if E.shape[1] == 3 and E.shape != (3, 3):
                E = E.T
            return cmoment_tensor_conversion.E_GD(E.astype(np.float64))
        except Exception:
            logger.exception('Error with C extension')
    else:
        logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
    if isinstance(E, np.ndarray):
        E = np.squeeze(np.array(E))
    if len(E.shape) > 1 and E.shape[1] > 1:
        gamma = np.empty(E.shape[1])
        delta = np.empty(E.shape[1])
        for i in range(E.shape[1]):
            gamma[i], delta[i] = E_GD(E[:, i])
    else:
        if len(np.unique(E)) == 1:
            return 0, np.sign(np.unique(E))*np.pi/2
        idx = E.argsort()[::-1]
        E = E[idx]
        gamma = np.arctan2(-E[0]+2*E[1]-E[2], np.sqrt(3)*(E[0]-E[2]))
        beta = np.arccos(np.sum(E)/(np.sqrt(3)*np.sqrt(np.sum(E*E))))
        delta = np.real(np.pi/2-beta)
    return gamma, delta


def GD_basic_cdc(gamma, delta):
    """
    Convert gamma, delta to basic crack+double-couple parameters

    Gamma and delta are the source type parameters from the Tape parameterisation.

    Args
        gamma: numpy array of gamma values
        delta: numpy array of delta values

    Returns:
        (numpy.array, numpy.array): tuple of alpha, poisson
    """

    alpha = np.arccos(-np.sqrt(3)*np.tan(gamma))
    poisson = (1+np.sqrt(2) * (np.tan((np.pi/2)-delta) * np.sin(gamma)))
    poisson /= (2-(np.sqrt(2) * (np.tan((np.pi/2)-delta) * np.sin(gamma))))
    return alpha, poisson


def TNP_SDR(T, N, P):
    """
    Convert the T,N,P vectors to the strike, dip and rake in radians

    Args
        T: numpy matrix of T vectors.
        N: numpy matrix of N vectors.
        P: numpy matrix of P vectors.

    Returns
        (float, float, float): tuple of strike, dip and rake angles of fault plane in radians

    """
    if cmoment_tensor_conversion:
        try:
            return cmoment_tensor_conversion.TP_SDR(T.astype(np.float64), P.astype(np.float64))
        except Exception:
            logger.exception('Error with C extension')
    else:
        logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
    N1, N2 = TP_FP(T, P)
    return FP_SDR(N1, N2)


def TP_FP(T, P):
    """
    Convert the 3x3 Moment Tensor to the fault normal and slip vectors.

    Args
        T: numpy matrix of T vectors.
        P: numpy matrix of P vectors.

    Returns
        (numpy.matrix, numpy.matrix): tuple of Normal and slip vectors
    """
    if T.ndim == 1:
        T = np.matrix(T)
    if P.ndim == 1:
        P = np.matrix(P)
    if T.shape[0] != 3:
        T = T.T
    if P.shape[0] != 3:
        P = P.T
    TP1 = T+P
    TP2 = T-P
    N1 = (TP1)/np.sqrt(np.einsum('ij,ij->j', TP1, TP1))
    N2 = (TP2)/np.sqrt(np.einsum('ij,ij->j', TP2, TP2))
    return (N1, N2)


def FP_SDR(normal, slip):
    """
    Convert fault normal and slip to strike, dip and rake

    Coordinate system is North East Down.

    Args
        normal: numpy matrix - Normal vector
        slip: numpy matrix - Slip vector


    Returns
        (float, float, float): tuple of strike, dip and rake angles in radians

    """
    if not isinstance(slip, np.matrixlib.defmatrix.matrix):
        slip = slip/np.sqrt(np.sum(slip*slip, axis=0))
    else:
        # Do we need to replace this with einsum
        slip = slip/np.sqrt(np.einsum('ij,ij->j', slip, slip))
    if not isinstance(normal, np.matrixlib.defmatrix.matrix):
        normal = normal/np.sqrt(np.sum(normal*normal, axis=0))
    else:
        normal = normal/np.sqrt(np.einsum('ij,ij->j', normal, normal))
    slip[:, np.array(normal[2, :] > 0).flatten()] *= -1
    normal[:, np.array(normal[2, :] > 0).flatten()] *= -1
    normal = np.array(normal)
    slip = np.array(slip)
    strike, dip = normal_SD(normal)
    rake = np.arctan2(-slip[2], slip[0]*normal[1]-slip[1]*normal[0])
    strike[dip > np.pi/2] += np.pi
    rake[dip > np.pi/2] = 2*np.pi-rake[dip > np.pi/2]
    dip[dip > np.pi/2] = np.pi-dip[dip > np.pi/2]
    strike = np.mod(strike, 2*np.pi)
    rake[rake > np.pi] -= 2*np.pi
    rake[rake < -np.pi] += 2*np.pi
    return (np.array(strike).flatten(), np.array(dip).flatten(), np.array(rake).flatten())


def basic_cdc_GD(alpha, poisson=0.25):
    """
    Convert alpha and poisson ratio to gamma and delta

    alpha is opening angle, poisson : ratio lambda/(2(lambda+mu)) Defaults to 0.25.
    Uses basic crack+double-couple model of Minson et al (Seismically and geodetically
    determined nondouble-couple source mechanisms from the 2000 Miyakejima volcanic
    earthquake swarm, Minson et al, 2007, JGR 112) and Tape and Tape (The classical
    model for moment tensors, Tape and Tape, 2013, GJI)

    Args
        alpha: Opening angle in radians (between 0 and pi/2)
        poisson:[0.25] Poisson ratio on the fault surface.

    Returns:
        (numpy.array, numpy.array): tuple of gamma, delta

    """
    gamma = np.arctan((-1/np.sqrt(3.))*np.cos(alpha))
    beta = np.arccos(np.sqrt(2./3)*np.cos(alpha)*(1.+poisson) /
                     np.sqrt(((1.-2.*poisson)*(1.-2.*poisson)) +
                             np.cos(alpha)*np.cos(alpha)*(1.+2.*poisson*poisson)))
    try:
        # fix for odd cos division problems
        beta[alpha == np.pi/2] = np.pi/2
    except Exception:
        if alpha == np.pi/2:
            beta = np.pi/2
    delta = np.pi/2-beta
    return (gamma, delta)


def GD_E(gamma, delta):
    """
    Convert the Tape parameterisation gamma and delta to the eigenvalues.


    Args
        gamma: numpy array of gamma values
        delta: numpy array of delta values

    Returns
        numpy.array: array of eigenvalues

    """
    U = (1/np.sqrt(6))*np.matrix([[np.sqrt(3), 0, -np.sqrt(3)],
                                  [-1, 2, -1],
                                  [np.sqrt(2), np.sqrt(2), np.sqrt(2)]])
    gamma = np.array(gamma).flatten()
    delta = np.array(delta).flatten()
    X = np.matrix([np.multiply(np.cos(gamma), np.sin((np.pi/2)-delta)),
                   np.multiply(np.sin(gamma), np.sin((np.pi/2)-delta)),
                   np.cos((np.pi/2)-delta)]).T
    if X.shape[1] == 3:
        X = X.T
    return np.array((U.T*X))


def SDR_TNP(strike, dip, rake):
    """
    Convert strike, dip  rake to TNP vectors

    Coordinate system is North East Down.

    Args
        strike: float radians
        dip: float radians
        rake: float radians

    Returns
        (numpy.matrix, numpy.matrix, numpy.matrix): tuple of T,N,P vectors.

    """
    strike = np.array(strike).flatten()
    dip = np.array(dip).flatten()
    rake = np.array(rake).flatten()
    N1 = np.matrix([(np.cos(strike)*np.cos(rake))+(np.sin(strike)*np.cos(dip)*np.sin(rake)),
                    (np.sin(strike)*np.cos(rake)) -
                    np.cos(strike)*np.cos(dip)*np.sin(rake),
                    -np.sin(dip)*np.sin(rake)])
    N2 = np.matrix([-np.sin(strike)*np.sin(dip), np.cos(strike)*np.sin(dip), -np.cos(dip)])
    return FP_TNP(N1, N2)


def SDR_SDR(strike, dip, rake):
    """
    Convert strike, dip  rake to strike, dip  rake for other fault plane

    Coordinate system is North East Down.

    Args
        strike: float radians
        dip: float radians
        rake: float radians

    Returns
        (float, float, float): tuple of strike, dip and rake angles of alternate fault
                        plane in radians

    """
    if cmoment_tensor_conversion:
        try:
            return cmoment_tensor_conversion.SDR_SDR(strike.astype(np.float64), dip.astype(np.float64), rake.astype(np.float64))
        except Exception:
            logger.exception('Error with C extension')
    else:
        logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
    # Handle multiple inputs
    N1, N2 = SDR_FP(strike, dip, rake)
    s1, d1, r1 = FP_SDR(N1, N2)
    s2, d2, r2 = FP_SDR(N2, N1)
    # This should be ok to return s2,d2,r2 but doesn't seem to work
    try:
        r2[np.abs(strike-s2) < 1] = r1[np.abs(strike-s2) < 1]
        d2[np.abs(strike-s2) < 1] = d1[np.abs(strike-s2) < 1]
        s2[np.abs(strike-s2) < 1] = s1[np.abs(strike-s2) < 1]
        return s2, d2, r2
    except Exception:
        if np.abs(strike-s1) < 1:
            return (s2, d2, r2)
        else:
            return (s1, d1, r1)


def FP_TNP(normal, slip):
    """
    Convert fault normal and slip to TNP axes

    Coordinate system is North East Down.

    Args
        normal: numpy matrix - normal vector
        slip: numpy matrix - slip vector

    Returns
        (numpy.matrix, numpy.matrix, numpy.matrix): tuple of T, N, P vectors
    """
    T = (normal+slip)
    T = T/np.sqrt(np.einsum('ij,ij->j', T, T))
    P = (normal-slip)
    P = P/np.sqrt(np.einsum('ij,ij->j', P, P))
    N = np.matrix(-np.cross(T.T, P.T)).T
    return (T, N, P)


def SDSD_FP(strike1, dip1, strike2, dip2):
    """
    Convert strike and dip pairs to fault normal and slip

    Converts the strike and dip pairs in radians to the fault normal and slip.

    Args
        strike1: float strike angle of fault plane 1 in radians
        dip1: float dip angle of fault plane 1 in radians
        strike2: float strike angle of fault plane 2 in radians
        dip2: float dip  of fault plane 2 in radians

    Returns
        (numpy.matrix, numpy.matrix): tuple of Normal and slip vectors
    """
    strike1 = np.array(strike1).flatten()
    dip1 = np.array(dip1).flatten()
    strike2 = np.array(strike2).flatten()
    dip2 = np.array(dip2).flatten()
    N1 = np.matrix([-np.sin(strike2)*np.sin(dip2),
                    np.cos(strike2)*np.sin(dip2),
                    -np.cos(dip2)])
    N2 = np.matrix([-np.sin(strike1)*np.sin(dip1),
                    np.cos(strike1)*np.sin(dip1),
                    -np.cos(dip1)])
    return (N1, N2)


def SDR_FP(strike, dip, rake):
    """
    Convert the strike, dip  and rake in radians to the fault normal and slip.

    Args
        strike: float strike angle of fault plane  in radians
        dip: float dip angle of fault plane  in radians
        rake: float rake angle of fault plane  in radians

    Returns
        (numpy.matrix, numpy.matrix): tuple of Normal and slip vectors
    """
    T, N, P = SDR_TNP(strike, dip, rake)
    return TP_FP(T, P)


def SDR_SDSD(strike, dip, rake):
    """
    Convert the strike, dip  and rake to the strike and dip pairs (all angles in radians).

    Args
        strike: float strike angle of fault plane  in radians
        dip: float dip angle of fault plane  in radians
        rake: float rake angle of fault plane  in radians

    Returns
        (float, float, float, float): tuple of strike1, dip1, strike2, dip2 angles in radians
    """
    N1, N2 = SDR_FP(strike, dip, rake)
    return FP_SDSD(N1, N2)


def FP_SDSD(N1, N2):
    """
    Convert the the fault normal and slip vectors to the strike and dip pairs
    (all angles in radians).

    Args
        Normal: numpy matrix - Normal vector
        Slip: numpy matrix - Slip vector

    Returns
        (float, float, float, float): tuple of strike1, dip1, strike2, dip2 angles
                        in radians
    """
    s1, d1 = normal_SD(N1)
    s2, d2 = normal_SD(N2)
    return (s1, d1, s2, d2)


def Tape_MT33(gamma, delta, kappa, h, sigma, **kwargs):
    """
    Convert Tape parameters to a 3x3 moment tensor

    Args
        gamma: float, gamma parameter (longitude on funamental lune takes values
                    between -pi/6 and pi/6).
        delta: float, delta parameter (latitude on funamental lune takes values
                    between -pi/2 and pi/2)
        kappa: float, strike (takes values between 0 and 2*pi)
        h: float, cos(dip) (takes values between 0 and 1)
        sigma: float, slip angle (takes values between -pi/2 and pi/2)

    Returns
        numpy.matrix: 3x3 moment tensor

    """
    E = GD_E(gamma, delta)
    D = np.diag(E.flatten())
    T, N, P = SDR_TNP(kappa, np.arccos(h), sigma)
    L = np.matrix(
        [np.array(T).flatten(), np.array(N).flatten(), np.array(P).flatten()]).T
    MT33 = L*D*L.T
    return MT33


def Tape_MT6(gamma, delta, kappa, h, sigma):
    """
    Convert the Tape parameterisation to the moment tensor six-vectors.

    Args
        gamma: Gamma parameter (longitude on funamental lune takes values
               between -pi/6 and pi/6).
        delta: Delta parameter (latitude on funamental lune takes values
               between -pi/2 and pi/2)
        kappa: Strike (takes values between 0 and 2*pi)
        h: Cos(dip) (takes values between 0 and 1)
        sigma: Slip angle (takes values between -pi/2 and pi/2)

    Returns
        np.array: Array of MT 6-vectors

    """
    if cmoment_tensor_conversion:
        try:
            return cmoment_tensor_conversion.Tape_MT6(gamma.astype(np.float64),
                                                      delta.astype(np.float64),
                                                      kappa.astype(np.float64),
                                                      h.astype(np.float64),
                                                      sigma.astype(np.float64))
        except Exception:
            logger.exception('Error with C extension')
    else:
        logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
    if isinstance(gamma, (list, np.ndarray)):
        MT6 = np.empty((6, len(gamma)))
        for i in range(len(gamma)):
            MT6[:, i] = np.squeeze(MT33_MT6(Tape_MT33(gamma[i],
                                                      delta[i],
                                                      kappa[i],
                                                      h[i],
                                                      sigma[i])))
        return MT6
    MT33 = Tape_MT33(gamma, delta, kappa, h, sigma)
    return MT33_MT6(MT33)


def Tape_TNPE(gamma, delta, kappa, h, sigma):
    """
    Convert the Tape parameterisation to the T,N,P vectors and the eigenvalues.

    Args
        gamma: Gamma parameter (longitude on funamental lune takes values
               between -pi/6 and pi/6).
        delta: Delta parameter (latitude on funamental lune takes values
               between -pi/2 and pi/2)
        kappa: Strike (takes values between 0 and 2*pi)
        h: Cos(dip) (takes values between 0 and 1)
        sigma: Slip angle (takes values between -pi/2 and pi/2)

    Returns
        (numpy.matrix, numpy.matrix, numpy.matrix, numpy.array): T,N,P vectors
                and Eigenvalues tuple
    """
    E = GD_E(gamma, delta)
    T, N, P = SDR_TNP(kappa, np.arccos(h), sigma)
    return (T, N, P, E)


def normal_SD(normal):
    """
    Convert a plane normal to strike and dip

    Coordinate system is North East Down.

    Args
        normal: numpy matrix - Normal vector


    Returns
        (float, float): tuple of strike and dip angles in radians
    """
    if not isinstance(normal, np.matrixlib.defmatrix.matrix):
        normal = np.array(normal)/np.sqrt(np.sum(normal*normal, axis=0))
    else:
        normal = normal/np.sqrt(np.diag(normal.T*normal))
    normal[:, np.array(normal[2, :] > 0).flatten()] *= -1
    normal = np.array(normal)
    strike = np.arctan2(-normal[0], normal[1])
    dip = np.arctan2((normal[1]**2+normal[0]**2),
                     np.sqrt((normal[0]*normal[2])**2+(normal[1]*normal[2])**2))
    strike = np.mod(strike, 2*np.pi)
    return strike, dip


def toa_vec(azimuth, plunge, radians=False):
    """
    Convert the azimuth and plunge of a vector to a cartesian description of the vector

    Args
        azimuth: float, vector azimuth
        plunge: float, vector plunge

    Keyword Arguments
        radians: boolean, flag to use radians [default = False]

    Returns
        np.matrix: vector
    """
    if not radians:
        azimuth = np.pi*np.array(azimuth)/180.
        plunge = np.pi*np.array(plunge)/180.
    if not isinstance(plunge, np.ndarray):
        plunge = np.array([plunge])
    try:
        return np.matrix([np.cos(azimuth)*np.sin(plunge),
                          np.sin(azimuth)*np.sin(plunge),
                          np.cos(plunge)])
    except Exception:
        return np.array([np.cos(azimuth)*np.sin(plunge),
                         np.sin(azimuth)*np.sin(plunge),
                         np.cos(plunge)])


def output_convert(mts):
    """
    Convert the moment tensors into several different parameterisations

    The moment tensor six-vectors are converted into the Tape gamma,delta,kappa,h,sigma
    parameterisation; the Hudson u,v parameterisation; and the strike, dip and rakes
    of the two fault planes are calculated.

    Args
        mts: numpy array of moment tensor six-vectors

    Returns
        dict: dictionary of numpy arrays for each parameter
    """
    if cmoment_tensor_conversion:
        try:
            return cmoment_tensor_conversion.MT_output_convert(mts.astype(np.float64))
        except Exception:
            logger.exception('Error with C extension')
    else:
        logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
    g = np.empty((mts.shape[1],))
    d = np.empty((mts.shape[1],))
    k = np.empty((mts.shape[1],))
    h = np.empty((mts.shape[1],))
    s = np.empty((mts.shape[1],))
    u = np.empty((mts.shape[1],))
    v = np.empty((mts.shape[1],))
    s1 = np.empty((mts.shape[1],))
    d1 = np.empty((mts.shape[1],))
    r1 = np.empty((mts.shape[1],))
    s2 = np.empty((mts.shape[1],))
    d2 = np.empty((mts.shape[1],))
    r2 = np.empty((mts.shape[1],))
    rad_cor = 180/np.pi
    for i in range(mts.shape[1]):
        MT33 = MT6_MT33(mts[:, i])
        T, N, P, E = MT33_TNPE(MT33)
        u[i], v[i] = tk_uv(*E_tk(E))
        g[i], d[i] = E_GD(E)
        k[i], dip, s[i] = TNP_SDR(T, N, P)
        s1[i] = k[i]
        d1[i] = dip
        r1[i] = s[i]

        if np.abs(s[i]) > np.pi/2:
            k[i], dip, s[i] = SDR_SDR(k[i], dip, s[i])
        h[i] = np.cos(dip)
        s2[i], d2[i], r2[i] = SDR_SDR(s1[i], d1[i], r1[i])
        s1[i] *= rad_cor
        d1[i] *= rad_cor
        r1[i] *= rad_cor
        s2[i] *= rad_cor
        d2[i] *= rad_cor
        r2[i] *= rad_cor
    g = np.array(g).flatten()
    d = np.array(d).flatten()
    k = np.array(k).flatten()
    h = np.array(h).flatten()
    s = np.array(s).flatten()
    u = np.array(u).flatten()
    v = np.array(v).flatten()
    s1 = np.array(s1).flatten()
    d1 = np.array(d1).flatten()
    r1 = np.array(r1).flatten()
    s2 = np.array(s2).flatten()
    d2 = np.array(d2).flatten()
    r2 = np.array(r2).flatten()
    return {'g': g, 'd': d, 'k': k, 'h': h, 's': s, 'u': u, 'v': v,
            'S1': s1, 'D1': d1, 'R1': r1, 'S2': s2, 'D2': d2, 'R2': r2}


# Bi-axes
def isotropic_c(lambda_=1, mu=1, c=False):
    """
    Calculate the isotropic stiffness tensor

    Calculate isotropic stiffness parameters. The input parameters are either
    the two lame parameters lambda and mu, or is a full 21 element stiffness tensor::

        C(21) = ( C_11, C_12, C_13, C_14, C_15, C_16,
                        C_22, C_23, C_24, C_25, C_26,
                              C_33, C_34, C_35, C_36,
                                    C_44, C_45, C_46,
                                          C_55, C_56,
                                                C_66 )

                 (upper triangular part of Voigt stiffness matrix)

    If the full stiffness tensor is used, the "average" isotropic approximation is
    calculated using Eqns 81a and 81b from Chapman, C and Leaney,S, 2011. A new
    moment-tensor decomposition for seismic events in anisotropic media, GJI, 188(1),
    343-370.

    Args
        lambda_: lambda value
        mu: mu value
        c: list or numpy array of the 21 element input stiffness tensor (overrides
            lambda and mu arguments)

    Returns
        list: list of 21 elements of the stiffness tensor
    """
    if cmoment_tensor_conversion:
        try:
            if isinstance(c, bool):
                c = []
            return cmoment_tensor_conversion.isotropic_c(lambda_, mu, c)
        except Exception:
            logger.exception('Error with C extension')
    else:
        logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
    # Calculate isotropic approximation
    if not isinstance(c, bool) and len(c) == 21:
        # Eqns 81a and 81b from chapman and leaney 2011
        mu = ((c[0]+c[6]+c[11])+3*(c[15]+c[18]+c[20])-(c[1]+c[2]+c[7]))/15
        lambda_ = ((c[0]+c[6]+c[11])-2*(c[15]+c[18]+c[20])+4*(c[1]+c[2]+c[7]))/15
    n = lambda_+2*mu
    c = [n, lambda_, lambda_, 0, 0, 0, n, lambda_, 0, 0, 0, n, 0, 0, 0, mu, 0, 0, mu, 0, mu]
    return c


def MT6_biaxes(MT6, c=isotropic_c(lambda_=1, mu=1)):
    """
    Convert six vector to bi-axes

    Convert the moment tensor 6-vector to the bi-axes decomposition from Chapman, C and
    Leaney,S, 2011. A new moment-tensor decomposition for seismic events in anisotropic
    media, GJI, 188(1), 343-370.
    The 6-vector has the form::

        Mxx
        Myy
        Mzz
        sqrt(2)*Mxy
        sqrt(2)*Mxz
        sqrt(2)*Myz

    The stiffness tensor can be provided as an input, as a list of the 21 elements of the
    upper triangular part of the Voigt stiffness matrix::

        C(21) = ( C_11, C_12, C_13, C_14, C_15, C_16,
                        C_22, C_23, C_24, C_25, C_26,
                              C_33, C_34, C_35, C_36,
                                    C_44, C_45, C_46,
                                          C_55, C_56,
                                                C_66 )

                 (upper triangular part of Voigt stiffness matrix)

    Alternatively, the default isotropic parameters can be used or a possible isotropic
    stiffness tensor can be genereated using (isotropic_c)

    Args
        MT6: numpy matrix Moment tensor 6-vector
        c: list or numpy array of the 21 element input stiffness tensor

    Returns
        (numpy.array, numpy.array, numpy.array): tuple of phi (bi-axes) vectors,
                explosion value and area_displacement value.
    """
    if isinstance(c, bool):
        c = isotropic_c(1, 1)
    if len(MT6.shape) > 1 and MT6.shape[1] > 1:
        phi = np.empty((MT6.shape[1], 3, 2))
        explosion = np.empty((MT6.shape[1],))
        area_displacement = np.empty((MT6.shape[1],))
        for i in range(MT6.shape[1]):
            try:
                phi[i, :, :], explosion[i], area_displacement[i] = MT6_biaxes(MT6[:, i], c[i])
            except Exception:
                phi[i, :, :], explosion[i], area_displacement[i] = MT6_biaxes(MT6[:, i], c)
    else:
        if np.prod(MT6.shape) != 6:
            MT6 = np.asarray([MT6]).T
        if cmoment_tensor_conversion:
            try:
                return cmoment_tensor_conversion.MT6_biaxes(MT6.flatten().astype(np.float64), c)
            except Exception:
                pass
        else:
            logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
        lambda2mu = (
            3*(c[0]+c[6]+c[11])+4*(c[15]+c[18]+c[20])+2*(c[1]+c[2]+c[7]))/15
        mu = ((c[0]+c[6]+c[11])+3*(c[15]+c[18]+c[20])-(c[1]+c[2]+c[7]))/15
        lambda_ = lambda2mu-2*mu
        T, N, P, E = MT6_TNPE(MT6)
        isotropic = (lambda_+mu)*E[1]/mu-lambda_*(E[0]+E[2])/(2*mu)
        if is_isotropic_c(c):
            explosion = isotropic
        else:
            def isotropic_solve(iso):

                iso6 = np.squeeze(iso)*np.array([[1], [1], [1], [0], [0], [0]])
                if iso6.shape != MT6.shape:
                    iso6 = iso6.T
                T, N, P, E = MT6_TNPE(MT6c_D6(np.squeeze(MT6-iso6), c))
                return E[1]
            explosion = fsolve(isotropic_solve, isotropic)

        explosion6 = np.squeeze(explosion)*np.array([[1], [1], [1], [0], [0], [0]])
        if explosion6.shape != MT6.shape:
            explosion6 = explosion6.T
        T, N, P, E = MT6_TNPE(MT6c_D6(np.squeeze(MT6-explosion6), c))
        area_displacement = E[0]-E[2]
        phi = np.zeros((3, 2))
        if area_displacement != 0:      # to avoid undefined
            cphi = np.squeeze(np.sqrt(E[0]/area_displacement))
            sphi = np.squeeze(np.sqrt(-E[2]/area_displacement))
            phi[:, 0] = np.array(cphi*T+sphi*P).flatten()
            phi[:, 1] = np.array(cphi*T-sphi*P).flatten()
    return phi, explosion, area_displacement


def MT6c_D6(MT6, c=isotropic_c(lambda_=1, mu=1)):
    """
    Convert the moment tensor 6-vector to the potency tensor.
    The 6-vector has the form::

        Mxx
        Myy
        Mzz
        sqrt(2)*Mxy
        sqrt(2)*Mxz
        sqrt(2)*Myz

    The stiffness tensor can be provided as an input, as a list of the 21 elements of the
    upper triangular part of the Voigt stiffness matrix::

        C(21) = ( C_11, C_12, C_13, C_14, C_15, C_16,
                        C_22, C_23, C_24, C_25, C_26,
                              C_33, C_34, C_35, C_36,
                                    C_44, C_45, C_46,
                                          C_55, C_56,
                                                C_66 )

                 (upper triangular part of Voigt stiffness matrix)

    Alternatively, the default isotropic parameters can be used or a possible isotropic
    stiffness tensor can be genereated using (isotropic_c).

    Args
        MT6: numpy matrix Moment tensor 6-vector
        c: list or numpy array of the 21 element input stiffness tensor

    Returns
        numpy.array: numpy array of the potency 6 vector (in the same ordering as the
                moment tensor six vector)

    """
    if cmoment_tensor_conversion:
        try:
            return np.asarray(cmoment_tensor_conversion.MT6c_D6(MT6.astype(np.float64), c))
        except Exception:
            logger.exception('Error with C extension')
    else:
        logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
    mtvoigt = MT6[np.array([0, 1, 2, 5, 4, 3])]
    mtvoigt = np.matrix(mtvoigt)
    if mtvoigt.shape[1] == 6:
        mtvoigt = mtvoigt.T
    # Convert to voigt
    dvoigt = np.linalg.solve(np.matrix(c21_cvoigt(c)), mtvoigt)
    return dvoigt[np.array([0, 1, 2, 5, 4, 3])]


def is_isotropic_c(c):
    """
    Evaluate if an input stiffness tensor is isotropic

    The stiffness tensor needs to be provided as a list of the 21 elements of the
    upper triangular part of the Voigt stiffness matrix::

        C(21) = ( C_11, C_12, C_13, C_14, C_15, C_16,
                        C_22, C_23, C_24, C_25, C_26,
                              C_33, C_34, C_35, C_36,
                                    C_44, C_45, C_46,
                                          C_55, C_56,
                                                C_66 )

                 (upper triangular part of Voigt stiffness matrix)

    Args
        c:  input list of stiffness parameters (21 required)

    Returns
        bool: result of is_isotropic check
    """
    if cmoment_tensor_conversion:
        try:
            return cmoment_tensor_conversion.is_isotropic_c(c)
        except Exception:
            logger.exception('Error with C extension')
    else:
        logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
    tol = 1.e-6*c_norm(c)
    return ((abs(c[3]) < tol) and (abs(c[4]) < tol) and (abs(c[5]) < tol) and
            (abs(c[8]) < tol) and (abs(c[9]) < tol) and (abs(c[10]) < tol) and
            (abs(c[12]) < tol) and (abs(c[13]) < tol) and (abs(c[14]) < tol) and
            (abs(c[16]) < tol) and (abs(c[17]) < tol) and (abs(c[19]) < tol) and
            (abs(c[0]-c[6]) < tol) and (abs(c[6]-c[11]) < tol) and
            (abs(c[15]-c[18]) < tol) and (abs(c[18]-c[20]) < tol) and
            (abs(c[1]-c[2]) < tol) and (abs(c[2]-c[7]) < tol) and
            (abs(c[0]-c[1]-2*c[15]) < tol))


def c21_cvoigt(c):
    """
    Convert an input stiffness tensor to voigt form

    The stiffness tensor needs to be provided as a list of the 21 elements of the
    upper triangular part of the Voigt stiffness matrix::

        C(21) = ( C_11, C_12, C_13, C_14, C_15, C_16,
                        C_22, C_23, C_24, C_25, C_26,
                              C_33, C_34, C_35, C_36,
                                    C_44, C_45, C_46,
                                          C_55, C_56,
                                                C_66 )

                 (upper triangular part of Voigt stiffness matrix)

    Args
        c:  input list of stiffness parameters (21 required)

    Returns
        numpy.array: voigt form of the stiffness tensor
    """
    if cmoment_tensor_conversion:
        try:
            return np.asarray(cmoment_tensor_conversion.c21_cvoigt(c))
        except Exception:
            logger.exception('Error with C extension')
    else:
        logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
    return np.array([[c[0], c[1], c[2], np.sqrt(2)*c[3], np.sqrt(2)*c[4], np.sqrt(2)*c[5]],
                     [c[1], c[6], c[7], np.sqrt(
                         2)*c[8], np.sqrt(2)*c[9], np.sqrt(2)*c[10]],
                     [c[2], c[7], c[11], np.sqrt(
                         2)*c[12], np.sqrt(2)*c[13], np.sqrt(2)*c[14]],
                     [np.sqrt(2)*c[3], np.sqrt(2)*c[8], np.sqrt(2)*c[12],
                      2*c[15], 2*c[16], 2*c[17]],
                     [np.sqrt(2)*c[4], np.sqrt(2)*c[9], np.sqrt(2)*c[13],
                      2*c[16], 2*c[18], 2*c[19]],
                     [np.sqrt(2)*c[5], np.sqrt(2)*c[10], np.sqrt(2)*c[14],
                      2*c[17], 2*c[19], 2*c[20]]])


def c_norm(c):
    """
    Calculate norm of the stiffness tensor.

    The stiffness tensor needs to be provided as a list of the 21 elements of the
    upper triangular part of the Voigt stiffness matrix::

        C(21) = ( C_11, C_12, C_13, C_14, C_15, C_16,
                        C_22, C_23, C_24, C_25, C_26,
                              C_33, C_34, C_35, C_36,
                                    C_44, C_45, C_46,
                                          C_55, C_56,
                                                C_66 )

                 (upper triangular part of Voigt stiffness matrix)

    Args
        c:  input list of stiffness parameters (21 required)

    Returns
       float: Euclidean norm of the tensor

    """
    if cmoment_tensor_conversion:
        try:
            return cmoment_tensor_conversion.c_norm(c)
        except Exception:
            logger.exception('Error with C extension')
    else:
        logger.info(C_EXTENSION_FALLBACK_LOG_MSG)
    return np.sqrt(c[0]**2 + c[6]**2 + c[11]**2 + 2*(c[1]**2+c[2]**2+c[7]**2) +
                   4*(c[3]**2+c[4]**2+c[5]**2+c[8]**2+c[9]**2+c[10]**2 +
                      c[12]**2+c[13]**2+c[14]**2+c[15]**2+c[18]**2+c[20]**2) +
                   8*(c[16]**2+c[17]**2+c[19]**2))
