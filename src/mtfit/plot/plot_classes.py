"""
plot_classes
************

Plotting classes for moment tensor plotting.
"""


# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.


import re
import logging
from distutils.version import StrictVersion

import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
from matplotlib import cm, patches
from matplotlib import __version__ as matplotlib_version
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

from MTfit.inversion import station_angles
from MTfit.convert import MT33_MT6, toa_vec, tk_uv, E_tk, TNP_SDR, MT6_TNPE, E_GD, TP_FP, normal_SD, SDR_SDR, MT6_biaxes, E_uv
from MTfit.utilities.extensions import get_extensions
try:
    from MTfit.sampling import unique_columns, _6sphere_prior, ln_bayesian_evidence
except Exception:
    pass  # SCIPY ERROR -CAN'T USE

from .spherical_projection import equal_area, equal_angle

logger = logging.getLogger('MTfit.plot')

# Default colors and colormaps
DEFAULT_COLOR = 'purple'
DEFAULT_HIST_COLORMAP = 'viridis'

if StrictVersion(matplotlib_version) < StrictVersion('1.5.0'):
    DEFAULT_HIST_COLORMAP = 'CMRmap'

DEFAULT_AMP_COLORMAP = 'bwr'

rad_deg = 180/np.pi


# Vector class


class Arrow3D(FancyArrowPatch):
    """3D arrow class"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super(Arrow3D, self).__init__((0, 0), (0, 0), *args, **kwargs)
        self._3d_vertices = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._3d_vertices
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super(Arrow3D, self).draw(renderer)


class MTData(object):

    """
    MTData object manages the moment tensors.

    Additionally it enables conversion between different parameterisations
    and calculation of several statistics.

    This is used by all of the plot classes for handling the moment tensor,
    and is transparent to converting and accessing the moment tensors, as
    it behaves like a numpy array (e.g. indexing and properties like shape
    are given for the moment tensor).

    The MTData object also stores the probability, and is used for
    calculating the orientation mean.

    """

    def __init__(self, MTs, probability=[], c=False, **kwargs):
        """
        MTData initialisation

        Parameters corresponding to the moment tensor samples can be set
        as kwargs. These parameters include:
            T, N, P, E, u, v, tau, k, gamma, delta, kappa, h, sigma,
            strike1, dip1, rake1, strike2, dip2, rake2, N1, N2

        Args
            MTs: numpy array of moment tensor six vectors with shape (6,n).
                Alternatively, the input can be a dictionary from the MTfit output.


        Keyword Args
            probability: list or numpy array of probabilities - should be
                         the same length as the number of moment tensors,
                         or empty (Default is [])
            **kwargs: Attributes that correspond to converted parameters
                      of the moment tensor, as described above


        """
        self._calculable = ['T', 'N', 'P', 'E', 'u', 'v', 'tau', 'k', 'gamma', 'delta',
                            'kappa', 'h', 'sigma', 'strike', 'strike1', 'strike2',
                            'dip', 'dip1', 'dip2', 'rake', 'rake1', 'rake2', 'N1', 'N2',
                            'clustered_N1', 'clustered_N2', 'clustered_rake1',
                            'clustered_rake2', 'mean_strike', 'mean_dip', 'mean_rake',
                            'mean_normal', 'cov_clustered_N1', 'var_clustered_rake1',
                            'bayesian_evidence', 'ln_bayesian_evidence', 'mean', 'covariance',
                            'max_probability', 'phi', 'explosion', 'area_displacement']
        self.c = c  # Stiffness tensor
        if isinstance(MTs, dict):
            # Assume output dictionary from MTfit
            try:
                try:
                    self.MTs = self._check(MTs['moment_tensor_space'])
                    key_map = {'g': 'gamma', 'd': 'delta', 'k': 'kappa', 's': 'sigma', 'S1': 'strike', 'D1': 'dip',
                               'R1': 'rake', 'S2': 'strike2', 'D2': 'dip2', 'R2': 'rake2', 'total_number_samples': 'total_number_samples'}
                    self._set_probability(MTs['probability'])
                    try:
                        self._set_ln_pdf(MTs['ln_pdf'])
                    except Exception:
                        pass

                except Exception:
                    self.MTs = self._check(MTs['MTSpace'])
                    key_map = {'g': 'gamma', 'd': 'delta', 'k': 'kappa', 's': 'sigma', 'S1': 'strike', 'D1': 'dip',
                               'R1': 'rake', 'S2': 'strike2', 'D2': 'dip2', 'R2': 'rake2', 'NSamples': 'total_number_samples'}
                    self._set_probability(MTs['Probability'])
                    try:
                        self._set_ln_pdf(MTs['ln_pdf'])
                    except Exception:
                        pass

                for key in MTs.keys():
                    try:
                        mapped_key = key
                        if key in key_map.keys():
                            mapped_key = key_map[key]
                        if mapped_key in self._calculable or mapped_key == 'total_number_samples':
                            setattr(self, mapped_key, MTs[key])
                    except Exception:
                        logger.exception('Error mapping keys at key: {}'.format(key))
                        pass
            except Exception:
                pass
        else:
            self.MTs = self._check(MTs)
            self._set_probability(probability)
        for key in kwargs:
            if key in self._calculable or key == 'total_number_samples':
                setattr(self, key, kwargs[key])
        self._prior = kwargs.get('prior', _6sphere_prior)

    def _set_probability(self, probability):
        """Set probability samples, if the probability is the same size as the moment tensor."""
        if not isinstance(probability, (np.ndarray, list)):
            probability = [probability]
        if len(probability) > 1:
            try:
                assert len(probability) == len(self)
            except AssertionError:
                raise ValueError('probability must either be 1 or have same length as the number of moment tensors')
            # Delete probability dependent parameters
            self._clear_dependent_parameters()
        self.probability = np.array(probability).flatten()

    def _set_ln_pdf(self, ln_pdf):
        """Set ln_pdf samples, if the ln_pdf is the same size as the moment tensor."""
        if not isinstance(ln_pdf, (list, np.ndarray)):
            ln_pdf = [ln_pdf]
        try:
            assert len(ln_pdf) == len(self)
        except AssertionError:
            raise ValueError('ln_pdf must either be 1 or have same length as the number of moment tensors')
        # Delete ln_pdf dependent parameters
        self._clear_dependent_parameters()
        self.ln_pdf = ln_pdf

    def _clear_dependent_parameters(self):
        """
        Clears dependent variables when updating the MTData object with new MTs etc.

        These variables are dependent on the MTs so need to be recalculated if these are changed.
        """
        for attr in ['mean_normal', 'mean_strike', 'mean_dip', 'mean_rake', 'var_clustered_rake1',
                     'cov_clustered_N2', 'cov_clustered_N1', 'var_clustered_rake2',
                     'ln_bayesian_evidence', 'mean', 'covariance', 'max_probability']:
            try:
                delattr(self, attr)
            except Exception:
                pass

    def _check(self, MTs):
        """
        Check the moment tensor dimensions and convert to MT 6 vector form.

        Returns
            numpy array: numpy array of moment tensor 6 vectors
        """
        # convert MTs to MT6
        if not isinstance(MTs, np.ndarray):
            MTs = np.array(MTs)
        if 6 not in MTs.shape:
            # MT 33 as (:,3,3) 3 dimensional array
            if MTs.shape[-2:] == (3, 3):
                # 3x3 matrix
                if len(MTs.shape) == 3:
                    # multiple MT 3 vectors
                    new_mts = np.empty((6, MTs.shape[0]))
                    for i in range(MTs.shape[0]):
                        new_mts[:, i] = np.array(
                            MT33_MT6(MTs[i, :, :])).flatten()
                else:
                    new_mts = MT33_MT6(MTs)
            else:
                raise ValueError('MTs shape, {} cannot be processed. The last two elements need to be 3,3 or it needs to be in six vector form'.format(MTs.shape))
        else:
            new_mts = MTs
        # make into an array
        if len(new_mts.shape) < 2:
            new_mts = np.array([new_mts])
        # Get 6,n format
        if new_mts.shape[0] != 6:
            new_mts = new_mts.T
        return new_mts

    def __getitem__(self, arg):
        """
        MTData[x]<=>MTData.__getitem__(x)

        Handles moment tensor indexing, including any converted parameters.

        Returns
             MTData: new object with the corresponding moment tensors and any converted parameters.
        """
        converted = self._get_converted(arg)
        probability = []
        if len(self.probability) > 1:
            try:
                probability = self.probability.__getitem__(arg)
            except Exception:
                probability = self.probability.__getitem__(arg[1])
        return self.__class__(self.MTs.__getitem__(arg), probability, **converted)

    def __setitem__(self, arg, value):
        """
        MTData[x]=... <=>MTData.__setitem__(x,...)

        Sets the moment tensors using the argument and also sets any converted parameters.
        """
        if len(self.probability) > 1:
            raise ValueError('Probability set so cannot set items')
        self.MTs.__setitem__(arg, value)
        self._set_converted(arg)

    def __delitem__(self, arg):
        """
        del MTData[x]

        raises ValueError as cannot delete numpy array elements
        """
        raise ValueError('Cannot delete numpy array elements')

    def __len__(self):
        """
        len(x)<=>x.__len__()

        Returns
            int: number of moment tensors
        """
        return self.MTs.shape[1]

    def __str__(self):
        """
        str(x)<=>x.__str__()

        Returns
            str: moment tensor 6 vector array
        """
        return str(self.MTs)

    def __repr__(self):
        """
        repr(x)<=>x.__repr__()

        Returns
            str: moment tensor 6 vector array
        """
        return repr(self.MTs)

    def __getattr__(self, attr):
        """
        a.x<=>a.__getattr__(x)

        Handles Moment tensor conversion if the attribute is calculable, as well as the mean orientation
        """
        if attr == 'mean_orientation':
            return (self.mean_strike, self.mean_dip, self.mean_rake)
        if attr == 'bayesian_evidence':
            return np.exp(self.ln_bayesian_evidence)
        MTindexes = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
        if attr in MTindexes:
            return self.MTs[MTindexes.index(attr), :]
        if attr == 'probability':
            return []
        if attr == 'shape':
            return getattr(self.MTs, attr)
        if attr in self._calculable:
            return self._convert(attr)
        return super(MTData, self).__getattribute__(attr)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all(self.MTs == other.MTs) and all(self.probability == other.probability)
        elif isinstance(other, np.ndarray):
            return all(self.MTs == other)

    def _get_converted(self, arg=None):
        """Gets converted parameters

        Keyword Args
            arg: slice or indexing to sample. If None (default), all converted samples are returned

        Returns
            dict: Dictionary of the calculated converted moment tensor parameters.

        """
        converted = {}
        for key in self._calculable:
            if key in self.__dict__ and key not in ['mean', 'covariance', 'max_probability', 'mean_strike', 'mean_dip', 'mean_rake', 'mean_normal', 'cov_clustered_N1', 'var_clustered_rake1', 'ln_bayesian_evidence', 'bayesian_evidence']:
                if arg is None:
                    converted[key] = getattr(self, key)
                else:
                    try:
                        converted[key] = getattr(self, key).__getitem__(arg)
                    except Exception:
                        converted[key] = getattr(self, key).__getitem__(arg[1])
        return converted

    def _del_converted(self, arg=None):
        """
        Deletes converted parameters - NOT CALLED AS NUMPY ARRAYS CANNOT DELETE ITEMS

        Keyword Args
            arg: slice or indexing to sample. If None (default), all converted samples are returned

        Returns
            dict: Dictionary of the calculated converted moment tensor parameters, with deleted values.

        """
        self._clear_dependent_parameters()
        converted = {}
        for key in self._calculable:
            if key in self.__dict__:
                if arg is None:
                    converted[key] = delattr(self, key)
                else:
                    try:
                        try:
                            converted[key] = getattr(self, key).__delitem__(arg)
                        except Exception:
                            converted[key] = getattr(self, key).__delitem__(arg[1])
                    except Exception:
                        try:
                            converted[key] = np.delete(getattr(self, key), arg)
                        except Exception:
                            converted[key] = np.delete(getattr(self, key), arg[1])
        return converted

    def _set_converted(self, arg):
        """
        Sets the converted parameters for a new MTs set using __setitem__

        If a parameter has been set, the new moment tensors are converted to that parameter and set

        (Includes parent parameters, i.e. those required to calculate the other parameters)

        Args
            arg: index or slice of parameters to set.
        """
        self._clear_dependent_parameters()
        T, N, P, E = MT6_TNPE(self.MTs[arg])
        if 'phi1' in self.__dict__.keys():
            phi, explosion, area_displacement = MT6_biaxes(
                self.MTs[arg], self.c)
            try:
                self.phi1[arg] = (phi[:, :, 0]).T
            except Exception:
                self.phi1[arg[1]] = (phi[:, :, 0]).T
            try:
                self.phi2[arg] = (phi[:, :, 1]).T
            except Exception:
                self.phi2[arg[1]] = (phi[:, :, 1]).T
            try:
                self.area_displacement[arg] = area_displacement
            except Exception:
                self.area_displacement[arg[1]] = area_displacement
            try:
                self.explosion[arg] = explosion
            except Exception:
                self.explosion[arg[1]] = explosion
        if 'T' in self.__dict__.keys():
            try:
                self.T[arg] = np.array(np.matrix(T))
                self.N[arg] = np.array(np.matrix(N))
                self.P[arg] = np.array(np.matrix(P))
            except ValueError:
                self.T[arg] = T.flatten()
                self.N[arg] = N.flatten()
                self.P[arg] = P.flatten()
        if 'N1' in self.__dict__.keys():
            N1, N2 = TP_FP(self.T, self.P)
            try:
                self.N1[arg] = np.matrix(N1)
                self.N2[arg] = np.matrix(N2)
            except ValueError:
                self.N1[arg] = N1.flatten()
                self.N2[arg] = N2.flatten()

        if 'strike' in self.__dict__.keys():
            strike, dip, rake = TNP_SDR(T, N, P)
            try:
                self.strike[arg] = strike*rad_deg
            except Exception:
                self.strike[arg[1]] = strike*rad_deg
            try:
                self.dip[arg] = dip*rad_deg
            except Exception:
                self.dip[arg[1]] = dip*rad_deg
            try:
                self.rake[arg] = rake*rad_deg
            except Exception:
                self.rake[arg[1]] = rake*rad_deg

        if 'strike2' in self.__dict__.keys():
            try:
                strike2, dip2, rake2 = SDR_SDR(strike, dip, rake)
            except Exception:
                strike, dip, rake = TNP_SDR(T, N, P)
                strike2, dip2, rake2 = SDR_SDR(strike, dip, rake)

            try:
                self.strike2[arg] = strike2*rad_deg
            except Exception:
                self.strike2[arg[1]] = strike2*rad_deg
            try:
                self.dip2[arg] = dip2*rad_deg
            except Exception:
                self.dip2[arg[1]] = dip2*rad_deg
            try:
                self.rake2[arg] = rake2*rad_deg
            except Exception:
                self.rake2[arg[1]] = rake2*rad_deg
        if 'gamma' in self.__dict__.keys():
            gamma, delta = E_GD(E)
            try:
                self.gamma[arg] = gamma
            except Exception:
                self.gamma[arg[1]] = gamma
            try:
                self.delta[arg] = delta
            except Exception:
                self.delta[arg[1]] = delta

        if 'tau' in self.__dict__.keys():
            tau, k = E_tk(E)
            try:
                self.tau[arg] = tau
            except Exception:
                self.tau[arg[1]] = tau
            try:
                self.k[arg] = k
            except Exception:
                self.k[arg[1]] = k

        if 'u' in self.__dict__.keys():
            try:
                u, v = E_uv(tau, k)
            except Exception:
                tau, k = E_tk(E)
                u, v = E_uv(tau, k)
            try:
                self.u[arg] = u
            except Exception:
                self.u[arg[1]] = u
            try:
                self.v[arg] = v
            except Exception:
                self.v[arg[1]] = v

        if 'kappa' in self.__dict__.keys():
            try:
                kappa = strike
                kappa[np.abs(rake) > 90] = strike2[np.abs(rake) > 90]
                h = dip
                h[np.abs(rake) > 90] = dip2[np.abs(rake) > 90]
                h = np.cos(h/rad_deg)
                sigma = rake
                sigma[np.abs(rake) > 90] = rake2[np.abs(rake) > 90]
                kappa = kappa/rad_deg
                sigma = sigma/rad_deg
            except Exception:
                strike, dip, rake = TNP_SDR(T, N, P)
                strike2, dip2, rake2 = SDR_SDR(strike, dip, rake)
                kappa = strike
                kappa[np.abs(rake) > 90] = strike2[np.abs(rake) > 90]
                h = dip
                h[np.abs(rake) > 90] = dip2[np.abs(rake) > 90]
                h = np.cos(h/rad_deg)
                sigma = rake
                sigma[np.abs(rake) > 90] = rake2[np.abs(rake) > 90]
                kappa = kappa/rad_deg
                sigma = sigma/rad_deg
            try:
                self.kappa[arg] = kappa
            except Exception:
                self.kappa[arg[1]] = kappa
            try:
                self.h[arg] = h
            except Exception:
                self.h[arg[1]] = h
            try:
                self.sigma[arg] = sigma
            except Exception:
                self.sigma[arg[1]] = sigma

    def _convert(self, attr, *args, **kwargs):
        """
        Converts the parameters for a given attribute

        Args
            attr: str attribute to convert the moment tensor to

        Returns
            numpy array: values for the attribute
        """
        if attr in ['strike1', 'dip1', 'rake1']:
            return getattr(self, attr.rstrip('1'))
        if attr in ['kappa', 'h', 'sigma']:
            kappa = self.strike1.copy()
            kappa[np.abs(self.rake1) > 90] = self.strike2[
                np.abs(self.rake1) > 90]
            h = self.dip1.copy()
            h[np.abs(self.rake1) > 90] = self.dip2[np.abs(self.rake1) > 90]
            h = np.cos(h/rad_deg)
            sigma = self.rake1.copy()
            sigma[np.abs(self.rake1) > 90] = self.rake2[
                np.abs(self.rake1) > 90]
            self.kappa = kappa/rad_deg
            self.h = h
            self.sigma = sigma/rad_deg
        if attr in ['strike2', 'dip2', 'rake2']:
            strike2, dip2, rake2 = SDR_SDR(
                self.strike1/rad_deg, self.dip1/rad_deg, self.rake1/rad_deg)
            self.strike2 = strike2*rad_deg
            self.dip2 = dip2*rad_deg
            self.rake2 = rake2*rad_deg
        if attr in ['gamma', 'delta']:
            gamma, delta = E_GD(self.E)
            self.gamma = gamma
            self.delta = delta
        if attr in ['strike', 'dip', 'rake']:
            strike, dip, rake = TNP_SDR(self.T, self.N, self.P)
            self.strike = strike*rad_deg
            self.dip = dip*rad_deg
            self.rake = rake*rad_deg
        if attr in ['T', 'N', 'P', 'E']:
            T, N, P, E = MT6_TNPE(self.MTs)
            self.T = np.array(T)
            self.N = np.array(N)
            self.P = np.array(P)
            self.E = np.array(E)
        if attr in ['N1', 'N2']:
            # N1 is T+P N2 is T-P - strike1, dip1,rake1 correspond to N1 and
            # strike2,dip2,rake2 to N2
            N1, N2 = TP_FP(self.T, self.P)
            self.N1 = np.matrix(N1)
            self.N2 = np.matrix(N2)
        if attr in ['clustered_N1', 'clustered_N2', 'clustered_rake1', 'clustered_rake2']:
            self.cluster_normals()
        if attr in ['mean_strike', 'mean_dip', 'mean_rake', 'mean_orientation', 'mean_normal', 'cov_clustered_N1', 'var_clustered_rake1']:
            self.get_mean_orientation()
        if attr in ['mean', 'covariance']:
            self.get_mean()
        if attr in ['max_probability']:
            self.get_max_probability()
        if attr in ['u', 'v']:
            u, v = tk_uv(self.tau, self.k)
            self.u = u
            self.v = v
        if attr in ['phi1', 'phi2', 'area_displacement', 'explosion']:
            phi, explosion, area_displacement = MT6_biaxes(self.MTs, self.c)
            try:
                self.phi1 = np.squeeze(phi[:, :, 0]).T
                self.phi2 = np.squeeze(phi[:, :, 1]).T
            except Exception:
                self.phi1 = np.atleast_2d(phi[:, 0]).T
                self.phi2 = np.atleast_2d(phi[:, 1]).T
            self.area_displacement = area_displacement
            self.explosion = explosion
        if attr in ['tau', 'k']:
            tau, k = E_tk(self.E)
            self.tau = tau
            self.k = k
        if attr in ['ln_bayesian_evidence']:
            try:
                self.ln_bayesian_evidence = ln_bayesian_evidence({'g': self.gamma, 'd': self.delta, 'ln_pdf': self.ln_pdf},
                                                                 self.total_number_samples, self._prior)
            except Exception:
                raise ValueError('Bayesian Evidence calculation requires probability and total_number_samples attributes to be set.')
        return getattr(self, attr)

    def is_dc(self):
        """Returns array of DC values"""
        return np.multiply(np.abs(self.gamma) < 10**-10, np.abs(self.delta) < 10**-10,)

    def cluster_normals(self):
        """
        Cluster the normals.

        This is a very simplistic algorithm that clusters the normals by dividing them into two groups with the shortest distance between them.

        N.B. This depends on the initial values given, so if the normals are not well clustered, or the initial normals chosen are outliers, the results will not be a good clustering.

        Consequently, only use this approach for well clustered fault plane normals (in at least one normal direction). The more tightly clustered normals are set to clustered_N1.

        Returns
            numpy array, numpy array, numpy array, numpy array: tuple of numpy arrays, clustered_N1, clustered_N2, clustered_rake1,clustered_rake2
        """
        # Use max probability as base if it exists
        if len(self.probability) > 1:
            max_prob_idx = self.probability == self.probability.max()
        else:
            max_prob_idx = 0
        n1 = np.matrix(np.zeros((self.T.shape[0], self.T.shape[1]+1)))
        n2 = np.matrix(np.zeros((self.T.shape[0], self.T.shape[1]+1)))
        n1[:, 0] = self.N1[:, max_prob_idx][:, 0]
        n2[:, 0] = self.N2[:, max_prob_idx][:, 0]
        rake1 = self.rake1.copy()
        rake2 = self.rake2.copy()
        for i in range(0, self.N1.shape[1]):
            # np.abs allows vector flipping - if cos theta is negative then the
            # negative vector would have the equivalent positive value
            n1_1cos_theta = np.abs(self.N1[:, i].T*n1).max()
            n1_2cos_theta = np.abs(self.N2[:, i].T*n1).max()
            n2_1cos_theta = np.abs(self.N1[:, i].T*n2).max()
            n2_2cos_theta = np.abs(self.N2[:, i].T*n2).max()
            dist = np.array([n1_1cos_theta, n1_2cos_theta, n2_1cos_theta, n2_2cos_theta])
            ind = np.nonzero(dist == max(dist))[0]
            if (len(ind == 1) and (ind[0] == 0 or ind[0] == 3)) or (ind == [0, 3]).all():
                if (self.N1[:, i].T*n1).max() < 0:
                    N1 = -self.N1[:, i]
                else:
                    N1 = self.N1[:, i]
                if (self.N2[:, i].T*n2).max() < 0:
                    N2 = -self.N2[:, i]
                else:
                    N2 = self.N2[:, i]
                n1[:, i+1] = N1
                n2[:, i+1] = N2
            else:
                if (self.N1[:, i].T*n1).max() < 0:
                    N1 = -self.N1[:, i]
                else:
                    N1 = self.N1[:, i]
                if (self.N2[:, i].T*n2).max() < 0:
                    N2 = -self.N2[:, i]
                else:
                    N2 = self.N2[:, i]
                n2[:, i+1] = N1
                r2 = rake2[i]
                rake2[i] = rake1[i]
                n1[:, i+1] = N2
                rake1[i] = r2
        n1 = n1[:, 1:]
        n2 = n2[:, 1:]
        if (np.var(np.mean(n1, 1).T*n1) > np.var(np.mean(n2, 1).T*n2)) and ((np.var(n1, 1)).max() > (np.var(n2, 1)).max()):
            nx = n1
            r1 = rake1
            n1 = n2
            rake1 = rake2
            n2 = nx
            rake2 = r1
        self.clustered_N1 = n1
        self.clustered_N2 = n2
        self.clustered_rake1 = rake1
        self.clustered_rake2 = rake2
        return self.clustered_N1, self.clustered_N2, self.clustered_rake1, self.clustered_rake2

    def get_mean_orientation(self):
        """Get the mean orientation from the clustered normals and rake parameters using the more tightly clustered parameters.

        This also calculates the variance of the rake distributions and covariance of the clustered normals (using the probability if it is set)

        Returns
            numpy array, numpy array, numpy array: tuple of the mean strike dip and rake of the fault planes.
        """
        self.clustered_N1[:, np.array(self.clustered_N1[:, 0].T*self.clustered_N1).flatten() < 0] = -self.clustered_N1[:, np.array(self.clustered_N1[:, 0].T*self.clustered_N1).flatten() < 0]
        if len(self.probability) > 1:
            n = np.sum(np.multiply(self.clustered_N1, self.probability), 1)/np.sum(self.probability)
            r = np.sum(np.multiply(self.clustered_rake1, self.probability))/np.sum(self.probability)
            self.var_clustered_rake1 = (np.sum(np.multiply(np.multiply(self.clustered_rake1-r, self.clustered_rake1-r), self.probability)) /
                                        (np.sum(self.probability)-(np.sum(np.multiply(self.probability, self.probability))/np.sum(self.probability))))
            r2 = np.sum(np.multiply(self.clustered_rake1, self.probability))/np.sum(self.probability)
            self.var_clustered_rake2 = (np.sum(np.multiply(np.multiply(self.clustered_rake2-r2, self.clustered_rake2-r2), self.probability)) /
                                        (np.sum(self.probability)-(np.sum(np.multiply(self.probability, self.probability))/np.sum(self.probability))))
            if StrictVersion(np.__version__) < StrictVersion('1.10.0'):
                cov = ((np.sum(self.probability)/(np.sum(self.probability)**2-np.sum(np.multiply(self.probability, self.probability)))) *
                       (np.matrix(np.multiply(self.clustered_N1-n, self.probability)) * np.matrix(self.clustered_N1-n).T))
                self.cov_clustered_N1 = cov
                n2 = np.sum(np.multiply(self.clustered_N2, self.probability), 1)/np.sum(self.probability)
                cov = ((np.sum(self.probability)/(np.sum(self.probability)**2-np.sum(np.multiply(self.probability, self.probability)))) *
                       (np.matrix(np.multiply(self.clustered_N2-n2, self.probability)) * np.matrix(self.clustered_N2-n2).T))
                self.cov_clustered_N2 = cov
            else:
                self.cov_clustered_N1 = np.cov(self.clustered_N1, aweights=self.probability, ddof=1)
                self.cov_clustered_N2 = np.cov(self.clustered_N2, aweights=self.probability, ddof=1)
        else:
            n = np.mean(self.clustered_N1, 1)
            r = np.mean(self.clustered_rake1)
            self.var_clustered_rake1 = np.var(self.clustered_rake1, ddof=1)
            self.var_clustered_rake2 = np.var(self.clustered_rake2, ddof=1)
            self.cov_clustered_N1 = np.cov(self.clustered_N1, ddof=1)
            self.cov_clustered_N2 = np.cov(self.clustered_N2, ddof=1)
        s, d = normal_SD(n)
        self.mean_normal = n
        self.mean_strike = s*rad_deg
        self.mean_dip = d*rad_deg
        self.mean_rake = r
        return s, d, r

    def get_unique_McMC(self):
        """Gets the unique McMC samples, with the probability scaled by the number of samples

        Returns
            MTData: New MTData object with unique MT samples and probabilities corresponding to the moment tensor frequencies.
        """
        _MTs, counts, idx = unique_columns(self.MTs, True, True)
        new = self[:, idx]
        new._set_probability(counts)
        return new

    def get_max_probability(self, single=False):
        """Returns an MTData object containing the maximum probability solutions

        Keyword Args
            single: bool - flag to return only one moment tensor (the first maximum probability moment tensor)

        Returns:
            MTData:  New MTData object with max probability MT samples.
        """
        if not len(self.probability) > 1:
            raise ValueError('maximum probability requires probability to be set.')
        idx = self.probability == self.probability.max()
        new = self[:, idx]
        if single and len(new) > 1:
            new = new[:, 0]
        self.max_probability = new
        return new

    def get_mean(self):
        """
        Calculates the mean moment tensor and variance of the moment tensor samples

        Returns:
            numpy array,numpy array: tuple of numpy arrays for the mean moment tensor six vector and the moment tensor six vector covariance.
        """
        if len(self.probability) > 1:
            mean = np.sum(
                np.multiply(self.MTs, self.probability), 1)/np.sum(self.probability)
            mean = np.array(np.matrix(mean).T)
            if StrictVersion(np.__version__) < StrictVersion('1.10.0'):
                covariance = (np.sum(self.probability)/(np.sum(self.probability)**2-np.sum(np.multiply(self.probability, self.probability))))*(
                    np.matrix(np.multiply(self.MTs-mean, self.probability))*np.matrix(self.MTs-mean).T)
                self.covariance = covariance
            else:
                self.covariance = np.cov(self.MTs, aweights=self.probability, ddof=1)
        else:
            mean = np.mean(self.MTs, 1)
            mean = np.array(np.matrix(mean).T)
            covariance = np.cov(self.MTs, ddof=1)
            self.covariance = covariance
        self.mean = mean
        return self.mean, self.covariance


class MTplot(object):

    """MTplot class - handles plotting of the moment tensor using  different plot_classes

    This object acts as a handle round a matplotlib figure, handling the moment tensors and creating the plot classes for axes.
    """

    def __init__(self, MTs, plot_type='beachball', stations={}, plot=True, label=False, save_file='', save_dpi=200, *args, **kwargs):
        """
        Initialises the MTplot object.

        Multiple plots (subplots) are handled using the matplotlib GridSpec - to create multiple plots, use nested lists for the MTs, with the inner lists corresponding to each column and the outer each row, e.g.::

                [[x,y=1,1;x,y=2,1],[x,y=1,2;x,y=2,2]]

        The total number of columns is given by the max number of points in the nested list. Lists with fewer points will have the last plot stretched to cover the remaining columns.
        Setting None in the list will stretch the plot to it's left to cover the columns.

        The args and kwargs, which are passed through to the plot_class object can also be nested in this way. Additionally, if the parameters are equal to the number of columns,
        or number of rows, the same values are used for each plot in the corresponding column or row. If the dimension of the multiplot is square, the parameters are assumed to correspond to the rows.

        Args
            MTs: Moment tensor samples for creating MTData object, such as a numpy array of MT 6 vectors (see MTData initialisation for different types, and MTplot docstring for handling multiple plots

        Keyword Args
            plot_type: str - plot type selection (Default is 'beachball')
            stations: dict - Dictionary of stations, containing keys: 'names','azimuth','takeoff_angle' and 'polarity' or an empty dict (default)
            plot: bool - flag to actually plot and show the figure
            label: bool - flag to show axis labels if multiple plots are being shown
            save_file: string  - filename to save plot to (if set)
            save_dpi: int - output dpi for file save
            args: passed through to plot_class initialisation
            kwargs: passed through to plot_class initialisation
        """
        # Get number of plots
        x = 1
        y = 1
        self.show = kwargs.pop('show', True)
        if isinstance(MTs, list):
            # Handling structured data, multiplot using grid spec
            if any([isinstance(u, list) for u in MTs]):
                y = len(MTs)
                x = max([len(u) for u in MTs if isinstance(u, list)])
            else:
                y = 1
                x = len(MTs)
                MTs = [MTs]
        else:
            MTs = [[MTs]]
        # Prep data:
        indices = []
        for i in range(y):
            indices.append([])
            while len(MTs[i]) < x:
                MTs[i].append(None)
            for j in range(x):
                if MTs is not None:
                    indices[-1].append(1)
                else:
                    indices[-1].append(0)
        self.yx = (y, x, indices)
        plot_type = self._prep_data(plot_type, 'plot_type')
        stations = self._prep_data(stations, 'stations')
        new_args = []
        for i, arg in args:
            new_args.append(self._prep_data(arg, 'arg '+str(i)))
        new_kwargs = {}
        for key in kwargs.keys():
            new_kwargs[key] = self._prep_data(kwargs[key], key)
        self.plot_classes = []
        from matplotlib import pyplot as plt
        self.fig = plt.figure()
        self.grid_spec = gridspec.GridSpec(y, x)
        self.label = label
        self.save_file = save_file
        self.save_dpi = save_dpi
        self.labels = []
        for i in range(y):
            ind = np.nonzero(self.yx[2][i])[0]
            for u, j in enumerate(ind):
                try:
                    gs = self.grid_spec[i, j:ind[u+1]]
                except Exception:
                    gs = self.grid_spec[i, j:]
                plot_args = []
                plot_kwargs = {}
                for arg in new_args:
                    plot_args.append(arg[i][j])
                for key in new_kwargs.keys():
                    plot_kwargs[key] = new_kwargs[key][i][j]
                class_mapping_key = plot_type[i][j].lower().replace(' ', '').replace('-', '').replace('_', '')
                if class_mapping_key not in class_mapping.keys():
                    raise ValueError('Plot type: '+plot_type[i][j]+' not recognised')
                self.plot_classes.append(class_mapping[class_mapping_key](gs, self.fig, MTs[i][j], stations=stations[i][j], *plot_args, **plot_kwargs))

        # default is to plot, but can be stopped by adding plot=False to MTplot
        # initialisation arguments
        if plot:
            self.plot(**kwargs)

    def _prep_data(self, param, name):
        """
        Prepares the data to the given multi-plot dimensions

        Returns:
            list: nested list of parameter values of the correct dimensions for the multiplot.
        """
        try:
            if isinstance(param, list):
                if all([isinstance(u, list) for u in param]) and len(param) == self.yx[0]:
                    assert len(param) == self.yx[0]
                    for i, u in enumerate(param):
                        # assert len(param)==sum(self.yx[2][i])
                        for j in range(self.yx[1]):
                            if self.yx[2][i][j] is None:
                                param[i].insert(j, None)
                        assert len(param[i]) == self.yx[1]
                    return param
                elif len(param) == sum(self.yx[2][0]):
                    # X axis params (default)
                    new_param = [param]
                    for i in range(self.yx[0]):
                        new_param.append(param)
                        for j in range(self.yx[1]):
                            if self.yx[2][i][j] is None:
                                new_param[i].insert(j, None)
                    return new_param
                elif len(param) == self.yx[1]:
                    # Y axis params
                    new_param = []
                    for i in range(self.yx[0]):
                        new_param.append([])
                        for j in range(self.yx[1]):
                            if self.yx[2][i][j] is None:
                                new_param[i].append(None)
                            else:
                                new_param[i].append(param[i])
                    return new_param
            # param is individual so need to expand to full region
            new_param = []
            for i in range(self.yx[0]):
                new_param.append([])
                for j in range(self.yx[1]):
                    if self.yx[2][i][j] is None:
                        new_param[i].append(None)
                    else:
                        new_param[i].append(param)
            return new_param
        except AssertionError:
            raise ValueError(name+' incorrect size - needs to be a nested list of size '+str(self.yx[1])+','+str(self.yx[0])+' or an individual value, or to match the x or y axis sizes')

    def plot(self, *args, **kwargs):
        """
        Plots the figure and shows it.

        Calls the plot functions for each plot_class in the multiplot.
        """
        for plot_class in self.plot_classes:
            plot_class(*args, **kwargs)
        self.fig.patch.set_facecolor('w')
        if self.show:
            self.fig.show()
        self.ax_labels(self.show)
        if len(self.save_file):
            self.fig.savefig(self.save_file, dpi=self.save_dpi)
            from matplotlib import pyplot as plt
            plt.close(self.fig)

    def ax_labels(self, show=True):
        """Set or update axis label to axis corners."""
        if self.label:
            if show:
                self.fig.show()
            plot_class_ind = 0
            top_max = []
            for i in range(self.yx[0]):
                ind = np.nonzero(self.yx[2][i])[0]
                t_max = 0
                for u, j in enumerate(ind):
                    t_max = max(
                        [t_max, self.plot_classes[plot_class_ind].ax.get_position().corners()[1, 1]])
                    plot_class_ind += 1
                top_max.append(t_max)
            label = ord('a')
            bot, top, left, right = self.grid_spec.get_grid_positions(self.fig)
            k = 0
            for i in range(self.yx[0]):
                ind = np.nonzero(self.yx[2][i])[0]
                for u, j in enumerate(ind):
                    try:
                        self.labels[k].set_position((left[k], top_max[i]))
                    except Exception:
                        self.labels.append(self.fig.text(left[k], top_max[i],
                                           '('+chr(label)+')', horizontalalignment='right',
                                           verticalalignment='bottom'))
                    label += 1
                    k += 1
            if show:
                self.fig.show()


class _BasePlot(object):

    """
    This is the base class for MT plotting

    This class handles plotting to a single axis including projection and conversion of the moment tensor, setting the figure background etc.

    Should be subclassed
    """
    single = False

    def __init__(self, subplot_spec, fig, MTs, *args, **kwargs):
        """
        BasePlot initialisation

        Args
            subplot_spec: matplotlib subplot spec
            fig: matplotlib figure
            MTs: moment tensor samples (see MTData initialisation docstring for formats)

        Keyword Args
            colormap: str - matplotlib colormap selection (using matplotlib.cm.get_cmap())
            fontsize: int - fontsize for text
            linewidth: float - base linewidth (sometimes thinner or thicker values are used, but relative to this parameter)
            text: bool - flag to show or hide text on the plot
            axis_lines: bool - flag to show or hide axis lines on the plot
            resolution: int - resolution for spherical sampling etc
        """
        show = kwargs.pop('show', True)
        try:
            self.dimension = int(kwargs.get('dimension', 2))
            if self.dimension not in [2, 3]:
                raise Exception
        except Exception:
            raise ValueError('dimension options must be an integer, either 2 or 3, not '+str(kwargs.get('dimension', 2)))
        from matplotlib import pyplot as plt
        if not fig:
            self.fig = plt.figure()
            self.show = show
        else:
            self.fig = fig
            self.show = False
        if not subplot_spec:
            self.grid_spec = gridspec.GridSpec(1, 1)
        else:
            self.grid_spec = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=subplot_spec)
        if self.dimension < 3:
            ax = self.fig.add_subplot(self.grid_spec[0, 0])
            ax = self.fig.gca()
        else:
            ax = self.fig.add_subplot(self.grid_spec[0, 0], projection='3d')
            ax = self.fig.gca()
        self.ax = ax
        if isinstance(MTs, MTData):
            self.MTs = MTs
        else:
            self.MTs = MTData(MTs)

        if self.single and len(self.MTs) > 1:
            raise ValueError('Can only plot single moment tensors in an '+(' '.join(re.findall(
                '[A-Z][^A-Z]*', str(self.__class__).split('.')[-1].lstrip('_').rstrip("'>")))).lower())
        self.colormap = cm.get_cmap(kwargs.get('colormap', DEFAULT_AMP_COLORMAP))
        self.colormap.set_under([0, 0, 0, 0])
        self.fontsize = kwargs.get('fontsize', 10)
        self.kwargs = kwargs
        self.linewidth = float(kwargs.get('linewidth', 1.5))
        self.text = bool(kwargs.get('text', True))
        self.axis_lines = bool(kwargs.get('axis_lines', True))
        self.resolution = int(kwargs.get('resolution', 1000))

    def __call__(self, *args, **kwargs):
        """X(*args,**kwargs)=X.__call__(*args,**kwargs)

        Plots the result
        """
        self.plot(*args, **kwargs)

    def plot(self, MTs=False, *args, **kwargs):
        """Plots the result

        Keyword Args
            MTs: Moment tensors to plot (see MTData initialisation docstring for formats)
            args: args passed to the _ax_plot function (e.g. set local parameters to be different from initialisation values)
            kwargs: kwargs passed to the _ax_plot function (e.g. set local parameters to be different from initialisation values)
        """
        if MTs:
            self.MTs = MTs
        self._convert()
        # Let's remove kwargs that aren't valid
        plot_kwargs = {u: v for u, v in kwargs.items() if u not in ['nodal_line', 'fault_plane', 'TNP',
                                                                    'axis_lines', 'projection', 'lower',
                                                                    'full_sphere', 'show_stations', 'station_distribution',
                                                                    'show_zero_polarity', 'station_markersize', 'station_colors',
                                                                    'show_mean', 'max_number_planes', 'probability_cutoff',
                                                                    'grid_lines', 'marginalised', 'hex_bin', 'type_label',
                                                                    'zorder', 'weights', 'parameter', 'hex_extent',
                                                                    'show_max_likelihood', 'probability',
                                                                    'resolution', 'markersize', 'fontsize',
                                                                    'text', 'colormap', 'linewidth',
                                                                    'hex_bin']}
        handle = self._ax_plot(*args, **plot_kwargs)
        self._background(handle)
        if self.show:
            self.fig.show()
        return handle

    def _convert(self):
        """Handles MT conversion to coordinates (setting xdata,ydata,zdata (if required) and cdata attributes)
        """
        x, y, z, c = self._convert_mts()
        self.xdata = x
        self.ydata = y
        if not isinstance(z, bool):
            self.zdata = z
        self.cdata = c

    def _convert_mts(self):
        """Convert the moment tensors as required"""
        if isinstance(self.__class__, _BasePlot):
            raise NotImplementedError('Must be implemented in the subclass')

    def _ax_plot(self, *args, **kwargs):
        """
        Main plotting function

        Plots the axis
        """
        if isinstance(self.__class__, _BasePlot):
            raise NotImplementedError('Must be implemented in the subclass')

    def _background(self, handle):
        """Create the plot background"""
        if isinstance(self.__class__, _BasePlot):
            raise NotImplementedError('Must be implemented in the subclass')

    def _eigenvector_matrix(self, e1, e2, e3):
        """
        create an eigenvector matrix

        Args
            e1: numpy array/matrix - first eigenvector
            e2: numpy array/matrix - second eigenvector
            e3: numpy array/matrix - third eigenvector

        Returns
            numpy matrix: eigenvector matrix
        """
        return np.matrix([np.array(e1).flatten(), np.array(e2).flatten(), np.array(e3).flatten()])
    # Deep plotting functions

    def _surf_plot(self, x, y, z, c, zorder=0, **kwargs):
        """
        Call to plot a surface
        (handles 2 or 3 dimensions as required)

        Args
            x: numpy array - x coordinates
            y: numpy array - y coordinates
            z: numpy array - z coordinates
            c: numpy array/str- c values

        Keyword Args
            zorder: float - z index of the surface (positive higher)

        """
        if self.dimension < 3:
            return self._2d_surf_plot(x, y, c, **kwargs)
        return self._3d_surf_plot(x, y, z, c, **kwargs)

    def _scatter_plot(self, x, y, z, c, marker='.', markersize=10, zorder=0, **kwargs):
        """
        Call to plot scatter
        (handles 2 or 3 dimensions as required)

        Args
            x: numpy array - x coordinates
            y: numpy array - y coordinates
            z: numpy array - z coordinates
            c: numpy array/str- c values

        Keyword Args
            marker: str - marker description (see matplotlib documentation)
            markersize: float - marker width (squared to correspond to markersize parameter)
            zorder: float - z index of the surface (positive higher)

        """
        try:
            if isinstance(c, np.ndarray) and c.shape == x.shape:
                c = c[~np.isnan(x)]
        except Exception:
            pass
        nans = np.isnan(x)
        if not np.isscalar(x):
            y = y[~nans]
            x = x[~nans]
        if self.dimension < 3:
            return self._2d_scatter_plot(x, y, c, marker, markersize, zorder=zorder, **kwargs)
        if not np.isscalar(z):
            z = z[~nans]
        return self._3d_scatter_plot(x, y, z, c, marker, markersize, zorder=zorder, **kwargs)

    def _line_plot(self, x, y, z, c, linestyle='-', linewidth=1, zorder=0, **kwargs):
        """
        Call to plot a line
        (handles 2 or 3 dimensions as required)

        Args
            x: numpy array - x coordinates
            y: numpy array - y coordinates
            z: numpy array - z coordinates
            c: numpy array/str- c value

        Keyword Args
            linestyle: str - linestyle description (see matplotlib documentation)
            linewidth: float - line width
            zorder: float - z index of the surface (positive higher)

        """
        if self.dimension < 3:
            return self._2d_line_plot(x, y, c, linestyle, linewidth, zorder=zorder, **kwargs)
        return self._3d_line_plot(x, y, z, c, linestyle, linewidth, zorder=zorder, **kwargs)

    def _text(self, x, y, z, text, zorder=2, **kwargs):
        """
        Call to write text
        (handles 2 or 3 dimensions as required)

        Args
            x: numpy array - x coordinates
            y: numpy array - y coordinates
            z: numpy array - z coordinates
            text: str - text value

        Keyword Args
            zorder: float - z index of the surface (positive higher)

        """
        try:
            for i, x_i in enumerate(x):
                try:
                    if isinstance(text, str):
                        raise Exception()
                    t = text[i]
                except Exception:
                    t = text
                self._text(x_i, y[i], z[i], t, **kwargs)
            return
        except Exception:
            # Not iterable
            pass
        if np.isnan(x) or np.isnan(y):
            return
        kwargs['fontsize'] = kwargs.get('fontsize', self.fontsize)
        if self.dimension < 3:
            return self._2d_text(x, y, text, zorder=zorder, **kwargs)
        if np.isnan(z):
            return
        return self._3d_text(x, y, z, text, zorder=zorder, **kwargs)

    def _2d_surf_plot(self, x, y, c, zorder=0, **kwargs):
        """2d surface plot (see _surf_plot documentation)"""
        kwargs.pop('colormap', None)
        kwargs.pop('bins', None)
        # Need to handle nans in x and y here
        valid = ~np.isnan(x)
        assert (valid == ~np.isnan(y)).all(), 'Expect both x and y to have the same NaN values'
        valid_rows = np.bitwise_and.reduce(valid, axis=1)
        assert (valid_rows == np.bitwise_or.reduce(valid, axis=1)).all(), 'Expect to set each row to either nan or non nan'
        return self.ax.pcolormesh(x[valid_rows, :], y[valid_rows, :], c[valid_rows, :], cmap=self.colormap, shading='flat', zorder=zorder, **kwargs)

    def _2d_scatter_plot(self, x, y, c, marker='.', markersize=10, zorder=0, **kwargs):
        """2d scatter plot (see _scatter_plot documentation)"""
        kwargs.pop('color', None)
        kwargs.pop('bins', None)
        return self.ax.scatter(x, y, markersize*markersize, color=c, marker=marker, zorder=zorder, **kwargs)

    def _2d_line_plot(self, x, y, c, linestyle, linewidth, zorder=0, **kwargs):
        """2d line plot (see _line_plot documentation)"""
        kwargs.pop('color', None)
        kwargs.pop('bins', None)
        return self.ax.plot(x, y, color=c, linestyle=linestyle, linewidth=linewidth, zorder=zorder, **kwargs)

    def _2d_text(self, x, y, text, fontsize, zorder=2, **kwargs):
        """2d text plot (see _text_plot documentation)"""
        kwargs.pop('bins', None)
        return self.ax.text(x, y, text, fontsize=fontsize, zorder=zorder, **kwargs)
    # 3D

    def _3d_surf_plot(self, x, y, z, c, zorder=0, **kwargs):
        """3d surface plot (see _surf_plot documentation)"""
        kwargs.pop('colormap', None)
        kwargs.pop('bins', None)
        try:
            return self.ax.plot_surface(x, y, z, facecolors=self.colormap(c), rstride=1, cstride=1,
                                        linewidth=0, antialiased=False, zorder=zorder, **kwargs)
        except Exception:
            return self.ax.plot_surface(x, y, z, color=c, rstride=1, cstride=1,
                                        linewidth=0, antialiased=False, zorder=zorder, **kwargs)

    def _3d_scatter_plot(self, x, y, z, c, marker='.', markersize=10, zorder=0, **kwargs):
        """3d scatter plot (see _scatter_plot documentation)"""
        kwargs.pop('color', None)
        kwargs.pop('bins', None)
        return self.ax.scatter(x, y, z, color=c, marker=marker, **kwargs)

    def _3d_line_plot(self, x, y, z, c, linestyle, linewidth, zorder=0, **kwargs):
        """3d line plot (see _line_plot documentation)"""
        kwargs.pop('color', None)
        kwargs.pop('bins', None)
        return self.ax.plot(x, y, z, kwargs.get('markersize', 2)**2, color=c, linestyle=linestyle, linewidth=linewidth, zorder=zorder, **kwargs)

    def _3d_text(self, x, y, z, text, fontsize, zorder=2, **kwargs):
        """3d text plot (see _text_plot documentation)"""
        kwargs.pop('bins', None)
        return self.ax.text(x, y, z, text, fontsize=fontsize, zorder=zorder, **kwargs)


class _FocalSpherePlot(_BasePlot):

    """Base class for plotting on a focal sphere"""

    def __init__(self, subplot_spec, fig, MTs, stations={}, phase='P', *args, **kwargs):
        """
        Args
            subplot_spec: matplotlib subplot spec
            fig: matplotlib figure
            MTs: moment tensor samples (see MTData initialisation docstring for formats)

        Keyword Args
            phase: str - phase to plot (default='p')
            lower: bool - project lower hemisphere (ie. downward hemisphere)
            full_sphere: bool - plot the full sphere
            fault_plane: bool - plot the fault planes
            nodal_line: bool - plot the nodal lines
            stations: dict - station dict containing keys: 'names','azimuth','takeoff_angle' and 'polarity'
            station_distribution: list - list of station dictionaries corresponding to a location PDF distribution
            show_zero_polarity: bool - flag to show zero polarity receivers
            show_stations: bool - flag to show stations when station_distribution present
            station_markersize: float - station marker size (squared to get area)
            station_colors: tuple - tuple of colors for negative, no, and positive polarity
            TNP: bool - show the TNP axes on the plot
            colormap: str - matplotlib colormap selection (using matplotlib.cm.get_cmap())
            fontsize: int - fontsize for text
            linewidth: float - base linewidth (sometimes thinner or thicker values are used, but relative to this parameter)
            text: bool - flag to show or hide text on the plot
            axis_lines: bool - flag to show or hide axis lines on the plot
            resolution: int - resolution for spherical sampling etc
        """
        super(_FocalSpherePlot, self).__init__(subplot_spec, fig, MTs, *args, **kwargs)
        self.phase = phase
        try:
            self.projection = kwargs.get('projection', 'equalarea').replace('-', '').replace('_', '').replace(' ', '').lower()
            if self.projection not in ['equalarea', 'equalangle']:
                raise ValueError('Projection {} not recognised'.format(self.projection))
        except Exception:
            raise ValueError('projection options must be either "equalarea" or "equalangle", not '+str(kwargs.get('projection', 'equalarea')))
        if self.dimension < 3:
            self._projection_fn = {'equalarea': equal_area, 'equalangle': equal_angle}[self.projection]
        self.lower = kwargs.get('lower', True)
        self.full_sphere = kwargs.get('full_sphere', False)
        self.fault_plane = float(kwargs.get('fault_plane', True))
        self.nodal_line = float(kwargs.get('nodal_line', False))
        self.stations = stations
        self.show_stations = kwargs.get('show_stations', True)
        self.station_distribution = kwargs.get('station_distribution', False)
        self.show_zero_polarity = kwargs.get('show_zero_polarity', True)
        if self.station_distribution:
            # Join together
            self.station_distribution_pdf = np.squeeze(np.array([self.station_distribution['probability']]))
            n_samples = len(self.station_distribution['distribution'])
            n_stations = len(self.station_distribution['distribution'][0]['names'])
            azimuth = np.empty((n_stations, n_samples))
            takeoff_angle = np.empty((n_stations, n_samples))
            names = sorted(self.station_distribution['distribution'][0]['names'])
            for i, station in enumerate(self.station_distribution['distribution']):
                try:
                    name_match = station['names'] == names
                except Exception:
                    try:
                        name_match = (station['names'] == names).any()
                    except Exception:
                        name_match = False
                if not name_match:
                    _name, azimuth[:, i], takeoff_angle[:, i] = zip(*sorted(zip(station['names'], np.array(station['azimuth']).flatten().tolist(),
                                                                                np.array(station['takeoff_angle']).flatten()),
                                                                            key=lambda l: names.index(l[0])))
                else:
                    azimuth[:, i], takeoff_angle[:, i] = (np.array(station['azimuth']).flatten().tolist(), np.array(station['takeoff_angle']).flatten())
            if self.stations and names == self.stations['names']:
                polarity = self.stations['polarity']
            else:
                polarity = []
                if self.stations:
                    station_pol = np.array(self.stations['polarity'])
                    if len(station_pol.shape) > 1:
                        station_pol = np.array(station_pol).flatten()
                for name in names:
                    if self.stations and name in self.stations['names']:
                        polarity.append(station_pol[self.stations['names'].index(name)])
                    else:
                        polarity.append(0)
                polarity = np.array(polarity)
            idx = np.argsort(self.station_distribution_pdf)
            self.station_distribution_pdf = self.station_distribution_pdf[idx]
            # sort to station_distribution_pdf order (min -> max)
            self.station_distribution = {'names': [], 'azimuth': azimuth[:, idx], 'takeoff_angle': takeoff_angle[:, idx], 'polarity': polarity}
        self.station_markersize = kwargs.get('station_markersize', 10)
        self.station_colors = kwargs.get('station_colors', ('k', 'w', 'k'))
        self.TNP = bool(kwargs.get('TNP', True))

    def _convert_mts(self):
        if isinstance(self.__class__, _FocalSpherePlot):
            raise NotImplementedError('Must be implemented in the subclass')

    def _ax_plot(self, *args, **kwargs):
        if isinstance(self.__class__, _FocalSpherePlot):
            raise NotImplementedError('Must be implemented in the subclass')

    def plot_plane(self, strike, dip, c, radians=False, zorder=0, *args, **kwargs):
        """
        Plot a plane on the focal sphere

        Args
            strike - strike of the plane
            dip - dip of the plane
            c - color description(see matplotlib documentation)

        Keyword Args
            radians: bool - strike and dip in radians or degrees
            zorder: float - z order of the plane (larger is on top)
        """
        if not radians:
            strike = strike*np.pi/180.
            dip = dip*np.pi/180.
        v1 = toa_vec((strike+np.pi/2), (np.pi/2-dip), radians=True)
        v2 = toa_vec(strike, np.pi/2, radians=True)
        [x, y, z] = self._get_great_circle(v1, v2)
        kwargs['linewidth'] = kwargs.get('linewidth', 0.75*self.linewidth)
        kwargs['linestyle'] = kwargs.get('linestyle', '-')
        if self.dimension < 3:
            # switched so north is up and east right
            y, x = self._projection_fn(x, y, z, lower=self.lower, full_sphere=self.full_sphere)
        return self._line_plot(x, y, z, c, zorder=zorder, *args, **kwargs)

    def _get_great_circle(self, v1, v2, az=(0, 2*np.pi), n=500):
        """
        Get the great circle described by the two vectors v1 and v2

        Args
            v1: numpy matrix/tuple - vector on the great circle
            v2: numpy matrix/tuple - a different vector on the great circle

        Keyword Args
            n_quadrants: int - number of quadrants of the great circle to calculate

        Returns
            numpy array, numpy array, numpy array: tuple of coordinates (x,y,z)
        """
        if isinstance(v1, tuple):
            v1 = np.matrix([[v1[0]], [v1[1]], [v1[2]]])
        if isinstance(v2, tuple):
            v2 = np.matrix([[v2[0]], [v2[1]], [v2[2]]])
        az = np.linspace(*az, num=n)
        if np.abs(np.matrix(v1).T*np.matrix(v2)) > 10**-10:
            v3 = np.cross(v1.T, v2.T).T
            v3 /= np.sqrt(np.sum(v3*v3))
            v2 = np.cross(v3.T, v1.T).T
        r = v1*np.cos(az)+v2*np.sin(az)
        return np.array(r[0, :]).flatten(), np.array(r[1, :]).flatten(), np.array(r[2, :]).flatten()

    def _get_small_circle(self, pole, alpha, az=(0, 2*np.pi)):
        """
        Get small circle about the pole with opening angle alpha

        Args
            pole: numpy matrix - pole of the small circle
            alpha: float - angle of the small circle from the pole

        Keyword Args
            az: tuple - lower and upper limits of the azimuth to calculate for the small circle

        Returns
            numpy array, numpy array, numpy array: tuple of coordinates (x,y,z)
        """
        # Calculate about z axis and rotate
        n = 360
        X = np.ones((3, n))
        az = np.linspace(az[0], az[1], n)
        X[0, :] = np.sin(alpha)*np.cos(az)
        X[1, :] = np.sin(alpha)*np.sin(az)
        X[2, :] = X[2, :]*np.cos(alpha)
        # Calculate rotation matrix - simple rotation about a rotation axis
        R = self._rotation_matrix(pole)
        v = R*np.matrix(X)
        return v[0, :], v[1, :], v[2, :]

    def _rotation_matrix(self, zaxis):
        """
        Calculate rotation matrix to rotate the vector zaxis onto the z-axis

        Args
            zaxis: numpy matrix - vector to be rotated onto the z-axis

        Returns
            numpy matrix: rotation matrix
        """
        rAxis = np.cross(np.matrix(zaxis).T, np.matrix([[0], [0], [1]]).T).T
        if rAxis.any():
            theta = -np.arcsin(np.sqrt(np.sum(rAxis.T*rAxis))/(np.sqrt(np.sum(zaxis.T*zaxis))))
            rAxis = rAxis/np.sqrt(np.sum(rAxis.T*rAxis))
            rX = rAxis[0]
            rY = rAxis[1]
            rZ = rAxis[2]
            R = (np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])*np.cos(theta)+np.sin(theta) *
                 np.matrix([[0, -rZ, rY], [rZ, 0, -rX], [-rY, rX, 0]])+(1-np.cos(theta))*(rAxis.T*rAxis))
        else:
            R = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        return R

    def _get_nodal_line(self, T, N, P, E):
        """
        Gets the nodal line (values with zero amplitude)

        Args
            T: numpy matrix - Tension (eigen)vector
            N: numpy matrix - Neutral (eigen)vector
            P: numpy matrix - Pressure (eigen)vector
            E: numpy array - Eigenvalues (E[0]>=E[1]>=E[2])

        Returns
            numpy array, numpy array, numpy array - tuple of x y z coordinates of the nodal lines
        """
        sign_check = False
        # Get unique values and counts
        if StrictVersion(np.__version__) < StrictVersion('1.10.0'):
            unique, inverse = np.unique(E, return_inverse=True)
            counts = np.zeros(len(unique), np.int)
            np.add.at(counts, inverse, 1)
        else:
            unique, counts = np.unique(E, return_counts=True)
            idx2 = np.flipud(np.argsort(unique))
            counts = counts[idx2]
            unique = unique[idx2]
        if min(counts) > 1:
            raise ValueError('Cannot plot nodal lines as no unique eigenvalue')
        if len(unique) == 3:
            sign_check = True
            # all unique so want unique sign
            if StrictVersion(np.__version__) < StrictVersion('1.10.0'):
                unique, inverse = np.unique(np.sign(E), return_inverse=True)
                counts = np.zeros(len(unique), np.int)
                np.add.at(counts, inverse, 1)
            else:
                unique, counts = np.unique(np.sign(E), return_counts=True)
                idx2 = np.flipud(np.argsort(unique))
                counts = counts[idx2]
                unique = unique[idx2]
        # sort low -> high
        idx = np.argsort(unique)
        unique = unique[idx]
        counts = counts[idx]
        # Get the unqiue eigenvalue (and unique sign if possible)
        E3 = unique[counts == counts.min()][0]
        if sign_check:
            # Get eigenvalue with unique sign
            E3 = E[np.sign(E) == E3]
        idx = np.nonzero(E == E3)[0][0]
        # Handle RH coordinate system allocation - convert TNP axes and
        # eigenvalues to RH system with e3 corresponding to E3
        if idx == 0:
            # T axis is e3
            e3 = T
            E2 = E[2]
            e2 = P
            E1 = E[1]
            e1 = N
        elif idx == 1:
            # N axis is e3
            e3 = N
            E2 = E[0]
            e2 = T
            E1 = E[2]
            e1 = P
        elif idx == 2:
            # P axis is e3
            e3 = P
            E2 = E[1]
            e2 = N
            E1 = E[0]
            e1 = T
        # Get nodal lines in eigenvector basis
        n = 500
        X = np.ones((3, n))
        az = np.linspace(0, 2*np.pi, n)
        if min(E) > 0:
            raise ValueError('Smallest eigenvalue must be less than zero')
        az = np.array(az).flatten()
        # Eigenvalue ordering independent - E3 chosen as unique eigenvalue and rotate around that.
        # P amplitude nodal line (solved from diagonalised P amplitudes)
        theta = np.arctan(
            np.sqrt(-E3/(E1*np.cos(az)*np.cos(az)+E2*np.sin(az)*np.sin(az))))
        X = np.matrix(
            [np.cos(az)*np.sin(theta), np.sin(az)*np.sin(theta), np.cos(theta)])
        # Rotate to geographical coordinate bases (NED)
        R = self._eigenvector_matrix(e1, e2, e3).T  # R
        v = R*np.matrix(X)
        return np.array(v[0, :]).flatten(), np.array(v[1, :]).flatten(), np.array(v[2, :]).flatten()

    def _axis_lines(self, r):
        """Plots the axis lines through the origin"""
        self._line_plot([-r, r], [0, 0], [0, 0], 'k')
        self._line_plot([0, 0], [-r, r], [0, 0], 'k')
        if self.dimension < 3:
            self._line_plot([0, 0], [0, 0], [-r, r], 'k')

    def _boundary_lines(self, r):
        """
        Plots the boundary lines, fault planes, nodal lines and axis lines as selected by the initialisation parameters

        Args
            r: float - circle radius (depends on the projection and determined in _background())
        """
        if self.dimension < 3:
            circle1 = patches.Circle((0, 0), r, facecolor='none', edgecolor='k', linewidth=self.linewidth, zorder=10, axes=self.ax)
            self.ax.add_artist(circle1)
        if self.fault_plane:
            self._fault_plane(self.MTs, 'k')
        if self.nodal_line:
            self._nodal_line(self.MTs, 'k')
        if self.axis_lines:
            self._axis_lines(r)

    def _fault_plane(self, MT, c, zorder=0, *args, **kwargs):
        """
        Plots the fault planes for a given moment tensor

        Args
            MT: MTData - Moment tensor to plot
            c: str - color description (see matplotlib documentation)

        Keyword Args
            zorder: float - z order for the plane (larger is higher)
            args: passed through to the plot_plane (and _line_plot) calls so can set e.g. linewidth
            kwargs: passed through to the plot_plane (and _line_plot) calls so can set e.g. linewidth
        """
        self.plot_plane(MT.strike1, MT.dip1, c, radians=False, zorder=zorder, *args, **kwargs)
        self.plot_plane(MT.strike2, MT.dip2, c, radians=False, zorder=zorder, *args, **kwargs)

    def _plot_TNP(self, MT, c=['w', 'k', 'w'], marker=('*', '*', '*'), markersize=10, zorder=0, *args, **kwargs):
        """
        Plots the TNP axes for the given moment tensor

        Args
            MT: MTData - Moment tensor to plot

        Keyword Args
            c: tuple - color description for T N P axes (see matplotlib documentation)
            marker: tuple - marker description for T N P axes (see matplotlib documentation)
            markersize: float - marker width - squared to give marker area
            zorder: float - z order for the plane (larger is higher)
            args: passed through to the plot_plane (and _line_plot) calls so can set e.g. linewidth
            kwargs: passed through to the plot_plane (and _line_plot) calls so can set e.g. linewidth
        """
        name = ['T', 'N', 'P']
        for i, vec in enumerate([self.MTs.T, self.MTs.N, self.MTs.P]):
            vec = vec.copy()
            if self.dimension < 3:
                vec = np.append(vec, -vec, 1)
                y, x = self._projection_fn(vec[0, :], vec[1, :], vec[2, :], lower=self.lower, full_sphere=self.full_sphere)
                vec[0, :] = x
                vec[1, :] = y
            self._scatter_plot(np.array(vec[0, :]).flatten(), np.array(vec[1, :]).flatten(), np.array(
                vec[2, :]).flatten(), c[i], marker=marker[i], markersize=markersize, zorder=zorder, *args, **kwargs)
            if self.text:
                delta = 0.05
                self._text(np.array(vec[0, :]).flatten()+delta, np.array(vec[1, :]).flatten()+delta,
                           np.array(vec[2, :]).flatten()+delta, name[i], zorder=zorder+1)

    def _nodal_line(self, MTs, c, zorder=0, *args, **kwargs):
        """
        Plots the nodal line for a given moment tensor

        Args
            MT: MTData - Moment tensor to plot
            c: str - color description (see matplotlib documentation)

        Keyword Args
            zorder: float - z order for the line (larger is higher)
            args: passed through to the  _line_plot call so can set e.g. linewidth
            kwargs: passed through to the _line_plot call so can set e.g. linewidth
        """
        if self.phase.lower() == 'p':
            if MTs.E.min() < 0:
                x, y, z = self._get_nodal_line(MTs.T, MTs.N, MTs.P, MTs.E)
                x1, y1, z1 = self._get_nodal_line(-MTs.T, -MTs.N, -MTs.P, MTs.E)
                if self.dimension < 3:
                    # switched so north is up and east right
                    y, x = self._projection_fn(x, y, z, lower=self.lower, full_sphere=self.full_sphere)
                    # switched so north is up and east right
                    y1, x1 = self._projection_fn(x1, y1, z1, lower=self.lower, full_sphere=self.full_sphere)
                self._line_plot(x, y, z, c, '-', 0.75*self.linewidth, zorder=zorder, *args, **kwargs)
                self._line_plot(x1, y1, z1, c, '-', 0.75*self.linewidth, zorder=zorder, *args, **kwargs)

    def _background(self, handle):
        """
        Plots the axes background

        Args
            handle: handle of the plot object (for surface plots)
        """
        super(_FocalSpherePlot, self)._background(handle)
        if self.dimension < 3 and self.full_sphere:
            lim = 2
        elif self.dimension < 3:
            lim = {'equalarea': np.sqrt(2), 'equalangle': 1}[self.projection]
        else:
            lim = 1
        self._boundary_lines(lim)
        self._stations(colors=self.station_colors, lim=lim)
        if self.TNP:
            self._plot_TNP(self.MTs)
        lim *= 1.1
        self.ax.set_aspect(1)
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        if self.dimension > 2:
            self.ax.set_zlim(-lim, lim)
        self.ax.set_axis_off()
        try:
            self.fig.patch.set_facecolor('w')
        except Exception:
            pass

    def _stations(self, lim=np.sqrt(2), colors=('k', 'w', 'k'), marker=('v', 'o', '^')):
        """
        Plots stations on the focal sphere

        Keyword Args
            lim: float - limit of the focal sphere (projection dependent)
            colors: tuple - tuple of color descriptions for stations with negative, no and positive polarity
            marker: tuple - tuple of marker descriptions for stations with negative, no and positive polarity
        """
        if self.station_distribution:
            text = self.text
            station_markersize = self.station_markersize
            self.station_markersize = 3
            self.text = False
            self.station_distribution_pdf = self.station_distribution_pdf/self.station_distribution_pdf.max()
            self.station_distribution_pdf = np.array(np.matrix(np.squeeze(self.station_distribution_pdf)))
            zeros = 0*self.station_distribution_pdf
            r = np.append(np.append(0.8*self.station_distribution_pdf+0.2, zeros, 0), zeros, 0).T
            g = np.append(np.append(zeros, 0.8*self.station_distribution_pdf+0.2, 0), zeros, 0).T
            b = np.append(np.append(zeros, zeros, 0), 0.8*self.station_distribution_pdf+0.2, 0).T
            self._plot_stations(marker=('.', '.', '.'), colors=(b, g, r), lim=lim,
                                zorder=self.station_distribution_pdf/self.station_distribution_pdf.max(),
                                **self.station_distribution)
            self.station_markersize = station_markersize
            self.text = text
            if not self.show_stations:
                return

        if not len(self.stations):
            return  # no stations
        # Check polarity probabilities
        if 'polarity' in self.stations.keys() and not np.all(np.array(self.stations['polarity']).flatten().astype(np.int) == np.array(self.stations['polarity']).flatten()):
            # self.stations['polarity'] are polarity probabilities
            # Color by value (blue for negative red for positive)
            # Scale so 0.5 is white
            scaled_probabilities = np.abs(np.atleast_2d(2*(np.abs(self.stations['polarity'])-0.5)))
            ones = 0*scaled_probabilities+1
            r = np.append(np.append(ones, 0.8*(1-scaled_probabilities)+0.2, 0), 0.8*(1-scaled_probabilities)+0.2, 0).T
            w = np.append(np.append(ones, ones, 0), ones, 0).T
            b = np.append(np.append(0.8*(1-scaled_probabilities)+0.2, 0.8*(1-scaled_probabilities)+0.2, 0), ones, 0).T
            marker = ('o', 'o', 'o')
            colors = (b, w, r)
        self._plot_stations(marker=marker, colors=colors, lim=lim, zorder=2, **self.stations)

    def _plot_stations(self, names, azimuth, takeoff_angle, polarity, marker=('v', 'o', '^'), colors=('k', 'w', 'k'), radians=False, lim=np.sqrt(2), zorder=0):
        """
        Handles the station plotting on  the focal sphere.

        If self.text is True then the station names are annotated using radial arrows
        Args
            names: list - list of station names
            azimuth: numpy array - array of station azimuths
            takeoff_angle: numpy array - array of station takeoff angles
            polarity: numpy array - array of station polarities

        Keyword Args
            radians: bool - flag for station angles in radians [default is False]
            lim: float - limit of the focal sphere (projection dependent)
            colors: tuple - tuple of color descriptions for stations with negative, no and positive polarity
            marker: tuple - tuple of marker descriptions for stations with negative, no and positive polarity
        """
        v = toa_vec(azimuth, takeoff_angle, radians=radians)
        x = np.squeeze(np.array(v[0, :]))
        y = np.squeeze(np.array(v[1, :]))
        z = np.squeeze(np.array(v[2, :]))
        if self.dimension < 3:
            if not self.full_sphere:
                try:
                    if not isinstance(colors[0],  str) and np.squeeze(v[0, :]).shape[1] == colors[0].shape[0]:
                        colors = list(colors)
                        colors[0] = np.append(colors[0], colors[0], 0)
                        colors[1] = np.append(colors[1], colors[1], 0)
                        colors[2] = np.append(colors[2], colors[2], 0)
                except Exception:
                    pass
                try:
                    x = np.append(x, -x, 1)
                    y = np.append(y, -y, 1)
                    z = np.append(z, -z, 1)
                except IndexError:
                    x = np.append(x, -x, 0)
                    y = np.append(y, -y, 0)
                    z = np.append(z, -z, 0)

            # switched so north is up and east right
            y, x = self._projection_fn(x, y, z, lower=self.lower, full_sphere=self.full_sphere)
            idx = ~np.isnan(x)
            if len(names):
                names = np.array(names)
                try:
                    names = np.append(names, names, 1)
                except IndexError:
                    names = np.append(names, names, 0)
                names = names[idx]
            z = z[idx]
            y = y[idx]
            x = x[idx]
        # if mapping by zorder, then loop over all zorder values rather than
        # polarities
        if marker == ('.', '.', '.') and len(np.squeeze(zorder)) > 1:
            try:
                zorder = np.append(zorder, zorder, 1)
            except IndexError:
                zorder = np.append(zorder, zorder, 0)
            if len(idx.shape) > 1:
                if not isinstance(colors[0], str):
                    col = np.array(colors)[polarity+1, :, :][idx, :]
                polarity_array = (np.kron(polarity, np.ones((idx.shape[1], 1))).T)[idx]
                zorder = np.squeeze(np.kron(zorder, np.ones((idx.shape[0], 1)))[idx])
                for zo in np.unique(zorder):
                    if not self.show_zero_polarity:
                        self._scatter_plot(x[(zorder == zo)*(polarity_array != 0)], y[(zorder == zo)*(polarity_array != 0)], z[(zorder == zo)*(polarity_array != 0)],
                                           col[(zorder == zo)*(polarity_array != 0)], '.', edgecolor=col[(zorder == zo)*(polarity_array != 0)],
                                           markersize=self.station_markersize, zorder=zo)
                    else:
                        self._scatter_plot(x[(zorder == zo)], y[(zorder == zo)], z[(zorder == zo)], col[(zorder == zo)], '.', edgecolor=col[(zorder == zo)],
                                           markersize=self.station_markersize, zorder=zo)
        # check polarity probability
        elif not np.all(np.array(polarity).flatten().astype(np.int) == np.array(polarity).flatten()):
            if self.dimension < 3 and len(polarity):
                polarity = np.array(polarity)
                try:
                    polarity = np.append(polarity, polarity, 1)
                except Exception:
                    polarity = np.append(polarity, polarity, 0)
                polarity = polarity[idx]
            for i, pol in enumerate(polarity.flatten()):
                pol = pol-np.sign(pol)*0.5
                if pol < -0.5 or 0 < pol < 0.5:
                    pol = 0
                elif 0 > pol > -0.5 or pol > 0.5:
                    pol = 2
                else:
                    pol = 1
                col = colors[pol]
                if marker[pol] == '.':
                    self._scatter_plot(x[i], y[i], z[i], np.atleast_2d(col[i]), marker[pol], edgecolor=col[i], markersize=self.station_markersize, zorder=zorder)
                else:
                    self._scatter_plot(x[i], y[i], z[i], np.atleast_2d(col[i]), marker[pol], edgecolor='k', markersize=self.station_markersize, zorder=zorder)

        else:
            if self.dimension < 3 and len(polarity):
                polarity = np.array(polarity)
                try:
                    polarity = np.append(polarity, polarity, 1)
                except Exception:
                    polarity = np.append(polarity, polarity, 0)
                polarity = polarity[idx]
            for pol in np.unique(polarity):
                if pol == 0 and not self.show_zero_polarity:
                    continue
                col = colors[pol]
                # polarity
                polarity_array = polarity
                if len(idx.shape) > 1:
                    if not isinstance(colors[0], str):
                        col = np.kron(colors[pol], np.ones((idx.shape[0], 1, 1)))[idx, :]
                    polarity_array = (np.kron(polarity, np.ones((idx.shape[1], 1))).T)[idx]
                if marker[pol] == '.':
                    self._scatter_plot(x[polarity_array == pol], y[polarity_array == pol], z[polarity_array == pol], col, marker[pol], edgecolor=col,
                                       markersize=self.station_markersize, zorder=zorder)
                else:
                    self._scatter_plot(x[polarity_array == pol], y[polarity_array == pol], z[polarity_array == pol], col, marker[pol], edgecolor='k',
                                       markersize=self.station_markersize, zorder=zorder)
        delta = 0.05
        if self.text and len(names):
            # Add radial arrows to the station point
            for i, name in enumerate(names):
                # mult=0.55+(0.03*(count%3))
                if not self.show_zero_polarity and not polarity[i]:
                    continue
                if self.dimension < 3:
                    mult = 1.2*lim
                    norm = np.sqrt(x[i]*x[i]+y[i]*y[i])
                    # Zero polarity for labelling
                    text = self.ax.annotate(name, xy=(x[i], y[i]), xycoords='data', xytext=(mult*x[i]/norm, mult*y[i]/norm), textcoords='data',
                                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', lw=1),
                                            size=10, ha="center", path_effects=[PathEffects.withStroke(linewidth=2, foreground="w")])

                    text.arrow_patch.set_path_effects([PathEffects.Stroke(linewidth=3, foreground="w"),
                                                       PathEffects.Normal()])
                else:
                    self._text(x[i]+delta, y[i]+delta, z[i]+delta, name)


class _AmplitudePlot(_FocalSpherePlot):

    """
    Amplitude plotting (beachball)

    parameter single is set to True, as can only plot one source per axis
    """
    single = True

    def _convert_mts(self):
        """
        Calculate and convert the moment tensor to the amplitude on the focal sphere

        Returns
            numpy array, numpy array, numpy array, numpy array: tuple of x,y,z and (scaled) c values
        """
        # convert mts to x,y,z,c
        phi = np.linspace(0.001, 2*np.pi-0.001, self.resolution)
        theta = np.linspace(0.001, np.pi-0.001, self.resolution)
        phi, theta = np.meshgrid(phi, theta)
        a = station_angles((np.reshape(phi, (np.prod(phi.shape), 1)), np.reshape(theta, (np.prod(theta.shape), 1))),
                           self.phase.lower(), radians=True)
        x = np.sin(theta)*np.cos(phi)  # North
        y = np.sin(theta)*np.sin(phi)  # East
        z = np.cos(theta)
        c = np.array(np.reshape(np.matrix(self.MTs.MTs).T*np.matrix(a).T, phi.shape))
        return x, y, z, c/np.abs(c).max()

    def _ax_plot(self, *args, **kwargs):
        """Projects and plots the amplitude surface

        Returns
             matplotlib surface object
        """
        if self.dimension < 3:
            self.ydata, self.xdata = self._projection_fn(
                self.xdata, self.ydata, self.zdata, lower=self.lower, full_sphere=self.full_sphere)  # switched so north is up and east right
        return self._surf_plot(self.xdata, self.ydata, self.zdata, self.cdata, *args, **kwargs)

    def _background(self, handle):
        """Sets the colormap to be symmetrical and plots the axes background.

        Args
            handle: handle of the plot object (for surface plots)
        """
        try:
            data = handle.get_array()
            vmin = data[~np.isnan(data)].min()
            vmax = data[~np.isnan(data)].max()
            clim = max([abs(vmin), abs(vmax)])
            handle.set_clim(-clim, clim)
        except Exception:
            pass
        super(_AmplitudePlot, self)._background(handle)
    # helper functions (histogram etc)


class _RadiationPlot(_AmplitudePlot):

    """Plot the radiation pattern
    """

    def _convert_mts(self):
        """Scales the x,y,z coordinates by the amplitude

        Returns
            numpy array, numpy array, numpy array, numpy array: tuple of x,y,z and (scaled) c values
        """
        x, y, z, c = super(_RadiationPlot, self)._convert_mts()
        return x*c, y*c, z*c, c

    def _boundary_lines(self, r):
        """Only plots the axis lines
        """
        if self.axis_lines:
            self._axis_lines(r)

    def _stations(self, *args, **kwargs):
        """Doesn't plot any stations
        """
        return


class _FaultPlanePlot(_FocalSpherePlot):

    """Plots the fault plane distribution

    parameter single is set to False as multiple sources can be plotted per axes
    """
    single = False

    def __init__(self, subplot_spec, fig, MTs, stations={}, probability=[], phase='p', *args, **kwargs):
        """
        Args
            subplot_spec: matplotlib subplot spec
            fig: matplotlib figure
            MTs: moment tensor samples (see MTData initialisation docstring for formats)

        Keyword Args
            probability: numpy array - moment tensor probabilities
            phase: str - phase to plot (default='p')
            lower: bool - project lower hemisphere (ie. downward hemisphere)
            full_sphere: bool - plot the full sphere
            fault_plane: bool - plot the fault planes
            nodal_line: bool - plot the nodal lines
            stations: dict - station dict containing keys: 'names','azimuth','takeoff_angle' and 'polarity'
            station_distribution: list - list of station dictionaries corresponding to a location PDF distribution
            show_zero_polarity: bool - flag to show zero polarity receivers
            show_stations: bool - flag to show stations when station_distribution present
            station_markersize: float - station marker size (squared to get area)
            station_colors: tuple - tuple of colors for negative, no, and positive polarity
            TNP: bool - show the TNP axes on the plot
            colormap: str - matplotlib colormap selection (using matplotlib.cm.get_cmap())
            fontsize: int - fontsize for text
            linewidth: float - base linewidth (sometimes thinner or thicker values are used, but relative to this parameter)
            text: bool - flag to show or hide text on the plot
            axis_lines: bool - flag to show or hide axis lines on the plot
            resolution: int - resolution for spherical sampling etc
            show_max_likelihood: bool - show the maximum likelihood solution in the default color
            show_mean: bool - show the mean orientation source in green (thicker plane is better constrained)
            color: set color for maximum likelihood fault plane (show_max_likelihood)
        """
        super(_FaultPlanePlot, self).__init__(subplot_spec, fig, MTs, stations, phase, *args, **kwargs)
        self.station_colors = kwargs.get('station_colors', ('w', 'w', 'w'))
        self.show_max_likelihood = kwargs.get('show_max_likelihood', False)
        self.show_mean = kwargs.get('show_mean', False)
        self.color = kwargs.get('color', DEFAULT_COLOR)
        self.max_number_planes = kwargs.get('max_number_planes', 40000)
        self.probability_cutoff = kwargs.get('probability_cutoff', 0.01)
        # Scale probability and sort the MTs
        if not len(probability):
            probability = self.MTs.probability
        if len(probability) > 1:
            if probability.max() > 1:
                probability /= probability.max()
            assert(len(probability) == self.MTs.shape[1])
            idx = np.argsort(probability)
            probability = probability[idx]
            self.MTs = self.MTs[:, idx]
            self.MTs._set_probability(probability)
        if self.probability_cutoff > 0 and len(probability):
            self.MTs = self.MTs[:, self.MTs.probability > self.probability_cutoff*self.MTs.probability.max()]
        if self.max_number_planes > 0 and len(self.MTs) > self.max_number_planes:
            self.MTs = self.MTs[:, np.random.choice(np.arange(0, len(self.MTs)),
                                                    self.max_number_planes,
                                                    replace=False)]
        if len(probability) > 1:
            self.MTs.probability /= self.MTs.probability.max()
            idx = np.argsort(self.MTs.probability)
            self.MTs = self.MTs[:, idx]

    def _convert_mts(self):
        """Returns tuple of None as not used to plot the source"""
        return None, None, None, None

    def _ax_plot(self, *args, **kwargs):
        """
        Plots the fault planes/nodal lines/TNP axes with color scaled by the probability values (if set)

        Darker colors are more likely
        """
        # plot fault planes scaled by probability
        if self.fault_plane:
            for i in range(self.MTs.shape[1]):
                if len(self.MTs.probability) > 1:
                    probability = self.MTs.probability[i]
                elif len(self.MTs.probability):
                    probability = self.MTs.probability
                else:
                    probability = 1
                self._fault_plane(self.MTs[:, i], str(1-probability), zorder=float(i)-self.MTs.shape[1],
                                  *args, **kwargs)
        elif self.nodal_line:
            for i in range(self.MTs.shape[1]):
                if len(self.MTs.probability) > 1:
                    probability = self.MTs.probability[i]
                elif len(self.MTs.probability):
                    probability = self.MTs.probability
                else:
                    probability = 1
                self._nodal_line(self.MTs[:, i], str(1-probability), zorder=float(i)-self.MTs.shape[1],
                                 *args, **kwargs)
        if self.TNP:
            text_flag = self.text
            self.text = False
            for i in range(self.MTs.shape[1]):
                if len(self.MTs.probability) > 1:
                    probability = self.MTs.probability[i]
                elif len(self.MTs.probability):
                    probability = self.MTs.probability
                else:
                    probability = 1
                self._plot_TNP(self.MTs[:, i], ([0, 0, probability], [0, probability, 0], [probability, 0, 0]),
                               marker=('.', '.', '.'), markersize=2, zorder=float(i)-self.MTs.shape[1], *args, **kwargs)
            self.text = text_flag
        if self.show_max_likelihood and len(self.MTs.probability) > 1:
            if self.fault_plane:
                self._fault_plane(self.MTs[:, self.MTs.probability == self.MTs.probability.max()], self.color,
                                  zorder=1, *args, **kwargs)
            elif self._nodal_line:
                self._nodal_line(self.MTs[:, self.MTs.probability == self.MTs.probability.max()], self.color,
                                 zorder=1, *args, **kwargs)
        if self.show_mean and self.fault_plane:
            kwargs['linewidth'] = 2*kwargs.get('linewidth', self.linewidth)
            self.plot_plane(self.MTs.mean_strike, self.MTs.mean_dip,
                            'g', radians=False, zorder=1, *args, **kwargs)
            strike2, dip2, rake2 = SDR_SDR(self.MTs.mean_strike/rad_deg, self.MTs.mean_dip/rad_deg,
                                           self.MTs.mean_rake/rad_deg)
            kwargs['linewidth'] = 0.25*kwargs.get('linewidth', self.linewidth)
            self.plot_plane(strike2*rad_deg, dip2*rad_deg, 'g', zorder=1, radians=False, *args, **kwargs)

    def _background(self, handle):
        """
        Plots the background (ignoring fault planes, nodal lines and TNP axes)

        Args
            handle: handle of the plot object (for surface plots)
        """
        fp = self.fault_plane
        nl = self.nodal_line
        TNP = self.TNP
        self.fault_plane = False
        self.TNP = False
        self.nodal_line = False
        super(_FaultPlanePlot, self)._background(handle)
        self.fault_plane = fp
        self.TNP = TNP
        self.nodal_line = nl


class _HistPlot(_BasePlot):

    """
    Base histogram plot class

    Used by plot classes which histogram the data

    """

    def __init__(self, subplot_spec, fig, MTs, probability=[], **kwargs):
        """
        Args
            subplot_spec: matplotlib subplot spec
            fig: matplotlib figure
            MTs: moment tensor samples (see MTData initialisation docstring for formats)

        Keyword Args
            probability: numpy array - moment tensor probabilities
            colormap: str - matplotlib colormap selection (using matplotlib.cm.get_cmap())
            fontsize: int - fontsize for text
            linewidth: float - base linewidth (sometimes thinner or thicker values are used, but relative to this parameter)
            text: bool - flag to show or hide text on the plot
            axis_lines: bool - flag to show or hide axis lines on the plot
            resolution: int - resolution for spherical sampling etc
            grid_lines: bool - show the interior grid lines
            marginalised: bool - marginalise the PDF (defailt is True)
            color: set marker color
            type_label: bool - show the label of the different types
            hex_bin: bool - use the hex-bin histogram type (slightly smoother)
            bins: int/array/list of arrays - bins for numpy histogram call
        """
        kwargs['colormap'] = kwargs.get('colormap', DEFAULT_HIST_COLORMAP)
        super(_HistPlot, self).__init__(subplot_spec, fig, MTs, **kwargs)
        self.grid_lines = kwargs.get('grid_lines', True)
        self.marginalised = kwargs.get('marginalised', True)
        self.color = kwargs.get('color', DEFAULT_COLOR)
        self.type_label = bool(kwargs.get('type_label', True))
        self.hex_bin = bool(kwargs.get('hex_bin', True))
        self.bins = bool(kwargs.get('bins', 10))
        if len(probability) > 1:
            if probability.max() > 1:
                probability /= probability.max()
            self.MTs._set_probability(probability)

    def _ax_plot(self, *args, **kwargs):
        """
        Main plotting function

        Plots the axis
        """
        if type(self.data_y) != bool:
            return self._2d_hist(self.data_x, self.data_y, self.data_c, *args, **kwargs)
        else:
            return self._hist(self.data_x, self.data_c, *args, **kwargs)

    def _hist(self, data, c=None, bins=10, **kwargs):
        """
        1D histogram

        Args
            data: numpy array - data to bin

        Keyword Args
            bins: int - number of bins to use
            c: numpy array - weights to use for each sample in the histogram

        Returns
            matplotlib histogram object
        """
        # bins can be a range- arguments are passed to plt.hist
        if c is not None:
            kwargs['weights'] = c
        if bins is not None and len(bins) == 0:
            bins = None
        return self.ax.hist(data, bins, **kwargs)

    def _max_2d_for_hist(self, data_x, data_y, c, bins):
        """
        Get Max value for 2d hist (silhouette type plot rather than marginalised
            (see Pugh, D J, 2015. Bayesian Source Inversion of Microseismic Events, Thesis))

        Returns
            numpy array, numpy array, numpy array: tuple of numpy arrays describing the silhouette coordinates and maximum probability values for each point.
        """
        # Check bins
        # Get max values
        xbins = bins[0]
        ybins = bins[1]
        xindices = np.digitize(data_x, xbins)
        yindices = np.digitize(data_y, ybins)
        coords = np.array([xindices, yindices])
        inds = unique_columns(coords)
        c2 = np.empty(inds.shape[1])
        reduced_data_x = np.empty(inds.shape[1])
        reduced_data_y = np.empty(inds.shape[1])
        for i in range(inds.shape[1]):
            c2[i] = (c[(coords[0, :] == inds[0, i])*(coords[1, :] == inds[1, i])]).max()
            reduced_data_x[i] = data_x[(coords[0, :] == inds[0, i])*(coords[1, :] == inds[1, i])*(c == c2[i])][0]
            reduced_data_y[i] = data_y[(coords[0, :] == inds[0, i])*(coords[1, :] == inds[1, i])*(c == c2[i])][0]
        return reduced_data_x.flatten(), reduced_data_y.flatten(), c2

    def _2d_hist(self, data_x, data_y, c=None, bins=10, nx_hex=100, hex_extent=[-1, 1, -1, 1], **kwargs):
        """
        2D histogram

        Args
            data_x: numpy array - x data to bin
            data_y: numpy array - y data to bin

        Keyword Args
            bins: int - number of bins to use
            c: numpy array - weights to use for each sample in the histogram
            nx_hex: number of hexes to use when hex_bin is True
            hex_extent: minimum extent of the hexagon region
            hex_bin: use the matplotlib hexbin function to histogram (2D only)

        Returns
            matplotlib histogram/hexbin object
        """
        if kwargs.get('hex_bin', self.hex_bin) and self.dimension < 3:
            hex_extent[0] = min([min(data_x), hex_extent[0]])
            hex_extent[1] = max([max(data_x), hex_extent[1]])
            hex_extent[2] = min([min(data_y), hex_extent[2]])
            hex_extent[3] = max([max(data_y), hex_extent[3]])
            if self.marginalised:
                reduce_C_function = np.sum
            else:
                reduce_C_function = np.max
            return self.ax.hexbin(data_x, data_y, c, gridsize=nx_hex, extent=hex_extent, cmap=self.colormap, reduce_C_function=reduce_C_function, **kwargs)
        else:
            if c is not None:
                if not self.marginalised:
                    N, xedge, yedge = np.histogram2d(data_x, data_y, bins=bins, **kwargs)
                    bins = (xedge, yedge)
                    data_x, data_y, c = self._max_2d_for_hist(data_x, data_y, c, bins)
                kwargs['weights'] = c
            N, xedge, yedge = np.histogram2d(data_x, data_y, bins=bins, **kwargs)

            try:
                kwargs.pop('weights')
            except Exception:
                pass
            # Transpose due to variation between array dimensions ([y,x]) and
            # plot [x,y]
            N = N.T
            N[N == 0] = np.nan
            return self._surf_plot(xedge, yedge, 0, N, **kwargs)

    def _background(self, handle):
        """
        Plots the axes background

        Args
            handle: handle of the plot object (for surface plots)
        """
        super(_HistPlot, self)._background(self)
        try:
            data = handle.get_array()
            vmin = min([data[~np.isnan(data)].min(), 0])
            vmax = data[~np.isnan(data)].max()
            handle.set_clim(vmin, vmax)
        except Exception:
            pass
        self.ax.set_axis_off()
        try:
            self.fig.patch.set_facecolor('w')
        except Exception:
            pass


class _LunePlot(_FocalSpherePlot, _HistPlot):

    """
    Plots the source on the fundamental eigenvalue lune

    Can be either a histogram or scatter plot.

    parameter single is set to False as multiple sources can be plotted per axes
    """
    single = False

    def _convert_mts(self):
        """
        Converts the moment tensors for plotting

        Returns
            numpy array, numpy array, 0, numpy array: tuple of gamma,delta,0,probability values
        """
        return self.MTs.gamma.copy(), self.MTs.delta.copy(), 0, self.MTs.probability

    def _ax_plot(self, *args, **kwargs):
        """
        Plots the sources on the lune (as a 2D histogram of the PDF if the probability values are set)

        Returns
            matplotlib scatter/2dhist object
        """
        if len(self.MTs.probability):
            kwargs['bins'] = kwargs.get('bins', [np.linspace(-np.pi/6, np.pi/6, 34), np.arccos(np.linspace(1, -1, 101))])
            try:
                kwargs['bins'] = [np.linspace(-np.pi/6, np.pi/6, kwargs['bins']/3), np.arccos(np.linspace(1, -1, kwargs['bins']))]
            except Exception:
                pass
            return self._2d_hist(self.xdata, self.ydata, self.cdata, *args, **kwargs)
        else:
            kwargs['color'] = kwargs.get('color', self.color)
            color = kwargs.pop('color')
            # project to vectors
            x, y, z = self._lune_coords(self.xdata, self.ydata)
            if self.dimension < 3:
                x, y = self._projection_fn(x, y, z, False, False)
            return self._scatter_plot(x, y, z, color, marker='o', *args, **kwargs)

    def _lune_coords(self, gamma, delta):
        """
        Calculates the lune coordinates for a given gamma and delta values

        Args
            gamma: numpy array/float - gamma values
            delta: numpy array/float - delta values

        Returns
            numpy array, numpy array, numpy array: tuple of x, y and z coordinates
        """
        beta = np.pi/2-delta
        if not isinstance(gamma, np.ndarray):
            gamma = np.array(gamma)
        if not isinstance(beta, np.ndarray):
            beta = np.array(beta)
        oshape = gamma.shape
        try:
            gamma = np.reshape(gamma, np.prod(gamma.shape))
        except Exception:
            pass
        try:
            beta = np.reshape(beta, np.prod(beta.shape))
        except Exception:
            pass
        if self.dimension < 3:
            R = np.matrix([[0, 1, 0], [0, 0, 1], [-1, 0, 0]])
            X = np.matrix([np.cos(gamma)*np.sin(beta), np.sin(gamma)*np.sin(beta), np.cos(beta)])
            if X.shape[0] != 3:
                X = X.T
            vec = R*X
            x = vec[0, :]
            y = vec[1, :]
            z = vec[2, :]
        else:
            x = (np.cos(gamma)*np.sin(beta))
            y = (np.sin(gamma)*np.sin(beta))
            z = (np.cos(beta))
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        if 1 in gamma.shape or len(gamma.shape) <= 1:
            x = x.flatten()
            y = y.flatten()
            z = z.flatten()
        try:
            x = np.reshape(x, oshape)
        except Exception:
            pass
        try:
            y = np.reshape(y, oshape)
        except Exception:
            pass
        try:
            z = np.reshape(z, oshape)
        except Exception:
            pass
        return x, y, z

    def _2d_hist(self, data_g, data_d, c=None, bins=10, nx_hex=100, hex_extent=[-0.5, 0.5, -1.5, 1.5], **kwargs):
        """
        2D histogram - handles gamma delta conversion and plotting

        Args
            data_g: numpy array - g data to bin
            data_d: numpy array - d data to bin

        Keyword Args
            bins: int - number of bins to use
            c: numpy array - weights to use for each sample in the histogram
            nx_hex: number of hexes to use when hex_bin is True
            hex_extent: minimum extent of the hexagon region
            hex_bin: use the matplotlib hexbin function to histogram (2D only)

        Returns
            matplotlib histogram/hexbin object
        """
        if kwargs.get('hex_bin', self.hex_bin) and self.dimension < 3:
            kwargs['zorder'] = kwargs.get('zorder', 100)
            data_x, data_y = self._projection_fn(*self._lune_coords(data_g, data_d), lower=False, full_sphere=False)
            hex_extent[0] = min([min(data_x), hex_extent[0]])
            hex_extent[1] = max([max(data_x), hex_extent[1]])
            hex_extent[2] = min([min(data_y), hex_extent[2]])
            hex_extent[3] = max([max(data_y), hex_extent[3]])
            if self.marginalised:
                reduce_C_function = np.sum
            else:
                reduce_C_function = np.max
            artist = self.ax.hexbin(np.array(data_x).flatten(), np.array(data_y).flatten(), c, gridsize=nx_hex, extent=hex_extent, cmap=self.colormap, reduce_C_function=reduce_C_function, **kwargs)
            artist.set_zorder(kwargs.get('zorder', 100))
            return artist
        else:
            data_b = np.pi/2-data_d
            if c is not None:
                if not self.marginalised:
                    N, gedge, bedge = np.histogram2d(data_g, data_b, bins=bins, **kwargs)
                    bins = (gedge, bedge)
                    data_g, data_b, c = self._max_2d_for_hist(data_g, data_b, c, bins)
                kwargs['weights'] = c
            N, gedge, bedge = np.histogram2d(data_g, data_b, bins=bins, **kwargs)
            kwargs.pop('weights')
            g, b = np.meshgrid(gedge, bedge)
            [x, y, z] = self._lune_coords(g, np.pi/2-b)
            if self.dimension < 3:
                x, y = self._projection_fn(x, y, z, lower=False, full_sphere=False)
            # Transpose due to variation between array dimensions ([y,x]) and
            # plot [x,y]
            N = N.T
            N[N == 0] = np.nan
            return self._surf_plot(x, y, z, N, **kwargs)

    def _get_small_circle(self, pole, alpha, az=(0, 2*np.pi)):
        """
        Get and convert a small circle onto the Lune

        Args
            pole: numpy matrix - pole of the small circle
            alpha: float - angle of the small circle from the pole

        Keyword Args
            az: tuple - lower and upper limits of the azimuth to calculate for the small circle

        Returns
            numpy array, numpy array, numpy array: tuple of coordinates (x,y,z)
        """
        x, y, z = super(_LunePlot, self)._get_small_circle(pole, alpha, az)
        if self.dimension < 3:
            R = np.matrix([[0, 1, 0], [0, 0, 1], [-1, 0, 0]])
            X = np.matrix([np.array(x).flatten(), np.array(y).flatten(), np.array(z).flatten()])
            if X.shape[0] != 3:
                X = X.T
            vec = R*X
            x = vec[0, :]
            y = vec[1, :]
            z = vec[2, :]
        return x, y, z

    def _boundary_lines(self, *args):
        """
        Get and plot boundary lines

        Also plots type labels if type_label flag set on initialisation
        """
        # Exterior lines
        [x, y, z] = self._get_great_circle(self._lune_coords(0, np.pi/2), self._lune_coords(np.pi/6, 0))
        ind = np.abs(np.arctan2(y, x)) < np.pi/2
        x = x[ind]
        y = y[ind]
        z = z[ind]
        if self.dimension < 3:
            x, y = self._projection_fn(x, y, z, False, False)
            if len(self.MTs.probability):
                x1 = np.append(x, np.array([0, 1, 1, 0]))
                y1 = np.append(y, np.array([-1.5, -1.5, 1.5, 1.5]))
                p = patches.Polygon(np.array([x1, y1]).T, closed=False, edgecolor='w', facecolor='w', zorder=1)
                p.set_zorder(1)
                self.ax.add_artist(p)

        self._line_plot(x, y, z, 'k', zorder=101)
        [x, y, z] = self._get_great_circle(self._lune_coords(0, np.pi/2), self._lune_coords(-np.pi/6, 0))
        ind = np.abs(np.arctan2(y, x)) > np.pi/2
        x = x[ind]
        y = y[ind]
        z = z[ind]
        if self.dimension < 3:
            x, y = self._projection_fn(x, y, z, False, False)
            if len(self.MTs.probability):
                x1 = np.append(x, np.array([0, -1, -1, 0]))
                y1 = np.append(y, np.array([-1.5, -1.5, 1.5, 1.5]))
                p = patches.Polygon(np.array([x1, y1]).T, closed=False, edgecolor='w', facecolor='w', zorder=1)
                p.set_zorder(1)
                self.ax.add_artist(p)
        self._line_plot(x, y, z, 'k', zorder=101)
        # Interior lines
        grey = '0.6'
        if self.grid_lines:
            n = 10
            for i in range(n):
                x, y, z = self._get_small_circle(np.matrix([0, 0, 1]).T, (i+1)*np.pi/(n+2), [-np.pi/6, np.pi/6])
                ls = '--'
                if self.dimension < 3:
                    x, y = self._projection_fn(x, y, z, lower=False, full_sphere=True)
                if float(i+1)/(n+2) == 0.5:
                    ls = '-'
                self._line_plot(x.flatten(), y.flatten(), z.flatten(), grey, ls, zorder=-5)
            for i in range(5):
                theta = -3*np.pi/18+(i+1)*np.pi/18
                [x, y, z] = self._get_great_circle(self._lune_coords(0, np.pi/2), self._lune_coords(theta, 0))
                ls = '--'
                zorder = -5
                if theta < 0:
                    ind = np.abs(np.arctan2(y, x)) < np.pi/2
                else:
                    ind = np.abs(np.arctan2(y, x)) >= np.pi/2
                if theta == 0:
                    ls = '-'
                    zorder = -4
                x = x[ind]
                y = y[ind]
                z = z[ind]
                if self.dimension < 3:
                    x, y = self._projection_fn(x, y, z, lower=True, full_sphere=False)
                    ind = np.argsort(y)
                    x = x[ind]
                    y = y[ind]

                self._line_plot(x, y, z, c=grey, linestyle=ls, zorder=zorder)
        if self.type_label:
            for label, coord, ha, va in [('CLVD', [2, -1, -1], 'right', 'center'), ('CLVD', [-2, 1, 1], 'left', 'center'), ('Explosion', [1, 1, 1], 'center', 'bottom'),
                                         ('Implosion', [-1, -1, -1], 'center', 'top'), ('DC', [1, 0, -1], 'center', 'bottom'), (r'TC$_+$', [3, 1, 1], 'right', 'bottom'),
                                         (r'TC$_-$', [-3, -1, -1], 'left', 'top')]:
                coord.append(0)
                coord.append(0)
                coord.append(0)
                mt = MTData(coord)
                delta = 0.06
                x, y, z = self._lune_coords(mt.gamma, mt.delta)
                if self.dimension < 3:
                    x, y = self._projection_fn(x, y, z, lower=False, full_sphere=True)
                    if ha == 'right':
                        x -= delta
                    elif ha == 'left':
                        x += delta
                    if va == 'top':
                        y -= delta
                    elif va == 'bottom':
                        y += delta
                self._text(x, y, z, text=label, horizontalalignment=ha, verticalalignment=va, fontsize=self.fontsize, color='k', zorder=10001)
    # Pass through functions

    def _stations(self, *args, **kwargs):
        """Pass-through function - nothing plotted"""
        pass

    def _plot_TNP(self, *args, **kwargs):
        """Pass-through function - nothing plotted"""
        pass

    def _nodal_line(self, *args, **kwargs):
        """Pass-through function - nothing plotted"""
        pass


class _HudsonPlot(_HistPlot):

    """
    Hudson plot class

    Plots the source on the u-v or tau-k Hudson plot
    """

    def __init__(self, subplot_spec, fig, MTs, projection='uv', probability=[], **kwargs):
        """
        Args
            subplot_spec: matplotlib subplot spec
            fig: matplotlib figure
            MTs: moment tensor samples (see MTData initialisation docstring for formats)

        Keyword Args
            projection: str - select the projection type from uv or tauk [default is uv]
            probability: numpy array - moment tensor probabilities
            colormap: str - matplotlib colormap selection (using matplotlib.cm.get_cmap())
            fontsize: int - fontsize for text
            linewidth: float - base linewidth (sometimes thinner or thicker values are used, but relative to this parameter)
            text: bool - flag to show or hide text on the plot
            axis_lines: bool - flag to show or hide axis lines on the plot
            resolution: int - resolution for spherical sampling etc
            grid_lines: bool - show the interior grid lines
            marginalised: bool - marginalise the PDF (defailt is True)
            color: set marker color
            type_label: bool - show the label of the different types
            hex_bin: bool - use the hex-bin histogram type (slightly smoother)
            bins: int/array/list of arrays - bins for numpy histogram call
        """
        super(_HudsonPlot, self).__init__(subplot_spec, fig, MTs, probability, **kwargs)
        self.xy = {'uv': ('u', 'v'), 'tauk': ('tau', 'k'), 'tk': ('tau', 'k'), 'equalarea': ('u', 'v')}[projection]

    def _convert_mts(self):
        """
        Converts the moment tensors to the chosen source parameters

        Returns
            numpy array, numpy array, 0, numpy array: tuple of hudson parameters, 0 and probability
        """
        return getattr(self.MTs, self.xy[0]).copy(), getattr(self.MTs, self.xy[1]).copy(), 0, self.MTs.probability

    def _ax_plot(self, *args, **kwargs):
        """
        Plots the sources on the hudson plot (as a 2D histogram of the PDF if the probability values are set)

        Returns
            matplotlib scatter/2dhist object
        """
        self.dimension = 2
        if len(self.MTs.probability):
            kwargs['bins'] = kwargs.get('bins', np.linspace(-4./3, 4./3, 100))
            try:
                kwargs['bins'] = np.linspace(-4./3, 4./3, kwargs['bins'])
            except Exception:
                pass
            return self._2d_hist(self.xdata, self.ydata, self.cdata, *args, **kwargs)
        else:
            kwargs['color'] = kwargs.get('color', self.color)
            color = kwargs.pop('color')
            return self._scatter_plot(self.xdata, self.ydata, self.zdata, color, marker='o', *args, **kwargs)

    def _background(self, handle):
        """
        Sets limits and plots the axes background

        Args
            matplotlib surface object

        """
        if self.xy == ('u', 'v'):
            xlim = 4./3.
            ylim = 1
        else:
            xlim = 1
            ylim = 1
        self._boundary_lines()
        xlim *= 1.1
        ylim *= 1.1
        self.ax.set_aspect(1)
        self.ax.set_xlim(-xlim, xlim)
        self.ax.set_ylim(-ylim, ylim)
        super(_HudsonPlot, self)._background(handle)

    def _boundary_lines(self, *args):
        """
        Get and plot boundary lines

        Also plots type labels if type_label flag set on initialisation
        """
        # Outer lines
        # Axis lines
        # Grid lines
        if self.xy == ('u', 'v'):
            return self._uv_boundary_lines()
        else:
            return self._tk_boundary_lines()

    def _uv_boundary_lines(self):
        """
        Get and plot boundary lines for the u-v plot

        Also plots type labels if type_label flag set on initialisation
        """
        # outer patches
        if len(self.MTs.probability):
            p = patches.Polygon(np.array([[0, 0, 1.5, 1.5, 4/3.], [-1, -1.5, -1.5, 1/3., 1/3.]]).T,
                                closed=False, edgecolor='w', facecolor='w', zorder=100)
            self.ax.add_artist(p)
            p = patches.Polygon(np.array([[0, 0, 1.5, 1.5, 4/3.], [1, 1.5, 1.5, 1/3., 1/3.]]).T,
                                closed=False, edgecolor='w', facecolor='w', zorder=100)
            self.ax.add_artist(p)
            p = patches.Polygon(np.array([[0, 0, -1.5, -1.5, -4/3.], [1, 1.5, 1.5, -1/3., -1/3.]]).T,
                                closed=False, edgecolor='w', facecolor='w', zorder=100)
            self.ax.add_artist(p)
            p = patches.Polygon(np.array([[0, 0, -1.5, -1.5, -4/3.], [-1, -1.5, -1.5, -1/3., -1/3.]]).T,
                                closed=False, edgecolor='w', facecolor='w', zorder=100)
            self.ax.add_artist(p)
        s = 0.5
        # outer lines
        if self.axis_lines:
            self._line_plot(np.array([0, 0]), np.array([-1, 1]), 0, c='k', linewidth=s*self.linewidth, zorder=-1)
            self._line_plot(np.array([-1, 1]), np.array([0, 0]), 0, c='k', linewidth=s*self.linewidth, zorder=-1)
        self._line_plot(np.array([0, 4/3.]), np.array([-1, 1/3.]), 0, c='k', linewidth=self.linewidth, zorder=101)
        self._line_plot(np.array([0, -4/3.]), np.array([-1, -1/3.]), 0, c='k', linewidth=self.linewidth, zorder=101)
        self._line_plot(np.array([-4/3., 0]), np.array([-1/3., 1]), 0, c='k', linewidth=self.linewidth, zorder=101)
        self._line_plot(np.array([0, 4/3.]), np.array([1, 1/3.]), 0, c='k', linewidth=self.linewidth, zorder=101)
        grey = '0.6'
        if self.grid_lines:
            # diagonal line
            self._line_plot(np.array([-4/3., 4/3.]), np.array([-1/3., 1/3.]),
                            0, c=grey, linestyle='--', linewidth=s*self.linewidth, zorder=-2)
            n = 4
            for i in range(n):
                T = float(i*(1./(n)))
                k = float(i*(1./(n)))
                # second/fourth quadrant have k=v
                self._line_plot(np.array([0, (1-(k))]), np.array([-k, -k]), 0, c=grey, linestyle='--',
                                linewidth=s*self.linewidth, zorder=-2)
                self._line_plot(np.array([0, -(1-(k))]), np.array([k, k]), 0, c=grey, linestyle='--',
                                linewidth=s*self.linewidth, zorder=-2)
                self._line_plot(np.array([0, -(T)]), np.array([1, 0]), 0, c=grey, linestyle='--',
                                linewidth=s*self.linewidth, zorder=-2)
                self._line_plot(np.array([0, (T)]), np.array([-1, 0]), 0, c=grey, linestyle='--',
                                linewidth=s*self.linewidth, zorder=-2)
                # first quadrant
                # A
                self._line_plot(np.array([0, 4.*T/(4.-T)]), np.array([1, T/(4.-T)]), 0, c=grey,
                                linestyle='--', linewidth=s*self.linewidth, zorder=-2)
                if i <= np.floor(n/4.):
                    self._line_plot(np.array([0, 4.*k/(1-2*k)]), np.array([k, k/(1-2.*k)]), 0, c=grey,
                                    linestyle='--', linewidth=s*self.linewidth, zorder=-2)
                else:
                    self._line_plot(np.array([0, 2*(1-k)/(1+k)]), np.array([k, 2.*k/(1+k)]), 0, c=grey,
                                    linestyle='--', linewidth=s*self.linewidth, zorder=-2)
                # B
                self._line_plot(np.array([T, T/(1-0.25*T)]), np.array([0, T/(4.-T)]), 0, c=grey,
                                linestyle='--', linewidth=s*self.linewidth, zorder=-2)
                # third quadrant
                # A
                self._line_plot(np.array([0, -4.*T/(4.-T)]), np.array([-1, -T/(4.-T)]), 0, c=grey,
                                linestyle='--', linewidth=s*self.linewidth, zorder=-2)
                if i <= np.floor(n/4.):
                    self._line_plot(np.array([0, -4*k/(1-2*k)]), np.array([-k, -k/(1-2*k)]), 0, c=grey,
                                    linestyle='--', linewidth=s*self.linewidth, zorder=-2)
                else:
                    self._line_plot(np.array([0, -2.*(1-k)/(1+k)]), np.array([-k, -2.*k/(1+k)]), 0, c=grey,
                                    linestyle='--', linewidth=s*self.linewidth, zorder=-2)
                # B
                self._line_plot(np.array([-T, -T/(1-0.25*T)]), np.array([0, -T/(4.-T)]), 0, c=grey,
                                linestyle='--', linewidth=s*self.linewidth, zorder=-2)
            # extra k lines in B part of first and third quadrant
            # self._line_plot(np.array([0.5,1.125]),np.array([0.125,0.125]),0,c=grey,linestyle='--',linewidth=s*self.linewidth)
            # self._line_plot(np.array([-0.5,-1.125]),np.array([-0.125,-0.125]),0,c=grey,linestyle='--',linewidth=s*self.linewidth)

        if self.type_label:
            self._text(-1.05, 0, 0.5, 'CLVD ', horizontalalignment='right',
                       verticalalignment='center', fontsize=self.fontsize, color='k', zorder=101)
            self._text(1.05, 0, 0.5, ' CLVD', horizontalalignment='left',
                       verticalalignment='center', fontsize=self.fontsize, color='k', zorder=101)
            self._text(0, 1.05, 0.5, 'Explosion', horizontalalignment='center',
                       verticalalignment='bottom', fontsize=self.fontsize, color='k', zorder=101)
            self._text(0, -1.05, 0.5, 'Implosion', horizontalalignment='center',
                       verticalalignment='top', fontsize=self.fontsize, color='k', zorder=101)
            self._text(0, 0, 0.5, 'DC', horizontalalignment='center',
                       verticalalignment='bottom', fontsize=self.fontsize, color='k', zorder=101)
            self._text(-0.4444, 0.5556, 0.5, r'TC$_+$', horizontalalignment='right',
                       verticalalignment='bottom', fontsize=self.fontsize, color='k', zorder=101)
            self._text(0.4444, -0.5556, 0.5, r'TC$_-$', horizontalalignment='left',
                       verticalalignment='top', fontsize=self.fontsize, color='k', zorder=101)

    def _tk_boundary_lines(self):
        """Get and plot boundary lines for the tau-k plot

        Also plots type labels if type_label flag set on initialisation
        """
        p = patches.Polygon(np.array([[0, 0, 1.5, 1.5, 1], [-1, -1.5, -1.5, 0, 0]]).T,
                            edgecolor='w', facecolor='w', zorder=100)
        self.ax.add_artist(p)
        p = patches.Polygon(np.array([[0, 0, 1.5, 1.5, 1], [1, 1.5, 1.5, 0, 0]]).T,
                            edgecolor='w', facecolor='w', zorder=100)
        self.ax.add_artist(p)
        p = patches.Polygon(np.array([[0, 0, -1.5, -1.5, -1], [1, 1.5, 1.5, 0, 0]]).T,
                            edgecolor='w', facecolor='w', zorder=100)
        self.ax.add_artist(p)
        p = patches.Polygon(np.array([[0, 0, -1.5, -1.5, -1], [-1, -1.5, -1.5, 0, 0]]).T,
                            edgecolor='w', facecolor='w', zorder=100)
        self.ax.add_artist(p)
        s = 0.5
        if self.axis_lines:
            self._line_plot(np.array([0, 0]), np.array([-1, 1]), 0, c='k', linewidth=s*self.linewidth, zorder=-1)
            self._line_plot(np.array([-1, 1]), np.array([0, 0]), 0, c='k', linewidth=s*self.linewidth, zorder=-1)

        self._line_plot(np.array([0, 1]), np.array([-1, 0]), 0, c='k', linewidth=self.linewidth, zorder=101)
        self._line_plot(np.array([0, 1]), np.array([1, 0]), 0, c='k', linewidth=self.linewidth, zorder=101)
        self._line_plot(np.array([0, -1]), np.array([1, 0]), 0, c='k', linewidth=self.linewidth, zorder=101)
        self._line_plot(np.array([0, -1]), np.array([-1, 0]), 0, c='k', linewidth=self.linewidth, zorder=101)
        grey = '0.6'
        if self.grid_lines:
            n = 4
            d = float(1)/n
            for i in range(n):
                i = float(i)
                self._line_plot(np.array([0, i*d]), np.array([-1, 0]), 0, c=grey, linestyle='--', linewidth=s*self.linewidth, zorder=-2)
                self._line_plot(np.array([0, i*-d]), np.array([-1, 0]), 0, c=grey, linestyle='--', linewidth=s*self.linewidth, zorder=-2)
                self._line_plot(np.array([0, i*d]), np.array([1, 0]), 0, c=grey, linestyle='--', linewidth=s*self.linewidth, zorder=-2)
                self._line_plot(np.array([0, i*-d]), np.array([1, 0]), 0, c=grey, linestyle='--', linewidth=s*self.linewidth, zorder=-2)
            self._line_plot(np.array([-0.8, 0.8]), np.array([-0.2, 0.2]), 0, c=grey, linestyle='--', linewidth=s*self.linewidth, zorder=-2)
        if self.type_label:
            self._text(-1.0, 0, 0.5, 'CLVD ', horizontalalignment='right',
                       verticalalignment='center', fontsize=self.fontsize, color='k', zorder=101)
            self._text(1.0, 0, 0.5, ' CLVD', horizontalalignment='left',
                       verticalalignment='center', fontsize=self.fontsize, color='k', zorder=101)
            self._text(0, 1.05, 0.5, 'Explosion', horizontalalignment='center',
                       verticalalignment='bottom', fontsize=self.fontsize, color='k', zorder=101)
            self._text(0, -1.05, 0.5, 'Implosion', horizontalalignment='center',
                       verticalalignment='top', fontsize=self.fontsize, color='k', zorder=101)
            self._text(0, 0, 0.5, 'DC', horizontalalignment='center',
                       verticalalignment='bottom', fontsize=self.fontsize, color='k', zorder=101)
            self._text(-0.4444, 0.5556, 0.5, r'TC$_+$', horizontalalignment='right',
                       verticalalignment='bottom', fontsize=self.fontsize, color='k', zorder=101)
            self._text(0.4444, -0.5556, 0.5, r'TC$_-$', horizontalalignment='left',
                       verticalalignment='top', fontsize=self.fontsize, color='k', zorder=101)


class _RiedeselJordanPlot(_FocalSpherePlot):

    """Plots the source as a Riedesel-Jordan type plot"""

    def __init__(self, subplot_spec, fig, MTs, *args, **kwargs):
        """
        Args
            subplot_spec: matplotlib subplot spec
            fig: matplotlib figure
            MTs: moment tensor samples (see MTData initialisation docstring for formats)

        Keyword Args
            phase: str - phase to plot (default='p')
            lower: bool - project lower hemisphere (ie. downward hemisphere)
            full_sphere: bool - plot the full sphere
            fault_plane: bool - plot the fault planes
            nodal_line: bool - plot the nodal lines
            show_stations: bool - flag to show stations when station_distribution present
            station_markersize: float - sets the source type marker sizes (squared to get area)
            station_colors: tuple - tuple of colors for negative, no, and positive polarity
            TNP: bool - show the TNP axes on the plot
            colormap: str - matplotlib colormap selection (using matplotlib.cm.get_cmap())
            color: str - matplotlib color for source region
            fontsize: int - fontsize for text
            linewidth: float - base linewidth (sometimes thinner or thicker values are used, but relative to this parameter)
            text: bool - flag to show or hide text on the plot
            axis_lines: bool - flag to show or hide axis lines on the plot
            resolution: int - resolution for spherical sampling etc
        """
        super(_RiedeselJordanPlot, self).__init__(subplot_spec, fig, MTs, *args, **kwargs)
        self.color = kwargs.get('color', DEFAULT_COLOR)

    def _convert_mts(self):
        """
        Converts the moment tensor and calculates the base vectors

        Returns tuple of None.
        """
        # if self.show_amplitude:
        #     pass
        E = self.MTs.E
        T = np.array(self.MTs.T)
        N = np.array(self.MTs.N)
        P = np.array(self.MTs.P)
        self.mt = E[0][0]*T+E[1][0]*N+E[2][0]*P
        self.mt = self.mt.flatten()
        self.mt /= np.sqrt(self.mt.T*self.mt)
        self.dc = (T-P)/np.sqrt(2)
        self.clvd1 = (0.5*T+0.5*N-P)/np.sqrt(3./2)
        self.clvd2 = (T-0.5*N-0.5*P)/np.sqrt(3./2)
        self.iso = (T+N+P)/np.sqrt(3)
        return None, None, None, None

    def _ax_plot(self, *args, **kwargs):
        """
        Plots the sources on the Riedesel-Jordan plot (on the focal sphere)

        Returns
            None
        """
        kwargs['color'] = kwargs.get('color', 'k')
        color = kwargs.pop('color')
        i = 0
        kwargs['markersize'] = kwargs.get(
            'markersize', self.station_markersize)
        markersize_kw = kwargs.pop('markersize')
        delta = 0.04
        for attr, marker in [('mt', 'o'), ('dc', 'v'), ('clvd1', 'v'), ('clvd2', 'v'), ('iso', 'v')]:
            x = getattr(self, attr)[0]
            y = getattr(self, attr)[1]
            z = getattr(self, attr)[2]
            if self.dimension < 3:
                x, y = self._projection_fn(
                    x, y, z, lower=True, full_sphere=False, back_project=True)
            c = color
            markersize = markersize_kw
            # Raising Deprecation Warning
            if (isinstance(attr, str) and attr != 'mt') and np.all(self.mt == getattr(self, attr)):
                c = 'w'
                markersize = markersize_kw/2
            self._scatter_plot(
                x, y, z, c=c, marker=marker, markersize=markersize, zorder=i, **kwargs)
            if self.text:
                x = x+delta
                z = z+delta
                if attr == 'mt':
                    y = y-2*delta
                else:
                    y = y+delta
                self._text(x, y, z, attr, zorder=6+i)
            i += 1
        kwargs['markersize'] = markersize_kw

    def _background(self, *args):
        """Plot the axes background ignoring the fault plane, TNP and nodal line flags"""
        fp = self.fault_plane
        nl = self.nodal_line
        TNP = self.TNP
        self.fault_plane = False
        self.TNP = False
        self.nodal_line = False
        super(_RiedeselJordanPlot, self)._background(*args)
        self.fault_plane = fp
        self.TNP = TNP
        self.nodal_line = nl

    def _stations(self, *args, **kwargs):
        return

    def _boundary_lines(self, r):
        """
        Plots the boundary lines, and the valid source region with color set by the color flag.

        Args
            r: float - circle radius (depends on the projection and determined in _background())
        """
        # Great circle
        proj_dc_x, proj_dc_y = self._projection_fn(self.dc[0], self.dc[1], self.dc[2], lower=True, full_sphere=False, back_project=True)
        lower_flag = False
        upper_flag = False
        for i, clvd in enumerate([self.clvd1, self.clvd2]):
            [x, y, z] = self._get_great_circle(self.iso, clvd)
            if self.dimension < 3:
                x, y = self._projection_fn(x, y, z, lower=True, full_sphere=False, back_project=False)
                # Find nearest match to dc x
                idx_nan = ~np.isnan(x)
                x = x[idx_nan]
                y = y[idx_nan]
                z = z[idx_nan]
                if (y != x).any():
                    theta = np.arctan2(y, x)
                    pos = theta[theta > 0]
                    neg = theta[theta < 0]
                    if len(neg) and len(pos) and not (np.abs(neg).min() < np.pi/6 and np.abs(pos).min() < np.pi/6):
                        theta = np.mod(theta, 2*np.pi)
                    idx2 = np.argsort(theta)
                else:  # Straight line so sort by r and sign
                    # biggest to smallest
                    idx2 = np.flipud(np.argsort((y**2+x**2)*np.sign(np.arctan2(y, x))))
                x = x[idx2]
                y = y[idx2]
                z = z[idx2]
                idx = [np.abs(x-proj_dc_x[0]) == (np.abs(x-proj_dc_x[0])).min()]
                if x.min() > proj_dc_x[0] or y[idx] < proj_dc_y:
                    if lower_flag and x.min() > proj_dc_x[0]:
                        upper = [x.copy(), y.copy(), z.copy()]
                    elif lower_flag:
                        # TODO check this path
                        # upper = lower[:]
                        lower = [x.copy(), y.copy(), z.copy()]
                    else:
                        lower = [x.copy(), y.copy(), z.copy()]
                    lower_flag = True
                else:
                    if upper_flag and x.min() > proj_dc_x[0]:
                        # TODO - check this path
                        # lower = upper[:]
                        upper = [x.copy(), y.copy(), z.copy()]
                    elif upper_flag:
                        lower = [x.copy(), y.copy(), z.copy()]
                    else:
                        upper = [x.copy(), y.copy(), z.copy()]
                        upper_flag = True
            self._line_plot(x, y, z, c=self.color, linestyle='-', linewidth=self.linewidth, zorder=-1)
            # Check here for lower upper
        # Add deviatoric circle
        [x, y, z] = self._get_great_circle(self.dc, self.clvd1, n=100)
        costheta = x*self.dc[0]+y*self.dc[1]+z*self.dc[2]
        costheta0 = self.clvd1[0]*self.dc[0]+self.clvd1[1]*self.dc[1]+self.clvd1[2]*self.dc[2]
        x[np.abs(costheta) < costheta0] = np.nan
        y[np.abs(costheta) < costheta0] = np.nan
        z[np.abs(costheta) < costheta0] = np.nan
        if self.dimension < 3:
            x, y = self._projection_fn(x, y, z, lower=True, full_sphere=False, back_project=True)
        # Scatter to allow for non continuous line
        self._scatter_plot(x, y, z, c='k', marker='.', markersize=1, zorder=-1)
        # Get fill area
        if self.dimension < 3:
            # Check points (and reverse if necessary)
            dc_iso_gc_x, dc_iso_gc_y, dc_iso_gc_z = self._get_great_circle(
                self.dc, self.iso)
            proj_dc_iso_gc_x, proj_dc_iso_gc_y = self._projection_fn(
                dc_iso_gc_x, dc_iso_gc_y, dc_iso_gc_z, lower=True, full_sphere=False)
            idx_nan = ~np.isnan(proj_dc_iso_gc_x)
            proj_dc_iso_gc_x = proj_dc_iso_gc_x[idx_nan]
            proj_dc_iso_gc_y = proj_dc_iso_gc_y[idx_nan]
            if (proj_dc_iso_gc_y != proj_dc_iso_gc_x).any():
                theta = np.arctan2(proj_dc_iso_gc_y, proj_dc_iso_gc_x)
                pos = theta[theta > 0]
                neg = theta[theta < 0]
                if len(neg) and len(pos) and (np.abs(neg).min() > np.pi/6 or np.abs(pos).min() > np.pi/6):
                    theta = np.mod(theta, 2*np.pi)
                idx2 = np.argsort(theta)
            else:
                idx2 = np.flipud(np.argsort((proj_dc_iso_gc_y**2+proj_dc_iso_gc_x**2)*np.sign(
                    np.arctan2(proj_dc_iso_gc_y, proj_dc_iso_gc_x))))  # biggest to smallest
            proj_dc_iso_gc_x = proj_dc_iso_gc_x[idx2]
            proj_dc_iso_gc_y = proj_dc_iso_gc_y[idx2]
            phi_dc_iso = np.arctan2(proj_dc_iso_gc_y[-1], proj_dc_iso_gc_x[-1])
            phi_iso_dc = np.arctan2(proj_dc_iso_gc_y[0], proj_dc_iso_gc_x[0])
            phi_dc_l0 = np.arctan2(lower[1][0], lower[0][0])
            phi_dc_l1 = np.arctan2(lower[1][-1], lower[0][-1])
            phi_dc_u0 = np.arctan2(upper[1][0], upper[0][0])
            phi_dc_u1 = np.arctan2(upper[1][-1], upper[0][-1])
            if not ((phi_dc_u0 >= phi_dc_iso >= phi_dc_l0 and not phi_dc_u0 >= phi_iso_dc >= phi_dc_l0) or
                    (phi_dc_u0 <= phi_dc_iso <= phi_dc_l0 and not phi_dc_u0 <= phi_iso_dc <= phi_dc_l0) or
                    (phi_dc_u0 >= phi_iso_dc >= phi_dc_l0 and not phi_dc_u0 >= phi_dc_iso >= phi_dc_l0) or
                    (phi_dc_u0 <= phi_iso_dc <= phi_dc_l0 and not phi_dc_u0 <= phi_dc_iso <= phi_dc_l0)):
                upper = [np.flipud(u) for u in upper]  # Arbitrarily reverse upper
                u1 = phi_dc_u1
                phi_dc_u1 = phi_dc_u0
                phi_dc_u0 = u1
            if (phi_dc_u0 >= phi_dc_iso >= phi_dc_l0) or (phi_dc_u0 >= phi_iso_dc >= phi_dc_l0):
                phi1 = np.linspace(phi_dc_l0, phi_dc_u0, 100)
                phi2 = np.linspace(phi_dc_l1, phi_dc_u1, 100)

            # if (phi_dc_u0<=phi_dc_iso<=phi_dc_l0) or
            # (phi_dc_u0<=phi_iso_dc<=phi_dc_l0):
            else:
                if phi_dc_u0*phi_dc_l0 < 0:
                    phi_dc_l0 = np.mod(phi_dc_l0, 2*np.pi)
                if phi_dc_u1*phi_dc_l1 < 0:
                    phi_dc_l1 = np.mod(phi_dc_l1, 2*np.pi)
                phi1 = np.linspace(phi_dc_u0, phi_dc_l0, 100)
                phi2 = np.linspace(phi_dc_u1, phi_dc_l1, 100)
            edge1 = (r*np.cos(phi1), r*np.sin(phi1), np.zeros(100))
            edge2 = (r*np.cos(phi2), r*np.sin(phi2), np.zeros(100))

            for i in range(len(lower)):
                if (phi_dc_u0 >= phi_dc_iso >= phi_dc_l0) or (phi_dc_u0 >= phi_iso_dc >= phi_dc_l0):
                    upper[i] = np.append(edge1[i], upper[i])
                    lower[i] = np.append(lower[i], edge2[i])
                else:
                    lower[i] = np.append(edge1[i], lower[i])
                    upper[i] = np.append(upper[i], edge2[i])
            x = np.append(lower[0], np.flipud(upper[0]))
            y = np.append(lower[1], np.flipud(upper[1]))
            z = np.append(lower[2], np.flipud(upper[2]))
            p = patches.Polygon(np.array([x, y]).T, facecolor=self.color, edgecolor=None, alpha=0.7, zorder=-100, closed=True)
            self.ax.add_artist(p)
        super(_RiedeselJordanPlot, self)._boundary_lines(r)


class _ParameterHistPlot(_HistPlot):

    def __init__(self, *args, **kwargs):
        """
        Args
            subplot_spec: matplotlib subplot spec
            fig: matplotlib figure
            MTs: moment tensor samples (see MTData initialisation docstring for formats)

        Keyword Args
            probability: numpy array - moment tensor probabilities
            colormap: str - matplotlib colormap selection (using matplotlib.cm.get_cmap())
            fontsize: int - fontsize for text
            linewidth: float - base linewidth (sometimes thinner or thicker values are used, but relative to this parameter)
            text: bool - flag to show or hide text on the plot
            axis_lines: bool - flag to show or hide axis lines on the plot
            resolution: int - resolution for spherical sampling etc
            grid_lines: bool - show the interior grid lines
            marginalised: bool - marginalise the PDF (defailt is True)
            color: set marker color
            type_label: bool - show the label of the different types
            hex_bin: bool - use the hex-bin histogram type (slightly smoother)
            bins: int/array/list of arrays - bins for numpy histogram call
            parameter: str/tuple parameters to plot.
        """
        super(_ParameterHistPlot, self).__init__(*args, **kwargs)
        # Check parameters
        self.parameter = kwargs.get('parameter', 'gamma')
        if not isinstance(self.parameter, (list, tuple)):
            self.dimension = 1
        elif len(self.parameter) == 1:
            self.dimension = 1
            self.parameter = self.parameter[0]
        else:
            self.dimension = 2
        self.parameter_limits = {'gamma': (-np.pi/6, np.pi/6), 'delta': (-np.pi/2, np.pi/2), 'kappa': (0, 2*np.pi), 'h': (0, 1), 'sigma': (-np.pi/2, np.pi/2),
                                 'strike': (0, 360), 'dip': (0, 90), 'rake': (0, 360), 'strike1': (0, 360), 'dip1': (0, 90), 'rake1': (0, 360),
                                 'strike2': (0, 360), 'dip2': (0, 90), 'rake2': (0, 360), 'u': (-4/3., 4/3.)}

    def _ax_plot(self, *args, **kwargs):
        """
        Main plotting function

        Plots the axis
        """
        # Set up bins and hex_extent
        if self.dimension > 1:
            hex_extent = [-1, 1, -1, 1]
            hex_extent = kwargs.get('hex_extent', hex_extent)
            hex_extent[0] = min([min(self.parameter_limits[self.parameter[0]]), hex_extent[0]])
            hex_extent[1] = max([max(self.parameter_limits[self.parameter[0]]), hex_extent[1]])
            hex_extent[2] = min([min(self.parameter_limits[self.parameter[1]]), hex_extent[2]])
            hex_extent[3] = max([max(self.parameter_limits[self.parameter[1]]), hex_extent[3]])
            kwargs['hex_extent'] = hex_extent
            # Bins
            n0 = np.diff(self.parameter_limits[self.parameter[0]])
            n1 = np.diff(self.parameter_limits[self.parameter[1]])
            if n0 > n1:
                r = float(n1)/n0
                n0 = 100
                n1 = n0*r
            else:
                r = float(n0)/n1
                n1 = 100
                n0 = n1*r
            kwargs['bins'] = kwargs.get('bins', [np.linspace(*self.parameter_limits[self.parameter[0]], num=n0),
                                                 np.linspace(*self.parameter_limits[self.parameter[1]], num=n1)])
            try:
                n0 = np.diff(self.parameter_limits[self.parameter[0]])
                n1 = np.diff(self.parameter_limits[self.parameter[1]])
                if n0 > n1:
                    r = float(n1)/n0
                    n0 = kwargs['bins']
                    n1 = n0*r
                else:
                    r = float(n0)/n1
                    n1 = kwargs['bins']
                    n0 = n1*r
                kwargs['bins'] = [np.linspace(*self.parameter_limits[self.parameter[0]], num=n0),
                                  np.linspace(*self.parameter_limits[self.parameter[1]], num=n1)]
            except Exception:
                pass
        else:
            kwargs['bins'] = kwargs.get('bins', np.linspace(*self.parameter_limits[self.parameter], num=100)[0])
            try:
                kwargs['bins'] = np.linspace(*self.parameter_limits[self.parameter], num=kwargs['bins'])
            except Exception:
                pass
        return super(_ParameterHistPlot, self)._ax_plot(*args, **kwargs)

    def _convert(self):
        """Handles MT conversion to coordinates (setting xdata,ydata and cdata attributes)"""
        if self.dimension == 1:
            self.data_x = self.MTs.__getattr__(self.parameter)
            self.data_y = False
        else:
            self.data_x = self.MTs.__getattr__(self.parameter[0])
            self.data_y = self.MTs.__getattr__(self.parameter[1])
        if len(self.MTs.probability) > 1:
            self.data_c = self.MTs.probability
        else:
            self.data_c = None

    def _background(self, handle):
        """Create the plot background
        """
        parameter_mapping = {'gamma': '$\gamma$', 'delta': '$\delta$', 'kappa': '$\kappa$', 'sigma': '$\sigma$'}

        # Draw box
        if self.dimension == 1:
            try:
                self.ax.set_xlabel(parameter_mapping[self.parameter])
            except KeyError:
                self.ax.set_xlabel(self.parameter)
            try:
                self.ax.set_xlim(self.parameter_limits[self.parameter])
            except KeyError:
                self.ax.set_xlim((-1, 1))
            self.ax.set_yticks([])
        else:
            try:
                self.ax.set_xlabel(parameter_mapping[self.parameter[0]])
            except KeyError:
                self.ax.set_xlabel(self.parameter[0])
            try:
                self.ax.set_xlim(self.parameter_limits[self.parameter[0]])
            except KeyError:
                self.ax.set_xlim((-1, 1))
            try:
                self.ax.set_ylabel(parameter_mapping[self.parameter[1]])
            except KeyError:
                self.ax.set_xlabel(self.parameter[1])
            try:
                self.ax.set_ylim(self.parameter_limits[self.parameter[1]])
            except KeyError:
                self.ax.set_ylim((-1, 1))
        # Limits to plot
        super(_ParameterHistPlot, self)._background(handle)
        self.ax.set_axis_on()


class _TapePlot(_HistPlot):

    def __init__(self, subplot_spec, fig, MTs, *args, **kwargs):
        """
        Args
            subplot_spec: matplotlib subplot spec
            fig: matplotlib figure
            MTs: moment tensor samples (see MTData initialisation docstring for formats)

        Keyword Args
            probability: numpy array - moment tensor probabilities
            colormap: str - matplotlib colormap selection (using matplotlib.cm.get_cmap())
            fontsize: int - fontsize for text
            linewidth: float - base linewidth (sometimes thinner or thicker values are used, but relative to this parameter)
            text: bool - flag to show or hide text on the plot
            axis_lines: bool - flag to show or hide axis lines on the plot
            resolution: int - resolution for spherical sampling etc
            grid_lines: bool - show the interior grid lines
            marginalised: bool - marginalise the PDF (defailt is True)
            color: set marker color
            type_label: bool - show the label of the different types
            hex_bin: bool - use the hex-bin histogram type (slightly smoother)
            bins: int/array/list of arrays - bins for numpy histogram call
        """
        super(_TapePlot, self).__init__(subplot_spec, fig, MTs, *args, **kwargs)
        self.parameter_classes = []
        if not subplot_spec:
            self.grid_spec = gridspec.GridSpec(2, 6, wspace=0.3, hspace=0.4)
        else:
            self.grid_spec = gridspec.GridSpecFromSubplotSpec(
                2, 6, subplot_spec=subplot_spec, wspace=0.3, hspace=0.4)
        # Convert MTS
        self.MTs._convert('gamma')
        self.MTs._convert('kappa')
        self.axs = []
        self.ax.set_axis_off()
        # Plot 1,1 split into gamma (2,0:3),delta (2,3:)
        self.parameter_classes.append(_ParameterHistPlot(self.grid_spec[0, 0:3], self.fig, self.MTs, parameter='gamma', *args, **kwargs))
        self.axs.append(self.parameter_classes[-1].ax)
        self.parameter_classes.append(_ParameterHistPlot(self.grid_spec[0, 3:6], self.fig, self.MTs, parameter='delta', *args, **kwargs))
        self.axs.append(self.parameter_classes[-1].ax)
        # Plot 2,1 split into kappa(2,0:2),h (2,2:4),sigma (2,4:)
        self.parameter_classes.append(_ParameterHistPlot(self.grid_spec[1, 0:2], self.fig, self.MTs, parameter='kappa', *args, **kwargs))
        self.axs.append(self.parameter_classes[-1].ax)
        self.parameter_classes.append(_ParameterHistPlot(self.grid_spec[1, 2:4], self.fig, self.MTs, parameter='h', *args, **kwargs))
        self.axs.append(self.parameter_classes[-1].ax)
        self.parameter_classes.append(_ParameterHistPlot(self.grid_spec[1, 4:6], self.fig, self.MTs, parameter='sigma', *args, **kwargs))
        self.axs.append(self.parameter_classes[-1].ax)

    def plot(self, *args, **kwargs):
        """
        Plots the result

        Keyword Args
            MTs: Moment tensors to plot (see MTData initialisation docstring for formats)
            args: args passed to the _ParameterHistPlot._ax_plot functions (e.g. set local parameters to be different from initialisation values)
            kwargs: kwargs passed to the _ParameterHistPlot._ax_plot functions (e.g. set local parameters to be different from initialisation values)
        """
        bins = kwargs.get('bins', None)
        for plot_class in self.parameter_classes:
            plot_class(*args, bins=bins)
        self.fig.patch.set_facecolor('w')
        if self.show:
            self.fig.show()
        return


class_mapping = {'amplitude': _AmplitudePlot, 'beachball': _AmplitudePlot,
                 'radiation': _RadiationPlot, 'faultplane': _FaultPlanePlot,
                 'lune': _LunePlot, 'hudson': _HudsonPlot, 'riedeseljordan': _RiedeselJordanPlot,
                 'tape': _TapePlot, 'parameter': _ParameterHistPlot}
class_mapping = get_extensions(group='MTfit.plot', defaults=class_mapping)[1]
