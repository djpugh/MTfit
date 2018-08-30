"""
sampling.py
***********
module containing basic Sample object for storing samples from the source pdf.
"""


# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.

import os
import sys
import shutil


import numpy as np


from .probability import LnPDF
from .probability import dkl_estimate
from .probability.probability import _6sphere_prior
from .utilities.file_io import convert_keys_to_unicode
from .utilities.file_io import unique_columns
from .convert import output_convert

# Because long doesn't exist in python 3 our isinstance tests fail
if sys.version_info.major >= 3:
    long = int


__all__ = ['Sample', 'FileSample', 'ln_bayesian_evidence', '_convert', '_6sphere_prior']


class Sample(object):
    """Sample object for storing source pdf samples."""

    def __init__(self, initial_sample_size=100000, number_events=1, prior=_6sphere_prior):
        """
        Sample initialisation

        Args
            initial_sample_size:[100000] Initial size for MT vector
            number_events:[1] Number of events in sample (allows for sampling multiple event joint PDF)
            prior:[6sphere]

        Returns
            Sample object

        """
        self.moment_tensors = np.matrix(np.zeros((number_events*6, initial_sample_size)))
        self.n = 0
        self.ln_pdf = LnPDF()
        self.number_events = number_events
        self._initial_sample_size = initial_sample_size
        self._i = 0
        self._prior = prior

    def append(self, moment_tensors, ln_pdf, n, scale_factor=False, extensions_scale_factor=False):
        """
        Appends new samples to the Sample object, extending the MT array if the limit is reached.

        Args
            moment_tensors: numpy matrix of moment tensors to append.
            ln_pdf: numpy matrix/array or PDF object containing ln_pdf samples corresponding to the moment tensors in moment_tensors (must be same length).
            n: number of tried samples (including zero probability samples).
            scale_factor:[False] scale_factor estimates for relative amplitude inversion.
            extensions_scale_factor:[False] scale_factor estimates for extensions carrying out relative inversion.

        Returns
            None

        """
        # If scale_factor is passed as an argument, and no scale_factor data stored, create the array
        if not isinstance(scale_factor, bool) and not hasattr(self, 'scale_factor'):
            self.scale_factor = np.array([])
        if isinstance(extensions_scale_factor, dict) and len(extensions_scale_factor) and not hasattr(self, 'extensions_scale_factor'):
            self.extensions_scale_factor = {}
        # Check if moment tensors are a list (multiple events) and convert to the correct format
        if isinstance(moment_tensors, list) and isinstance(moment_tensors[0], np.ndarray):
            moment_tensors = np.array(moment_tensors)
            moment_tensors = np.matrix(moment_tensors.reshape(moment_tensors.shape[0]*moment_tensors.shape[1], moment_tensors.shape[2]))
        # Check moment tensor and PDF shape
        if isinstance(ln_pdf, (float, long, int, np.float64, np.float32)) and ((isinstance(moment_tensors, np.ndarray) and moment_tensors.shape[1] != 1) or (isinstance(moment_tensors, list) and max([mt.shape[1] for mt in moment_tensors]) != 1)):
            raise ValueError('Moment Tensor shape must be 6x1 when using a float ln_pdf')
        elif not isinstance(ln_pdf, (float, long, int, np.float64, np.float32)) and moment_tensors.shape[1] != ln_pdf.shape[1]:
            raise ValueError('Moment Tensor shape[2] and ln_pdf shape[2] must be the same')
        # Add number of samples (n) to total number of samples (self.n)
        self.n += n
        # Convert ln_pdf to ln_pdf form
        if isinstance(ln_pdf, (np.ndarray, float, int, np.float64, np.float32)):
            ln_pdf = LnPDF(ln_pdf)
        # Get non-zero probability samples
        moment_tensors = moment_tensors[:, ln_pdf.nonzero()]
        # Check scale factor
        if not isinstance(scale_factor, bool):
            if not isinstance(scale_factor, dict):  # single event
                scale_factor = scale_factor[ln_pdf.nonzero()]
        if isinstance(extensions_scale_factor, dict) and len(extensions_scale_factor):
            for key in extensions_scale_factor.keys():
                extensions_scale_factor[key] = extensions_scale_factor[key][ln_pdf.nonzero()]
        # Check if there are non-zero moment tensor samples
        if moment_tensors.shape[0]*moment_tensors.shape[1] > 0:  # Has samples
            if len(ln_pdf.nonzero()) != moment_tensors.shape[1]:
                raise ValueError('Moment Tensor shape[2] and ln_pdf shape[2] must be the same')
            # Check if moment tensor sampling array has enough spare elements
            size_check = self.moment_tensors.shape[1]-self._i > moment_tensors.shape[1]
            # If size-check fails, append new zero samples to the array
            while not size_check:
                self.moment_tensors = np.append(self.moment_tensors, np.zeros((self.number_events*6, self._initial_sample_size)),
                                                axis=1)
                size_check = self.moment_tensors.shape[1]-self._i > moment_tensors.shape[1]
            # Set moment tensor samples in array
            self.moment_tensors[:, self._i:self._i+moment_tensors.shape[1]] = moment_tensors
            # Handle scale factors
            if not isinstance(scale_factor, bool):
                # Scale factor dict containing mu and s
                self.scale_factor = np.append(self.scale_factor, scale_factor)
            if isinstance(extensions_scale_factor, dict) and len(extensions_scale_factor):
                # Scale factor dict containing mu and s
                for key in extensions_scale_factor.keys():
                    if key in self.extensions_scale_factor.keys():
                        self.extensions_scale_factor[key] = np.append(self.extensions_scale_factor[key], extensions_scale_factor[key])
                    else:
                        self.extensions_scale_factor[key] = extensions_scale_factor[key]
            # Append to PDF
            self.ln_pdf.append(ln_pdf[:, ln_pdf.nonzero()])
            # Update moment tensor index
            self._i += moment_tensors.shape[1]

    def output(self, normalise=True, convert=False, n_samples=0, discard=10000, mcmc=False):
        """
        Returns nonzero probability samples in a dictionary

        Keyword Arguments
            normalise:[True] boolean flag for normalising the PDF output or not.
            convert:[False] boolean flag for converting the moment tensor output into different parameterisations
            n_samples:[0] Total number of samples (not using self.pdf.n as multiple samplings for e.g. MPI inversions)
            discard:[10000] float, if non-zero discard is a multiplier on n_samples so that samples with probability less than 1/(n_samples*discard) are discarded as negligeable

        Returns
            Dictionary of nonzero probability samples as:
                {'MTSpace':nonzeromoment_tensors,'Probability':nonzeroProbability,'dV':volumeElement}

        Raises
            ValueError: No nonzero probability samples.

        """
        # Gets normalised and unnormalised pdfs
        ln_pdf = self.ln_pdf.output(normalise)
        un_normalised_ln_pdf = self.ln_pdf.output(normalise=False)
        output_string = '\n------------MTfit Forward Model Output------------\n\n'
        # Check if any non-zero samples
        if not np.prod(ln_pdf._ln_pdf.shape):
            output_string += 'No Non-Zero probability samples\n\n'
            return {'probability': []}, output_string
        # Get max probability solutions
        output_string += 'Sample Max Probability: '+str(ln_pdf.max())+'\n'
        moment_tensors = self.moment_tensors[:, :self._i]
        if len(ln_pdf.shape) > 1:
            max_p_mt = unique_columns(moment_tensors[:, np.array(ln_pdf == np.max(ln_pdf._ln_pdf))[0]])
        else:
            max_p_mt = unique_columns(moment_tensors[:, np.array(ln_pdf == np.max(ln_pdf._ln_pdf))])
        output_string += 'Sample Max Probability MT:\n'+str(max_p_mt)+'\n\n'
        # Check if discarding samples
        if discard and n_samples:
            output_string += 'Discarding samples less than 1/'+str(discard*n_samples)+' of the maximum likelihood value as negligeable\n\n'
        # Check if there are samples
        if len(ln_pdf):
            # Get non_zero samples
            non_zero = ln_pdf.nonzero(discard=discard, n_samples=n_samples)
            if discard and n_samples:
                output_string += 'After discard, '+str(non_zero.shape[0])+' samples remain\n\n'
            if len(ln_pdf.shape) > 1:
                probability = np.exp(ln_pdf[:, non_zero])
            else:
                probability = np.exp(ln_pdf[non_zero])
            if len(un_normalised_ln_pdf.shape) > 1:
                un_normalised_ln_pdf = un_normalised_ln_pdf[:, non_zero]
            else:
                un_normalised_ln_pdf = un_normalised_ln_pdf[non_zero]
            output = {'probability': probability, 'dV': ln_pdf.dV, 'ln_pdf': un_normalised_ln_pdf}
            # Multiple events
            if self.number_events > 1:
                if hasattr(self, 'scale_factor'):
                    output['scale_factors'] = list(self.scale_factor[non_zero])
                # Create MTspace for multiple events
                for i in range(self.number_events):
                    output['moment_tensor_space_'+str(i+1)] = moment_tensors[6*i:6*(i+1), non_zero]
                    if convert:
                        # convert results
                        output.update(_convert(output['moment_tensor_space_'+str(i+1)], i+1))
            else:
                output['moment_tensor_space'] = moment_tensors[:, non_zero]
                if convert:
                    # convert results
                    output.update(_convert(output['moment_tensor_space']))
            # Bayesian evidence calculation
            try:
                if not mcmc:
                    output['ln_bayesian_evidence'] = ln_bayesian_evidence(output, n_samples, self._prior)
            except Exception:  # Should fail when McMC and not MC
                pass
            try:
                if not mcmc:
                    # DKL
                    V = 1.0
                    if 'g' not in output.keys():
                        # Multiple events
                        g = sorted([key for key in output.keys() if key == 'g' or ('g' in key and (key[0] == 'g' and key[1] == '_'))], key=lambda u: int(u.split('_')[-1]))
                        d = sorted([key for key in output.keys() if key == 'd' or ('d' in key and (key[0] == 'd' and key[1] == '_'))], key=lambda u: int(u.split('_')[-1]))
                        for i, gi in enumerate(g):
                            if np.max(gi)-np.min(gi) < 0.000001 and np.abs(np.mean(gi)) < 0.000001 and np.max(d[i])-np.min(d[i]) < 0.000001 and np.abs(np.mean(d[i])) < 0.000001:
                                V *= (2*np.pi*np.pi)
                            else:
                                V *= (np.pi*np.pi*np.pi)
                    else:
                        # Single event
                        if np.max(output['g'])-np.min(output['g']) < 0.000001 and np.abs(np.mean(output['g'])) < 0.000001 and np.max(output['d'])-np.min(output['d']) < 0.000001 and np.abs(np.mean(output['d'])) < 0.000001:
                            V *= (2*np.pi*np.pi)
                        else:
                            V *= (np.pi*np.pi*np.pi)
                    output_string += str(V)+'\n'
                    output['dkl'] = dkl_estimate(self.ln_pdf, V, n_samples)
                    output_string += 'PDF Kullback-Leibler Divergence Estimate (Dkl): '+str(output['dkl'])+'\n\n'
            except Exception:
                pass
            if isinstance(output['ln_pdf'], LnPDF):
                output['ln_pdf'] = output['ln_pdf']._ln_pdf
            return output, output_string
        else:
            raise ValueError('No non zero probability samples found.')

    def __len__(self):
        """
        Returns number of tried samples.

        Returns
            n: number of forward modelled MT samples.

        """
        return self.n

    def nonzero(self):
        """Return non zero indices"""
        return self.ln_pdf.nonzero()


class FileSample(Sample):
    """
    FileSample object stores sample values on disk rather than in memory (writes to disk each time)

    These then need to be cleaned up but should allow for simple restore and recover values.

    NEEDS hdf5 (MATLAB -v7.3) storage
    """

    def __init__(self, fname='MTfit_run', file_safe=True, *args, **kwargs):
        """
        FileSample initialisation

        Args
            fname:['MTfit_run'] Filename for storage.
            file_safe:[True] Boolean flag for file safe output (i.e. write and move).

        Returns
            FileSample object

        """
        super(FileSample, self).__init__(*args, **kwargs)
        # Check if hdf5storage is installed
        try:
            from hdf5storage import savemat
            from hdf5storage import loadmat
            self.savemat = savemat
            self.loadmat = loadmat
        except Exception:
            raise ImportError('hdf5storage module missing - required for FileSample sample type')
        # Set default filename
        if not isinstance(fname, str):
            fname = 'MTfit_run'
        self.fname = fname.split('.mat')[0]+'_in_progress.mat'
        self.n = 0
        self._i = 1
        self.non_zero_samples = 0
        self.pdf = LnPDF()
        self.moment_tensor = np.matrix([[]])
        self.file_safe = file_safe
        # Load data if file exists (recover)
        if os.path.exists(self.fname):
            data = self.loadmat(self.fname)
            try:
                self.n = data['n']
                self.non_zero_samples = data['non_zero_samples']
                self._i = data['i']
            except Exception:
                pass

    def append(self, moment_tensors, ln_pdf, n, scale_factor=False, extensions_scale_factor=False):
        """
        Appends new samples to the FileSample file.

        Args
            moment_tensors: numpy matrix of moment tensors to append.
            ln_pdf: numpy matrix/array or PDF object containing ln_pdf samples corresponding to the moment tensors in moment_tensors (must be same length).
            n: number of tried samples (including zero probability samples).
            scale_factor:[False] scale_factor estimates for relative amplitude inversion.
            extensions_scale_factor:[False] scale_factor estimates for extensions carrying out relative inversion.

        Returns
            None

        """
        if isinstance(moment_tensors, list) and isinstance(moment_tensors[0], np.ndarray):
            moment_tensors = np.array(moment_tensors)
            moment_tensors = np.matrix(moment_tensors.reshape(moment_tensors.shape[0]*moment_tensors.shape[1], moment_tensors.shape[2]))
        if isinstance(ln_pdf, (float, long, int, np.float64, np.float32)) and moment_tensors.shape[1] != 1:
            raise ValueError('Moment Tensor shape must be 6x1 when using a float ln_pdf')
        elif not isinstance(ln_pdf, (float, long, int, np.float64, np.float32)) and moment_tensors.shape[1] != ln_pdf.shape[1]:
            raise ValueError('Moment Tensor shape[2] and ln_pdf shape[2] must be the same')

        self.n += n
        if isinstance(ln_pdf, (np.ndarray, float, int, np.float64, np.float32)):
            ln_pdf = LnPDF(ln_pdf)
        moment_tensors = moment_tensors[:, ln_pdf.nonzero()]
        if not isinstance(scale_factor, bool):
            if isinstance(scale_factor, list):
                scale_factor = np.array(scale_factor)
            scale_factor = scale_factor[ln_pdf.nonzero()]
        if isinstance(extensions_scale_factor, dict) and len(extensions_scale_factor):
            # Scale factor dict containing mu and s
            for key in extensions_scale_factor.keys():
                extensions_scale_factor[key] = convert_keys_to_unicode(list(extensions_scale_factor[key][ln_pdf.nonzero()]))
        if self.file_safe:
            old_fname = self.fname
            try:
                shutil.copy(self.fname, self.fname+'~')
            except IOError:
                pass
            self.fname += '~'
        if moment_tensors.shape[0]*moment_tensors.shape[1] > 0:  # Check there are samples
            if len(ln_pdf.nonzero()) != moment_tensors.shape[1]:
                raise ValueError('Moment Tensor shape[2] and ln_pdf shape[2] must be the same')
            self.savemat(self.fname, {'MTSpace_'+str(self._i): moment_tensors}, appendmat=not self.file_safe, store_python_metadata=True)
            if not isinstance(scale_factor, bool):
                # dict with p mu and s
                self.savemat(self.fname, {'scale_factor_'+str(self._i): convert_keys_to_unicode(list(scale_factor))}, appendmat=not self.file_safe, store_python_metadata=True)
            if isinstance(extensions_scale_factor, dict) and len(extensions_scale_factor):
                # dict with p mu and s
                self.savemat(self.fname, {'extensions_scale_factor_'+str(self._i): extensions_scale_factor}, appendmat=not self.file_safe, store_python_metadata=True)
            self.savemat(self.fname, {'LnPDF_'+str(self._i): ln_pdf[:, ln_pdf.nonzero()]}, appendmat=not self.file_safe, store_python_metadata=True)
            self.non_zero_samples += len(ln_pdf.nonzero())
            self.savemat(self.fname, {'non_zero_samples': self.non_zero_samples}, appendmat=not self.file_safe, store_python_metadata=True)
            self._i += 1
        self.savemat(self.fname, {'i': self._i}, appendmat=not self.file_safe, store_python_metadata=True)
        self.savemat(self.fname, {'n': self.n}, appendmat=not self.file_safe, store_python_metadata=True)
        if self.file_safe:
            shutil.copy(self.fname, old_fname)
            self.fname = old_fname

    def output(self, *args, **kwargs):
        """
        Returns nonzero probability samples in a dictionary

        Returns
            Dictionary of nonzero probability samples as:
                {'MTSpace':nonzero_moment_tensors,'Probability':nonzero_probability,'dV':volume_element}

        Raises
            ValueError: No nonzero probability samples.

        """
        # load data
        data = self.loadmat(self.fname)
        # Set up attributes for Sample.output call
        self.moment_tensors = np.matrix(np.zeros((self.number_events*6, self.non_zero_samples)))
        self.scale_factor = False
        if len([key for key in data if 'scale_factor' in key]):
            self.scale_factor = []
        _x = 0
        self.n = data['n']
        for key in [key for key in data.keys() if 'MTSpace' in key]:
            i = key.split('_')[-1]
            self.ln_pdf.append(LnPDF(data['LnPDF_'+str(i)]))
            moment_tensors = data['MTSpace_'+str(i)]
            self.moment_tensors[:, _x: _x+moment_tensors.shape[1]] = moment_tensors
            if isinstance(self.scale_factor, list):
                self.scale_factor.append(data['scale_factor_'+str(i)])
            _x += moment_tensors.shape[1]
        self.scale_factor = np.array(self.scale_factor)
        output, output_string = super(FileSample, self).output(*args, **kwargs)
        try:
            os.remove(self.fname)
        except Exception:
            pass
        return output, output_string


def ln_bayesian_evidence(output, n_samples, prior=_6sphere_prior):
    """
    Bayesian evidence calculation for the output

    The priors are accessible as a pkg_resource, along with the sampling distribution
    (MTfit.algorithms.__base__), so new sampling priors can be added.
    If the sampling is from the prior distribution, the prior in thisfunction is
    just the uniform prior.

    Args
        output: dictionary from Sample.output.
        n_samples: total number of samples test_append.
        prior:['6sphere'] the prior distribution to use (reflects the sampling).

    Returns
        float: returns the Bayesian evidence as a log value.
    """
    # Get prior
    if 'g' not in output.keys():
        # Multiple events
        p = 1.0
        g = sorted([key for key in output.keys() if key == 'g' or ('g' in key and (key[0] == 'g' and key[1] == '_'))], key=lambda u: int(u.split('_')[-1]))
        d = sorted([key for key in output.keys() if key == 'd' or ('d' in key and (key[0] == 'd' and key[1] == '_'))], key=lambda u: int(u.split('_')[-1]))
        for i, gi in enumerate(g):
            p *= prior(output[gi], output[d[i]])
    else:
        # Single event
        p = prior(output['g'], output['d'])
    if not isinstance(output['ln_pdf'], LnPDF):
        output['ln_pdf'] = LnPDF(output['ln_pdf'])
    return np.log((output['ln_pdf']+np.log(p)-output['ln_pdf']._ln_pdf.max()).exp().sum())+output['ln_pdf']._ln_pdf.max()-np.log(n_samples)


def _convert(moment_tensors, i=None):
    """
    Converts the moment tensor to Tape parameters, Hudson u,v and strike dip rake pairs.

    Args
        moment_tensors: moment tensor matrix.
        i:[None] event index if using multiple events.

    Returns
        output dictionary containing converted parameters
    """
    output = output_convert(moment_tensors)
    if isinstance(i, int):
        # Multiple events
        for key in output.keys():
            output[key+'_'+str(i)] = output.pop(key)
    return output
