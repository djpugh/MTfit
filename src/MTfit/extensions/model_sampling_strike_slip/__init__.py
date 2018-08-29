"""
model_sampling_strike_slip
**************************

Example extension module for generating strike slip events
"""
import numpy as np
from MTfit.convert import Tape_MT6


def random_strike_slip(self, n_samples):
    kappa = np.random.rand(n_samples)*2*np.pi
    h = np.ones(n_samples,)
    sigma = np.random.rand(n_samples)*np.pi-np.pi/2
    return Tape_MT6(np.zeros(n_samples), np.zeros(n_samples), kappa, h, sigma)
