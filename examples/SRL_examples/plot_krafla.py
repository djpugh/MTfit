import numpy as np
from numpy.random import rand, randn
import scipy.stats as sp

from mtfit import plot
from mtfit import convert
# Make Figure 2
# Load Data
station_distribution = plot.read(
     'krafla_event_ppolarityDCStationDistribution.mat',
     station_distribution=True)
DCs, DCstations = plot.read('krafla_event_ppolarityDC.mat')
MTs, MTstations = plot.read('krafla_event_ppolarityMT.mat')
# Plot
plot = plot.MTplot([np.array([1, 0, -1, 0, 0, 0]), DCs, MTs],
                   stations=[DCstations, DCstations, MTstations],
                   station_distribution=[station_distribution, False, False],
                   plot_type=['faultplane', 'faultplane', 'hudson'],
                   fault_plane=[False, True, False],
                   show_mean=False, show_max=True, grid_lines=True,
                   TNP=False, text=[False, False, True])
# Make Figure 4
n = 100
# Make Data
DCs = convert.tape_MT6(np.zeros(n), np.zeros(n), np.pi+0.1*randn(n),
                       0.5+0.01*randn(n), 0.1*randn(n))
n = 10000
g = -np.pi/12+0.01*randn(n)
d = np.pi/3+0.1*randn(n)
MTs = convert.tape_MT6(g, d, np.pi+0.1*randn(n), 0.5+0.01*randn(n),
                       0.1*randn(n))
prob_MTs = sp.norm.pdf(g, -np.pi/12, 0.01)*sp.norm.pdf(d, np.pi/3, 0.1)
# Plot
plot = plot.MTplot([np.array([1, 0, 1, -1, 0, 0]), DCs, MTs,
                   MTs, np.array([1, 0, 1, -1, 0, 0])],
                   plot_type=['beachball', 'faultplane',
                              'hudson', 'lune', 'riedeseljordan'],
                   probability=[False, rand(n), prob_MTs, prob_MTs, False],
                   colormap=['bwr', 'bwr', 'viridis', 'viridis', 'bwr'],
                   stations=[{'names': ['S01', 'S02', 'S03', 'S04'],
                              'azimuth': np.array([120., 45., 238., 341.]),
                              'takeoff_angle': np.array([12., 56., 37., 78.]),
                              'polarity': [1, 0, -1, -1]}, {}, {}, {}, {}],
                   show_mean=True, show_max=True, grid_lines=True,
                   TNP=False, label=True,  fontsize=6,
                   station_markersize=2, markersize=2)
