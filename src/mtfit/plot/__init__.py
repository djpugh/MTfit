"""plot
============================
Submodule for moment tensor plotting.

There are several different plot types available:

    * beachball
    * radiation
    * fault plane
    * hudson
    * lune
    * riedesel jordan

The default format is for 2D plots although 3D plots are possible (dimension=3)

The MTData class provides transparent handling of moment tensor dat
including conversion to different parameters and the calculation of some statistics.
"""

# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.

import sys
import os

# CHECK DISPLAY
if 'DISPLAY' not in os.environ.keys() and 'win32' not in sys.platform:
    import matplotlib as mpl
    mpl.use('Agg')  # Default no backend

from .core import MTplot, MTData, run, read  # noqa E402, F401
