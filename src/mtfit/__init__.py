"""
**Restricted:  For Non-Commercial Use Only**
This code is protected intellectual property and is available solely for teaching
and non-commercially funded academic research purposes.

Applications for commercial use should be made to Schlumberger or the University of Cambridge.
"""
import sys
import multiprocessing
import json

from ._version import get_versions
__version__ = get_versions()['version']
from .run import MTfit  # noqa F401

__all__ = ['algorithms', 'convert', 'extensions', 'plot', 'utilities', 'inversion', 'probability', 'sampling']

del get_versions


def get_details():
    # Check if the different components are present
    c_extensions = []
    try:
        from MTfit.algorithms import cmarkov_chain_monte_carlo
        del cmarkov_chain_monte_carlo
        c_extensions.append('cmarkov_chain_monte_carlo')
    except Exception:
        pass
    try:
        from MTfit.probability import cprobability
        del cprobability
        c_extensions.append('cprobability')
    except Exception:
        pass
    try:
        from MTfit.convert import cmoment_tensor_conversion
        del cmoment_tensor_conversion
        c_extensions.append('cmoment_tensor_conversion')
    except Exception:
        pass
    try:
        from MTfit.extensions import cscatangle
        del cscatangle
        c_extensions.append('cscatangle')
    except Exception:
        pass
    dependency_versions = {}
    import numpy as np
    dependency_versions['numpy'] = np.__version__
    import scipy
    dependency_versions['scipy'] = scipy.__version__
    try:
        import matplotlib as mpl
        dependency_versions['matplotlib'] = mpl.__version__
    except ImportError:
        pass
    except AttributeError:
        dependency_versions['matplotlib'] = 'unknown'
    try:
        import cython
        dependency_versions['cython'] = cython.__version__
    except ImportError:
        pass
    except AttributeError:
        dependency_versions['cython'] = 'unknown'
    try:
        import pyqsub
        dependency_versions['pyqsub'] = pyqsub.__version__
    except ImportError:
        pass
    except AttributeError:
        dependency_versions['pyqsub'] = 'unknown'
    try:
        import sphinx
        dependency_versions['sphinx'] = sphinx.__version__
    except ImportError:
        pass
    except AttributeError:
        dependency_versions['sphinx'] = 'unknown'
    try:
        import h5py
        dependency_versions['h5py'] = h5py.__version__
    except ImportError:
        pass
    except AttributeError:
        dependency_versions['h5py'] = 'unknown'
    try:
        import hdf5storage
        dependency_versions['hdf5storage'] = hdf5storage.__version__
    except ImportError:
        pass
    except AttributeError:
        dependency_versions['hdf5storage'] = 'unknown'
    details = {'version': __version__,
               'c_extensions present': c_extensions,
               'platform': sys.platform,
               'num_threads': multiprocessing.cpu_count(),
               'python version': sys.version,
               'python version info': sys.version_info,
               'dependency info': dependency_versions}
    if sys.platform.startswith('win'):
        details['windows version'] = sys.getwindowsversion()
    if sys.version_info.major < 3:
        details['python version info'] = str(details['python version info'])
        if sys.platform.startswith('win'):
            details['windows version'] = str(details['windows version'])
    return details


def get_details_json():
    return json.dumps(get_details(), indent=4)
