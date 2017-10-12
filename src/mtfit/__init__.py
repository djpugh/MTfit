"""
**Restricted:  For Non-Commercial Use Only**
This code is protected intellectual property and is available solely for teaching
and non-commercially funded academic research purposes.

Applications for commercial use should be made to Schlumberger or the University of Cambridge.
"""

from ._version import get_versions
__all__ = ['algorithms', 'convert', 'extensions', 'plot', 'utilities', 'inversion', 'probability', 'sampling']

__version__ = get_versions()['version']
del get_versions
