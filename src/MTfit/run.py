"""
run.py
******
Core module for MTfit - handles all the command line parsing logic and calling the forward model based inversion.
"""

# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.


import os
import subprocess
import warnings
import traceback

# Test flag for running qsub flags in test
try:
    import pyqsub  # python module for cluster job submission using qsub.
    _PYQSUB = True
except Exception:
    _PYQSUB = False

from .inversion import Inversion
from .inversion import combine_mpi_output
from .utilities.extensions import get_extensions
from .extensions import default_pre_inversions
from .extensions import default_post_inversions
from .extensions import default_extensions
from .utilities.argparser import MTfit_parser


WARNINGS_MAP = {'a': 'always', 'd': 'default',
                'e': 'error', 'i': 'ignore',
                'm': 'module', 'o': 'once'}


ERROR_MESSAGE = """
**********************************************
Error running MTfit:

If this is a recurring error, please post an issue at https://github.com/djpugh/MTfit/issues/ (if one doesn't already exist)
including the traceback and the following information to help with diagnosis:

## Environment Information

```
{}
```

## Traceback

```
{}
```

**********************************************
"""


# Main MTfit function

def MTfit(data={}, data_file=False, location_pdf_file_path=False, algorithm='Time', parallel=True, n=0, phy_mem=8, dc=False, **kwargs):
    """
    Runs MTfit

    Creates an MTfit.inversion.Inversion object for the given arguments and runs the forward model based inversion.
    For a simple method of initialising the inversion use the command line approach (see MTfit docs).

    Args
        Data: Data dictionary (see MTfit.inversion.Inversion for structure).
        data_file: File or list of files which are pickled Data dictionaries.
        location_pdf_file_path: Path to angle scatter files or List of Angle Scatter Files (for Monte carlo marginalisation over location and model uncertainty)
            Can be generated from NonLinLoc *.scat files.
        algorithm:['Time'] Default search algorithm, for more information on the different algorithms see the MTfit.inversion.Inversion docstrings.
        parallel:['True'] Selects whether to run the inversion in parallel or on a single thread.
        n:[0] Number of threads to use, 0 defaults to all available threads reported by the system (from multiprocessing.cpu_count()).
        phy_mem:[8Gb] Estimated physical memory to use (used for determining array sizes, it is likely that more memory will be used, and if so no errors are forced).
        dc:[False] If true constrains the inversion to the double-couple space only.

    Keyword Arguments
        Test:[False] If true, runs unittests rather than the inversion.
        Other arguments to pass to the MTfit.inversion.Inversion object or for the algorithm (for more information see the MTfit.inversion.Inversion docstrings).

    Returns
        0

    """
    try:
        kwargs['data'] = data
        kwargs['data_file'] = data_file
        kwargs['location_pdf_file_path'] = location_pdf_file_path
        kwargs['algorithm'] = algorithm
        kwargs['parallel'] = parallel
        kwargs['n'] = n
        kwargs['phy_mem'] = phy_mem
        kwargs['dc'] = dc
        # GET PLUGINS
        pre_inversion_names, pre_inversions = get_extensions('MTfit.pre_inversion', default_pre_inversions)
        post_inversion_names, post_inversions = get_extensions('MTfit.post_inversion', default_post_inversions)
        extension_names, extensions = get_extensions('MTfit.extension', default_extensions)
        # Default extensions
        for ext in extensions.values():
            result = ext(**kwargs)
            if result != 1:
                return result
        # Check combine mpi output
        if kwargs.get('combine_mpi_output', False):
            combine_mpi_output(kwargs.get('path', ''), kwargs.get('output_format', 'matlab'), **kwargs)  # binary_file_version flag set here
            return 0
        if len(data) or (isinstance(data_file, str) and not os.path.isdir(data_file)) or isinstance(data_file, list):
            warnings.filterwarnings(WARNINGS_MAP[kwargs.get('warnings', 'd')])
            # Pre-inversion
            for plugin in pre_inversions.values():
                kwargs = plugin(**kwargs)
            if not kwargs.get('_mpi_call', False):
                print('Running MTfit.')
            # Effectively
            # inversion = Inversion(data, data_file, location_pdf_file_path, algorithm, parallel, n, phy_mem, dc, **kwargs)
            # but allowing the pre inversion plugin to change the kwargs
            inversion = Inversion(**kwargs)
            inversion.forward()
            if kwargs.get('dc_mt', False):
                # inversion = Inversion(data, data_file, location_pdf_file_path, algorithm, parallel, n, phy_mem, not dc, **kwargs)
                kwargs['dc'] = not kwargs['dc']
                inversion = Inversion(**kwargs)
                inversion.forward()
            # Post-inversion
            for plugin in post_inversions.values():
                plugin(**kwargs)
            return 0
    except Exception as e:
        from . import get_details_json
        print(ERROR_MESSAGE.format(get_details_json(), traceback.format_exc()))
        raise e


def run(args=None):
    """
    Runs inversion from command line arguments

    Runs the command line options parser and either submits job to cluster or runs the inversion on the local machine depending on the command line arguments

    Args
        args:[None] Input arguments - if not given, defaults to using sys.argv

    Returns
        0

    """
    options, options_map = MTfit_parser(args)
    if options['qsub'] and _PYQSUB:
        options_map['data_file'] = options_map['DATAFILE']
        options['singlethread'] = not options.pop('parallel')
        options['_mpi_call'] = options['mpi']
        if 'mcmc' in options['algorithm'].lower():
            options['max_samples'] = options['chain_length']
        return pyqsub.submit(options, options_map, __name__)
    else:
        for key in list(options.keys()):
            if 'qsub' in key:
                options.pop(key)
        if options['mpi'] and not options['_mpi_call']:
            try:
                # could add extra changeable MPI python handling here (and
                # elsewhere?)
                import mpi4py  # noqa F401
            except Exception:
                raise ImportError('MPI module mpi4py not found, unable to run in mpi')
            # restart python as mpirun
            options['_mpi_call'] = True
            print('Running MTfit using mpirun')
            optstring = pyqsub.make_optstr(options, options_map)
            mpiargs = ["mpirun", "-n", str(options['n']), "MTfit"]
            mpiargs.extend(optstring.split())
            return subprocess.call(mpiargs)
        return MTfit(**options)
