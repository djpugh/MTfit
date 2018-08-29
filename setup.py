#!/usr/bin/env python
"""
Setup script for MTfit
***********************
Call from command line as: python setup.py install if on unix
or python setup.py install
"""

import sys
import os
import subprocess
import distutils

import numpy as np
from setuptools import setup
from setuptools import Extension
from setuptools import find_packages

import versioneer

_CYTHON = False

try:
    from Cython.Distutils import build_ext
    _CYTHON = True
except ImportError:
    pass


__author__ = 'David J Pugh'
__email__ = 'david.j.pugh@cantab.net'


long_description = """
Bayesian moment tensor inversion code based on the approach described in Pugh, D J, 2015,
Bayesian Source Inversion of Microseismic Events, PhD Thesis, Department of Earth Sciences,
University of Cambridge."""


def setup_package():
    """
    setup_package()

    Function to call distutils setup to install the package to the default location for third party modules.

    Checks to see if required modules are installed and if not tries to install them (apart from Basemap)
    """
    # Extensions
    # try:
    build_extensions()
    # extensions = True
    # print('\n\n**********\nError building extensions, will just use native python implementation\nCaution this is slower and less efficient\n\nError {}\n\n**********'.format(e))
    version = versioneer.get_version()
    kwargs = dict(name='MTfit',
                  version=version,
                  cmdclass=versioneer.get_cmdclass(),
                  author=__author__,
                  author_email=__email__,
                  classifiers=[
                      'Environment :: Console',
                      'Development Status :: 4 - Beta',
                      'Intended Audience :: Science/Research',
                      'License :: Free for non-commercial use',
                      'Operating System :: Microsoft :: Windows',
                      'Operating System :: MacOS :: MacOS X',
                      'Operating System :: POSIX',
                      'Programming Language :: Cython',
                      'Topic :: Scientific/Engineering',
                      'Programming Language :: Python :: 2',
                      'Programming Language :: Python :: 2.7',
                      'Programming Language :: Python :: 3',
                      'Programming Language :: Python :: 3.5',
                      'Programming Language :: Python :: 3.6'],
                  packages=find_packages('src'),
                  package_dir={'': 'src'},
                  requires=['numpy', 'scipy'],
                  install_requires=['numpy>=1.7.0', 'scipy', 'cython>=0.20.2', 'setuptools', 'pyqsub>=1.0.2'],
                  provides=['MTfit'],
                  test_suite='MTfit.tests.test_suite',
                  description='MTfit: Bayesian Moment Tensor Inversion Code',
                  long_description=long_description,
                  package_data={'': ['*.rst', 'examples/README', 'examples/command_line.sh', 'examples/command_line.bat', '*.pyx', '*.pxd', '*.c',
                                     'extensions/example_parser/*', 'extensions/model_sampling_strike_slip/*', 'extensions/model_sampling_clvd/*',
                                     'docs/man/*', 'docs/epub/*.epub', 'docs/pdf/*', 'docs/html/*.*', 'docs/html/.doctrees/*.*', 'docs/html/_images/*.*',
                                     'docs/html/_downloads/*.*', 'docs/html/_modules/*.*', 'docs/html/_modules/MTfit/*.*', 'docs/html/_sources/*.*', 'docs/html/_static/*.*']}
                  )
    kwargs['extras_require'] = {'MATLAB -v7.3': ['h5py', 'hdf5storage'], 'HTML documentation': ['sphinx>=1.3.1'], 'Cluster': ['pyqsub>=1.0.2'], 'Plotting': ['matplotlib>=1.4.0']}
    kwargs['entry_points'] = {}
    kwargs['entry_points'] = {'console_scripts': ['MTfit = MTfit.run:run', 'MTplot = MTfit.plot.core:run'],
                              'MTfit.parsers': ['.csv = MTfit.utilities.file_io:parse_csv', '.hyp = MTfit.utilities.file_io:parse_hyp'],
                              'MTfit.location_pdf_parsers': ['.scatangle = MTfit.extensions.scatangle:location_pdf_parser'],
                              'MTfit.output_formats': ['matlab=MTfit.utilities.file_io:MATLAB_output', 'pickle=MTfit.utilities.file_io:pickle_output', 'hyp=MTfit.utilities.file_io:hyp_output'],
                              'MTfit.output_data_formats': ['full_pdf=MTfit.utilities.file_io:full_pdf_output_dicts', 'hyp=MTfit.utilities.file_io:hyp_output_dicts'],
                              'MTfit.cmd_defaults': ['scatangle=MTfit.extensions.scatangle:cmd_defaults'],
                              'MTfit.cmd_opts': ['scatangle=MTfit.extensions.scatangle:cmd_opts'],
                              'MTfit.pre_inversion': ['scatangle=MTfit.extensions.scatangle:pre_inversion']
                              }
    kwargs['include_dirs'] = [np.get_include()]
    extra_compile_args = ['-O3', '-march=native']
    libraries = ["m"]
    if 'win32' in sys.platform:
        extra_compile_args = []
        libraries = []
    if _CYTHON and 'build_ext' in sys.argv:
        kwargs['ext_modules'] = [Extension('MTfit.probability.cprobability', sources=['src/MTfit/probability/cprobability.pyx'], libraries=libraries, extra_compile_args=extra_compile_args, optional=True),
                                 Extension('MTfit.convert.cmoment_tensor_conversion', sources=[
                                           'src/MTfit/convert/cmoment_tensor_conversion.pyx'], libraries=libraries, extra_compile_args=extra_compile_args, optional=True),
                                 Extension('MTfit.extensions.cscatangle', sources=[
                                           'src/MTfit/extensions/cscatangle.pyx'], libraries=libraries, extra_compile_args=extra_compile_args, optional=True),
                                 Extension('MTfit.algorithms.cmarkov_chain_monte_carlo', sources=['src/MTfit/algorithms/cmarkov_chain_monte_carlo.pyx'], libraries=libraries, extra_compile_args=extra_compile_args, optional=True)]
        kwargs['cmdclass'] = {"build_ext": build_ext}
    else:
        kwargs['ext_modules'] = [Extension('MTfit.probability.cprobability', sources=['src/MTfit/probability/cprobability.c'], libraries=libraries, extra_compile_args=extra_compile_args, optional=True),
                                 Extension('MTfit.convert.cmoment_tensor_conversion', sources=[
                                           'src/MTfit/convert/cmoment_tensor_conversion.c'], libraries=libraries, extra_compile_args=extra_compile_args, optional=True),
                                 Extension('MTfit.extensions.cscatangle', sources=['src/MTfit/extensions/cscatangle.c'], libraries=libraries, extra_compile_args=extra_compile_args, optional=True),
                                 Extension('MTfit.algorithms.cmarkov_chain_monte_carlo', sources=['src/MTfit/algorithms/cmarkov_chain_monte_carlo.c'], libraries=libraries, extra_compile_args=extra_compile_args, optional=True)]
    setup(**kwargs)


def setup_help():
    # Run setup with cmd args (DEFAULT)
    if '--help' in sys.argv or '-h' in sys.argv:
        print("""setup.py script for MTfit

MTfit can be installed from the source by calling:

    $ python setup.py install

To install it as a non root user (or without using sudo):

    $ python setup.py install --user

This will install the module to the user site-packages directory. Alternatively, to install the module to a specific location use:

     $ python setup.py install --prefix=/path/to/top_level_directory

""")


def build_extensions():
    if 'build_ext' in sys.argv:
        print('------\nCLEANING EXTENSION FILES\n-----\n')
        exts = []
        if _CYTHON:
            exts.append('c')
        if 'win32' in sys.platform:
            exts.append('pyd')
        else:
            exts.append('so')
        for filename in ['src/MTfit/probablity/cprobability.', 'src/MTfit/algorithms/cmarkov_chain_monte_carlo.', 'src/MTfit/extensions/cscatangle.', 'src/MTfit/convert/cmoment_tensor_conversion.']:
            for ext in exts:
                if os.path.exists(filename+ext):
                    os.remove(filename+ext)
    elif 'develop' in sys.argv or 'sdist' in sys.argv or 'bdist_wheel' in sys.argv  :
        print('------\nBUILDING EXTENSIONS\n-----\n')
        argv = [sys.executable, "setup.py", "build_ext", "--inplace"]
        res = subprocess.call(argv)
        if res:
            print('------\nEXTENSIONS FAILED\n-----\n')


if __name__ == "__main__":
    setup_package()
