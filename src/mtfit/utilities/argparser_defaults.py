"""
argparser_defaults.py
***********************

Dictionaries of MTfit and MTPlot argparser defaults and
default types

"""


# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.

from distutils.version import StrictVersion

from matplotlib import __version__ as matplotlib_version
#
# MTfit defaults
#
# Parameters for command line argument parser
# Defaults are set in this dictionary and updated by site and user defaults

MTFIT_PARSER_DEFAULTS = {
    'data_file': False,
    'location_pdf_file_path': False,
    'algorithm': 'time',
    'single_threaded': False,
    'number_workers': 0,
    'number_stations': 0,
    'memory': 8,
    'double-couple': False,
    'compare_constrained': False,
    'nstations': 0,
    'number_location_samples': 0,
    'disk_sample': False,
    'no_file_safe': False,
    'inversion_options': False,
    'output_file': False,
    'max_iteration_samples': 6000000,
    'max_time': 600,
    'multiple_events': False,
    'relative_amplitude': False,
    'marginalise_relative': False,
    'recover': False,
    'mpi': False,
    'benchmark': False,
    'test': False,
    'quality': False,
    'quality_check_value': 10,
    'debug': False,
    'diagnostics': False,
    'no_dist': False,
    'learning_length': 5000,
    'warnings': 'default',
    'acceptance_rate_window': 500,
    'initial_sampling': 'grid',
    'jump_probability': 0.01,
    'mcmc_min_number_initialisation_samples': 5000,
    'transdmcmc_min_number_initialisation_samples': 5000,
    'min_number_check_samples': 30000,
    'mcmc_max_acceptance_rate': 0.5,
    'mcmc_min_acceptance_rate': 0.3,
    'transdmcmc_max_acceptance_rate': 0.2,
    'transdmcmc_min_acceptance_rate': 0.05,
    'mcmc_chain_length': 10000,
    'transdmcmc_chain_length': 100000,
    'data_extension': 'inv',
    'location_extension': 'scatangle',
    'min_number_intersections': 2,
    'verbosity': 0,
    'nodes': 1,
    'ppn': 8,
    'pmem': 2,
    'walltime': '24:00:00',
    'queue': 'batch',
    'email': False,
    'email_options': 'bae',
    'output_format': 'matlab',
    'results_format': 'full_pdf',
    'no_normalise': False,
    'convert': False,
    'discard': 10000,
    'dc_prior': 0.5,
    'sampling': False,
    'sample_distribution': False,
    'sampling_prior': False,
    'mpi_output': False,
    'c_generate': False,
    'combine_mpi_output': False,
    'relative_loop': False
}

# Default types for the ARGPARSER_DEFAULTS structure, used for checking
# option types
MTFIT_PARSER_DEFAULT_TYPES = {
    'data_file': [str, bool],
    'location_pdf_file_path': [str, bool],
    'algorithm': [str],
    'single_threaded': [bool],
    'number_workers': [int],
    'number_stations': [int],
    'memory': [float],
    'double-couple': [bool],
    'compare_constrained': [bool],
    'nstations': [int],
    'number_location_samples': [int],
    'disk_sample': [bool],
    'no_file_safe': [bool],
    'inversion_options': [str, bool],
    'output_file': [str, bool],
    'max_iteration_samples': [int],
    'max_time': [float, int],
    'multiple_events': [bool],
    'relative_amplitude': [bool],
    'marginalise_relative': [bool],
    'recover': [bool],
    'mpi': [bool],
    'benchmark': [bool],
    'test': [bool],
    'quality': [bool],
    'quality_check_value': [float],
    'debug': [bool],
    'diagnostics': [bool],
    'mcmc_min_acceptance_rate': [float],
    'mcmc_max_acceptance_rate': [float],
    'no_dist': [bool],
    'learning_length': [int],
    'warnings': 'default',
    'acceptance_rate_window': [int],
    'initial_sampling': [str],
    'jump_probability': [float],
    'min_number_check_samples': [int],
    'mcmc_min_number_initialisation_samples': [int],
    'transdmcmc_min_number_initialisation_samples': [int],
    'data_extension': [str],
    'location_extension': [str],
    'min_number_intersections': [int],
    'verbosity': [int],
    'nodes': [int],
    'ppn': [int],
    'pmem': [float],
    'walltime': [str],
    'queue': [str],
    'email': [str, bool],
    'email_options': [str, bool],
    'transdmcmc_min_acceptance_rate': [float],
    'transdmcmc_max_acceptance_rate': [float],
    'mcmc_chain_length': [int],
    'transdmcmc_chain_length': [int],
    'output_format': [str],
    'results_format': [str],
    'no_normalise': [bool],
    'discard': [float],
    'convert': [bool],
    'dc_prior': [float],
    'sampling': [bool, str],
    'sampling_prior': [bool, str],
    'sample_distribution': [bool, str],
    'mpi_output': [bool],
    'c_generate': [bool],
    'combine_mpi_output': [bool],
    'relative_loop': [bool]
}

# MTplot defaults are set in this dictionary and updated by site and user
# defaults
DEFAULT_HIST_COLORMAP = 'viridis'
if StrictVersion(matplotlib_version) < StrictVersion('1.5.0'):
    DEFAULT_HIST_COLORMAP = 'CMRmap'
DEFAULT_AMP_COLORMAP = 'bwr'

MTPLOT_PARSER_DEFAULTS = {
    'plot_type': 'beachball',
    # Uses DEFAULT_AMP_COLORMAP or DEFAULT_HIST_COLORMAP as set above
    'colormap': False,
    'fontsize': 11,
    'linewidth': 1,
    'text': False,
    'resolution': 500,
    'bins': 100,
    'fault_plane': False,
    'nodal_line': True,
    'TNP': False,
    'markersize': 10,
    'station_markersize': 10,
    'show_max_likelihood': False,
    'show_mean': False,
    'grid_lines': False,
    'color': 'purple',
    'type_label': False,
    'hex_bin': False,
    'projection': 'equalarea',
    'save_file': '',
    'save_dpi': 200,
    'hide': False
}

# Default types for the MTplot parser structure, used for checking option types
MTPLOT_PARSER_DEFAULT_TYPES = {
    'plot_type': [str],
    'colormap': [str],
    'fontsize': [int],
    'linewidth': [int],
    'text': [bool],
    'resolution': [int],
    'bins': [int],
    'fault_plane': [bool],
    'nodal_line': [bool],
    'TNP': [bool],
    'markersize': [int],
    'station_markersize': [int],
    'show_max_likelihood': [bool],
    'show_mean': [bool],
    'grid_lines': [bool],
    'color': [str],
    'type_label': [bool],
    'hex_bin': [bool],
    'projection': [str],
    'save_file': [str],
    'save_dpi': [int],
    'hide': [bool],
}
