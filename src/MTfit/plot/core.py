# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.


import os


from .plot_classes import MTplot, MTData  # noqa F401
from ..utilities.file_io import read_matlab_output
from ..utilities.file_io import read_pickle_output
from ..utilities.file_io import read_hyp_output
from ..utilities.file_io import read_scatangle_output
from ..utilities.argparser import MTplot_parser as _parser
from ..utilities.extensions import get_extensions


def read(filename, station_distribution=False, *args, **kwargs):
    """
    Read filename or list of filenames (can be nested for multiplot MTplot call)

    Args
        filename: str/list filenames to read

    Returns
        list/dict, list/dict: tuple of event and station data
    """
    if type(filename) == list:
        events = []
        stations = []
        for fn in filename:
            if station_distribution:
                stations.append(read(fn))
            else:
                # recursive to allow for nested list structure for multiplot
                ev, st = read(fn)
                events.append(ev)
                stations.append(st)
    elif station_distribution:
        return _read(filename, station_distribution=station_distribution, *args, **kwargs)
    else:
        events, stations = _read(filename)
    if station_distribution:
        return stations
    return events, stations


def _read(filename, file_format=False, station_distribution=False, *args, **kwargs):
    """
    Read filename

    Args
        filename: str filenames to read

    Keyword Args
        file_format: str file format (default is to try and use ending)

    Returns
        dict, dict: tuple of event and station data
    """
    default_formats = {'.hyp': read_hyp_output, '.mt': read_hyp_output, '.mat': read_matlab_output,
                       '.out': read_pickle_output, '.pkl': read_pickle_output, '.scatangle': read_scatangle_output}
    parser_names, parsers = get_extensions('MTfit.plot_read', default_formats)
    if not file_format:
        file_format = os.path.splitext(filename)[-1]
    return parsers[file_format](filename, station_distribution=station_distribution, *args, **kwargs)


def plot(filename, *args, **kwargs):
    """
    Plot the filename using MTplot

    Args
        filename: str/list filenames to read
        kwargs: passed to MTplot call
        args: passed to MTplot call

    Returns
        MTplot object

    """
    # Handle File IO
    MTs, stations = read(filename)
    # Plot results
    return MTplot(MTs, stations=stations, *args, **kwargs)


def run(input_args=[]):
    """
    Run from the command line

    Has a blocking plot.show call, so will wait until the plot is closed.

    Keyword Args:
        input_args: list of input arguments

    Returns
        MTplot object
    """
    options = _parser(input_args)
    filename = options.pop('data_file')
    h = plot(filename, **options)
    return h
