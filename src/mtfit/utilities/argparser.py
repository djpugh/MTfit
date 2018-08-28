"""
argparser.py
*************

Command line argument parser code for MTfit

"""


# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.

import optparse
import textwrap
import glob
import os
import sys
import multiprocessing
import subprocess
import logging

try:
    # Check if argparse present
    import argparse
    _ARGPARSE = True
except ImportError:
    _ARGPARSE = False
# Test flag for running qsub flags in test
try:
    # python module for cluster job submission using qsub.
    import pyqsub
    _PYQSUB = True
except ImportError:
    _PYQSUB = False

from .extensions import get_extensions
from .extensions import evaluate_extensions
from .. import __version__
from ..extensions import default_cmd_opts
from ..extensions import default_cmd_defaults
from .argparser_defaults import MTFIT_PARSER_DEFAULTS
from .argparser_defaults import MTFIT_PARSER_DEFAULT_TYPES
from .argparser_defaults import DEFAULT_HIST_COLORMAP
from .argparser_defaults import DEFAULT_AMP_COLORMAP
from .argparser_defaults import MTPLOT_PARSER_DEFAULTS
from .argparser_defaults import MTPLOT_PARSER_DEFAULT_TYPES


logger = logging.getLogger('MTfit')

# qsub test flag
_QSUBTEST = False


def get_details_json():
    from .. import get_details_json as _get_details_json
    return _get_details_json()


if _ARGPARSE:
    class ArgparseIndentedHelpFormatterWithNewLines(argparse.RawDescriptionHelpFormatter):

        """
        argparse help formatted with new lines

        Formats the argparse help with newlines.
        """

        def _format_action(self, action):
            """Formats the action with newlines. Adapted from base function."""
            # determine the required width and the entry label
            help_position = min(self._action_max_length + 2,
                                self._max_help_position)
            help_width = self._width - help_position
            action_width = help_position - self._current_indent - 2
            action_header = self._format_action_invocation(action)

            # no help; start on same line and add a final newline
            if not action.help:
                tup = self._current_indent, '', action_header
                action_header = '%*s%s\n' % tup

            # short action name; start on the same line and pad two spaces
            elif len(action_header) <= action_width:
                tup = self._current_indent, '', action_width, action_header
                action_header = '%*s%-*s  ' % tup
                indent_first = 0

            # long action name; start on the next line
            else:
                tup = self._current_indent, '', action_header
                action_header = '%*s%s\n' % tup
                indent_first = help_position

            # collect the pieces of the action help
            parts = [action_header]
            # if there was help for the action, add lines of help text
            if action.help:
                help_text = self._expand_help(action)
                help_lines = []
                for para in help_text.split("\n"):
                    if not len(textwrap.wrap(para, help_width)):
                        help_lines.extend(' ')
                    else:
                        help_lines.extend(textwrap.wrap(para, help_width))

                help_lines.extend(' ')
                help_lines.extend(' ')
                parts.append('%*s%s\n' % (indent_first, '', help_lines[0]))
                for line in help_lines[1:]:
                    parts.append('%*s%s\n' % (help_position, '', line))

            # or add a newline if the description doesn't end with one
            elif not action_header.endswith('\n'):
                parts.append('\n')

            # if there are any sub-actions, add their help as well
            for subaction in self._iter_indented_subactions(action):
                parts.append(self._format_action(subaction))

            # return a single string
            return self._join_parts(parts)

        def _format_action_invocation(self, action):
            result = super(ArgparseIndentedHelpFormatterWithNewLines, self)._format_action_invocation(action)
            if len(result) > self._width+5-2:
                checked = []
                ind = 0
                while ind < len(result):
                    if len(result)-ind <= self._width+5-2:
                        checked.append('  '+result[ind:].lstrip(' '))
                        break

                    checked.append(
                        '  '+result[ind:ind+result[ind:ind+self._width].rfind(', -')+1].lstrip(' '))
                    ind = ind+result[ind:ind+self._width].rfind(', -')+1
                checked[0] = checked[0].lstrip(' ')
                if len(checked[-1]) > self._current_indent:
                    checked.append('')
                result = '\n'.join(checked)
            return result


class OptparseIndentedHelpFormatterWithNewLines(optparse.IndentedHelpFormatter):

    """
    optparse help formatted with new lines

    Formats the optparse help with newlines.
    """

    def format_description(self, description):
        """Format the description with newlines. Adapted from base function."""
        if not description:
            return ""
        desc_width = self.width - self.current_indent
        indent = " "*self.current_indent
    # the above is still the same
        bits = description.split('\n')
        formatted_bits = [
            textwrap.fill(bit,
                          desc_width,
                          initial_indent=indent,
                          subsequent_indent=indent)
            for bit in bits]
        result = "\n".join(formatted_bits) + "\n"
        return result

    def format_option(self, option):
        """Format the option with newlines. Adapted from base function."""
        result = []
        opts = self.option_strings[option]
        opt_width = self.help_position - self.current_indent - 2
        if len(opts) > opt_width:
            opts = "%*s%s\n" % (self.current_indent, "", opts)
            indent_first = self.help_position
        else:  # start help on same line as opts
            opts = "%*s%-*s  " % (self.current_indent, "", opt_width, opts)
            indent_first = 0
        result.append(opts)
        if option.help:
            help_text = self.expand_default(option)
        # Everything is the same up through here
            help_lines = []
            for para in help_text.split("\n"):
                if not len(textwrap.wrap(para, self.help_width)):
                    help_lines.extend(' ')
                else:
                    help_lines.extend(textwrap.wrap(para, self.help_width))
            help_lines.extend(' ')
            help_lines.extend(' ')
        # Everything is the same after here
            result.append("%*s%s\n" % (
                indent_first, "", help_lines[0]))
            result.extend(["%*s%s\n" % (self.help_position, "", line)
                           for line in help_lines[1:]])
        elif opts[-1] != "\n":
            result.append("\n")
        return "".join(result)


def get_MTfit_defaults(test=False, extension_defaults={}, extension_default_types={}):
    return _get_env_defaults(test, extension_defaults, extension_default_types, "MTfit")


def get_MTplot_defaults(test=False, extension_defaults={}, extension_default_types={}):
    return _get_env_defaults(test, extension_defaults, extension_default_types, "MTplot")


def _get_env_defaults(test=False, extension_defaults={}, extension_default_types={}, app="MTfit"):
    """
    Gets environment defaults either from file or otherwise. Looks in the following locations in this order, with
    later files overriding the defaults:
        1. ~/MTfitdefaults (MTplotdefaults for MTplot)
        2. At the location specified by the MTFITDEFAULTSPATH (or MTPLOTDEFAULTSPATH)
        3. for a defaults file in the current working directory

    Default Format is simple on each line key:attr.
    Keys are given in the MTfit.utilities.argparser_defaults module
    """
    MTfit_flag = False
    if app.lower() == "mtplot":
        defaults_fname = ".MTplotdefaults"
        env_variable = "MTPLOTDEFAULTSPATH"
    else:
        defaults_fname = ".MTfitdefaults"
        env_variable = "MTFITDEFAULTSPATH"
        MTfit_flag = True
    default_files = []

    if 'win32' in sys.platform:
        home_defaults = os.environ['HOMEPATH']+os.path.sep+defaults_fname
    else:
        home_defaults = os.environ['HOME']+os.path.sep+defaults_fname
    if os.path.exists(home_defaults):
        default_files.append(home_defaults)
    if env_variable in os.environ.keys():
        default_files.append(os.environ[env_variable])
    if os.path.exists(os.getcwd()+os.path.sep+defaults_fname):
        default_files.append(os.getcwd()+os.path.sep+defaults_fname)
    if MTfit_flag:
        defaults = MTFIT_PARSER_DEFAULTS.copy()
        default_types = MTFIT_PARSER_DEFAULT_TYPES.copy()
    else:
        defaults = MTPLOT_PARSER_DEFAULTS.copy()
        default_types = MTPLOT_PARSER_DEFAULT_TYPES.copy()
    defaults.update(extension_defaults)
    default_types.update(extension_default_types)

    output_string = ''
    if test:
        return defaults
    for i, filename in enumerate(default_files):
        if i == 0:
            output_string += '--------------\nUsing default file: {} for default parameters\n----------------\n'.format(filename)
        else:
            output_string += '--------------\nUpdating default parameters using default file: {}\n----------------\n'.format(filename)
        lines = open(filename).readlines()
        error = False
        for line in lines:
            ok = True
            try:
                key = line.split(':')[0]
                attr = ':'.join(line.split(':')[1:]).rstrip()
            except Exception:
                if len(line):
                    error = True
                    ok = False
                    output_string += 'Error parsing defaults line {} - value ignored\n'.format(line)
            if ok:
                if key in defaults.keys():
                    if type(attr) not in default_types[key]:
                        ok = False
                        for tp in default_types[key]:
                            if not ok:
                                try:
                                    attr = tp(attr)
                                    ok = True
                                except Exception:
                                    pass
                        if not ok:
                            error = True
                            output_string += 'Error parsing attribute type for {}, expecting{}\n'.format(key, default_types[key])
                    if ok:
                        defaults[key] = attr
                else:
                    ok = False
                    error = True
                    output_string += 'Error parsing defaults key {} not in default keys\n'.format(key)

        if error:
            if MTfit_flag:
                keys = MTFIT_PARSER_DEFAULTS.keys()
            else:
                keys = MTPLOT_PARSER_DEFAULTS.keys()
            keys.extend(extension_defaults.keys())
            keys = list(set(keys))
            output_string += 'Errors in parsing defaults. Expecting file of the form \n\nkey:attr\nkey:attr\n\nValid keys are:\n{}\n'.format('\n'.join(sorted(keys)))
    log(output_string)
    return defaults


#
# Command line parser helper code and main functions
#


def log(string):
    """MPI print function"""
    if os.environ.__contains__('OMPI_COMM_WORLD_RANK'):
        if int(os.environ['OMPI_COMM_WORLD_RANK']) == 0:
            logging.info(string)
    else:
        logging.info(string)


def lower_string(input_string):
    """
    Convert string to lower TestCase

    argparse type function

    Args
        input_string: Input string to converts

    Returns:
        lower case of input string
    """
    return input_string.lower()


def check_path(input_string):
    """
    Check if the input string is a path, if * wildcard  present, checks the path using glob.glob.

    Args
        input_string: Input string to check if a path, can contain * as a wildcard. Can submit a comma separated list.

    Returns
        abspath for input_string. If the input_string is a comma delimited list, returns a list for all the paths, and if it contains a * wildcard,
            returns the list from glob.glob.

    Raises
        argparse.ArgumentTypeError if path doesn't exist and argparse is installed
        ValueError if path doesn't exist and argparse is not installed

    """
    if not input_string:
        return input_string
    elif ',' in input_string:
        # list
        files = input_string.lstrip('[').rstrip(']').split(',')
        for i, f in enumerate(files):
            files[i] = check_path(f)
        return files
    elif input_string and '*' in input_string and os.path.exists(os.path.abspath(os.path.split(input_string)[0])):
        # Check abspath when making file list
        return [os.path.abspath(u) for u in glob.glob(os.path.abspath(os.path.split(input_string)[0])+os.path.sep+os.path.split(input_string)[1])]
    elif input_string and '*' not in input_string and os.path.exists(os.path.abspath(input_string)):
        return os.path.abspath(input_string)
    if _ARGPARSE:
        raise argparse.ArgumentTypeError('Path: "'+input_string+'" does not exist')
    else:
        raise ValueError('Path: "'+input_string+'" does not exist')


def _data_file_search(data_path='./', data_extension='inv', angle_extension='scatangle'):
    """
    Searches for data and angle scatter file pairs

    Searches on the given data path for data files paired to angle scatter files (same file name for initial characters). Searches for data files using gloc:
    data_path+*.+data_extension, so data_path must have a path separator (ie slash) if a folder.

    Args
        data_path:['./'] Data path to search over.
        data_extension:['inv'] Extension for data files.
        angle_extension:['scatangle'] Extension for angle scatter files.
    Returns
        data_files,scatter_files:A list of data_files and a list of angle scatter files if there are any otherwise scatter_files is set to False.


    """
    data_files = glob.glob(data_path+'*.'+data_extension)
    scatter_files = []
    scatout = False
    for i, file_name in enumerate(data_files):
        file_name = os.path.abspath(file_name)
        data_files[i] = file_name
        scatter = _search(file_name, angle_extension)
        if scatter:
            scatter_files.append(scatter)
            scatout = True
        else:
            scatter_files.append('')
    if not scatout:
        scatter_files = False
    if len(data_files):
        return data_files, scatter_files
    else:
        scatter_files = glob.glob(data_path+os.path.sep+"*."+angle_extension)
        return data_files, scatter_files


def _search(file_name, extension='scatangle', n=12, it=1, imax=7):
    """
    Basic search for a matching file with a given basename and extension

    Searches using glob for a matching file with the extension given, where the match is over some length, so that there is at most 5 matching characters.

    Args
        file_name:base file name to search for.
        extension:['scatangle'] file extension to match to.
        n:[12] default inital length to match to.
        it:[1] iteration number (do not adjust from 1).
        imax:[7] maximum number of iterations to try.

    Returns
        matched file name if at least one exists, otherwise False.

    """
    scatter = glob.glob(os.path.split(
        file_name)[0]+os.path.sep+os.path.split(file_name)[1][:n]+"*."+extension)
    if it > imax:
        if len(scatter):
            return scatter[0]
        else:
            return False
    if len(scatter) < 1:
        return _search(file_name, extension, n-1, it+1, imax)
    if len(scatter) > 1:
        return _search(file_name, extension, n+1, it+1, imax)
    return scatter[0]

# MTfit


def _MTfit_argparser(input_args=None, test=False):
    """
    Return arguments parsed from command line

    Creates a command line argument parser (using argparse if present otherwise optparse (python <=2.6))
    and parses the command line arguments (or input_args if provided as a list). Contains some help formatter classes.

    Args
        input_args:[None] If provided, these are parsed instead of the command line arguments

    Returns
        parser,options,options_map: Parser object, the parsed options and the mapping from the parser options to the command line flags

    Raises
        Exception: If there is an error parsing the arguments

    """
    description = """MTfit - Moment Tensor Inversion Code by David J Pugh

    MTfit is a forward modelling based probabilistic evaluator of the source PDF.
    The inversion can be constrained to double-couple solutions only, or allowed to use the full moment tensor range.

    The approach evaluates the probability for a possible source given the observed data consisting of Polarity and/or Amplitude Ratio Information.

    Location uncertainty and model uncertainty can be included in the inversion as these correspond to uncertainties in the station ray path angles. These are included with a suitable ray path angle PDF.



    """
    # optparse parser description extra
    optparse_description = """Arguments are set as below, syntax is -dTest.i or --datafile=Test.i
    """
    # argparse parser description extra
    argparse_description = """Arguments are set as below, syntax is -dTest.i or -d Test.i
    """
    # Help for the angle scatter option
    location_pdf_help = """Path to location scatter angle files - wild cards behave as normal.
    To include the model and location uncertainty, a ray path angle pdf file must be provided.
    This is of the form:
    probability1
    Station1    Azimuth1    TakeOffAngle1
    Station2    Azimuth2    TakeOffAngle2
    .
    .
    .
    StationN    AzimuthN    TakeOffAngleN

    probability2
    Station1    Azimuth1    TakeOffAngle1
    Station2    Azimuth2    TakeOffAngle2
    .
    .
    .
    StationN    AzimuthN    TakeOffAngleN

    e.g.:
    504.7
    S0529   326.1   131.7
    S0083   223.7   118.2
    S0595   42.6    117.8
    S0236   253.6   118.6
    &&
    504.7
    S0529   326.1   131.8
    S0083   223.7   118.2
    S0595   42.7    117.9
    S0236   253.5   118.7"""
    # Help for the data file structure
    data_file_arg_help = "Data file to use for the inversion, optional but must be specified either"
    data_file_arg_help += " as a positional argument or as an optional argument (see -d below) If not"
    data_file_arg_help += " specified defaults to all *.inv files in current directory, and searches"
    data_file_arg_help += " for all anglescatterfiles in the directory too. Inversion file extension "
    data_file_arg_help += "can be set using --invext option. Angle scatter file extension can be set "
    data_file_arg_help += "using --angleext option"
    data_file_help = """Data file to use for the inversion. Can be provided as a positional argument.
    There are several different data file types:

        * pickled dictionary
        * csv file
        * NLLOC hyp file

    The data file is a pickled python dictionary of the form:
      {'DataType':{'Stations':{'Name':['STA1','STA2',...],
        'Azimuth':np.matrix([[190],[40],...]),
        'TakeOffAngle':np.matrix([[70],[40],...])},
        'Measured':np.matrix([[1],[-1],...]),
        'Error':np.matrix([[0.01],[0.02],...])}}

    e.g.:
      {'P/SHRMSAmplitudeRatio':{'Stations':{'Name':['S0649',"S0162"],
        'Azimuth':np.array([90.0,270.0]),
        'TakeOffAngle':np.array([30.0,60.0])},
        'Measured':np.matrix([[1],[-1]]),
        'Error':np.matrix([[ 0.001,0.02],[ 0.001,0.001]])}}

    Or a CSV file with events split by blank lines, a header line showing which row corresponds to which information (default is as shown here),
    UID and data-type information stored in the first column,
    e.g.:
            UID=123,,,,
            PPolarity,,,,
            Name,Azimuth,TakeOffAngle,Measured,Error
            S001,120,70,1,0.01
            S002,160,60,-1,0.02
            P/SHRMSAmplitudeRatio,,,,
            Name,Azimuth,TakeOffAngle,Measured,Error
            S003,110,10,1,0.05 0.04
            ,,,,
            PPolarity ,,,,
            Name,Azimuth,TakeOffAngle,Measured,Error
            S003,110,10,1,0.05

    Is a CSV file with 2 events, one event UID of 123, and PPolarity data at S001 and S002 and P/SHRMSAmplitude data at S003,
    and a second event with no UID (will default to the event number, in this case 2) with PPolarity data at S003.

    This data format can be constructed manually or automatically."""
    # Help for the algorithm option
    algorithm_help = """Selects the algorithm used for the search. [default=time]
    Possible algorithms are:
        iterate (random sampling of the source space for a set number of samples)
        time (random sampling of the source space for a set time)
        mcmc (Markov chain Monte Carlo sampling) """
    # Help for the inversion options
    inversion_options_help = """Set the inversion data types to use: comma delimited.
    If not set, the inversion uses all the data types in the data file.
    e.g.
    PPolarity,P/SHRMSAmplitudeRatio

    Needs to correspond to the data types in the data file.

    If not specified can lead to independence errors: e.g.
    P/SH Amplitude Ratio and P/SV Amplitude Ratio can give SH/SV Amplitude Ratio.
    Therefore using SH/SV Amplitude Ratio in the inversion is reusing data and will artificially sharpen the PDF.
    This applies to all forms of dependent measurements.
    """
    warnings_help = """Sets the warning visibility.

    options are:

        * "e","error" - turn matching warnings into exceptions
        * "i","ignore" - never print matching warnings
        * "a","always" - always print matching warnings
        * "d","default" - print the first occurrence of matching warnings for each location where the warning is issued
        * "m","module" - print the first occurrence of matching warnings for each module where the warning is issued
        * "o","once" - print only the first occurrence of matching warnings, regardless of location
    """
    # Set up qsub defaults
    options_map = {}
    algorithm_options = ["iterate", "time", "mcmc", "transdmcmc"]
    output_options = ['matlab', 'pickle', 'hyp']
    output_formats = get_extensions('MTfit.output_formats')[0]
    output_options.extend(output_formats)
    output_options = list(set(output_options))
    output_data_options = ['full_pdf', 'hyp']
    output_data_formats = get_extensions('MTfit.output_data_formats')[0]
    output_data_options.extend(output_data_formats)
    output_data_options = list(set(output_data_options))
    # Get extension defaults
    cmd_defaults = {}
    cmd_default_types = {}
    results = evaluate_extensions('MTfit.cmd_defaults', default_cmd_defaults)
    for result in results:
        cmd_defaults.update(result[0])
        cmd_default_types.update(result[1])
    # Try loading MTfitdefaults from env var or ~/.MTfitdefaults
    defaults = get_MTfit_defaults(test, cmd_defaults, cmd_default_types)
    # Get extension options
    cmd_opt_names, cmd_opts = get_extensions('MTfit.cmd_opts', default_cmd_opts)
    extension_parser_checks = []
    arguments = [
        dict(flags=["-d", "--datafile", "--data_file"], help=data_file_help,
             type=check_path, dest='DATAFILE', default=defaults['data_file']),
        dict(flags=["-s", "--anglescatterfilepath", "--location_pdf_file_path", "--location_file_path", "--scatterfilepath", "--scatter_file_path"],
             type=check_path, default=defaults['location_pdf_file_path'], help=location_pdf_help, dest="location_pdf_file_path"),
        dict(flags=["-a", "--algorithm"], choices=algorithm_options, type=lower_string,
             default=defaults['algorithm'], dest="algorithm", help=algorithm_help),
        dict(flags=["-l", "--singlethread", "--single", "--single_thread"], action="store_true",
             default=defaults['single_threaded'], help="Flag to disable parallel computation", dest="singlethread"),
        dict(flags=["-n", "--numberworkers", "--number_workers"], default=defaults['number_workers'], type=int,
             help="Set the number of workers used in the parallel computation. [default=all available cpus]", dest="n"),
        dict(flags=["-m", "--mem", "--memory", "--physical_memory", "--physicalmemory"], default=defaults['memory'], type=float,
             help="Set the maximum memory used in Gb if psutil not available [default="+str(defaults['memory'])+"Gb]", dest='mem'),
        dict(flags=["-c", "--doublecouple", "--double-couple", "--double_couple", "--dc", "--DC"], action="store_true",
             default=defaults['double-couple'], help=" Flag to constrain the inversion to double-couple sources only", dest="dc"),
        dict(flags=["-b", "--compareconstrained", "--compare_constrained"], action="store_true", default=defaults['compare_constrained'],
             help=" Flag to run two inversions, one constrained to double-couple and one unconstrained", dest="dc_mt"),
        dict(flags=["--nstations"], type=int, default=defaults['number_stations'],
             help="Set the maximum number of stations without having to load an angle pdf file - used for calculating sample sizes and memory sizes, and can speed up the calculation a bit, but has no effect on result.", dest='number_stations'),
        dict(flags=["--nanglesamples", "--nlocationsamples", "--number_location_samples", "--number-location-samples"], type=int, default=defaults['number_location_samples'],
             help="Set the maximum number of angle pdf samples to use. If this is less than the total number of samples, a subset are randomly selected [default="+str(defaults['number_location_samples'])+"].", dest="number_location_samples"),
        dict(flags=["-f", "--file_sample", "--file-sample", "--filesample", "--disk_sample", "--disk-sample", "--disksample"], default=defaults['disk_sample'],
             action='store_true', help="Save sampling to disk (allows for easier recovery and reduces memory requirements, but can be slower)", dest='file_sample'),
        dict(flags=["--not_file_safe", "--not-file-safe", "--notfilesafe", "--no_file_safe", "--no-file-safe", "--nofilesafe"], default=defaults[
             'no_file_safe'], action='store_true', help="Disable file safe saving (i.e. copy and write to .mat~ then copy back", dest='no_file_safe'),
        dict(flags=["-i", "--inversionoptions", "--inversion_options"], default=defaults[
             'inversion_options'], help=inversion_options_help, dest="inversion_options"),
        dict(flags=["-o", "--out", "--fid", "--outputfile", "--outfile"], default=defaults[
             'output_file'], help="Set output file basename [default=MTfitOutput]", dest="fid"),
        dict(flags=["-x", "--samples", "--maxsamples", "--max_samples", "--chain_length", "--max-samples", "--chain-length", "--chainlength"], default=False, type=int, help="Iteration algorithm: Set maximum number of samples to use [default=" +
             str(defaults['max_iteration_samples'])+"]. McMC algorithms: Set chain length [default="+str(defaults['mcmc_chain_length'])+"], trans-d McMC [default="+str(defaults['transdmcmc_chain_length'])+"]", dest="max_samples"),
        dict(flags=["-t", "--time", "--maxtime", "--max_time"], default=defaults['max_time'], type=float,
             help="Time algorithm: Set maximum time to use [default="+str(defaults['max_time'])+"]", dest="max_time"),
        dict(flags=['-e', '--multiple_events', '--multiple-events'], default=defaults['multiple_events'],
             action='store_true', help="Run using events using joint PDF approach", dest='multiple_events'),
        dict(flags=['-r', '--relative_amplitude', '--relative-amplitude'], default=defaults['relative_amplitude'],
             action='store_true', help="Run using events using joint PDF approach", dest='relative_amplitude'),
        dict(flags=['--marginalise_relative', '--marginalise', '--marginalise-relative'], default=defaults['marginalise_relative'], action='store_true',
             help="Flag to marginalise location uncertainty in relative amplitude case [default="+str(defaults['marginalise_relative'])+"]", dest='marginalise_relative'),
        dict(flags=["-R", "--recover"], action="store_true", default=defaults['recover'],
             help="Recover crashed run (ie restart from last event not written out)]", dest="recover"),
        dict(flags=["--invext", "--dataextension", "--dataext", "--data-extension", "--data_extension"], default=defaults[
             'data_extension'], help="Set data file extension to search for when inverting on a folder", dest="data_extension"),
        dict(flags=["--angleext", "--locationextension", "--locationext", "--location-extension", "--location_extension"], default=defaults[
             'location_extension'], help="Set location sample file extension to search for when inverting on a folder", dest="angle_extension"),
        dict(flags=["-S", "--minimum_number_intersections", "--min_number_intersections", "--minimum-number-intersections", "--min-number-intersections"], type=int, default=defaults['min_number_intersections'],
             help="For relative amplitude inversion, the minimum number of intersecting stations required (must be greater than 1) [default="+str(defaults['min_number_intersections'])+"]", dest='minimum_number_intersections'),
        dict(flags=['-M', '--mpi', '--MPI'], default=defaults['mpi'], action='store_true',
             help="Run using mpi - will reinitialise using mpirun (mpi etc needs to be added to path)", dest='mpi'),
        dict(flags=['-B', '--benchmark', '--benchmarking'], default=defaults['benchmark'],
             action='store_true', help="Run benchmark tests for the event", dest='benchmark'),
        dict(flags=['-X', '--min_number_check_samples', '--min_number_initialisation_samples'], default=False, type=int,
             help="Minimum number of samples for McMC initialiser, or the minimum number of samples required when using quality check (-Q)", dest='min_number_initialisation_samples'),
        dict(flags=["-T", "--test", "--test"], default=defaults['test'], action='store_true',
             help="Run MTfit Test suite (if combined with -q runs test suite on cluster", dest="test"),
        dict(flags=["-Q", "--quality"], default=defaults['quality'], nargs='?', action='store',
             help="Run MTfit with quality checks enabled [default=False]. Checks if an event has a percentage of non-zero samples lower than the flag - values from 0-100.", dest="quality_check"),
        dict(flags=["-D", "--debug"], default=defaults['debug'],
             action='store_true', help="Run MTfit with debugging enabled.", dest="debug"),
        dict(flags=["-V", "--verbosity"], default=defaults['verbosity'], type=int,
             help="Set verbosity level for non-fatal errors [default=0].", dest="verbosity"),
        dict(flags=["-g", "--diagnostics"], default=defaults['diagnostics'], action='store_true',
             help="Run MTfit with diagnostic output. Outputs the full chain and sampling - wil make a large file.", dest="diagnostic_output"),
        dict(flags=["-j", "--jumpProbability", "--jumpProb", "--jumpprob", "--jumpProb", "--dimensionJumpProb", "--dimensionjumpprob"], default=defaults['jump_probability'],
             type=float, help="Sets the probability of making a dimension jump in the Trans-Dimensional McMC algorithm [default=0.01]", dest="dimension_jump_prob"),
        dict(flags=["-y", "--initialSampling"], default=defaults['initial_sampling'],
             help="Sets the initialisation sampling method for McMC algorithms choices:\ngrid - use grid based sampling to find non-zero initial sample [default=grid]", choices=['grid'], dest="initial_sample"),
        dict(flags=["-u", "--minAcceptanceRate", "--minacceptancerate", "--min_acceptance_rate"], default=False, type=float, help="Set the minimum acceptance rate for the McMC algorithm [mcmc default=" +
             str(defaults['mcmc_min_acceptance_rate'])+", transdmcmc default="+str(defaults['transdmcmc_min_acceptance_rate'])+"]", dest="min_acceptance_rate"),
        dict(flags=["-v", "--maxAcceptanceRate", "--maxacceptancerate", "--max_acceptance_rate"], default=False, type=float, help="Set the maximum acceptance rate for the McMC algorithm [mcmc default=" +
             str(defaults['mcmc_max_acceptance_rate'])+", transdmcmc default="+str(defaults['transdmcmc_max_acceptance_rate'])+"]", dest="max_acceptance_rate"),
        dict(flags=["-w", "--acceptanceLearningWindow", "--acceptancelearningwindow"], type=int, default=defaults['acceptance_rate_window'],
             help="Sets the window for calculating and updating the acceptance rate for McMC algorithms [default="+str(defaults['acceptance_rate_window'])+"]", dest='acceptance_rate_window'),
        dict(flags=["-W", "--warnings", "--Warnings"], type=str,
             default=defaults['warnings'], help=warnings_help, dest='warnings'),
        dict(flags=["-z", "--learningLength", "--learninglength", "--learning_length"], default=defaults['learning_length'], type=int,
             help="Sets the number of samples to discard as the learning period [default="+str(defaults['learning_length'])+"]", dest="learning_length"),
        dict(flags=["--version"], action="version",
             version="%(prog)s "+__version__),
        dict(flags=["--detail"], action="version",
             version=get_details_json()),
        dict(flags=["--mpi_call"], default=False, action='store_true',
             help='DO NOT USE - only for spawning mpi subprocess', dest='_mpi_call'),
        dict(flags=["--output-format", "--output_format", "--outputformat", "--format"], default=defaults['output_format'],
             choices=output_options, type=lower_string, help='Output file format [default='+str(defaults['output_format'])+']', dest='output_format'),
        dict(flags=["--results-format", "--results_format", "--resultsformat"], default=defaults['results_format'], choices=output_data_options,
             help='Output results data format (extensible) [default='+str(defaults['results_format'])+']', dest='results_format'),
        dict(flags=["--no-dist", "--no_dist", "--nodist"], default=defaults['no_dist'], action='store_true',
             help='Do not output station distribution if running location samples', dest='no_station_distribution'),
        dict(flags=["--dc-prior", "--dc_prior", "--dcprior"], default=defaults['dc_prior'], type=float,
             help='Prior probability for the double-couple model when using the Trans-Dimensional McMC algorithm', dest='dc_prior'),
        dict(flags=["--sampling", "--sampling", "--sampling"], default=defaults['sampling'],
             type=float, help='Random moment tensor sampling distribution', dest='sampling'),
        dict(flags=["--sample-models", "--sample_distribution", "--samplemodels"], default=defaults['sample_distribution'],
             type=str, help='Alternate models for random sampling (Monte Carlo algorithms only)', dest='sample_distribution'),
        dict(flags=["--sampling-prior", "--sampling_prior", "--samplingprior"], default=defaults['sampling_prior'], type=str,
             help='Prior probability for the model distribution when using the McMC algorithm, alternatively the prior distribution for the source type parameters gamma and delta for use by the Bayesian evidence calculation for the MC algorithms', dest='sampling_prior'),
        dict(flags=["--no-normalise", "--no-norm", "--no_normalise", "--no_norm"], default=defaults[
             'no_normalise'], action='store_true', help='Do not normalise the output pdf', dest='no_normalise'),
        dict(flags=["--convert"], default=defaults['convert'], action='store_true',
             help='Convert the output MTs to Tape parameters, hudson parameters and strike dip rakes.', dest='convert'),
        dict(flags=["--discard"], default=defaults['discard'], type=float,
             help='Fraction of maxProbability * total samples to discard as negligeable.', dest='discard'),
        dict(flags=["--mpioutput", "--mpi_output", "--mpi-output"], default=defaults['mpi_output'], action='store_true',
             help='When the mpi flag -M is used outputs each processor individually rather than combining', dest='mpi_output'),
        dict(flags=["--combine_mpi_output", "--combine-mpi-output", "--combinempioutput"], default=defaults['combine_mpi_output'], action='store_true',
             help='Combine the mpi output from the mpioutput flag. The data path corresponds to the root path for the mpi output', dest='combine_mpi_output'),
        dict(flags=["--c_generate", "--c-generate", "--generate"], default=defaults['c_generate'],
             action='store_true', help='Generate moment tensor samples in the probability evaluation', dest='c_generate'),
        dict(flags=["--relative_loop", "--relative-loop", "--relativeloop", "--loop"], default=defaults['relative_loop'], action='store_true',
             help='Loop over independent non-zero samples randomly to construct joint rather than joint samples', dest='relative_loop')
    ]
    if _ARGPARSE:
        parser = argparse.ArgumentParser(prog='MTfit', description=description+argparse_description, formatter_class=ArgparseIndentedHelpFormatterWithNewLines)
        parser.add_argument('data_file', type=check_path, help=data_file_arg_help, nargs="?")
        for arg in arguments:
            kwargs = {key: value for (key, value) in arg.items() if key != 'flags'}
            parser.add_argument(*arg['flags'], **kwargs)
        for (name, extension) in cmd_opts.items():
            group = parser.add_argument_group(name.capitalize(), description="\nCommands for the extension "+name)
            (group, extension_parser_check) = extension(group, _ARGPARSE, defaults)
            extension_parser_checks.append(extension_parser_check)
        if _PYQSUB:
            group = parser.add_argument_group('Cluster', description="\nCommands for using MTfit on a cluster environment using qsub/PBS")
            group = pyqsub.parser_group(module_name='MTfit', group=group, default_nodes=defaults['nodes'], default_ppn=defaults['ppn'], default_pmem=defaults['pmem'],
                                        default_walltime=defaults['walltime'], default_queue=defaults['queue'], default_email_options=defaults['email_options'],
                                        default_email=defaults['email'])
        for option in parser._actions:
            if len(option.option_strings):
                i = 0
                while i < len(option.option_strings) and '--' not in option.option_strings[i]:
                    i += 1
                options_map[option.dest] = option.option_strings[i]
        # For testing
        if input_args is not None:
            options = parser.parse_args(input_args)
        else:
            options = parser.parse_args()
        options = vars(options)
        options['parallel'] = not options.pop('singlethread')
        if not options['data_file'] and not options['DATAFILE']:
            if not options['_mpi_call']:
                log("Data file not provided, using current directory.")
            options['data_file'] = os.path.abspath('./')
        elif options['data_file'] and options['DATAFILE']:
            parser.error("Multiple data files specified.")
        elif options['DATAFILE']:
            options['data_file'] = options['DATAFILE']
        options.pop('DATAFILE')
    else:
        parser = optparse.OptionParser(prog='MTfit', description=description+optparse_description, formatter=OptparseIndentedHelpFormatterWithNewLines(), version="%(prog)s "+__version__, usage="%prog [options]\nUse -h to get more information")
        for arg in arguments:
            kwargs = {
                key: value for (key, value) in arg.items() if key != 'flags'}
            parser.add_option(*arg['flags'], **kwargs)
        for (name, extension) in cmd_opts.items():
            group = optparse.OptionGroup(
                parser, name, description="\nCommands for the extension "+name)
            (group, extension_parser_check) = extension(
                group, _ARGPARSE, defaults)
            parser.add_option_group(group)
            extension_parser_checks.append(extension_parser_check)
        if _PYQSUB:
            group = optparse.OptionGroup(
                parser, 'Cluster', description="\nCommands for using MTfit on a cluster environment using qsub/PBS")
            group = pyqsub.parser_group(module_name='MTfit', group=group, default_nodes=defaults['nodes'], default_ppn=defaults['ppn'], default_pmem=defaults[
                                        'pmem'], default_walltime=defaults['walltime'], default_queue=defaults['queue'], default_email_options=defaults['email_options'], default_email=defaults['email'])
            parser.add_option_group(group)
        for option in parser.option_list:
            options_map[option.dest] = option.get_opt_string()
        if input_args and len(input_args):
            (options, args) = parser.parse_args(input_args)
        else:
            (options, args) = parser.parse_args()
        options = vars(options)
        options['parallel'] = not options.pop('singlethread')
        options['data_file'] = False
        if len(args):
            options['data_file'] = args[0]

        if not options['data_file'] and not options['DATAFILE']:
            if not options['_mpi_call']:
                log("Data file not provided, using current directory.")
            options['data_file'] = os.path.abspath('./')
        elif options['data_file'] and options['DATAFILE']:
            parser.error("Multiple data files specified.")
        elif options['DATAFILE']:
            options['data_file'] = options['DATAFILE']
        options.pop('DATAFILE')
        if isinstance(options['quality_check'], list) and len(options['quality_check']):
            options['quality_check'] = float(options['quality_check'][0])
        elif isinstance(options['quality_check'], list):
            options['quality_check'] = None
        try:
            options['data_file'] = check_path(options['data_file'])
        except ValueError:
            if not options['bin_scatangle']:
                parser.error('Data file: "{}" does not exist'.format(options['data_file']))
        try:
            options['location_pdf_file_path'] = check_path(options['location_pdf_file_path'])
        except ValueError:
            parser.error('Angle Scatter file path: "{}" does not exist'.format(options['location_pdf_file_path']))
    # Check sampling_priors
    if not options['sampling_prior']:
        options.pop('sampling_prior')
    if not options['sampling']:
        options.pop('sampling')
    if not options['sample_distribution']:
        options.pop('sample_distribution')
    # Set path variable
    if (isinstance(options['data_file'], list) and os.path.isdir(options['data_file'][0])) or (not isinstance(options['data_file'], list) and os.path.isdir(options['data_file'])):
        options['path'] = os.path.abspath(options['data_file'])
    options['chain_length'] = options['max_samples']
    flags = []
    for extension_parser_check in extension_parser_checks:
        options, ext_flags = extension_parser_check(parser, options, defaults)
        flags.extend(ext_flags)
    flags = list(set(flags))
    return parser, options, options_map, defaults, flags


def MTfit_parser(input_args=False, test=False):
    """
    Parses the command line arguments using _argparser and handles and returns the options

    Args
        input_args:[False] If provided, these are parsed instead of the command line arguments

    Returns
        options,options_map: The parsed options and the mapping from the parser options to the command line flags

    Raises
        Exception: If there is an error parsing the arguments

    """
    parser, options, options_map, defaults, flags = _MTfit_argparser(input_args, test)
    if not _PYQSUB:
        options['qsub'] = False
    if options['relative_amplitude'] and not options['multiple_events']:
        options['multiple_events'] = True
    if options['relative_amplitude']:
        if options['minimum_number_intersections'] <= 1:
            parser.error(
                'Minimum number of stations for intersection must be greater than 1')
    if options['marginalise_relative'] and not options['relative_amplitude']:
        parser.error(
            'Marginalise relative flag requires -r relative flag too.')
    if options['quality_check'] is None:
        options['quality_check'] = defaults['quality_check_value']
    if options['quality_check']:
        options['quality_check'] = float(options['quality_check'])
        options['quality_check'] = min(options['quality_check'], 100.0)
        options['quality_check'] = max(options['quality_check'], 0.0)
    if options['recover']:
        if not options['_mpi_call']:
            log('Recovering most recent run')
        qsub = False
        p_files = []
        if isinstance(options['data_file'], list) and len([f for f in options['data_file'] if '.p' in f]):
            p_files = [f for f in options['data_file'] if '.p' in f]
        if '.p' in options['data_file']:
            # Have a pfile in options ['Datafile']
            qsub = True
            if '*' in options['data_file']:
                p_files = glob.glob(options['data_file'])
            elif not isinstance(options['data_file'], list):
                p_files = options['data_file'].split(',')
        elif not isinstance(options['data_file'], list):
            # Look for .p files in options['Datafile']
            p_files = glob.glob(options['data_file']+os.path.sep+'*.p*')
            qsub = True
        if len([key for key in os.environ.keys() if 'PBS' in key and key != 'PBS_DEFAULT']):
            qsub = False

        try:
            subprocess.check_call(["which", "qsub"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            # Allow testing of qsub output
            if not _QSUBTEST:
                qsub = False
        pbs_files = [pbs_file for pbs_file in p_files if '.py' not in pbs_file]
        if (qsub or _QSUBTEST) and len(pbs_files):
            # try looking for MTfit.p with highest number - corresponds to cluster environment
            # append r and re qsub unless on cluster
            # sort by ascending job no
            pbs_files.sort(key=lambda x: x.split('.p')[1], reverse=True)
            for pbs_file in pbs_files:
                if '##mtfit qsub script' in open(pbs_file).read().lower():
                    if not options['_mpi_call']:
                        log('Recovering Run: '+pbs_file)
                    # MTfit Script therefore read, parse and act
                    line = [l for l in open(pbs_file).readlines() if 'python -c "import' in l and '#' not in l[:10]][0].rstrip()
                    qsub_args = []
                    for arg in line.split('MTfit.__run__()"')[1].split(' '):
                        if '=' in arg or '--' in arg:
                            qsub_args.append(arg)
                        elif len(arg):
                            qsub_args[-1] += ' '+arg
                    _, qsub_options, _, dd, fl = _MTfit_argparser(qsub_args)
                    for option_name in qsub_options.keys():
                        if 'qsub' not in option_name:
                            options[option_name] = qsub_options[option_name]
                    options['qsub'] = True
                    options['recover'] = True
                    break
    if options['test']:
        if options['qsub']:
            if len([key for key in os.environ.keys() if 'PBS' in key and key != 'PBS_DEFAULT']) and not _QSUBTEST:
                parser.error('Cannot submit as already on cluster')
            try:
                subprocess.check_call(["which", "qsub"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception:
                # Allow testing of qsub output
                if not _QSUBTEST:
                    parser.error('Could not find qsub - cannot run in cluster mode')
        return options, options_map
    if not isinstance(options['data_file'], list) and os.path.isdir(options['data_file']):
        options['path'] = os.path.abspath(options['data_file'])
        if not options['combine_mpi_output']:
            data_files, scatter_files = _data_file_search(
                options['data_file']+os.path.sep, options['data_extension'], options['angle_extension'])
            if not len(data_files):
                options['data_file'] = False
                if 'no_data_file_ok' not in flags and not options['combine_mpi_output']:
                    parser.error('No Data files found in current folder')
            else:
                options['data_file'] = data_files
            if 'no_location_update' not in flags:
                options['location_pdf_file_path'] = scatter_files
    elif not isinstance(options['data_file'], list) and 'no_data_file_ok' not in flags:
        options['data_file'] = [options['data_file']]
    if options['dc'] and options['dc_mt']:
        parser.error('Cannot select --double_couple constraint and --compare_constrained flags.')
    options.pop('data_extension')
    # options.pop('angle_extension')
    if options['parallel'] and options['n'] == 1:
        if not options['_mpi_call']:
            log('Parallel running disabled as number of workers set to 1')
        options['parallel'] = False
    if options['inversion_options'] and len(options['inversion_options']):
        if 'win' in sys.platform:
            options['inversion_options'] = ','.join(
                options['inversion_options'].split(' ')).split(',')
        else:
            options['inversion_options'] = options[
                'inversion_options'].split(',')

    else:
        options['inversion_options'] = False
    if options['algorithm'] not in ['iterate']:
        # McMC algorithms use max samples argument for chain length
        if options['max_samples'] and options['algorithm'] in ['time']:
            if not options['_mpi_call']:
                log('max_samples argument ignored as {} algorithm selected.'.format(options['algorithm']))
        options.pop('max_samples')
    if options['algorithm'] != 'time':
        if options['max_time']:
            if not options['_mpi_call']:
                log('max_time argument ignored as {} algorithm selected.'.format(options['algorithm']))
        options.pop('max_time')
    if options['algorithm'] == 'time' and not options['max_time']:
        options.pop('max_time')
    if options['algorithm'] == 'iterate' and not options['max_samples']:
        options.pop('max_samples')
    if 'mcmc' in options['algorithm'] and not options['chain_length']:
        options['chain_length'] = defaults[
            options['algorithm'].lower()+'_chain_length']
    elif 'iterate' in options['algorithm'] and (('max_samples' in options.keys() and not options['max_samples']) or'max_samples' not in options.keys()):
        options['max_samples'] = defaults['max_iteration_samples']
    if 'mcmc' in options['algorithm'] and not options['min_acceptance_rate']:
        options['min_acceptance_rate'] = defaults[
            options['algorithm'].lower()+'_min_acceptance_rate']
    if 'mcmc' in options['algorithm'] and not options['max_acceptance_rate']:
        options['max_acceptance_rate'] = defaults[
            options['algorithm'].lower()+'_max_acceptance_rate']
    if 'mcmc' in options['algorithm'] and options['max_acceptance_rate'] <= options['min_acceptance_rate']:
        parser.error(
            'Acceptance rate values cannot be the same, maximum acceptance rate must be bigger than the minimum acceptance rate')
    if 'mcmc' in options['algorithm'] and not options['min_number_initialisation_samples']:
        options['min_number_initialisation_samples'] = defaults[
            options['algorithm'].lower()+'_min_number_initialisation_samples']
    options['min_number_check_samples'] = options[
        'min_number_initialisation_samples']
    if 'mcmc' not in options['algorithm'] and not options['min_number_initialisation_samples']:
        options.pop('min_number_initialisation_samples')
        options['min_number_check_samples'] = defaults[
            'min_number_check_samples']
    if 'mcmc' in options['algorithm']:
        if options['mpi']:
            log('Cannot run McMC with MPI - disabling mpi')
        options['mpi'] = False
    options['phy_mem'] = options.pop('mem')
    options_map['phy_mem'] = '--mem'
    if options['warnings'].lower() not in ['a', 'always', 'd', 'default', 'e', 'error', 'i', 'ignore', 'm', 'module', 'o', 'once']:
        parser.error('warnings flag not recognised')
    options['warnings'] = options['warnings'][0].lower()
    if options['warnings'][0] not in ['a', 'e', 'd', 'i', 'm', 'o']:
        options['warnings'] = 'd'
    # Check MPI vs number_workers:
    if options['mpi'] and multiprocessing.cpu_count() == 1:
        options['mpi'] = False
    if options['qsub']:
        options['qsub_mpi'] = options['mpi']
        if len([key for key in os.environ.keys() if 'PBS' in key and key != 'PBS_DEFAULT']) and not _QSUBTEST:
            parser.error('Cannot submit as already on cluster')
        try:
            subprocess.check_call(["which", "qsub"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            # Allow testing of qsub output
            if not _QSUBTEST:
                parser.error(
                    'Could not find qsub - cannot run in cluster mode')
        if options['qsub_mpi']:
            num_procs = options['qsub_nodes']*options['qsub_ppn']
        else:
            num_procs = options['qsub_ppn']
            if options['qsub_nodes'] > 1 and not options['_mpi_call']:
                parser.error(
                    'Warning MPI required for multinode running use -M/--mpi flag on command line')
        if options['n'] and options['n'] != num_procs:
            if not options['_mpi_call']:
                log(
                    'Number of workers does not match number of available processors, changing number of workers to: '+str(num_procs))
            options['n'] = num_procs
        else:
            if not options['_mpi_call']:
                log(
                    'Number of workers not set, changing to match number of available processors: '+str(num_procs))
            options['n'] = num_procs
        if options['qsub_mpi']:
            options['qsub_np'] = options['n']
        if (options['qsub_pmem'] and options['qsub_pmem'] > 0 and options['phy_mem'] < 0.9*options['n']*options['qsub_pmem']) or (options['qsub_pmem'] and options['qsub_pmem'] > 0 and options['phy_mem'] > options['n']*options['qsub_pmem']):
            phy_mem = 0.9*float(options['n'])*options['qsub_pmem']
            if not options['_mpi_call']:
                log('Memory limit set and qsub memory limit set. Not matching so changing sampling memory limit to: {}'.format(phy_mem))
            options['phy_mem'] = phy_mem
        elif not options['qsub_pmem'] or options['qsub_pmem'] <= 0:
            options['phy_mem'] = 8  # set to use 8GB for sampling...
            options['qsub_pmem'] = 0
        if options['qsub_walltime']:
            if len(options['qsub_walltime'].split(':')) != 3:
                parser.error('Walltime {} format incorrect, needs to be HH:MM:SS'.format(options['qsub_walltime']))
            walltime = 60.*60.*int(options['qsub_walltime'].split(':')[0])+60.*int(
                options['qsub_walltime'].split(':')[1])+int(options['qsub_walltime'].split(':')[2])
            if 'max_time' in options and walltime < options['max_time']:
                parser.error(
                    'Walltime '+options['qsub_walltime']+' shorter than maximum time: '+options['max_time']+'s')
            if 'max_time' in options and 0.7*walltime < options['max_time']:
                newHrs = int((10/7.)*options['max_time']/3600)
                newMin = int((((10/7.)*options['max_time'])-3600*newHrs)/60)
                newSec = int(
                    (((10/7.)*options['max_time'])-3600*newHrs-60*newMin))
                newWalltime = ':'.join([str(newHrs), str(newMin), str(newSec)])
                if not options['_mpi_call']:
                    log(
                        'Warning lengthening walltime as close to max time. Changing to '+newWalltime)
                options['qsub_walltime'] = newWalltime

    if options['file_sample'] and not options['qsub']:
        try:
            from hdf5storage import savemat, loadmat  # noqa F401
        except Exception:
            parser.error('file_sample option requires hdf5storage and h5py modules')
    if options['file_sample'] and 'mcmc' in options['algorithm']:
        parser.error('file_sample option not supported for mcmc algorithms.')
    if options['debug'] and options['qsub']:
        parser.error(
            'Cannot run debug mode as batch job. Run in interactive mode only')
    if options['debug'] and options['parallel']:
        if not options['_mpi_call']:
            log(
                'Cannot run debug in parallel (issues with terminal), setting parallel to false')
        options['parallel'] = False
    if 'transd' in options['algorithm'] and (options['dc_prior'] > 1 or options['dc_prior'] < 0):
        parser.error('dc_prior must be between 0 and 1')
    options['normalise'] = not options.pop('no_normalise')
    if options['results_format'] == 'hyp' or options['output_format'] == 'hyp':
        log('Output format set to NonLinLoc hyp file format')
        options['results_format'] = 'hyp'
        options['output_format'] = 'hyp'
    return options, options_map

# MTplot


def _MTplot_argparser(input_args=[], test=False):
    """
    Returns arguments parsed from command line

    Creates a command line argument parser (using argparse if present otherwise optparse (python <=2.6))
    and parses the command line arguments (or input_args if provided as a list). Contains some help formatter classes.

    Args
        input_args:[False] If provided, these are parsed instead of the command line arguments

    Returns
        parser,options: Parser object, the parsed options and the mapping from the parser options to the command line flags

    Raises
        Exception: If there is an error parsing the arguments

    """

    description = """MTPlot - Moment Tensor Plotting Code by David J Pugh

    MTPlot is the moment tensor plotting code linked to the MTfit moment tensor inversion code.


    """
    # optparse parser description extra
    optparse_description = """Arguments are set as below, syntax is -dTest.i or --datafile=Test.i
    """
    # argparse parser description extra
    argparse_description = """Arguments are set as below, syntax is -dTest.i or -d Test.i
    """
    # Get extension defaults
    cmd_defaults = {}
    cmd_default_types = {}
    results = evaluate_extensions('MTfit.cmd_defaults', default_cmd_defaults)
    for result in results:
        cmd_defaults.update(result[0])
        cmd_default_types.update(result[1])
    # Try loading MTfitdefaults from env var or ~/.MTfitdefaults
    defaults = get_MTplot_defaults(test, cmd_defaults, cmd_default_types)
    mtplot_data_file_help = """MTplot can read the output data from MTfit"""
    arguments = [
        dict(flags=["-d", "--datafile", "--data_file"],
             help=mtplot_data_file_help, type=str, dest='DATAFILE', default=False),
        dict(flags=["-p", "--plot_type", "--plottype", "--plot-type", "--type"],
             help="Type of plot to make", type=str, dest='plot_type', default=defaults['plot_type']),
        dict(flags=["-c", "--colormap", "--color_map", "--color-map"],
             help="Matplotlib colormap selection", type=str, dest='colormap', default=defaults['colormap']),
        dict(flags=["-f", "--font_size", "--fontsize", "--font-size"],
             help="Fontsize", type=float, dest='fontsize', default=defaults['fontsize']),
        dict(flags=["-l", "--line_width", "--linewidth", "--line-width"],
             help="Linewidth", type=float, dest='linewidth', default=defaults['linewidth']),
        dict(flags=["-t", "--text", "--show-text", "--show_text", "--showtext"], action="store_true",
             help="Flag to show text or not", dest='text', default=defaults['text']),
        dict(flags=["-r", "--resolution"], help="Resolution for the focal sphere plot types",
             type=int, dest='resolution', default=defaults['resolution']),
        dict(flags=["-b", "--bins"], help="Number of bins for the histogram plot types",
             type=int, dest='bins', default=defaults['bins']),
        dict(flags=["--fault_plane", "--faultplane", "--fault-plane"], help="Show the fault planes on a focal sphere type plot",
             action="store_true", dest='fault_plane', default=defaults['fault_plane']),
        dict(flags=["-n", "--nodal_line", "--nodal-line", "--nodalline"], help="Show the nodal lines on a focal sphere type plot",
             action="store_true", dest='nodal_line', default=defaults['nodal_line']),
        dict(flags=["--tnp", "--tp", "--pt"], help="Show TNP axes on focal sphere plots",
             action="store_true", dest='TNP', default=defaults['TNP']),
        dict(flags=["--marker_size", "--markersize", "--marker-size"], help="Set marker size",
             type=float, dest='markersize', default=defaults['markersize']),
        dict(flags=["--station_marker_size", "--stationmarkersize", "--station-marker-size", "--station_markersize", "--station-markersize"],
             help="Set station marker size", type=float, dest='station_markersize', default=defaults['station_markersize']),
        dict(flags=["--showmaxlikelihood", "--show_max_likelihood", "--show-max-likelihood"], help="Show the maximum likelihood solution on a fault plane plot (shown in color set by --color).",
             action="store_true", dest='show_max_likelihood', default=defaults['show_max_likelihood']),
        dict(flags=["--showmean", "--show-mean", "--show_mean"], help="Show the mean orientaion on a fault plane plot (shown in green).",
             action="store_true", dest='show_mean', default=defaults['show_mean']),
        dict(flags=["--grid_lines", "--gridlines", "--grid-lines"], help="Show interior lines on Hudson and lune plots",
             action="store_true", dest='grid_lines', default=defaults['grid_lines']),
        dict(flags=["--color"], help="Set default color",
             type=str, dest='color', default=defaults['color']),
        dict(flags=["--type_label", "--typelabel", "--type-label"], help="Show source type labels on Hudson and lune plots.",
             action="store_true", dest='type_label', default=defaults['type_label']),
        dict(flags=["--hex_bin", "--hexbin", "--hex-bin"], help="Use hex bin for histogram plottings",
             action="store_true", dest='hex_bin', default=defaults['hex_bin']),
        dict(flags=["--projection"], help="Projection choice for focal sphere plots",
             type=str, dest='projection', default=defaults['projection']),
        dict(flags=["--save", "--save_file", "--savefile", "--save-file"],
             help="Set the filename to save to (if set the plot is saved to the file)", type=str, dest='save_file', default=defaults['save_file']),
        dict(flags=["--save-dpi", "--savedpi", "--save_dpi"], help="Output file dpi",
             type=int, dest='save_dpi', default=defaults['save_dpi']),
        dict(flags=['-q', '--quiet', '--hide'], help="Don't show the plot on screen", action="store_true",
             dest='hide', default=defaults['hide']),
        dict(flags=["--version"], action="version",
             version="%(prog)s from MTfit "+__version__),
    ]
    if _ARGPARSE:
        parser = argparse.ArgumentParser(
            prog='MTplot', description=description+argparse_description, formatter_class=ArgparseIndentedHelpFormatterWithNewLines)
        parser.add_argument(
            'data_file', type=str, help="Data file to use for plotting, optional but must be specified either as a positional argument or as an optional argument (see -d below)", nargs="?")

        for arg in arguments:
            kwargs = {
                key: value for (key, value) in arg.items() if key != 'flags'}
            parser.add_argument(*arg['flags'], **kwargs)
        # For testing
        if input_args:
            options = parser.parse_args(input_args)
        else:
            options = parser.parse_args()
        options = vars(options)
        if not options['data_file'] and not options['DATAFILE']:
            parser.error("No data file specified.")
        elif options['data_file'] and options['DATAFILE']:
            parser.error("Multiple data files specified.")
        elif options['DATAFILE']:
            options['data_file'] = options['DATAFILE']
        options.pop('DATAFILE')
    else:
        parser = optparse.OptionParser(prog='MTplot', description=description+optparse_description, formatter=OptparseIndentedHelpFormatterWithNewLines(
        ), version="%(prog)s from MTfit "+__version__, usage="%prog [options]\nUse -h to get more information")
        for arg in arguments:
            kwargs = {
                key: value for (key, value) in arg.items() if key != 'flags'}
            parser.add_option(*arg['flags'], **kwargs)
        if input_args and len(input_args):
            (options, args) = parser.parse_args(input_args)
        else:
            (options, args) = parser.parse_args()
        options = vars(options)
        options['data_file'] = False
        if len(args):
            options['data_file'] = args[0]
        if not options['data_file'] and not options['DATAFILE']:
            parser.error("No data file specified.")
        elif options['data_file'] and options['DATAFILE']:
            parser.error("Multiple data files specified.")
        elif options['DATAFILE']:
            options['data_file'] = options['DATAFILE']
        options.pop('DATAFILE')
    if options.pop('hide'):
        options['show'] = False
    else:
        options['show'] = True
    return parser, options


def MTplot_parser(input_args=False, test=False):
    """
    Parses the command line arguments using _argparser and handles and returns the options

    Args
        input_args:[False] If provided, these are parsed instead of the command line arguments

    Returns
        options,options_map: The parsed options and the mapping from the parser options to the command line flags

    Raises
        Exception: If there is an error parsing the arguments

    """
    parser, options = _MTplot_argparser(input_args, test)
    if options['plot_type'].lower() == 'hudson':
        if options['projection'].lower().replace(' ', '').replace('-', '').replace('_', '') == 'equalarea':
            options['projection'] = 'uv'
        if options['projection'].lower().replace(' ', '').replace('-', '').replace('_', '') not in ['uv', 'tk', 'tauk']:
            parser.error(
                options['projection']+' not recognised for Hudson type plot')
    if not options['colormap']:
        if options['plot_type'].lower() in ['hudson', 'lune']:
            options['colormap'] = DEFAULT_HIST_COLORMAP
        else:
            options['colormap'] = DEFAULT_AMP_COLORMAP

    return options
