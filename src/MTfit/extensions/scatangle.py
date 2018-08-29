"""
scatangle.py
************
Extension for handling scatangle files (installed with the main module, but provides an example of an extension)
"""


# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.


import os
import glob
import time
import logging
import operator

import numpy as np

from ..utilities.multiprocessing_helper import JobPool

logger = logging.getLogger('MTfit.extensions.scatangle')

try:
    from . import cscatangle
except ImportError:
    cscatangle = None
except Exception:
    logger.exception('Error importing c extension')
    cscatangle = None

try:
    import argparse  # noqa F401
    _argparse = True  # noqa F811
except Exception:
    _argparse = False


def parse_scatangle(filename, number_location_samples=0, bin_size=0, _use_c=True):
    """
    Read station angles scatter file

    Reads the station angle scatter file. Expected format is given below. TakeOffAngle is 0 down (NED coordinate system).
    The probabilities are read in, so if Oct-tree or metropolis sampling is used, the probability values should be set to one.

    Args
        filename: Angle scatter file name
        location_samples:[Default=0] Number of samples to take from the full location PDF (0 means all samples)
        bin_size:[Default=1] Bin size for sample weighting in degrees (if 0 no binning used) - reduces the number of samples required for well constrained locations.

    Returns
        Records,Probability: Angle Records and the probability for each sample.

    Expected format is:
        Probability
        StationName Azimuth TakeOffAngle
        StationName Azimuth TakeOffAngle

        Probability
        .
        .
        .

    e.g.
        504.7
        S0271   231.1   154.7
        S0649   42.9    109.7
        S0484   21.2    145.4
        S0263   256.4   122.7
        S0142   197.4   137.6
        S0244   229.7   148.1
        S0415   75.6    122.8
        S0065   187.5   126.1
        S0362   85.3    128.2
        S0450   307.5   137.7
        S0534   355.8   138.2
        S0641   14.7    120.2
        S0155   123.5   117
        S0162   231.8   127.5
        S0650   45.9    108.2
        S0195   193.8   147.3
        S0517   53.7    124.2
        S0004   218.4   109.8
        S0588   12.9    128.6
        S0377   325.5   165.3
        S0618   29.4    120.5
        S0347   278.9   149.5
        S0529   326.1   131.7
        S0083   223.7   118.2
        S0595   42.6    117.8
        S0236   253.6   118.6

        502.7
        S0271   233.1   152.7
        S0649   45.9    101.7
        S0484   25.2    141.4
        S0263   258.4   120.7

    """
    # Read file
    with open(filename, 'r') as f:
        station_file = f.readlines()
    multipliers = []
    sample_records = []
    record = {'Name': [], 'Azimuth': [], 'TakeOffAngle': []}
    multiplier = 1.0
    # Loop over lines
    for line in station_file:
        if line.lstrip('\r') == '\n':
            if len(record['Name']) and multiplier:
                record['Azimuth'] = np.matrix(record['Azimuth']).T
                record['TakeOffAngle'] = np.matrix(record['TakeOffAngle']).T
                sample_records.append(record)
                # Using multipliers therefore prob = 1
                multipliers.append(multiplier)
            record = {'Name': [], 'Azimuth': [], 'TakeOffAngle': []}
        elif len(line.rstrip().rstrip('\r').split()) == 1:
            try:
                multiplier = float(line.rstrip().rstrip('\r'))
            except Exception:
                multiplier = 1.0
        else:
            record['Name'].append(line.split()[0])
            record['Azimuth'].append(float(line.split()[1]))
            record['TakeOffAngle'].append(float(line.rstrip().rstrip('\r').split()[2]))
    if len(record['Name']):
        record['Azimuth'] = np.matrix(record['Azimuth']).T
        record['TakeOffAngle'] = np.matrix(record['TakeOffAngle']).T
        sample_records.append(record)
        multipliers.append(multiplier)
    if number_location_samples and number_location_samples < len(sample_records):
        try:
            samples = np.random.choice(len(sample_records), number_location_samples, False)
        except AttributeError:
            i = 0
            samples = np.array(list(set(np.random.randint(0, len(sample_records), len(sample_records)).tolist())))
            while len(samples) < number_location_samples and i < 100:
                samples = np.array(list(set(np.random.randint(0, len(sample_records), len(sample_records)).tolist())))
                i += 1
            if len(samples) < number_location_samples:
                raise ValueError("Couldn't sample angle PDF")
            else:
                samples = samples[:len(sample_records)]
        # Sample randomly from records, probability
        sample_records = list(np.array(sample_records)[samples])
        multipliers = list(np.array(multipliers)[samples])
    new_multipliers = []
    if bin_size:
        old_size = len(sample_records)
        if cscatangle and _use_c:
            logger.info('C code used')
            t0 = time.time()
            sample_records, multipliers = cscatangle.bin_scatangle(sample_records, np.array(multipliers), bin_size)
            logger.info('Elapsed time = {}'.format(time.time()-t0))
        else:
            logger.info('Python code used')
            t0 = time.time()
            for i, record in enumerate(sample_records):
                if not (i+1) % 10:
                    logger.info('{} records completed'.format(i+1))
                j = i+1
                multiplier = multipliers[i]
                while j < len(sample_records):
                    toa_diff = sample_records[j]['TakeOffAngle']-record['TakeOffAngle']
                    az_diff = sample_records[j]['Azimuth']-record['Azimuth']
                    if np.max(np.abs(toa_diff)) < bin_size/2.0 and np.max(np.abs(az_diff)) < bin_size/2.0:
                        multiplier += multipliers.pop(j)
                        sample_records.pop(j)
                    else:
                        j += 1
                new_multipliers.append(multiplier)
            multipliers = new_multipliers
            logger.info('Elapsed time = {}'.format(time.time()-t0))
        logger.info('{} degree binning reduced {}  samples to {} samples.'.format(bin_size, old_size, len(sample_records)))
    return sample_records, multipliers


def _output_scatangle(filename, samples, probabilities):
    """
    Output scatangle file from samples and probabilities.

    Args
        filename: str name of file to output to.
        samples: list of location samples.
        probabilities: list of sample probabilities.
    """
    output = []
    # Loop over samples
    for i, sample in enumerate(samples):
        # Append multiplier (probability)
        output.append(str(probabilities[i]))
        # Loop over stations
        for j, st in enumerate(sample['Name']):
            output.append(st+'\t'+str(float(sample['Azimuth'][j]))+'\t'+str(float(sample['TakeOffAngle'][j])))
        output.append('')
    with open(filename, 'w') as f:
        f.write('\n'.join(output))


def bin_scatangle(filename, number_location_samples=0, bin_size=1):
    """
    Bin scatangle samples into bins of size given by the bin size argument, if all the differences in angles for each station between the two samples are within that range

    Args
        filename: str of filename to read.
        number_location_samples:[0} integer number of location samples to sub-sample (0 means to use all).
        bin_size:[1.0] float size of bin to stack samples over.
    """
    # Parse scatangle and bin
    sample_records, multipliers = parse_scatangle(filename, number_location_samples, bin_size)
    old_filename = filename
    # Add _bin_ to filename
    filename = ('_bin_'+str(bin_size)).join(os.path.splitext(filename))
    # Output to disk
    _output_scatangle(filename, sample_records, multipliers)
    return old_filename, filename


class BinScatangleTask(object):
    """
    Scatangle Binning task

    Bins and Saves scatangle file

    Initialisation
        Args
            fid: Filename for MATLAB output.
            number_location_samples: number_location_samples.
            bin_size: bin size.
    """

    def __init__(self, fid, number_location_samples=0, bin_size=1.0):
        """
        Initialisation of MatlabOutputTask

        Args
            fid: Filename for MATLAB output.
            output: Dictionary of output to be saved to fid.

        """
        self.fid = fid
        self.number_location_samples = number_location_samples
        self.bin_size = bin_size

    def __call__(self):
        """
        Runs the MATLAB output task

        Runs the MatlabOutputTask and returns a result code.

        Returns
            resultCode: 10 if successful, 20 if an exception is thrown.

        """
        try:
            return bin_scatangle(self.fid, self.number_location_samples, self.bin_size)
        except Exception:
            logger.exception('Scatangle Bin Error')
            return 20


def bin_scatangle_files(files, number_location_samples=0, bin_scatangle_size=1.0, parallel=True, mpi=False, **kwargs):
    """
    Bin scatangle samples into bins of size given by the bin size argument,
    if all the differences in angles for each station between the two samples are within that range

    Args
        filename: str of filename to read.
        number_location_samples:[0} integer number of location samples to sub-sample (0 means to use all).
        bin_scatangle_size:[1.0] float size of bin to stack samples over.
        parallel:[True] boolean to run in parallel using job pool (overridden by mpi option).
        mpi: [False] boolean to run using MPI (ignores parallel flag).

    Returns
        new_files:list of new file names
    """
    # check path and glob
    if not isinstance(files, list) and os.path.isdir(files):
        files = glob.glob('*.'+kwargs['angle_extension'])
    elif not isinstance(files, list) and '*' in files:
        files = glob.glob(files)
    # Make sure the files are a list
    if not isinstance(files, list):
        files = [files]
    # Set parallel flag
    parallel = parallel and not len(files) == 1
    # Get MPI parameters if running using MPI
    if mpi:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        except Exception:
            mpi = False
    # Otherwise create jobpool if running in parallel
    elif parallel:
        job_pool = JobPool(task=BinScatangleTask)
        nworkers = len(job_pool)
        if nworkers == 1:
            job_pool.close()
            parallel = False
    new_files = files[:]
    if mpi:
        # Sort files into size lists
        data = [[] for i in range(comm.Get_size())]
        j = 0
        for fn in files:
            data[j].append(fn)
            j = (j+1) % comm.Get_size()
        # Loop over files for each process
        for fn in data[comm.Get_rank()]:
            # If the filename is not blank, then bin it.
            if len(fn):
                new_files[new_files.index(fn)] = bin_scatangle(fn, number_location_samples, bin_scatangle_size)[1]
        end = True
        # Wait for end
        items = comm.gather(end, 0)
        if comm.Get_rank() == 0:
            for i in items:
                assert i is True
    elif parallel:
        # Loop over pool running tasks
        for fn in files:
            job_pool.task(fn, number_location_samples, bin_scatangle_size)
        # Get results
        results = job_pool.all_results()
        for new_fn in results:
            new_files[new_files.index(new_fn[0])] = new_fn[1]
        job_pool.close()
    else:
        # Run bin for each file
        for fn in files:
            new_files[new_files.index(fn)] = bin_scatangle(fn, number_location_samples, bin_scatangle_size)[1]
    return new_files


def convert_scatangle_to_MATLAB(scatangle_file, fid=False, data_file=False):
    """
    Converts scatangle file to MATLAB station distribution format

    Converts the scatangle file to the station distribution format separately from the inversion.

    Args
        scatangle_file: Scatangle filename.
        fid:[False] Output filename.
        data_file:[False] Data file path (means that only stations with observations are outputted).
    """
    from ..inversion import Inversion
    # Parse scatangle file
    X = parse_scatangle(scatangle_file)
    if not fid:
        fid = scatangle_file
    if data_file:
        # Check Stations for only those with observations
        data = Inversion({'UID': 1}, parallel=False)._load(data_file)
        original_samples = [u.copy() for u in X[0]]
        stations = []
        for key in [u for u in data.keys() if 'amplituderatio' in u.lower() or 'amplitude_ratio' in u.lower() or 'polarity' in u.lower()]:
            stations.extend(data[key]['Stations']['Name'])
        stations_set = set(stations)
        location_samples = [u.copy() for u in original_samples]
        selected_stations = list(set(location_samples[0]['Name']) & stations_set)
        indices = [location_samples[0]['Name'].index(u) for u in selected_stations]
        for i, sample in enumerate(location_samples):
            sample['Name'] = operator.itemgetter(*indices)(sample['Name'])
            sample['TakeOffAngle'] = sample['TakeOffAngle'][indices]
            sample['Azimuth'] = sample['Azimuth'][indices]
        # Output results
        Inversion(data_file=data_file, parallel=False).__output__({}, fid=fid, location_samples=location_samples,
                                                                  location_sample_multipliers=X[1], station_only=True)
    else:
        # Output results
        Inversion({'UID': 1}, parallel=False).__output__({}, fid=fid, location_samples=X[0],
                                                         location_sample_multipliers=X[1], station_only=True)


def parser_check(parser, options, defaults):
    flags = []
    if options['bin_scatangle']:
        if not options['location_pdf_file_path']:
            options['location_pdf_file_path'] = glob.glob(options['data_file']+os.path.sep+'*'+options['angle_extension'])
        if not isinstance(options['location_pdf_file_path'], list):
            options['location_pdf_file_path'] = [options['location_pdf_file_path']]
        flags = ['no_location_update']
    return options, flags


# MTfit pkg_resources EntryPoint functions


PARSER_DEFAULTS = {
          'bin_scatangle': False,
          'bin_size': 1.0,
          }
PARSER_DEFAULT_TYPES = {'bin_scatangle': [bool], 'bin_size': [float]}


def cmd_defaults():
    return(PARSER_DEFAULTS, PARSER_DEFAULT_TYPES)


def cmd_opts(group, argparser=_argparse, defaults=PARSER_DEFAULTS):
    """
    Adds parser group for scatangle arguments

    Returns
        group: argparse or optparse argument group

    """
    if argparser:
        group.add_argument("--bin-scatangle", "--binscatangle", "--bin_scatangle",  action="store_true",
                           default=defaults['bin_scatangle'], help="Bin the scatangle file to reduce the number of samples [default=False]. --bin-size Sets the bin size parameter .",
                           dest="bin_scatangle")
        group.add_argument("--bin-size", "--binsize", "--bin_size", type=float, default=defaults['bin_size'],
                           help="Sets the scatangle bin size parameter [default={}]".format(defaults['bin_size']), dest="bin_scatangle_size")
    else:
        group.add_option("--bin-scatangle", "--binscatangle", "--bin_scatangle",  action="store_true",
                         default=defaults['bin_scatangle'], help="Bin the scatangle file to reduce the number of samples [default=False]. --bin-size Sets the bin size parameter .",
                         dest="bin_scatangle")
        group.add_option("--bin-size", "--binsize", "--bin_size", type=float, default=defaults['bin_size'],
                         help="Sets the scatangle bin size parameter [default={}]".format(defaults['bin_size']), dest="bin_scatangle_size")
    return group, parser_check


def pre_inversion(**kwargs):
    if kwargs.get('bin_scatangle', False):
        try:
            kwargs['location_pdf_file_path'] = bin_scatangle_files(kwargs.get('location_pdf_file_path'), **kwargs)
            kwargs.pop('number_location_samples')
        except Exception:
            pass
    return kwargs


def location_pdf_parser(*args, **kwargs):
    return parse_scatangle(*args, **kwargs)
