"""
file_io
*******

Handles file input/output e.g. for the inversion results
"""


# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.


import os
import sys
import traceback
import struct
from datetime import datetime
import logging

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np

from ..convert.moment_tensor_conversion import MT6_Tape
from .. import __version__
from ..extensions.scatangle import parse_scatangle


logger = logging.getLogger('MTfit.utilities.file_io')

nlloc_polarity_inv_dict = {'P': {1: 'U', 0: '?', -1: 'D'}, 'S': {1: 'P', 0: '?', -1: 'N'}}
nlloc_polarity_dict = {'u': 1, '?': 0, 'd': -1, '+': 1, 'c': 1, '-': -1, '.': 0, 'p': 1, 'n': -1}

#
# Output Task Classes
#


class MATLABOutputTask(object):

    """
    MATLAB format output task

    Saves MATLAB output file to disk using hdf5storage or scipy savemat.

    Initialisation
        Args
            fid: Filename for MATLAB output.
            output: Dictionary of output to be saved to fid.
    """

    def __init__(self, fid, output_dict, version='7.3'):
        """
        Initialisation of MATLABOutputTask

        Args
            fid: Filename for MATLAB output.
            output: Dictionary of output to be saved to fid.
            version: Set MATLAB version (7.3 or 7) for hdf5storage or scipy storage, respectively

        """
        self.fid = fid
        self.output_dict = output_dict
        self.version = version

    def __call__(self):
        """
        Runs the MATLABOutputTask and returns a result code.

        Returns
            resultCode: 10 if successful, 20 if an exception is thrown.

        """
        # Check version and get savemat function
        if self.version == '7.3':
            try:
                from hdf5storage import savemat
            except Exception:
                self.version = '7'
        if self.version != '7.3':
            from scipy.io import savemat  # noqa F811
            self.version = '7'
        logging.info('Saving to {} in MATLAB version {}'.format(self.fid, self.version))
        # Try to save file
        try:
            # Convert to/from unicode for different functions (hdf5 needs
            # unicode) scipy needs non-unicode
            if self.version == '7.3':
                self.output_dict = convert_keys_to_unicode(self.output_dict)
            else:
                self.output_dict = convert_keys_from_unicode(self.output_dict)
            # Save data
            savemat(self.fid, self.output_dict)
            logger.info('Saved to '+self.fid)
            return 10
        except Exception:
            logger.exception('MATLAB Output Error')
            return 20


class PickleOutputTask(object):

    """
    Pickle format output task

    Saves output file to disk using pickle.

    Initialisation
        Args
            fid: Filename for output.
            output: Dictionary of output to be saved to fid.
    """

    def __init__(self, fid, output_dict,):
        """
        Initialisation of PickleOutputTask

        Args
            fid: Filename for MATLAB output.
            output: Dictionary of output to be saved to fid.

        """
        self.fid = fid
        self.output_dict = output_dict

    def __call__(self):
        """
        Runs the PickleOutputTask and returns a result code.

        Returns
            resultCode: 10 if successful, 20 if an exception is thrown.

        """
        logger.info('Saving to '+self.fid)
        try:
            with open(self.fid, 'wb') as f:
                pickle.dump(self.output_dict, f)
            logger.info('Saved to '+self.fid)
            return 10
        except Exception:
            logger.exception('Pickle Output Error')
            return 20


class HypOutputTask(object):

    """
    NLLOC hyp format output task

    Saves output file to disk.

    Initialisation
        Args
            fid: Filename for output.
            output: Data to be saved to fid.
            mt: Saving mt format.
    """

    def __init__(self, fid, output, binary=False):
        """
        Initialisation of HypOutputTask

        Args
            fid: Filename for MATLAB output.
            output: Data to be saved to fid.
            binary:[False] binary output file.

        """
        self.fid = fid
        self.output = output
        self.binary = binary

    def __call__(self):
        """
        Runs the PickleOutputTask and returns a result code.

        Returns
            resultCode: 10 if successful, 20 if an exception is thrown.

        """
        try:
            # Write binary as binary file,
            if self.binary:
                with open(self.fid, 'wb') as f:
                    f.write(self.output)
            # Write output as normal text
            else:
                with open(self.fid, 'w') as f:
                    f.write(self.output)
            return 10
        except Exception:
            logger.exception('Hyp Output Error')
            return 20

#
# Input file parsing
#


def parse_hyp(filename):
    """
    Parse NonLinLoc hyp file

    Reads NonLinLoc hyp file for input data.

    Args:
        filename:str hyp filename.
    """
    with open(filename) as f:
        lines = f.readlines()
    events_list = []
    event = []
    # Loop over file and split into events
    for line in lines:
        line = line.rstrip()
        line = line.split()
        if len(line):
            event.append(line)
            if line[0] == 'END_NLLOC':
                if len(event):
                    events_list.append(event)
                    event = []
    if len(event):
        events_list.append(event)
    # Parse events
    return _parse_hyp_events(events_list)


def _parse_hyp_events(events_list, polarity_error_multiplier=3.):
    """
    Parse events from NonLinLoc hyp file

    Parse the events read from a NonLinLoc hyp file.

    Args
        events_list: list of event lines read from NonLinLoc hyp file.
        polarity_error_multiplier:[3.] Multiplier to convert time error to polarity error estimates.

    """

    event_dict_list = []
    # Loop over events
    for event in events_list:
        event_dict = {}
        phase = False
        # For each line
        for line in event:
            # Generic key dictionary
            key_dict = {'Stations': {
                'Name': [], 'TakeOffAngle': [], 'Azimuth': []}, 'Measured': [], 'Error': []}
            # Get UID from GEOGRAPHIC line
            if line[0] == 'GEOGRAPHIC':
                sec = line[7]
                try:
                    event_dict['UID'] = line[2]+line[3]+line[4]+line[5]+line[6]+sec.split('.')[0]+sec.split('.')[1][:-1]
                except Exception:
                    pass
            # Phase flag
            if line[0] == 'PHASE':
                phase = True
                continue
            # End of phase flag
            if line[0] == 'END_PHASE':
                phase = False
            # If in phase section, read data
            if phase and len(line) >= 24:
                # Get station
                station = line[0]
                # Get phase
                phase_type = line[4]
                # Get polarity
                polarity = nlloc_polarity_dict[line[5].lower()]
                # Get uncertainty
                uncertainty = float(line[10])
                # Get amplitude (unused)
                amplitude = float(line[12])  # noqa F841
                if polarity != 0:
                    if not event_dict.__contains__(phase_type+'Polarity'):
                        event_dict[phase_type+'Polarity'] = key_dict.copy()
                    event_dict[
                        phase_type+'Polarity']['Stations']['Name'].append(station)
                    event_dict[
                        phase_type+'Polarity']['Measured'].append(polarity)
                    event_dict[
                        phase_type+'Polarity']['Error'].append(uncertainty*polarity_error_multiplier)
                    event_dict[
                        phase_type+'Polarity']['Stations']['TakeOffAngle'].append([float(line[24])])
                    event_dict[
                        phase_type+'Polarity']['Stations']['Azimuth'].append([float(line[23])])
        # End of event, make data into appropriate structures
        for key in event_dict.keys():
            if key != 'UID':
                event_dict[key]['Stations']['TakeOffAngle'] = np.matrix(
                    event_dict[key]['Stations']['TakeOffAngle'])
                event_dict[key]['Stations']['Azimuth'] = np.matrix(
                    event_dict[key]['Stations']['Azimuth'])
                event_dict[key]['Measured'] = np.matrix(
                    event_dict[key]['Measured']).T
                event_dict[key]['Error'] = np.matrix(
                    event_dict[key]['Error']).T
        # Sort keys and append event dict to list
        event_dict['hyp_file'] = event
        if sorted(event_dict.keys()) != ['hyp_file', 'UID']:
            event_dict_list.append(event_dict.copy())
            event_dict = {}
    return event_dict_list


def parse_csv(filename):
    """
    Parses CSV file to data dictionary

    CSV file format is to have events split by blank lines, a header line showing where the information is, UID and data-type information stored in the first column, e.g.

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

    Args:
        filename:str csv filename

    Returns:
        event_dict_list:list of event data dictionaries
    """
    # Read file
    with open(filename) as f:
        lines = f.readlines()
    events_list = []
    event = []
    # Loop over lines
    for l in lines:
        if not len(l.split(',')[0]) and not len(l.split(',')[1]):
            # End OF event
            if len(event):
                events_list.append(event)
            event = []
        else:
            event.append(l.rstrip())
    # If there is event data, append to the event list
    if len(event):
        events_list.append(event)
    # Return parsed events
    return _parse_csv_events(events_list)


def _parse_csv_events(events_list):
    """
    Parses CSV event list to data dictionary

    CSV file format is to have events split by blank lines, a header line showing where the information is, UID and data-type information stored in the first column, e.g.

        UID=123,,,,
        PPolarity,,,,
        Name,Azimuth,TakeOffAngle,Measured,Error
        S001,120,70,1,0.01
        S002,160,60,-1,0.02
        P/SHRMSAmplitudeRatio,,,,
        Name,Azimuth,TakeOffAngle,Measured,Error
        S003,110,10,1 2,0.05 0.04
        ,,,,
        PPolarity ,,,,
        Name,Azimuth,TakeOffAngle,Measured,Error
        S003,110,10,1,0.05

    Is a CSV file with 2 events, one event UID of 123, and PPolarity data at S001 and S002 and P/SHRMSAmplitude data at S003,
    and a second event with no UID (will default to the event number, in this case 2) with PPolarity data at S003.

    Args:
        filename:str csv filename

    Returns:
        event_dict_list:list of event data dictionaries
    """
    event_dict_list = []
    # Default keys and values
    key = 'PPolarity'
    name_index = 0
    measured_index = 3
    azimuth_index = 1
    take_off_angle_index = 2
    error_index = 4
    # Loop over events
    for event in events_list:
        event_dict = {'UID': str(events_list.index(event)+1)}
        # Set up generic dictionary
        key_dict = {'Stations': {
            'Name': [], 'TakeOffAngle': [], 'Azimuth': []}, 'Measured': [], 'Error': []}
        # Loop over lines
        for line in event:
            split_line = line.split(',')
            if len(split_line[0]) and not sum([len(u) for u in split_line[1:]]):
                # Try tp get UID
                if 'UID' in split_line[0]:
                    event_dict['UID'] = split_line[0].split(
                        'UID')[1].lstrip(':').lstrip('=').strip()
                else:
                    # End of event
                    if len(key_dict['Stations']['Name']):
                        key_dict['Stations']['TakeOffAngle'] = np.matrix(key_dict['Stations']['TakeOffAngle'])
                        key_dict['Stations']['Azimuth'] = np.matrix(key_dict['Stations']['Azimuth'])
                        key_dict['Measured'] = np.matrix(key_dict['Measured'])
                        key_dict['Error'] = np.matrix(key_dict['Error'])
                        event_dict[key] = key_dict
                    key = split_line[0].strip()
                    key_dict = {'Stations': {
                        'Name': [], 'TakeOffAngle': [], 'Azimuth': []}, 'Measured': [], 'Error': []}
            elif 'Name' in split_line:
                # Try to read header file and get key pieces
                split_line = ','.join(split_line).lower().split(',')
                name_index = split_line.index('name')
                measured_index = split_line.index('measured')
                azimuth_index = split_line.index('azimuth')
                take_off_angle_index = split_line.index('takeoffangle')
                error_index = split_line.index('error')
            else:
                # Append item to list
                key_dict['Stations']['Name'].append(split_line[name_index])
                key_dict['Stations']['TakeOffAngle'].append([float(split_line[take_off_angle_index].strip())])
                key_dict['Stations']['Azimuth'].append([float(split_line[azimuth_index].strip())])
                measured = []
                for _measured in split_line[measured_index].split():
                    measured.append(float(_measured.strip()))
                key_dict['Measured'].append(measured)
                error = []
                for err in split_line[error_index].split():
                    error.append(float(err.strip()))
                key_dict['Error'].append(error)
        # Sort output into correct structs
        if len(key_dict['Stations']['Name']):
            key_dict['Stations']['TakeOffAngle'] = np.matrix(key_dict['Stations']['TakeOffAngle'])
            key_dict['Stations']['Azimuth'] = np.matrix(key_dict['Stations']['Azimuth'])
            key_dict['Measured'] = np.matrix(key_dict['Measured'])
            key_dict['Error'] = np.matrix(key_dict['Error'])
            event_dict[key] = key_dict
        event_dict_list.append(event_dict)
    return event_dict_list

#
# Convert csv to inv
#


def csv2inv(filename):
    """
    Converts CSV file to inv file

    CSV file format is to have events split by blank lines, a header line showing where the information is, UID and data-type information stored in the first column, e.g.::

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

    Args:
        filename:str csv filename.

    """
    event_dict_list = parse_csv(filename)
    output_name = os.path.splitext(filename)[0]+'.inv'
    PickleOutputTask(output_name, event_dict_list)()

#
# Hyp output handling
#


def _convert_mt_space_to_struct(output_data, i=False):
    """
    Coverts mt space to binary struct

    Structure is:

        File version
        Nsamples(unsigned long int)
        Nmtspacesamples(unsigned long int)
        Converted(bool)
        Ln_BayesianEvidence
        DKL
        NSamples as:
            P(double)
            Ln_P(double)
            Mnn(double)
            Mee(double)
            Mdd(double)
            Mne(double)
            Mnd(double)
            Med(double)

            if Converted is true then each sample also contains:
                gamma(double)
                delta(double)
                kappa(double)
                h(double)
                sigma(double)
                u(double)
                v(double)
                strike1(double)
                dip1(double)
                rake1(double)
                strike2(double)
                dip2(double)
                rake2(double)

    Args
        output_data: output data dictionaries
        i:[False] event number for multiple events.

    Returns
        binary MT and scale factor outputs
    """
    file_version = 2
    sqrt2 = np.sqrt(2)
    # Check MT is single event
    if 'moment_tensor_space' in output_data:
        # single
        end = ''
    else:
        end = '_'+str(i)
    # Load converted data and set convert flag (for ending)
    try:
        g = output_data['g'+end]
        d = output_data['d'+end]
        k = output_data['k'+end]
        h = output_data['h'+end]
        s = output_data['s'+end]
        u = output_data['u'+end]
        v = output_data['v'+end]
        s1 = output_data['S1'+end]
        d1 = output_data['D1'+end]
        r1 = output_data['R1'+end]
        s2 = output_data['S2'+end]
        d2 = output_data['D2'+end]
        r2 = output_data['R2'+end]
        converted = True
    except Exception:
        converted = False
    # Add Number of samples and converted flag
    binary_output = struct.pack('QQQ?', file_version, output_data['total_number_samples'],
                                output_data['moment_tensor_space'+end].shape[1], converted)
    # Try to add Bayesian evidence
    try:
        binary_output += struct.pack('1d', output_data['ln_bayesian_evidence'])
    except Exception:
        binary_output += struct.pack('1d', np.nan)  # No Bayesian evidence
    # Try to add Dkl
    try:
        binary_output += struct.pack('1d', output_data['dkl'])
    except Exception:
        binary_output += struct.pack('1d', np.nan)  # No Dkl
    # Loop over MTs
    for i in range(output_data['moment_tensor_space'+end].shape[1]):
        # This seems slow in python 3
        # Add MT data
        if output_data['probability'].ndim == 2:
            p = output_data['probability'][0, i]
        else:
            p = output_data['probability'][i]
        if output_data['ln_pdf'].ndim == 2:
            ln_p = output_data['ln_pdf'][0, i]
        else:
            ln_p = output_data['ln_pdf'][i]
        binary_output += struct.pack('8d', p, ln_p, output_data['moment_tensor_space'+end][0, i], output_data['moment_tensor_space'+end][1, i],
                                     output_data['moment_tensor_space'+end][2, i], output_data['moment_tensor_space'+end][3, i]/sqrt2,
                                     output_data['moment_tensor_space'+end][4, i]/sqrt2,
                                     output_data['moment_tensor_space'+end][5, i]/sqrt2)
        if converted:
            # Add converted data
            binary_output += struct.pack('13d', g[i], d[i], k[i], h[i], s[i], u[i], v[i], s1[i], d1[i], r1[i], s2[i], d2[i], r2[i])
    # Handle scale_factors
    if sys.version_info.major > 2:
        sf_output = b''
    else:
        sf_output = ''
    if 'scale_factors' in output_data:
        n_events = output_data['scale_factors']['mu'].shape[1]
        sf_output = struct.pack('QQQ', output_data['total_number_samples'], len(output_data['scale_factors']), n_events)
        for i, sf in enumerate(output_data['scale_factors']):
            sf_output += struct.pack('dd', output_data['probability'][0, i], output_data['ln_pdf'][0, i])
            # Add data for each event - Loops over off diagonal elements
            k = 0
            _l = 1
            for j in range(int(n_events*(n_events-1)/2.)):
                sf_output += struct.pack('dd', sf['mu'][k, _l], sf['sigma'][k, _l])
                _l += 1
                if _l >= n_events:
                    k += 1
                    _l = k+1
    # Return output
    return binary_output, sf_output


def _generate_hyp_output_data(event_data, inversion_options=False, output_data=False, maxMT=np.matrix([[0], [0], [0], [0], [0], [0]])):
    """
    Generates the hyp output text data for the .hyp file

    Appends or creates the hyp output file (if a hyp file is used as input, the text is stored in the event data)

    Args
        event_data: Event data dictionary.
        inversion_options:[False] Inversion options list.
        output_data:[False] Output data results
        maxMT:[np.matrix([[0],[0],[0],[0],[0],[0]])] Maximum moment tensor solution.

    Returns
        list of event text outputs
    """
    # Try to get the maximum moment tensor
    from ..inversion import _polarity_misfit_check
    maxMT_tape = [0, 0, 0, 0, 0]
    try:
        maxMT = unique_columns(output_data['moment_tensor_space'][:, np.array(
            output_data['probability'] == output_data['probability'].max())[0]])
        if maxMT.shape[1] > 1:
            maxMT = maxMT[:, 0]
        maxMT_tape = MT6_Tape(maxMT)
    except Exception:
        traceback.print_exc()
    if all(maxMT == 0):
        maxMT_tape = [0, 0, 0, 0, 0]
    # Check if its a DC
    dc = False
    if abs(maxMT_tape[0]) < 1.0*10**-5 and abs(maxMT_tape[1]) < 1.0*10**-5:
        dc = True
    # Try to get the input hyp file if it exists
    if 'hyp_file' in event_data.keys():
        lines = event_data['hyp_file'][:]
    else:
        # Generate NLLOC type hyp file
        lines = []
        try:
            lines.append(['NLLOC', event_data['UID']])
        except Exception:
            lines.append(['NLLOC'])
        lines.append(['SIGNATURE', 'MTfit', __version__+'/'+datetime.now().isoformat()])
        lines.append(['COMMENT', 'MTfit inversion'])
        lines.append(['GRID', 'None'])
        lines.append(['SEARCH', 'None'])
        lines.append(['HYPOCENTER', 'x', '?', 'y', '?', 'z', '?',
                      'OT', '?', 'ix', '?', 'iy', '?', 'iz', '?'])
        lines.append(['GEOGRAPHIC', 'OT', '?', '?', '?', '?',
                      '?', '?', 'Lat', '?', 'Long', '?', 'Depth', '?'])
        lines.append(['QUALITY', 'Pmax', '?', 'MFmin', '?', 'MFmax', '?', 'RMS', '?',
                      'Nphs', '?', 'Gap', '?', 'Dist', '?', 'Mamp', '?', '?', 'Mdur', '?', '?'])
        lines.append(['VPVSRATIO', '?', '?'])
        lines.append(['STATISTICS', 'ExpectX', '?', 'Y', '?', 'Z', '?', 'CovXX', '?', 'XY', '?', 'XZ', '?', 'YY', '?',
                      'YZ', '?', 'ZZ', '?', 'EllAz1', '?', 'Dip1', '?', 'Len1', '?', 'Az2', '?', 'Dip2', '?', 'Len2', '?', 'Len3', '?'])
        lines.append(['TRANS', 'None'])
        lines.append(['PHASE', 'ID', 'Ins', 'Cmp', 'On', 'Pha', 'FM', 'Date', 'HrMn', 'Sec', 'Err', 'ErrMag', 'Coda', 'Amp',
                      'Per', '>', 'TTpred', 'Res', 'Weight', 'StaLoc(X', 'Y', 'Z)', 'SDist', 'SAzim', 'RAz', 'RDip', 'RQual', 'Tcorr'])
        # Station Data
        for phase in [key.split('Polarity')[0] for key in event_data.keys() if 'Polarity' in key and 'PolarityProb' not in key]:
            for i, station in enumerate(event_data[phase+'Polarity']['Stations']['Name']):
                polarity = nlloc_polarity_inv_dict[phase[0]][
                    event_data[phase+'Polarity']['Measured'][i, 0]]
                lines.append([station, '?', '?', '?', phase.upper(), polarity, '?', '?', '?', '?', '?', '?', '?', '?', '>', '?', '?', '?', '?', '?', '?', '?', '?',
                              '{:5.1f}'.format(event_data[phase+'Polarity']['Stations']['Azimuth'][i, 0]).lstrip(),
                              '{:5.1f}'.format(event_data[phase+'Polarity']['Stations']['TakeOffAngle'][i, 0]).lstrip(), '?', '?'])
        lines.append(['END_PHASE'])
        lines.append(['END_NLLOC'])
    phase_line_index = lines.index(['PHASE', 'ID', 'Ins', 'Cmp', 'On', 'Pha', 'FM', 'Date', 'HrMn', 'Sec', 'Err', 'ErrMag', 'Coda',
                                    'Amp', 'Per', '>', 'TTpred', 'Res', 'Weight', 'StaLoc(X', 'Y', 'Z)', 'SDist', 'SAzim', 'RAz', 'RDip', 'RQual', 'Tcorr'])
    end_phase_line_index = lines.index(['END_PHASE'])
    geog_line = [line for line in lines if line[0] == 'GEOGRAPHIC'][0]
    # GET PHASE LINE AND THEN PREPEND MT AND UPDATE misfit_checks
    # DC  FOCALMECH dlat dlong depth Mech dipDir dipAng rake mf misfit nObs numberObs
    # MT  MOMENTTENSOR dlat dlong depth MTNN Mnn EE Mee DD Mdd NE Mne ND Mnd
    # ED Med mf misfit nObs numberObs
    try:
        source_line_index = lines.index(
            [line for line in lines if line[0] == 'FOCALMECH'][0])
        lines.pop(source_line_index)
    except Exception:
        pass
    # Make source line
    if dc:
        source_line = ["FOCALMECH", geog_line[9], geog_line[11], geog_line[13], "Mech", str(float(maxMT_tape[2]*180/np.pi)),
                       str(float(np.arccos(maxMT_tape[3])*180/np.pi)), str(float(maxMT_tape[4]*180/np.pi)), "mf", "?", "nObs", "?"]
    else:
        source_line = ["MOMENTTENSOR", geog_line[9], geog_line[11], geog_line[13], "MTNN", str(float(maxMT[0])), "EE", str(float(maxMT[1])),
                       "DD", str(float(maxMT[2])), "NE", str(float(maxMT[3]/np.sqrt(2))), "ND", str(float(maxMT[4]/np.sqrt(2))), "ED",
                       str(float(maxMT[5]/np.sqrt(2))), "mf", "?", "nObs", "?"]
    # Insert before phases start
    lines.insert(phase_line_index, source_line)
    mf = 0
    nobs = 0
    # Run polarity misfit checks
    for i in range(phase_line_index+2, end_phase_line_index+1):
        if len(lines[i]) < 24:
            continue
        if _polarity_misfit_check(nlloc_polarity_dict[lines[i][5].lower()], float(lines[i][23]), float(lines[i][24]), lines[i][4], maxMT):
            mf += 1
            lines[i][5] = nlloc_polarity_inv_dict[lines[i][4].upper()[0]][nlloc_polarity_dict[lines[i][5].lower()]].lower()
        else:
            lines[i][5] = nlloc_polarity_inv_dict[lines[i][4].upper()[0]][nlloc_polarity_dict[lines[i][5].lower()]].upper()

        if lines[i][5] != '?':
            nobs += 1
    # Set number of misfits and number of observations
    lines[phase_line_index][-3] = str(mf)
    lines[phase_line_index][-1] = str(nobs)
    output = []
    for line in lines:
        output.append(' '.join(line))
    return output

#
# Results data format
#


def full_pdf_output_dicts(event_data, inversion_options=False, output_data=False, location_samples=False,
                          location_sample_multipliers=False, multiple_events=False, _diagnostic_output=False,
                          normalise=True, *args, **kwargs):
    """
    Create output dictionaries for full_pdf format

    This creates the output dictionaries for the full_pdf format from the output data.

    Args
        event_data: Event data containing stations and observations (input data for the inversion).
        inversion_options:[False] List of inversion data type options.
        output_data:[False] Output data dictionaries (output from algorithm.__output__()).
        location_samples:[False] List of location PDF samples.
        location_sample_multipliers:[False] List of location PDF sample probabilities.
        multiple_events:[False] Boolean flag for multiple events.
        _diagnostic_output:[False] Output angle coefficents and other data for testing and diagnostics.
        normalise:[True] Normalise the output probabilities.

    Keyword Arguments
        station_only:[False] Boolean flag to only output station data (no inversion data).

    Returns
        [mdict,sdict]: list of results dict and station distribution dict.
    """
    # Check if station only output (in kwargs)
    station_output = kwargs.get('station_only', False)
    # Set up output dicts
    stations = {}
    other = {}
    station_distribution = {}
    # Get number of events:
    n_events = 1
    if multiple_events:
        n_events = len(event_data)
    all_stations = []
    # Loop over event_data
    for i in range(n_events):
        if isinstance(event_data, list):
            ev_data = event_data[i]
        else:
            ev_data = event_data
        # If P Polarity or P Polarity probability data used, then get
        # polarities from that
        if (inversion_options and 'PPolarity' in inversion_options and 'PPolarity' in ev_data.keys()) \
                or ('PPolarity' in ev_data.keys() and len(ev_data['PPolarity']['Stations']['Name'])) \
                or (inversion_options and 'PPolarityProbability' in inversion_options and 'PPolarityProbability' in ev_data.keys()):
            if (inversion_options and 'PPolarityProbability' in inversion_options and 'PPolarityProbability' in ev_data.keys()):
                observations_dict = ev_data['PPolarityProbability']
                if observations_dict['Measured'].shape[1] == 2:
                    positive = np.array(
                        observations_dict['Measured'][:, 0]).flatten()
                    negative = np.array(
                        observations_dict['Measured'][:, 1]).flatten()
                else:
                    positive = np.array(
                        observations_dict['Measured'][0, :].T).flatten()
                    negative = np.array(
                        observations_dict['Measured'][1, :].T).flatten()
                measured = positive
                measured[positive < negative] = -negative[positive < negative]
                measured = np.matrix(measured).T
            else:
                observations_dict = ev_data['PPolarity']
                if len(observations_dict['Measured'].shape) > 1 and observations_dict['Measured'].shape[0] > observations_dict['Measured'].shape[1]:
                    measured = np.matrix(observations_dict['Measured'])
                else:
                    measured = np.matrix(observations_dict['Measured']).T
            number_stations = len(observations_dict['Stations']['Name'])
            # Make stations object
            stations = np.matrix(np.zeros((number_stations, 4), dtype=np.object))
            stations[:, 0] = np.matrix(observations_dict['Stations']['Name']).T
            # Check orientations of angles and observations
            if len(observations_dict['Stations']['Azimuth'].shape) > 1 and observations_dict['Stations']['Azimuth'].shape[0] > observations_dict['Stations']['Azimuth'].shape[1]:
                stations[:, 1] = np.matrix(observations_dict['Stations']['Azimuth'])
            else:
                stations[:, 1] = np.matrix(observations_dict['Stations']['Azimuth']).T
            if len(observations_dict['Stations']['TakeOffAngle'].shape) > 1 and observations_dict['Stations']['TakeOffAngle'].shape[0] > observations_dict['Stations']['TakeOffAngle'].shape[1]:
                stations[:, 2] = np.matrix(observations_dict['Stations']['TakeOffAngle'])
            else:
                stations[:, 2] = np.matrix(observations_dict['Stations']['TakeOffAngle']).T
            stations[:, 3] = measured
            # Append to array
            stations = np.array(stations)
        # Otherwise just get set of stations in data (loop over observations
        # dictionaries).
        else:
            # Get all observation dicts used in the inversion
            if inversion_options and inversion_options != 'False':
                observations_dicts = [
                    ev_data[method] for method in inversion_options if method in ev_data.keys()]
            # Otherwise get all data
            else:
                observations_dicts = [
                    val for key, val in ev_data.items() if key not in ['hyp_file', 'UID']]
            stations = np.matrix(np.zeros((0, 4), dtype=np.object))
            for j, observations_dict in enumerate(observations_dicts):
                try:
                    number_stations = len(observations_dict['Stations']['Name'])
                except Exception:
                    continue
                _stations = np.matrix(np.zeros((number_stations, 4), dtype=np.object))
                _stations[:, 0] = np.matrix(observations_dict['Stations']['Name']).T
                observations_dict['Stations']['Azimuth'] = np.matrix(observations_dict['Stations']['Azimuth'])
                if observations_dict['Stations']['Azimuth'].shape[0] < observations_dict['Stations']['Azimuth'].shape[1]:
                    observations_dict['Stations']['Azimuth'] = observations_dict['Stations']['Azimuth'].T
                observations_dict['Stations']['TakeOffAngle'] = np.matrix(observations_dict['Stations']['TakeOffAngle'])
                if observations_dict['Stations']['TakeOffAngle'].shape[0] < observations_dict['Stations']['TakeOffAngle'].shape[1]:
                    observations_dict['Stations']['TakeOffAngle'] = observations_dict['Stations']['TakeOffAngle'].T
                _stations[:, 1] = np.matrix(observations_dict['Stations']['Azimuth'])
                _stations[:, 2] = np.matrix(observations_dict['Stations']['TakeOffAngle'])
                _stations[:, 3] = np.zeros((len(observations_dict['Stations']['Name']), 1))
                try:
                    stations = np.append(stations, _stations, 0)
                except Exception:
                    pass
                stations = np.array(stations)
            # Remove duplicate stations
            if len(stations):
                sta, ind = np.unique(np.array(stations[:, 0].tolist()), True)
                stations = stations[ind, :]
        if n_events > 1:
            all_stations.append(stations)
        else:
            all_stations = stations
    # Station distributions
    if location_samples and location_sample_multipliers:
        station_distribution['Distribution'] = location_samples
        station_distribution['Probability'] = location_sample_multipliers
    # Add inversion options
    other['Inversions'] = inversion_options
    # Add diagnostic options if possible
    if _diagnostic_output:
        a_polarity = kwargs.get('a_polarity', False)
        other['a_polarity'] = a_polarity
        error_polarity = kwargs.get('error_polarity', False)
        other['error_polarity'] = error_polarity
        a1_amplitude_ratio = kwargs.get('a1_amplitude_ratio', False)
        other['a1_amplitude_ratio'] = a1_amplitude_ratio
        percentage_error1_amplitude_ratio = kwargs.get('percentage_error1_amplitude_ratio', False)
        other['percentage_error1_amplitude_ratio'] = percentage_error1_amplitude_ratio
        a2_amplitude_ratio = kwargs.get('a2_amplitude_ratio', False)
        other['a2_amplitude_ratio'] = a2_amplitude_ratio
        percentage_error2_amplitude_ratio = kwargs.get('percentage_error2_amplitude_ratio', False)
        other['percentage_error2_amplitude_ratio'] = percentage_error2_amplitude_ratio
    # Construct event dict
    event = {'NSamples': output_data.pop('total_number_samples'), 'dV': output_data.pop('dV'), 'Probability': output_data.pop('probability')}
    try:
        event['UID'] = event_data['UID']
    except Exception:
        pass
    output_data_keys = list(output_data.keys())
    # Handle multiple events data types
    if not station_output:
        for key in output_data_keys:
            if key == 'moment_tensor_space':
                event['MTSpace'] = output_data.pop('moment_tensor_space')
            elif 'moment_tensor_space' in key:
                event['MTSpace'+key.split('_')[-1]] = output_data.pop(key)
            if key in ['g', 'd', 'h', 'k', 's', 'u', 'v', 'S1', 'D1', 'R1', 'S2', 'R2', 'D2']:
                event[key] = output_data.pop(key)
            elif key.split('_')[0] in ['g', 'd', 'h', 'k', 's', 'u', 'v', 'S1', 'D1', 'R1', 'S2', 'R2', 'D2']:
                event[key] = output_data.pop(key)
        event.update(output_data)
    else:
        event = {'NSamples': 0}
    try:
        event['ln_pdf'] = event['ln_pdf']._ln_pdf
    except Exception:
        pass
    # Create mdict and sdict and return
    if (isinstance(all_stations, list) and not len(all_stations)) or (isinstance(all_stations, np.ndarray) and not np.prod(all_stations.shape)):
        all_stations = []
    mdict = {'Events': event, 'Stations': all_stations, 'Other': other}
    sdict = {'StationDistribution': station_distribution}
    if not len(sdict['StationDistribution']):
        sdict = False
    return [mdict, sdict]


def hyp_output_dicts(event_data, inversion_options=False, output_data=False, location_samples=False, location_sample_multipliers=False, multiple_events=False, _diagnostic_output=False, normalise=True, *args, **kwargs):
    """
    Create output dictionaries for hyp format

    This creates the output dictionaries for the hyp format from the output data.

    Args
        event_data: Event data containing stations and observations (input data for the inversion).
        inversion_options:[False] List of inversion data type options.
        output_data:[False] Output data dictionaries (output from algorithm.__output__()).
        location_samples:[False] List of location PDF samples.
        location_sample_multipliers:[False] List of location PDF sample probabilities.
        multiple_events:[False] Boolean flag for multiple events.
        _diagnostic_output:[False] Output angle coefficents and other data for testing and diagnostics.
        normalise:[True] Normalise the output probabilities.

    Keyword Arguments
        station_only:[False] Boolean flag to only output station data (no inversion data).

    Returns
        [output_data,output_mt_binary,output_scale_factor_binary]: list of results as text and binary mt and scale_factor structures.
    """
    # Check Multiple events
    if multiple_events:
        output_contents = []
        output_mt = []
        output_sf = []
        # Loop over events
        for i, ev_data in enumerate(event_data):
            if 'moment_tensor_space' not in output_data:
                try:
                    ind = np.array(output_data['probability'] == output_data['probability'].max())[0]
                    maxMT = unique_columns(output_data['moment_tensor_space_'+str(i+1)][:, ind])
                    if maxMT.shape[1] > 1:
                        maxMT = maxMT[:, 0]
                except Exception:
                    maxMT = np.matrix([[0], [0], [0], [0], [0], [0]])
            # Get hyp output data
            output_contents.append('\n'.join(_generate_hyp_output_data(ev_data, inversion_options, output_data, maxMT))+'\n\n')
            # Get mt and scale_factor outputs
            binary, sf = _convert_mt_space_to_struct(output_data, i+1)
            output_mt.append(binary)
            output_sf.append(sf)
    else:
        # Get hyp output data
        output_contents = ['\n'.join(_generate_hyp_output_data(event_data, inversion_options, output_data))+'\n']
        # Get mt and scale_factor outputs
        binary, sf = _convert_mt_space_to_struct(output_data)
        # Need to convert this to a string if using python 3
        output_mt = [binary]
        output_sf = [sf]
    if sys.version_info.major > 2:
        binary_spacer = b''
    else:
        binary_spacer = ''
    return '\n'.join(output_contents), binary_spacer.join(output_mt), binary_spacer.join(output_sf)

#
# Output file formats
#


def MATLAB_output(output_data, fid='MTfitOutput.mat', pool=False, version='7.3', *args, **kwargs):
    """
    Outputs event results to fid as .mat

    HDF5 is used as default (MATLAB -v7.3 file types), but this can be set using the version keyword.
    If hdf5storage is not installed then it defaults to pre version 7.3 - can lead to filesize problems with large files.

    Saves MATLAB output dictionary to fid

    Args
        output_data: data dictionary for the event_data.
        fid:['MTfitOutput.mat'] Filename for output.
        pool:[False] Use Jobpool for parallel output.
        version:[7.3] MATLAB file type to use (v 7.3 requires hdf5storage and h5py).

    Returns
        output_string, fid: String of information for stdout, filename of output file.

    """
    # Check MATLAB file version (7.3 requires hdf5storage and h5py but can
    # store files >2Gb)
    if version == '7.3':
        try:
            from hdf5storage import savemat
        except ImportError:
            from scipy.io import savemat
            version = '7'
    else:
        from scipy.io import savemat  # noqa F401
    try:
        from scipy.io import savemat as st_savemat
        st_ver = '7'
    except ImportError:
        from hdf5storage import savemat as st_savemat  # noqa F401
        st_ver = '7.3'
    output_string = 'Outputting data in MATLAB format --version={}\n'.format(version)
    # Get sdict and mdict
    sdict = False
    if isinstance(output_data, list):
        mdict = output_data[0]
        if len(output_data) > 1:
            sdict = output_data[1]
    else:
        mdict = output_data
    # Convert dict keys to or from unicode
    if version == '7.3':
        mdict = convert_keys_to_unicode(mdict)
    else:
        mdict = convert_keys_from_unicode(mdict)
    if st_ver == '7.3' and sdict:
        sdict = convert_keys_to_unicode(sdict)
    else:
        sdict = convert_keys_from_unicode(sdict)
    # Run output tasks (using pool or otherwise)
    if pool and (sys.version_info[:2] >= (2, 7, 4) or ('Events' in mdict.keys() and np.sum([np.prod(mdict['Events'][key].shape) for key in mdict['Events'] if 'MTSpace' in key])*8*8 < 2**30)):
        # Check for cPickle bug #13555 http://bugs.python.org/issue13555 which seems linked to multiprocessing issue #17560 http://bugs.python.org/issue17560
        # cannot pickle files longer than 2**31 (32 bit encoding used for
        # cPickle length)
        output_string += 'Using jobPool to save file to {}\n'.format(fid)
        if mdict:
            pool.custom_task(MATLABOutputTask, os.path.splitext(fid)[0]+'.mat', mdict, version)
        if sdict:
            pool.custom_task(MATLABOutputTask, os.path.splitext(fid)[0]+'StationDistribution.mat', sdict, st_ver)
    else:
        if mdict:
            MATLABOutputTask(os.path.splitext(fid)[0]+'.mat', mdict, version)()
        if sdict:
            try:
                MATLABOutputTask(os.path.splitext(fid)[0]+'StationDistribution.mat', sdict, st_ver)()
            except Exception:
                traceback.print_exc()
        output_string += 'Saved to {}\n'.format(fid)
    return output_string, fid


def pickle_output(output_data, fid='MTfitOutput.out', pool=False, *args, **kwargs):
    """
    Outputs event results to fid as .out (default)

    cPickle output of MATLAB format output.

    Saves output dictionary to fid

    Args
        output_data: data dictionary for the event_data.
        fid:['MTfitOutput.out'] Filename for output.
        pool:[False] Use Jobpool for parallel output.

    Returns
        output_string,fid: String of information for stdout, filename of output file.
    """
    # Set file ending
    if os.path.splitext(fid)[1] == '.mat':
        fid = os.path.splitext(fid)[0]+'.out'
    sdict = False
    if isinstance(output_data, list):
        mdict = output_data[0]
        if len(output_data) > 1:
            sdict = output_data[1]
    else:
        mdict = output_data
    output_string = 'Outputting data in python format\n'
    # Output structure to cPickle.
    if pool and (sys.version_info[:2] >= (2, 7, 4) or ('Events' in mdict.keys() and np.sum([np.prod(mdict['Events'][key].shape) for key in mdict['Events'] if 'MTSpace' in key])*8*8 < 2**30)):
        # Check for cPickle bug #13555 http://bugs.python.org/issue13555 which seems linked to multiprocessing issue #17560 http://bugs.python.org/issue17560
        # cannot pickle files longer than 2**31 (32 bit encoding used for
        # cPickle length)
        output_string += 'Using jobPool to save file to '+fid+'\n'
        pool.custom_task(PickleOutputTask, fid, mdict)
        if sdict:
            pool.custom_task(PickleOutputTask, os.path.splitext(fid)[0]+'StationDistribution'+os.path.splitext(fid)[1], sdict)
    else:
        PickleOutputTask(fid, mdict)()
        if sdict:
            PickleOutputTask(os.path.splitext(fid)[0]+'StationDistribution'+os.path.splitext(fid)[1], sdict)()
        output_string += 'Saved to '+fid+'\n'
    return output_string, fid


def hyp_output(output_data, fid='MTfitOutput.hyp', pool=False, *args, **kwargs):
    """
    Outputs event results to fid as .hyp (default)

    hyp format output.

    Saves output dictionary to fid

    Args
        output_data: data dictionary for the event_data.
        fid:['MTfitOutput.hyp'] Filename for output.
        pool:[False] Use Jobpool for parallel output.

    Returns
        output_string,fid: String of information for stdout, filename of output file.
    """
    # Set file ending
    if os.path.splitext(fid)[1] == '.mat':
        fid = os.path.splitext(fid)[0]+'.hyp'
    mt_data = False
    sf_data = False
    # Get data
    if isinstance(output_data, (list, tuple)):
        output = output_data[0]
        if len(output_data) > 1:
            mt_data = output_data[1]
        if len(output_data) > 2:
            sf_data = output_data[2]
    else:
        output = output_data
    output_string = 'Outputting data in NLLOC hyp format\n'
    # Output using pool or otherwise
    if pool:
        # Check for cPickle bug #13555 http://bugs.python.org/issue13555 which seems linked to multiprocessing issue #17560 http://bugs.python.org/issue17560
        # cannot pickle files longer than 2**31 (32 bit encoding used for
        # cPickle length)
        output_string += 'Using jobPool to save file to '+fid+'\n'
        pool.custom_task(HypOutputTask, fid, output, False)
        if mt_data and (sys.version_info[:2] >= (2, 7, 4) or len(mt_data)*128 < 2**30):
            pool.custom_task(
                HypOutputTask, fid.replace('.hyp', '.mt'), mt_data, True)
        if sf_data and (sys.version_info[:2] >= (2, 7, 4) or len(sf_data)*128 < 2**30):
            pool.custom_task(
                HypOutputTask, fid.replace('.hyp', '.sf'), sf_data, True)

    else:
        HypOutputTask(fid, output, False)()
        if mt_data:
            HypOutputTask(fid.replace('.hyp', '.mt'), mt_data, True)()
        if sf_data and len(sf_data):
            HypOutputTask(fid.replace('.hyp', '.sf'), sf_data, True)()
        output_string += 'Saved to '+fid+'\n'
    return output_string, fid

#
# Read inversion output
#


def read_binary_output(filename, version=2):
    """
    Reads binary output and converts to data dict.

    Reads the binary output created using hyp output format (_convert_mt_space_to_struct) and returns an output data dictionary.

    Args
        filename: str filename to read.

    Keyword Args
        version: int version of the file format (Outputs from versions before 1.0.3 did not contain
                 the DKL estimate or file version, so the version must be set to 1 for these,
                 otherwise the version is set automatically).

    Returns
        output_data: list of dictionaries of output data.
    """
    # Open file
    with open(filename, 'rb') as f:
        f.seek(0, 2)
        nmax = f.tell()
        f.seek(0, 0)
        output_data = []
        sqrt2 = np.sqrt(2)
        while f.tell() < nmax:
            # Read file version
            if version >= 2:
                version = struct.unpack('Q', f.read(8))[0]
            # Read total number and number of saved samples and converted flag
            total_number_samples, number_mt_samples, converted = struct.unpack('QQ?', f.read(17))
            # Get the Bayesian Evidence estimate
            ln_bayesian_evidence = struct.unpack('1d', f.read(8))[0]
            # Get the DKL estimate
            if version >= 2:
                dkl = struct.unpack('1d', f.read(8))[0]
            else:
                dkl = np.nan
            # Generate blank arrays
            MTSpace = np.matrix(np.zeros((6, number_mt_samples)))
            Probability = np.matrix(np.zeros((1, number_mt_samples)))
            Ln_P = np.matrix(np.zeros((1, number_mt_samples)))
            if converted:
                g = np.zeros((number_mt_samples,))
                d = np.zeros((number_mt_samples,))
                k = np.zeros((number_mt_samples,))
                h = np.zeros((number_mt_samples,))
                s = np.zeros((number_mt_samples,))
                u = np.zeros((number_mt_samples,))
                v = np.zeros((number_mt_samples,))
                s1 = np.zeros((number_mt_samples,))
                d1 = np.zeros((number_mt_samples,))
                r1 = np.zeros((number_mt_samples,))
                s2 = np.zeros((number_mt_samples,))
                d2 = np.zeros((number_mt_samples,))
                r2 = np.zeros((number_mt_samples,))
            # Loop over samples
            for i in range(number_mt_samples):
                Probability[0, i], Ln_P[0, i], Mnn, Mee, Mdd, Mne, Mnd, Med = struct.unpack('8d', f.read(64))
                MTSpace[0, i] = Mnn
                MTSpace[1, i] = Mee
                MTSpace[2, i] = Mdd
                MTSpace[3, i] = sqrt2*Mne
                MTSpace[4, i] = sqrt2*Mnd
                MTSpace[5, i] = sqrt2*Med
                if converted:
                    g[i], d[i], k[i], h[i], s[i], u[i], v[i], s1[i], d1[i], r1[
                        i], s2[i], d2[i], r2[i] = struct.unpack('13d', f.read(104))
            # create output data dict
            out = {'moment_tensor_space': MTSpace, 'dkl': dkl, 'probability':
                   Probability, 'ln_pdf': Ln_P, 'total_number_samples': total_number_samples}
            if converted:
                out.update({'g': g, 'd': d, 'k': k, 'h': h, 's': s, 'u': u, 'v': v, 'S1': s1, 'D1': d1,
                            'R1': r1, 'S2': s2, 'D2': d2, 'R2': r2, 'ln_bayesian_evidence': ln_bayesian_evidence})
            # Append to output data list
            output_data.append(out)
    return output_data


def read_sf_output(filename):
    """
    Reads binary scale factor output and converts to data dict.

    Reads the binary scalefactor output created using hyp output format (_convert_mt_space_to_struct) and returns an output data dictionary.
    The scalefactor pairs are stored as an array with the indices corresponding to the pairs 0,1 0,2 0,3 ... 1,2 1,3 .... etc.

    Args
        filename: str filename to read.

    Returns
        output_data: list of dictionaries of output data.
    """
    # Open file
    with open(filename, 'rb') as f:
        f.seek(0, 2)
        nmax = f.tell()
        f.seek(0, 0)
        output_data = []
        while f.tell() < nmax:
            # Read total number and number of saved samples and number of events
            total_number_samples, number_samples, n_events = struct.unpack('QQQ', f.read(24))
            # Generate blank arrays
            Mu = np.matrix(np.zeros((int(n_events*(n_events-1)/2.), number_samples)))
            S = np.matrix(np.zeros((int(n_events*(n_events-1)/2.), number_samples)))
            Ln_P = np.matrix(np.zeros((1, number_samples)))
            Probability = np.matrix(np.zeros((1, number_samples)))
            # Loop over samples
            for i in range(number_samples):
                Probability[0, i], Ln_P[0, i] = struct.unpack('2d', f.read(16))
                # Loop over event pairs (stored as 0,1 0,2 0,3 ... 1,2 1,3 ....)
                for j in range(int(n_events*(n_events-1)/2.)):
                    Mu[j, i], S[j, i] = struct.unpack('dd', f.read(16))
            # Creat output dict
            out = {'scale_factors': Mu, 'probability': Probability, 'ln_pdf':
                   Ln_P, 'total_number_samples': total_number_samples, 'sigma': S}
            # Append to output list
            output_data.append(out)
    return output_data


def read_matlab_output(filename, station_distribution=False):
    """
    Reads matlab output and converts to data dict.

    Reads the matlab output created using the matlab output format and returns an output data dictionary.

    Args
        filename: str filename to read.

    Returns
        (dict,dict): tuple of (events,data) dictionaries of output data.
    """
    try:
        hdf5 = False
        from scipy.io import loadmat
        data = loadmat(filename)
    except Exception:
        try:
            from hdf5storage import loadmat
            hdf5 = True
            data = loadmat(filename)
        except Exception:
            if hdf5:
                raise IOError('Input file '+filename+" can't be read as old or new style (hdf5) format")
            else:
                raise IOError('Input file '+filename+" can't be read - it could be a new style (hdf5) format, which requires the hdf5storage module to be installed.")
    if station_distribution:
        return _parse_matlab_station_distribution(data)
    return _parse_matlab_output(data)


def _parse_matlab_stations(stations_data):
    try:
        stations = {'names': stations_data[0]['Name'][0], 'azimuth': stations_data[0]['Azimuth'][0],
                    'takeoff_angle': stations_data[0]['TakeOffAngle'][0]}
        try:
            stations['polarity'] = stations_data[0]['Polarity'][0]
        except Exception:
            stations['polarity'] = np.zeros(len(stations['names']))
    except Exception:
        try:
            stations = {
                'names': [], 'azimuth': [], 'takeoff_angle': [], 'polarity': []}
            for st in stations_data:
                stations['names'].append(st[0][0])
                stations['azimuth'].append(st[1][0][0])
                stations['takeoff_angle'].append(st[2][0][0])
                try:
                    stations['polarity'].append(st[3][0][0])
                except Exception:
                    stations['polarity'].append(0)
        except Exception:
            try:
                stations = {
                    'names': [], 'azimuth': [], 'takeoff_angle': [], 'polarity': []}
                for st in stations_data:
                    stations['names'].append(st[0])
                    stations['azimuth'].append(st[1])
                    stations['takeoff_angle'].append(st[2])
                    try:
                        stations['polarity'].append(st[3])
                    except Exception:
                        stations['polarity'].append(0)
            except Exception:
                try:
                    stations = {
                        'names': [], 'azimuth': [], 'takeoff_angle': [], 'polarity': []}
                    for st in stations_data:
                        stations['names'].append(st[0][0])
                        stations['azimuth'].append(st[1][0][0])
                        stations['takeoff_angle'].append(st[2][0][0])
                        try:
                            stations['polarity'].append(st[3][0][0])
                        except Exception:
                            stations['polarity'].append(0)
                except Exception:
                    stations = {}
    try:
        stations['names'] = stations['names'].tolist()
    except Exception:
        pass
    return stations


def _parse_matlab_station_distribution(station_distribution_data):
    try:
        station_distribution = {'probability': station_distribution_data[
            'StationDistribution']['Probability'][0][0]}
        distribution = []
        for st in station_distribution_data['StationDistribution']['Distribution'][0][0][0]:
            st_sample = _parse_matlab_stations(st)
            try:
                st_sample.pop('polarity')
            except Exception:
                pass
            distribution.append(st_sample)
        station_distribution['distribution'] = distribution
    except Exception:
        station_distribution = convert_keys_from_unicode(
            station_distribution_data['StationDistribution'])
    return station_distribution


def _parse_matlab_output(data):
    """
    Converts the matlab structured data from the matlab file and returns an output data dictionary.

    Args
        data: dict matlab data.

    Returns
        (dict,dict): tuple of (events,data) dictionaries of output data.
    """
    n_events = 1
    if (isinstance(data['Events'], dict) and 'MTSpace' not in data['Events'].keys()) or (not isinstance(data['Events'], dict) and 'MTSpace' not in data['Events'].dtype.names):
        ev_ind = [u.replace('MTSpace', '') for u in data['Events'].dtype.names if 'MTSpace' in u]
        n_events = len(ev_ind)
    if n_events > 1:
        # Multiple Events
        events = []
        stations = []
        for ind in ev_ind:
            ev_data = {}
            key_map = dict([(u, u[:-1].replace('_', ''))
                            for u in data['Events'].dtype.names if u[-1] == ind])
            key_map['ln_pdf'] = 'ln_pdf'
            key_map['Probability'] = 'Probability'
            for i, key in enumerate(key_map.keys()):

                if key_map[key] not in ['MTSpace']:
                    ev_data[key_map[key]] = data['Events'][key][0, 0].flatten()
                    if len(ev_data[key_map[key]]) == 1:
                        ev_data[key_map[key]] = ev_data[key_map[key]][0]
                else:
                    ev_data[key_map[key]] = data['Events'][key][0, 0]
            try:
                st = _parse_matlab_stations(data['Stations'][0, int(ind)-1])
            except Exception:
                st = {}
            events.append(ev_data)
            stations.append(st)
        return events, stations
    # Single Event
    try:
        stations = _parse_matlab_stations(data['Stations'])
    except Exception:
        stations = {}
    events = {}
    try:
        for key in data['Events'].dtype.fields.keys():

            if key not in ['MTSpace']:
                events[key] = data['Events'][key][0, 0].flatten()
                if len(events[key]) == 1:
                    events[key] = events[key][0]
            else:
                events[key] = data['Events'][key][0, 0]
    except Exception:
        events = convert_keys_from_unicode(data['Events'])
    if events['Probability'].max() == 0:
        events['Probability'] = np.exp(events['ln_pdf']-events['ln_pdf'].max())
    return (events, stations)


def read_pickle_output(filename, station_distribution=False):
    """
    Reads pickle output and returns the data

    Args
        filename: str filename to read.

    Returns
        (dict,dict): tuple of (events,data) dictionaries of output data.
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    if station_distribution:
        station_distribution = data['StationDistribution']
        station_distribution['probability'] = station_distribution.pop('Probability')
        old_dist = station_distribution.pop('Distribution')
        station_distribution['distribution'] = []
        for st in old_dist:
            station_distribution['distribution'].append({'azimuth': st['Azimuth'],
                                                         'takeoff_angle': st['TakeOffAngle'],
                                                         'names': st['Name']})
        return station_distribution
    stations = {}
    if 'Stations' in data.keys():
        stations = _parse_matlab_stations(data.pop('Stations'))
    events = data
    if 'Events' in data.keys():
        events = data.pop('Events')
    return (events, stations)


def read_hyp_output(filename):
    """
    Reads hyp output and returns the data

    Args
        filename: str filename to read.

    Returns
        (dict,dict): tuple of (events,data) dictionaries of output data.
    """
    try:
        hyp_data = parse_hyp(os.path.splitext(filename)[0]+'.hyp')
    except Exception:
        hyp_data = {}
    events = read_binary_output(os.path.splitext(filename)[0]+'.mt')
    key_map = {'takeoffangle': 'takeoff_angle'}
    try:
        stations = hyp_data['PPolarity']['Stations']
        stations['polarity'] = hyp_data['PPolarity']['Measured']
        for key in stations:
            new_key = key.lower()
            if key.lower() in key_map.keys():
                new_key = key_map[key.lower()]
            stations[new_key] = stations.pop(key)
    except Exception:
        stations = {}
    return events, stations


def read_scatangle_output(filename, number_location_samples=0, bin_size=0, **kwargs):
    """
    Reads scatangle file for plotting

    Args
        filename: str filename to read

    Keyword Args
        number_location_samples: int number of location samples to sub-sample to.
        bin_size: float bin size for binning the angle samples in.

    Returns
        dict: station distribution
    """
    data = {'StationDistribution': {}}
    data['StationDistribution']['Distribution'], data['StationDistribution']['Probability'] = parse_scatangle(filename,
                                                                                                              number_location_samples=number_location_samples,
                                                                                                              bin_size=bin_size)
    station_distribution = data['StationDistribution']
    station_distribution['probability'] = station_distribution.pop('Probability')
    old_dist = station_distribution.pop('Distribution')
    station_distribution['distribution'] = []
    for st in old_dist:
        station_distribution['distribution'].append({'azimuth': st['Azimuth'], 'takeoff_angle': st['TakeOffAngle'], 'names': st['Name']})
    return station_distribution

# Dictionary conversion functions for file_io


def convert_keys_to_unicode(dictionary,):
    """
    Converts dictionary keys to unicode

    Required for MATLAB -v7.3 format output. This recursively changes strings in a dictionary to unicode.

    Args
        dictionary: Input dictionary.
    Returns
        dictionary: Converted dictionary.
    """
    if sys.version_info.major > 2:
        return dictionary
    if isinstance(dictionary, list):
        new_list = []
        for item in dictionary:
            new_list.append(convert_keys_to_unicode(item))
        return new_list
    elif isinstance(dictionary, dict):
        new_dictionary = {}
        for key, value in dictionary.items():
            if isinstance(value, list):
                new_value = []
                for item in value:
                    new_value.append(convert_keys_to_unicode(item))
                value = new_value
            if isinstance(value, dict):
                new_dictionary[key.decode('unicode-escape')] = convert_keys_to_unicode(value)
            else:
                new_dictionary[key.decode('unicode-escape')] = convert_keys_to_unicode(value)
        return new_dictionary
    return dictionary


def convert_keys_from_unicode(dictionary,):
    """
    Converts dictionary keys from  unicode

    Required for MATLAB -v7 format output. This recursively changes strings in a dictionary from unicode.

    Args
        dictionary: Input dictionary.
    Returns
        dictionary: Converted dictionary.
    """
    if sys.version_info.major > 2:
        return dictionary
    if isinstance(dictionary, list):
        new_list = []
        for item in dictionary:
            new_list.append(convert_keys_from_unicode(item))
        return new_list
    elif isinstance(dictionary, dict):
        new_dictionary = {}
        for key, value in dictionary.items():
            if isinstance(value, list):
                new_value = []
                for item in value:
                    new_value.append(convert_keys_from_unicode(item))
                value = new_value
            if isinstance(value, dict):
                new_dictionary[key.encode('utf-8')] = convert_keys_from_unicode(value)
            else:
                new_dictionary[key.encode('utf-8')] = convert_keys_from_unicode(value)
        return new_dictionary
    return dictionary


def unique_columns(data, counts=False, index=False):
    """Get unique columns in an array (used for max probability MT)"""
    output = []
    if float('.'.join(np.__version__.split('.')[:2])) >= 1.13:
        unique_results = np.unique(data, return_index=index, return_counts=counts, axis=1)
        if isinstance(unique_results, tuple):
            unique = unique_results[0]
        else:
            unique = unique_results
        if index:
            idx = unique_results[1]
        if counts:
            unique_counts = unique_results[-1]
    else:
        data = np.array(data).T
        ind = np.lexsort(data.T)
        data = np.squeeze(data)
        if len(ind.shape) > 1:
            ind = np.squeeze(ind)
        unique = data[ind[np.concatenate(([True], np.any(data[ind[1:]] != data[ind[:-1]], axis=1)))]].T
        if counts:
            indx = np.nonzero(np.concatenate(([True], np.any(data[ind[1:]] != data[ind[:-1]], axis=1))))[0]
            unique_counts = []
            for u, j in enumerate(indx):
                try:
                    unique_counts.append(indx[u+1]-j)
                except Exception:
                    unique_counts.append(1)
        if index:
            idx = ind[np.concatenate(([True], np.any(data[ind[1:]] != data[ind[:-1]], axis=1)))]
    output.append(np.matrix(unique))
    if counts:
        output.append(np.array(unique_counts))
    if index:
        output.append(idx)
    if len(output) == 1:
        return output[0]
    else:
        return tuple(output)
