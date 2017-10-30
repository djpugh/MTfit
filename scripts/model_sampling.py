"""
ModelSampling
=============
Model Sampling Code for Moment Tensor Inversion

The code can be called from the command line directly or from within python itself (see below)

The code requires Monte Carlo sampling for the scatangle files, and then samples from each model
scatangle with a probability given in the model_mapping dictionary.

Data Preparation
****************
The input arguments are model_mapping, a dictionary mapping the model scatangle files to the model
probability (in the range 0-1), N, the number of samples to generate, and output_path, an optional
argument for the output folder path.
If the model_mapping is not set, the current folder is used, with uniform probability for all the
*.scatangle files in it. The default value for N is 50000 samples

Output
******
The output is a file (models_scatter.scatangle) containing the scatter samples selected from the
different models. It is outputted to the current folder, unless the output_path arguments is set.

"""
import glob
import os
import sys

import numpy as np


def model_sample(data_path='.', model_mapping=False, N=50000, output_path=False):
    """
    Main Sampling code - sample over models

    Runs a monte carlo sampling over different earth model locations. Returns a scatangle file
    for the combined models sampling.

    Args:
        data_path: str data path
        model_mapping: dictionary mapping model scatangle files to probabilities between 0 and 1
                [default is all scatangle files in current folder with uniform probability].
        N: number of samples to generate [default=50000]
        output_path: path to folder to output models_scatter.scatangle file to [default is current folder]
    """
    os.chdir(data_path)
    if not model_mapping:
        # Assume uniform sampling
        # list all sample files
        files = glob.glob('*.scatangle')
        model_mapping = dict((f, 1.0/len(files)) for f in files)
    if output_path:
        output_name = os.path.abspath(output_path)
    else:
        output_name = os.path.abspath(os.getcwd())
    model_mapping = map_to_limits(model_mapping)
    samples = get_samples(model_mapping, N)
    if len(output_name):
        output_name += os.path.sep
    write_models_scatter_file(samples, output_name+'models_scatter.scatangle')


def map_to_limits(model_mapping):
    """
    Converts the model probabilities to limits for the model selection

    Args:
        model_mapping: dictionary of model files:probabilities

    Returns:
        dictionary of upper limits for each model (i.e. max value random number can take and still be
                considered selecting that model)

    The output limits are the max value that a random number between 0 and 1 can take and still select the model, e.g.

    If there are 3 models:
        Model1: p=0.2
        Model2: p=0.5
        Model3: p=0.3

    Then the limits dictionary would be {0.2:'Model1',0.7:'Model2',1.0:'Model3'}
    """
    limits = np.cumsum(model_mapping.values())/np.sum(model_mapping.values())
    return dict(zip(limits, model_mapping.keys()))


def random_model_sample(model_mapping):
    """
    Randomly select a model

    Args:
        model_mapping: dictionary mapping limits to model files (see map_to_limits docs for more info on
                dictionary structure)

    Returns:
        string model file name

    The model is selected by generating a random number between 0 and 1 and then the models are selected
    according to the different probabilities.
    """
    limits = np.array(sorted(model_mapping.keys()))
    return model_mapping[limits[np.random.rand() < limits][0]]  # Select model by choosing the lowest limit that the random number is less than.


def get_samples(model_mapping, N):
    """
    Get model samples

    Loops until N samples are generated from the model files.

    Args:
        model_mapping: dictionary of limits based model mapping (from map_to_limits)
        N: number of samples to generate

    Returns
        list of dictionaries for each sample.
    """
    records = []
    load_in_to_memory = check_memory(model_mapping)
    if load_in_to_memory:
        models = dict((model, read_station_scatter_file(model)) for model in model_mapping.values())
    i = 0
    while i < N:
        if not i % 100:
            print str(i)+' samples'
        model = random_model_sample(model_mapping)
        if load_in_to_memory:
            new_record = random_record(models[model])
        else:
            new_record = random_record(read_station_scatter_file(model))
        if len(new_record['Name']):
            records.append(new_record)
            i += 1
    return records


def random_record(records):
    """
    Selects random record from scatangle records.

    Args:
        records: input list of records

    Returns:
        dictionary scatangle record
    """
    index = np.random.randint(len(records))
    return records[index]


def check_memory(model_mapping):
    """
    Checks if loading in scatangle files into memory is efficient given available physical memory

    Args:
        model_mapping: dictionary mapping limits to model files

    Returns:
        boolean load in scatangle files.
    """
    load_in_multiplier = 3.6
    file_sizes = dict((f, os.path.getsize(f)/float(2**20)) for f in model_mapping.values())
    total_memory = sum(file_sizes.values())*load_in_multiplier
    try:
        import psutil
        free_memory = psutil.phymem_usage().available/float(2**20)
        if free_memory > 2*total_memory:
            return True
        return False
    except Exception:
        if 2*total_memory < 6000:
            return True
        return False


def read_station_scatter_file(filename):
        """
        Read station angles scatter file

        Reads the station angle scatter file. Expected format is given below. TakeOffAngle is
        0 down (NED coordinate system).

        Args:
            filename: Angle scatter file name

        Returns:
            records: Angle Records for each sample.

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
        import numpy as np
        with open(filename, 'r') as f:
            station_file = f.readlines()
        records = []
        record = {'Name': [], 'Azimuth': [], 'TakeOffAngle': []}
        for line in station_file:
            if line == '\n':
                if len(record['Name']):
                    record['Azimuth'] = np.matrix(record['Azimuth']).T
                    record['TakeOffAngle'] = np.matrix(record['TakeOffAngle']).T
                    records.append(record)
                record = {'Name': [], 'Azimuth': [], 'TakeOffAngle': []}
            elif len(line.rstrip().split()) > 1:
                record['Name'].append(line.split()[0])
                record['Azimuth'].append(float(line.split()[1]))
                record['TakeOffAngle'].append(float(line.rstrip().split()[2]))
        return records


def write_models_scatter_file(records, filename):
    """
    Writes the station angles scatter file

        Writes the station angle scatter file. Output format is given below. TakeOffAngle is 0 down
        (NED coordinate system).

        Args:
            records: angle records to output
            filename: Angle scatter file name

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
    output = ""
    for record in records:
        output += '1.00\n'
        for i, name in enumerate(record['Name']):
            output += name+'\t'+str(record['Azimuth'][i, 0])+'\t'+str(record['TakeOffAngle'][i, 0])+'\n'
        output += '\n'
    print(filename)
    with open(filename, 'w') as f:
        f.write(output)


def run():
    model_sample(*sys.argv[1:])


if __name__ == "__main__":
    run()
