********************************
MTfit command line options
********************************
Command line usage::

    MTfit [-h] [-d DATAFILE] [-s LOCATION_PDF_FILE_PATH]
    [-a {iterate,time,mcmc,transdmcmc}] [-l] [-n N] [-m MEM] [-c]
    [-b] [--nstations NUMBER_STATIONS]
    [--nanglesamples NUMBER_LOCATION_SAMPLES] [-f] [--not_file_safe]
    [-i INVERSION_OPTIONS] [-o FID] [-x MAX_SAMPLES] [-t MAX_TIME]
    [-e] [-r] [--marginalise_relative] [-R] [--invext DATA_EXTENSION]
    [--angleext ANGLE_EXTENSION] [-S MINIMUM_NUMBER_INTERSECTIONS]
    [-M] [-B] [-X MIN_NUMBER_INITIALISATION_SAMPLES] [-T]
    [-Q [QUALITY_CHECK]] [-D] [-V VERBOSITY] [-g]
    [-j DIMENSION_JUMP_PROB] [-y {grid}] [-u MIN_ACCEPTANCE_RATE]
    [-v MAX_ACCEPTANCE_RATE] [-w ACCEPTANCE_RATE_WINDOW]
    [-W WARNINGS] [-z LEARNING_LENGTH] [--version] [--mpi_call]
    [--output-format {matlab,pickle,hyp}]
    [--results-format {full_pdf,hyp}] [--no-dist]
    [--dc-prior DC_PRIOR] [--sampling SAMPLING]
    [--sample-models SAMPLE_DISTRIBUTION]
    [--sampling-prior SAMPLING_PRIOR] [--no-normalise] [--convert]
    [--discard DISCARD] [--mpioutput] [--combine_mpi_output]
    [--c_generate] [--relative_loop] [--bin-scatangle]
    [--bin-size BIN_SCATANGLE_SIZE] [-q] [--nodes QSUB_NODES]
    [--ppn QSUB_PPN] [--pmem QSUB_PMEM] [--email QSUB_M]
    [--emailoptions QSUB_M] [--name QSUB_N]
    [--walltime QSUB_WALLTIME] [--queue QSUB_Q]
    [--bladeproperties QSUB_BLADE_PROPERTIES]
    [--feature QSUB_BLADE_FEATURE]
    [data_file]


Positional Arguments:
============================

---------------------------

::

  data_file

Data file to use for the inversion, optional but must
be specified either as a positional argument or as an
optional argument (see -d below) If not specified
defaults to all *.inv files in current directory, and
searches for all anglescatterfiles in the directory
too. Inversion file extension can be set using
--invext option. Angle scatter file extension can be
set using --angleext option



Optional Arguments:
============================

---------------------------

::

  -h, --help

show this help message and exit


---------------------------

::

  -d DATAFILE, --datafile DATAFILE, --data_file DATAFILE

Data file to use for the inversion. Can be provided as
a positional argument.
There are several different data file types:

  * pickled dictionary
  * csv file
  * NLLOC hyp file

The data file is a pickled python dictionary of
the form::

    {'DataType':{'Stations':{'Name':['STA1','STA2',...],
    'Azimuth':np.matrix([[190],[40],...]),
    'TakeOffAngle':np.matrix([[70],[40],...])},
    'Measured':np.matrix([[1],[-1],...]),
    'Error':np.matrix([[0.01],[0.02],...])}}


e.g.::

    {'P/SHRMSAmplitudeRatio':{'Stations':{'Name':['S
    0649',"S0162"],
    'Azimuth':np.array([90.0,270.0]),
    'TakeOffAngle':np.array([30.0,60.0])},
    'Measured':np.matrix([[1],[-1]]),
    'Error':np.matrix([[ 0.001,0.02],[
    0.001,0.001]])}}


Or a CSV file with events split by blank lines, a
header line showing which row corresponds to which
information (default is as shown here),
UID and data-type information stored in the first
column,
e.g.::

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


Is a CSV file with 2 events, one event UID of 123,
and PPolarity data at S001 and S002 and
P/SHRMSAmplitude data at S003,
and a second event with no UID (will default to
the event number, in this case 2) with PPolarity data
at S003.

This data format can be constructed manually or
automatically.


---------------------------

::

  -s LOCATION_PDF_FILE_PATH, --anglescatterfilepath LOCATION_PDF_FILE_PATH,
  --location_pdf_file_path LOCATION_PDF_FILE_PATH,
  --location_file_path LOCATION_PDF_FILE_PATH,
  --scatterfilepath LOCATION_PDF_FILE_PATH,
  --scatter_file_path LOCATION_PDF_FILE_PATH


Path to location scatter angle files - wild cards
behave as normal.
To include the model and location uncertainty, a
ray path angle pdf file must be provided.
This is of the form::

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

e.g.::

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
    S0236   253.5   118.7



---------------------------

::

  -a {iterate,time,mcmc,transdmcmc}, --algorithm {iterate,time,mcmc,transdmcmc}

Selects the algorithm used for the search.
[default=time]
Possible algorithms are:
iterate (random sampling of the source space
for a set number of samples)
time (random sampling of the source space for
a set time)
mcmc (Markov chain Monte Carlo sampling)


---------------------------

::

  -l, --singlethread, --single, --single_thread

Flag to disable parallel computation


---------------------------

::

  -n N, --numberworkers N, --number_workers N

Set the number of workers used in the parallel
computation. [default=all available cpus]


---------------------------

::

  -m MEM, --mem MEM, --memory MEM, --physical_memory MEM, --physicalmemory MEM

Set the maximum memory used in Gb if psutil not
available [default=8Gb]


---------------------------

::

  -c, --doublecouple, --double-couple, --double_couple, --dc, --DC

Flag to constrain the inversion to double-couple
sources only


---------------------------

::

  -b, --compareconstrained, --compare_constrained

Flag to run two inversions, one constrained to
double-couple and one unconstrained


---------------------------

::

  --nstations NUMBER_STATIONS

Set the maximum number of stations without having to
load an angle pdf file - used for calculating sample
sizes and memory sizes, and can speed up the
calculation a bit, but has no effect on result.


---------------------------

::

  --nanglesamples NUMBER_LOCATION_SAMPLES,
  --nlocationsamples NUMBER_LOCATION_SAMPLES,
  --number_location_samples NUMBER_LOCATION_SAMPLES,
  --number-location-samples NUMBER_LOCATION_SAMPLES


Set the maximum number of angle pdf samples to use. If
this is less than the total number of samples, a
subset are randomly selected [default=0].


---------------------------

::

  -f, --file_sample, --file-sample, --filesample, --disk_sample,
  --disk-sample, --disksample


Save sampling to disk (allows for easier recovery and
reduces memory requirements, but can be slower)


---------------------------

::

  --not_file_safe, --not-file-safe, --notfilesafe, --no_file_safe,
  --no-file-safe, --nofilesafe


Disable file safe saving (i.e. copy and write to .mat~
then copy back


---------------------------

::

  -i INVERSION_OPTIONS, --inversionoptions INVERSION_OPTIONS,
  --inversion_options INVERSION_OPTIONS


Set the inversion data types to use: comma delimited.
If not set, the inversion uses all the data types
in the data file.
e.g.
PPolarity,P/SHRMSAmplitudeRatio

Needs to correspond to the data types in the data
file.

If not specified can lead to independence errors:
e.g.
P/SH Amplitude Ratio and P/SV Amplitude Ratio can
give SH/SV Amplitude Ratio.
Therefore using SH/SV Amplitude Ratio in the
inversion is reusing data and will artificially
sharpen the PDF.
This applies to all forms of dependent
measurements.



---------------------------

::

  -o FID, --out FID, --fid FID, --outputfile FID, --outfile FID

Set output file basename [default=MTfitOutput]


---------------------------

::

  -x MAX_SAMPLES, --samples MAX_SAMPLES, --maxsamples MAX_SAMPLES,
  --max_samples MAX_SAMPLES, --chain_length MAX_SAMPLES,
  --max-samples MAX_SAMPLES, --chain-length MAX_SAMPLES, --chainlength MAX_SAMPLES


Iteration algorithm: Set maximum number of samples to
use [default=6000000]. McMC algorithms: Set chain
length [default=10000], trans-d McMC [default=100000]


---------------------------

::

  -t MAX_TIME, --time MAX_TIME, --maxtime MAX_TIME, --max_time MAX_TIME

Time algorithm: Set maximum time to use [default=600]


---------------------------

::

  -e, --multiple_events, --multiple-events

Run using events using joint PDF approach


---------------------------

::

  -r, --relative_amplitude, --relative-amplitude

Run using events using joint PDF approach


---------------------------

::

  --marginalise_relative, --marginalise, --marginalise-relative

Flag to marginalise location uncertainty in relative
amplitude case [default=False]


---------------------------

::

  -R, --recover

Recover crashed run (ie restart from last event not
written out)]


---------------------------

::

  --invext DATA_EXTENSION, --dataextension DATA_EXTENSION,
  --dataext DATA_EXTENSION, --data-extension DATA_EXTENSION,
  --data_extension DATA_EXTENSION


Set data file extension to search for when inverting
on a folder


---------------------------

::

  --angleext ANGLE_EXTENSION, --locationextension ANGLE_EXTENSION,
  --locationext ANGLE_EXTENSION, --location-extension ANGLE_EXTENSION,
  --location_extension ANGLE_EXTENSION


Set location sample file extension to search for when
inverting on a folder


---------------------------

::

  -S MINIMUM_NUMBER_INTERSECTIONS,
  --minimum_number_intersections MINIMUM_NUMBER_INTERSECTIONS,
  --min_number_intersections MINIMUM_NUMBER_INTERSECTIONS,
  --minimum-number-intersections MINIMUM_NUMBER_INTERSECTIONS,
  --min-number-intersections MINIMUM_NUMBER_INTERSECTIONS


For relative amplitude inversion, the minimum number
of intersecting stations required (must be greater
than 1) [default=2]


---------------------------

::

  -M, --mpi, --MPI

Run using mpi - will reinitialise using mpirun (mpi
etc needs to be added to path)


---------------------------

::

  -B, --benchmark, --benchmarking

Run benchmark tests for the event


---------------------------

::

  -X MIN_NUMBER_INITIALISATION_SAMPLES,
  --min_number_check_samples MIN_NUMBER_INITIALISATION_SAMPLES,
  --min_number_initialisation_samples MIN_NUMBER_INITIALISATION_SAMPLES


Minimum number of samples for McMC initialiser, or the
minimum number of samples required when using quality
check (-Q)


---------------------------

::

  -T, --test, --test

Run MTfit Test suite (if combined with -q runs test
suite on cluster


---------------------------

::

  -Q [QUALITY_CHECK], --quality [QUALITY_CHECK]

Run MTfit with quality checks enabled [default=False].
Checks if an event has a percentage of non-zero
samples lower than the flag - values from 0-100.


---------------------------

::

  -D, --debug

Run MTfit with debugging enabled.


---------------------------

::

  -V VERBOSITY, --verbosity VERBOSITY

Set verbosity level for non-fatal errors [default=0].


---------------------------

::

  -g, --diagnostics

Run MTfit with diagnostic output. Outputs the full
chain and sampling - wil make a large file.


---------------------------

::

  -j DIMENSION_JUMP_PROB, --jumpProbability DIMENSION_JUMP_PROB,
  --jumpProb DIMENSION_JUMP_PROB, --jumpprob DIMENSION_JUMP_PROB,
  --jumpProb DIMENSION_JUMP_PROB, --dimensionJumpProb DIMENSION_JUMP_PROB,
  --dimensionjumpprob DIMENSION_JUMP_PROB


Sets the probability of making a dimension jump in the
Trans-Dimensional McMC algorithm [default=0.01]


---------------------------

::

  -y {grid}, --initialSampling {grid}

Sets the initialisation sampling method for McMC
algorithms choices:
grid - use grid based sampling to find non-zero
initial sample [default=grid]


---------------------------

::

  -u MIN_ACCEPTANCE_RATE, --minAcceptanceRate MIN_ACCEPTANCE_RATE,
  --minacceptancerate MIN_ACCEPTANCE_RATE,
  --min_acceptance_rate MIN_ACCEPTANCE_RATE


Set the minimum acceptance rate for the McMC algorithm
[mcmc default=0.3, transdmcmc default=0.05]


---------------------------

::

  -v MAX_ACCEPTANCE_RATE, --maxAcceptanceRate MAX_ACCEPTANCE_RATE,
  --maxacceptancerate MAX_ACCEPTANCE_RATE,
  --max_acceptance_rate MAX_ACCEPTANCE_RATE


Set the maximum acceptance rate for the McMC algorithm
[mcmc default=0.5, transdmcmc default=0.2]


---------------------------

::

  -w ACCEPTANCE_RATE_WINDOW,
  --acceptanceLearningWindow ACCEPTANCE_RATE_WINDOW,
  --acceptancelearningwindow ACCEPTANCE_RATE_WINDOW


Sets the window for calculating and updating the
acceptance rate for McMC algorithms [default=500]


---------------------------

::

  -W WARNINGS, --warnings WARNINGS, --Warnings WARNINGS

Sets the warning visibility.

options are:

  * "e","error" - turn matching warnings intoexceptions
  * "i","ignore" - never print matching warnings
  * "a","always" - always print matchingwarnings
  * "d","default" - print the first occurrenceof matching warnings for each location where thewarning is issued
  * "m","module" - print the first occurrence ofmatching warnings for each module where the warning isissued
  * "o","once" - print only the first occurrenceof matching warnings, regardless of location



---------------------------

::

  -z LEARNING_LENGTH, --learningLength LEARNING_LENGTH,
  --learninglength LEARNING_LENGTH, --learning_length LEARNING_LENGTH


Sets the number of samples to discard as the learning
period [default=5000]


---------------------------

::

  --version

show program's version number and exit


---------------------------

::

  --mpi_call

.. warning::

	Do not use - automatically set when spawning mpi subprocess


---------------------------

::

  --output-format {matlab,pickle,hyp}, --output_format {matlab,pickle,hyp},
  --outputformat {matlab,pickle,hyp}, --format {matlab,pickle,hyp}


Output file format [default=matlab]


---------------------------

::

  --results-format {full_pdf,hyp}, --results_format {full_pdf,hyp},
  --resultsformat {full_pdf,hyp}


Output results data format (extensible)
[default=full_pdf]


---------------------------

::

  --no-dist, --no_dist, --nodist

Do not output station distribution if running location
samples


---------------------------

::

  --dc-prior DC_PRIOR, --dc_prior DC_PRIOR, --dcprior DC_PRIOR

Prior probability for the double-couple model when
using the Trans-Dimensional McMC algorithm


---------------------------

::

  --sampling SAMPLING, --sampling SAMPLING, --sampling SAMPLING

Random moment tensor sampling distribution


---------------------------

::

  --sample-models SAMPLE_DISTRIBUTION,
  --sample_distribution SAMPLE_DISTRIBUTION, --samplemodels SAMPLE_DISTRIBUTION


Alternate models for random sampling (Monte Carlo
algorithms only)


---------------------------

::

  --sampling-prior SAMPLING_PRIOR, --sampling_prior SAMPLING_PRIOR,
  --samplingprior SAMPLING_PRIOR


Prior probability for the model distribution when
using the McMC algorithm, alternatively the prior
distribution for the source type parameters gamma and
delta for use by the Bayesian evidence calculation for
the MC algorithms


---------------------------

::

  --no-normalise, --no-norm, --no_normalise, --no_norm

Do not normalise the output pdf


---------------------------

::

  --convert

Convert the output MTs to Tape parameters, hudson
parameters and strike dip rakes.


---------------------------

::

  --discard DISCARD

Fraction of maxProbability * total samples to discard
as negligeable.


---------------------------

::

  --mpioutput, --mpi_output, --mpi-output

When the mpi flag -M is used outputs each processor
individually rather than combining


---------------------------

::

  --combine_mpi_output, --combine-mpi-output, --combinempioutput

Combine the mpi output from the mpioutput flag. The
data path corresponds to the root path for the mpi
output


---------------------------

::

  --c_generate, --c-generate, --generate

Generate moment tensor samples in the probability
evaluation


---------------------------

::

  --relative_loop, --relative-loop, --relativeloop, --loop

Loop over independent non-zero samples randomly to
construct joint rather than joint samples



Scatangle:
============================




---------------------------

::

  --bin-scatangle, --binscatangle, --bin_scatangle

Bin the scatangle file to reduce the number of samples
[default=False]. --bin-size Sets the bin size
parameter .


---------------------------

::

  --bin-size BIN_SCATANGLE_SIZE, --binsize BIN_SCATANGLE_SIZE,
  --bin_size BIN_SCATANGLE_SIZE


Sets the scatangle bin size parameter [default=1.0]



Cluster:
============================


---------------------------

::

  Commands for using MTfit on a cluster environment using qsub/PBS


---------------------------

::

  -q, --qsub, --pbs

Flag to set MTfit to submit to cluster


---------------------------

::

  --nodes QSUB_NODES

Set number of nodes to use for job submission.
[default=1]


---------------------------

::

  --ppn QSUB_PPN

Set ppn to use for job submission. [default=8]


---------------------------

::

  --pmem QSUB_PMEM

Set pmem (Gb) to use for job submission.
[default=2Gb]


---------------------------

::

  --email QSUB_M

Set user email address.


---------------------------

::

  --emailoptions QSUB_M

Set PBS -m mail options. Requires email address using
-M. [default=bae]


---------------------------

::

  --name QSUB_N

Set PBS -N job name options. [default=MTfit]


---------------------------

::

  --walltime QSUB_WALLTIME

Set PBS maximum wall time. Needs to be of the form
HH:MM:SS. [default=24:00:00]


---------------------------

::

  --queue QSUB_Q

Set PBS -q Queue options. [default=batch]


---------------------------

::

  --bladeproperties QSUB_BLADE_PROPERTIES

Set desired PBS blade properties. [default=False]


---------------------------

::

  --feature QSUB_BLADE_FEATURE

Set desired Torque feature arguments. [default=False]



.. only:: not latex

    :doc:`run`