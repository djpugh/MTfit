# !/usr/bin/env
"""p_sh_amplitude_ratio.py

Commented example script for P/SH Amplitude Ratio inversion
"""


def run(test=False):
    # test flag is to improve run speed when testing
    # Get data:
    from example_data import p_sh_amplitude_ratio_data
    data = p_sh_amplitude_ratio_data()

    print("Running P/SH Amplitude Ratio example\n\n\tInput data dictionary:")
    # Print data
    print(data)

    # Set inversion parameters
    # uses an iterative random sampling approach (see :ref:`iterative-sampling-label`).
    algorithm = 'iterate'
    # tries to run in parallel using Multiprocessing.
    parallel = True
    # uses a soft limit of 1Gb of RAM for estimating the sample sizes (This is only a soft limit, so no errors are thrown if the memory usage increases above this).
    phy_mem = 1
    # runs the full moment tensor inversion .
    dc = False
    # runs the inversion for 1,000,000 samples.
    max_samples = 1000000
    if test:
        max_samples = 1000
    # Set-up inversion object:
    from MTfit.inversion import Inversion
    # Inversion

    # Create the inversion object with the set parameters..
    inversion_object = Inversion(data, algorithm=algorithm, parallel=parallel,
                                 phy_mem=phy_mem, dc=dc, max_samples=max_samples,
                                 convert=True)

    # Run the forward model based inversion
    inversion_object.forward()
    # Run1 End

    # Change UID (and output file name)
    data['UID'] = 'P_SH_Amplitude_Ratio_Example_Time_Output'

    # Time sampling

    # runs the inversion for a given time period.
    algorithm = 'time'
    if test:
        # Run inversion for test
        max_time = 10
    else:
        # Length of time to run for in seconds.
        max_time = 300
    # Create the inversion object with the set parameters..
    inversion_object = Inversion(data, algorithm=algorithm, parallel=parallel,
                                 phy_mem=phy_mem, dc=dc, max_time=max_time, convert=True)
    # Run the forward model based inversion
    inversion_object.forward()
    # Run End


if __name__ == "__main__":
    run()
