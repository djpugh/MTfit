#!/usr/bin/env
"""time_inversion.py

Commented example script for Time limited inversion
"""


def run(test=False):
    # test flag is to improve run speed when testing
    # Get data:
    from example_data import time_inversion_data
    data = time_inversion_data()

    print("Running Time limited example\n\n\tInput data dictionary:")
    # Print data
    print(data)

    # print 'Data is pickled to Double_Couple_Example.inv'
    # import cPickle
    # cPickle.dump(data,open('Double_Couple_Example.inv','wb'))

    # Set parameters
    algorithm = 'time'  # uses an time limited random sampling approach (see :ref:`iterative-sampling-label`)
    parallel = False  # runs in a single thread.
    phy_mem = 1  # uses a soft limit of 1Gb of RAM for estimating the sample sizes (This is only a soft limit, so no errors are thrown if the memory usage increases above this)
    dc = False  # runs the inversion in the double-couple space.
    max_time = 120  # runs the inversion for 120 seconds.
    inversion_options = 'PPolarity,P/SHAmplitudeRatio'  # Just uses PPolarity and P/SHRMS Amplitude Ratios rather than all the data in the dictionary

    if test:
        max_time = 10

    # Set-up inversion object:
    from mtfit.inversion import Inversion
    # Inversion
    # Create the inversion object with the set parameters.
    inversion_object = Inversion(data, algorithm=algorithm, parallel=parallel,
                                 phy_mem=phy_mem, dc=dc, inversion_options=inversion_options,
                                 max_time=max_time, convert=True)

    # Run the forward model based inversion
    inversion_object.forward()
    # Run1 End
    # DC Inversion
    dc = True
    # Create the inversion object with the set parameters.
    inversion_object = Inversion(data, algorithm=algorithm, parallel=parallel,
                                 phy_mem=phy_mem, dc=dc, inversion_options=inversion_options,
                                 max_time=max_time, convert=True)
    # Run the forward model based inversion
    inversion_object.forward()
    # DC End


if __name__ == "__main__":
    run()
