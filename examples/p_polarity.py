#!/usr/bin/env
"""p_polarity.py

Commented example script for P Polarity inversion
"""


def run(test=False):
    # test flag is to improve run speed when testing
    # Get data:
    from example_data import p_polarity_data
    data = p_polarity_data()

    print("Running P Polarity example\n\n\tInput data dictionary:")
    # Print data
    print(data)

    # Set inversion parameters
    # uses an iterative random sampling approach (see :ref:`iterative-sampling-label`).
    algorithm = 'iterate'
    # tries to run in parallel using Multiprocessing.
    parallel = True
    # uses a soft limit of 500Mb of RAM for estimating the sample sizes (This is only a soft limit, so no errors are thrown if the memory usage increases above this).
    phy_mem = 0.5
    # runs the full moment tensor inversion.
    dc = False
    # runs the inversion for 1,000,000 samples.
    max_samples = 1000000
    # Test parameters
    if test:
        max_samples = 1000

    # Set-up inversion object:
    from mtfit.inversion import Inversion
    # Inversion
    # Create the inversion object with the set parameters..
    inversion_object = Inversion(data, algorithm=algorithm, parallel=parallel,
                                 phy_mem=phy_mem, dc=dc, max_samples=max_samples,
                                 convert=True)
    # Run the forward model based inversion
    inversion_object.forward()
    # Run1 End
    # Denser sampling
    # Runs the inversion for 10,000,000 samples.
    max_samples = 10000000

    data['UID'] = 'P_Polarity_Example_Dense_Output'  # Change UID (and output file name)

    if test:
        max_samples = 1000

    # Create the inversion object with the set parameters..
    inversion_object = Inversion(data, algorithm=algorithm, parallel=parallel,
                                 phy_mem=phy_mem, dc=dc, max_samples=max_samples,
                                 convert=True)
    # Run the forward model based inversion
    inversion_object.forward()
    # Run end


if __name__ == "__main__":
    run()
