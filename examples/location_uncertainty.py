#!/usr/bin/env
"""location_uncertainty.py

Commented example script for Location Uncertainty Inversion
"""


def run(test=False):
    # test flag is to improve run speed when testing
    # Get data:
    from example_data import location_uncertainty_data, location_uncertainty_angles
    data = location_uncertainty_data()

    print("Running Location Uncertainty example\n\n\tInput data dictionary:")
    # Print data
    print(data)

    # save location_uncertainty_angles
    with open('Location_Uncertainty.scatangle', 'w') as f:
        f.write(location_uncertainty_angles())

    # Set inversion parameters
    algorithm = 'time'  # uses an time limited random sampling approach (see :ref:`iterative-sampling-label`)
    parallel = False  # Doesn't run in multiple threads using :mod:`multiprocessing`.
    phy_mem = 1  # uses a soft limit of 1Gb of RAM for estimating the sample sizes (This is only a soft limit, so no errors are thrown if the memory usage increases above this)
    dc = False  # runs the inversion in the double-couple space.
    max_time = 60  # runs the inversion for 60 seconds.
    inversion_options = 'PPolarity'  # Just uses PPolarity rather than all the data in the dictionary

    if test:
        max_time = 10
        phy_mem = 0.01
    # Set-up inversion object:
    if test:
        import sys
        sys.path.insert(0, '../src')
        from mtfit.inversion import Inversion
    else:
        from mtfit.inversion import Inversion
    # Inversion
    # Location uncertainty path
    location_pdf_file_path = 'Location_Uncertainty.scatangle'
    # Create the inversion object with the set parameters..
    inversion_object = Inversion(data, location_pdf_file_path=location_pdf_file_path,
                                 algorithm=algorithm, parallel=parallel, phy_mem=phy_mem, dc=dc,
                                 inversion_options=inversion_options, max_time=max_time,
                                 convert=True)
    # Run the forward model based inversion
    inversion_object.forward()
    # Run1 End
    # Reduce the number of station samples to increase the
    # number of moment tensor samples tried
    number_station_samples = 10000
    # Create the inversion object with the set parameters..
    inversion_object = Inversion(data, location_pdf_file_path=location_pdf_file_path,
                                 algorithm=algorithm, parallel=parallel, phy_mem=phy_mem, dc=dc,
                                 inversion_options=inversion_options, max_time=max_time,
                                 number_station_samples=number_station_samples, convert=True)
    # Run the forward model based inversion
    inversion_object.forward()
    # End


if __name__ == "__main__":
    run()
