#!/usr/bin/env
"""double_couple.py

Commented example script for Double-Couple inversion
"""


def run(test=False):
    # test flag is to improve run speed when testing
    # Get data:
    from example_data import double_couple_data
    data = double_couple_data()
    if test:
        import cPickle
        with open('Double_Couple_Example.inv', 'wb') as f:
            cPickle.dump(data, f)
        max_samples = 1000
        algorithm = 'iterate'  # uses an iterative random sampling approach (see :ref:`iterative-sampling-label`)
        parallel = False  # Runs on a dingle thread.
        phy_mem = 1  # uses a soft limit of 1Gb of RAM for estimating the sample sizes (This is only a soft limit, so no errors are thrown if the memory usage increases above this)
        dc = True  # runs the inversion in the double-couple space.
        max_samples = 100000  # runs the inversion for 100,000 samples.
        # Set-up inversion object:
        import sys
        sys.path.insert(0, '../src')
        from mtfit.inversion import Inversion
        inversion_object = Inversion(data_file='Double_Couple_Example.inv', algorithm=algorithm,
                                     parallel=parallel, phy_mem=phy_mem, dc=dc, max_samples=max_samples,
                                     convert=True)  # Create the inversion object with the set parameters.
        inversion_object.forward()
        return
    print "Running Double-Couple example\n\n\tInput data dictionary:"
    # Print data
    print data
    print 'Data is pickled to Double_Couple_Example.inv'
    import cPickle
    with open('Double_Couple_Example.inv', 'wb') as f:
        cPickle.dump(data, f)
    # Set parameters
    algorithm = 'iterate'  # uses an iterative random sampling approach
    parallel = False  # Runs on a dingle thread.
    phy_mem = 1  # uses a soft limit of 1Gb of RAM for estimating the sample sizes (This is only a soft limit, so no errors are thrown if the memory usage increases above this)
    dc = True  # runs the inversion in the double-couple space.
    max_samples = 100000  # runs the inversion for 100,000 samples.
    # Inversion
    # Create the inversion object with the set parameters.
    inversion_object = Inversion(data_file='Double_Couple_Example.inv',
                                 algorithm=algorithm, parallel=parallel,
                                 phy_mem=phy_mem, dc=dc,
                                 max_samples=max_samples, convert=True)
    # Run the forward model based inversion
    inversion_object.forward()
    # End


if __name__ == "__main__":
    run()
