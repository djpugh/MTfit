#!/usr/bin/env
"""relative_event.py

Commented example script for inversion of a pair of colocated synthetic events including relative amplitudes, with more in depth explanation of the parameter choices
"""


def run(parallel=True, test=False):
    # Import inversion
    try:
        from mtfit.inversion import Inversion
    except:
        import sys
        sys.path.insert(0, '../src')
        from mtfit.inversion import Inversion
    # Get Data
    from example_data import relative_data
    data = relative_data()
    if test:
        # Identical code for running build test
        algorithm = 'iterate'
        parallel = parallel
        phy_mem = 1
        dc = True
        max_samples = 100
        inversion_options = ['PPolarity', 'PAmplitude']
        convert = True
        inversion_object = Inversion(data, algorithm=algorithm, parallel=parallel, inversion_options=inversion_options,
                                     phy_mem=phy_mem, dc=dc, max_samples=max_samples, convert=convert, multiple_events=True,
                                     relative=True)
        inversion_object.forward()
        algorithm = 'mcmc'
        chain_length = 100
        burn_length = 100
        max_samples = 100
        inversion_object = Inversion(data, algorithm=algorithm, parallel=False, inversion_options=inversion_options,
                                     phy_mem=phy_mem, dc=not dc, max_acceptance_rate=0.3, min_acceptance_rate=0.1,
                                     chain_length=chain_length, burn_length=burn_length, convert=convert, multiple_events=True,
                                     relative=True)
        inversion_object.forward()
        return
    # Begin
    # P Polarity and Relative P Amplitude Inversion
    # Set inversion parameters
    # Use an iteration random sampling algorithm
    algorithm = 'iterate'
    # Run in parallel if set on command line
    parallel = parallel
    # uses a soft memory limit of 1Gb of RAM for estimating the sample sizes
    # (This is only a soft limit, so no errors are thrown if the memory usage
    #       increases above this)
    phy_mem = 1
    # Run in double-couple space only
    dc = True
    # Run for 10 million samples - quite coarse for relative inversion of two events
    max_samples = 10000000
    # Set to only use P Polarity data
    inversion_options = ['PPolarity', 'PAmplitude']
    # Set the convert flag to convert the output to other source parameterisations
    convert = True
    # Create the inversion object with the set parameters.
    inversion_object = Inversion(data, algorithm=algorithm, parallel=parallel,
                                 inversion_options=inversion_options, phy_mem=phy_mem, dc=dc,
                                 max_samples=max_samples, convert=convert, multiple_events=True,
                                 relative_amplitude=True)
    # Run the forward model
    inversion_object.forward()
    # Run the full moment tensor inversion
    # Use mcmc algorithm for full mt space as random sampling can require a
    # prohibitive number of samples
    algorithm = 'mcmc'
    # Set McMC parameters
    burn_length = 30000
    chain_length = 100000
    min_acceptance_rate = 0.1
    max_acceptance_rate = 0.3
    # Create the inversion object with the set parameters.
    inversion_object = Inversion(data, algorithm=algorithm, parallel=False,
                                 inversion_options=inversion_options, phy_mem=phy_mem, dc=not dc,
                                 chain_length=chain_length, max_acceptance_rate=max_acceptance_rate,
                                 min_acceptance_rate=min_acceptance_rate, burn_length=burn_length,
                                 convert=convert, multiple_events=True, relative_amplitude=True)
    # Run the forward model
    inversion_object.forward()
    # Equivalent to pickling the data:
    #  >>> from example_data import relative_event
    #  >>> data=relative_event()
    #  >>> import cPickle
    #  >>> cPickle.dump(data,open('relative_event_data.inv','wb'))
    # And then calling from the command line
    #  mtfit --algorithm=iterate --pmem=1 --double-couple --max-samples=10000000 \
    #    --inversion-options=PPolarity,PAmplitude --convert --relative \
    #    --multiple-events relative_event_data.inv
    #  mtfit --algorithm=mcmc --pmem=1 --chain-length=100000  \
    #    --burn_in=30000 --min_acceptance_rate=0.1 \
    #    --max_acceptance_rate=0.3 --inversion-options=PPolarity,PAmplitude \
    #    --convert --relative --multiple-events relative_event_data.inv
    # End


if __name__ == "__main__":
    import sys
    parallel = '-l' not in sys.argv
    run(parallel)
