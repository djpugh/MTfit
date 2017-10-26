#!/usr/bin/env
"""synthetic_event.py

Commented example script for inversion of a synthetic event, with more in depth explanation of the parameter choices
"""


def run(case='PPolarity', parallel=True, test=False):
    # Import inversion
    try:
        from mtfit.inversion import Inversion
    except:
        import sys
        sys.path.insert(0, '../src')
        from mtfit.inversion import Inversion
    # Get Data
    from example_data import synthetic_event
    data = synthetic_event()
    if test:
        # Identical code for running build test
        if case.lower() == 'ppolarity':
            data['UID'] += '_ppolarity'
            algorithm = 'iterate'
            parallel = parallel
            phy_mem = 1
            dc = True
            max_samples = 100
            inversion_options = 'PPolarity'
            convert = True
            inversion_object = Inversion(data, algorithm=algorithm, parallel=parallel, inversion_options=inversion_options,
                                         phy_mem=phy_mem, dc=dc, max_samples=max_samples, convert=convert)
            inversion_object.forward()
            max_samples = 100
            inversion_object = Inversion(data, algorithm=algorithm, parallel=parallel, inversion_options=inversion_options,
                                         phy_mem=phy_mem, dc=not dc, max_samples=max_samples, convert=convert)
            inversion_object.forward()
        elif case.lower() == 'ar':
            data['UID'] += '_ar'
            algorithm = 'iterate'
            parallel = parallel
            phy_mem = 1
            dc = True
            max_samples = 100
            inversion_options = ['PPolarity', 'P/SHRMSAmplitudeRatio', 'P/SVRMSAmplitudeRatio']
            convert = False
            inversion_object = Inversion(data, algorithm=algorithm, parallel=parallel, inversion_options=inversion_options,
                                         phy_mem=phy_mem, dc=dc, max_samples=max_samples, convert=convert)
            inversion_object.forward()
            max_samples = 100
            inversion_object = Inversion(data, algorithm=algorithm, parallel=parallel, inversion_options=inversion_options,
                                         phy_mem=phy_mem, dc=not dc, max_samples=max_samples, convert=convert)
            inversion_object.forward()
        return
    if case.lower() == 'ppolarity':
        # P Polarity Inversion
        # print output data
        print data['PPolarity']
        data['UID'] += '_ppolarity'
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
        # Run for one hundred thousand samples
        max_samples = 100000
        # Set to only use P Polarity data
        inversion_options = 'PPolarity'
        # Set the convert flag to convert the output to other source parameterisations
        convert = True
        # Create the inversion object with the set parameters.
        inversion_object = Inversion(data, algorithm=algorithm, parallel=parallel,
                                     inversion_options=inversion_options, phy_mem=phy_mem, dc=dc,
                                     max_samples=max_samples, convert=convert)
        # Run the forward model
        inversion_object.forward()
        # Run the full moment tensor inversion
        # Increase the max samples due to the larger source space to 10 million samples
        max_samples = 10000000
        # Create the inversion object with the set parameters.
        inversion_object = Inversion(data, algorithm=algorithm, parallel=parallel,
                                     inversion_options=inversion_options, phy_mem=phy_mem, dc=not dc,
                                     max_samples=max_samples, convert=convert)
        # Run the forward model
        inversion_object.forward()
        # Equivalent to pickling the data:
        #  >>> from example_data import synthetic_event
        #  >>> data=synthetic_event()
        #  >>> import cPickle
        #  >>> cPickle.dump(data,open('synthetic_event_data.inv','wb'))
        # And then calling from the command line
        #  mtfit --algorithm=iterate --pmem=1 --double-couple --max-samples=100000 \
        #    --inversion-options=PPolarity --convert synthetic_event_data.inv
        #  mtfit --algorithm=iterate --pmem=1 --max-samples=10000000  \
        #    --inversion-options=PPolarity --convert synthetic_event_data.inv
        # End
    elif case.lower() == 'ar':
        # Amplitude Ratio Inversion
        # print output data
        print data['PPolarity']
        print data['P/SHRMSAmplitudeRatio']
        print data['P/SVRMSAmplitudeRatio']
        data['UID'] += '_ar'
        # Set inversion parameters
        # Use an iteration random sampling algorithm
        algorithm = 'iterate'
        # Run in parallel if set on command line
        parallel = parallel
        # uses a soft memory limit of 1Gb of RAM for estimating the sample sizes
        # (This is only a soft limit, so no errors are thrown if the memory usage
        #     increases above this)
        phy_mem = 1
        # Run in double-couple space only
        dc = True
        # Run for one hundred thousand samples
        max_samples = 100000
        # Set to only use P Polarity data
        inversion_options = ['PPolarity', 'P/SHRMSAmplitudeRatio',
                             'P/SVRMSAmplitudeRatio']
        # Set the convert flag to convert the output to other source parameterisations
        convert = False
        # Create the inversion object with the set parameters.
        inversion_object = Inversion(data, algorithm=algorithm, parallel=parallel,
                                     inversion_options=inversion_options, phy_mem=phy_mem, dc=dc,
                                     max_samples=max_samples, convert=convert)
        # Run the forward model
        inversion_object.forward()
        # Run the full moment tensor inversion
        # Increase the max samples due to the larger source space to 50 million samples
        max_samples = 50000000
        # Create the inversion object with the set parameters.
        inversion_object = Inversion(data, algorithm=algorithm, parallel=parallel,
                                     inversion_options=inversion_options, phy_mem=phy_mem, dc=not dc,
                                     max_samples=max_samples, convert=convert)
        # Run the forward model
        inversion_object.forward()
        # Equivalent to pickling the data:
        #  >>> from example_data import synthetic_event
        #  >>> data=synthetic_event()
        #  >>> import cPickle
        #  >>> cPickle.dump(data,open('synthetic_event_data.inv','wb'))
        # And then calling from the command line
        #  mtfit --algorithm=iterate --pmem=1 --double-couple --max-samples=100000 \
        #    --inversion-options=PPolarity,P/SHRMSAmplitudeRatio,P/SVRMSAmplitudeRatio \
        #    --convert synthetic_event_data.inv
        #  mtfit --algorithm=iterate --pmem=1 --max-samples=50000000  \
        #    --inversion-options=PPolarity,P/SHRMSAmplitudeRatio,P/SVRMSAmplitudeRatio \
        #    --convert synthetic_event_data.inv
        # End


if __name__ == "__main__":
    import sys
    case = [i for i, u in enumerate(sys.argv) if 'case=' in u]
    case = sys.argv[case[0]].split('case=')[1] if len(case) else 'ppolarity'
    parallel = '-l' not in sys.argv
    run(case, parallel)
