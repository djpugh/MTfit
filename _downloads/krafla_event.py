#!/usr/bin/env
"""krafla_event.py

Commented example script for inversion of a krafla event, with more in depth explanation of the parameter choices
"""


def run(case='PPolarity', parallel=True, test=False):
    # Import inversion
    from MTfit import MTfit

    if test:

        # Get Data
        from example_data import krafla_event, krafla_location
        data = krafla_event()
        # Get location pdf file
        with open('krafla_event.scatangle', 'w') as f:
            f.write(krafla_location())
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
            location_pdf_file_path = 'krafla_event.scatangle'
            number_location_samples = 50
            bin_scatangle = True
            MTfit(data, location_pdf_file_path=location_pdf_file_path, algorithm=algorithm, parallel=parallel,
                  inversion_options=inversion_options, phy_mem=phy_mem, dc=dc, max_samples=max_samples,
                  convert=convert, bin_scatangle=bin_scatangle, number_location_samples=number_location_samples)
            MTfit(data, location_pdf_file_path=location_pdf_file_path, algorithm=algorithm, parallel=parallel,
                  inversion_options=inversion_options, phy_mem=phy_mem, dc=not dc, max_samples=max_samples,
                  convert=convert, bin_scatangle=bin_scatangle, number_location_samples=number_location_samples)
        elif case.lower() == 'ppolarityprob':
            data['UID'] += '_ppolarityprob'
            algorithm = 'mcmc'
            parallel = False
            phy_mem = 0.1
            dc_mt = True
            chain_length = 100
            inversion_options = 'PPolarityProb'
            convert = True
            location_pdf_file_path = 'krafla_event.scatangle'
            number_location_samples = 50
            bin_scatangle = True
            MTfit(data, location_pdf_file_path=location_pdf_file_path, algorithm=algorithm, burn_length=100,
                  parallel=parallel, inversion_options=inversion_options, phy_mem=phy_mem, chain_length=chain_length,
                  convert=convert, bin_scatangle=bin_scatangle, dc_mt=dc_mt, number_location_samples=number_location_samples)
            data['UID'] += '_transd'
            algorithm = 'transdmcmc'
            MTfit(data, location_pdf_file_path=location_pdf_file_path, algorithm=algorithm, burn_length=100,
                  parallel=parallel, inversion_options=inversion_options, phy_mem=phy_mem, chain_length=chain_length,
                  convert=convert, bin_scatangle=bin_scatangle, number_location_samples=number_location_samples)
        return
    # Get Data
    from example_data import krafla_event, krafla_location
    data = krafla_event()
    # Get location pdf file
    with open('krafla_event.scatangle', 'w') as f:
        f.write(krafla_location())
    if case.lower() == 'ppolarity':
        # P Polarity Inversion
        # print output data
        print(data['PPolarity'])
        data['UID'] += '_ppolarity'
        # Set inversion parameters
        # Use an iteration random sampling algorithm
        algorithm = 'iterate'
        # Run in parallel if set on command line
        parallel = parallel
        # uses a soft memory limit of 1Gb of RAM for estimating the sample sizes
        # (This is only a soft limit, so no errors are thrown if the memory usage
        #         increases above this)
        phy_mem = 1
        # Run in double-couple space only
        dc = True
        # Run for one hundred thousand samples
        max_samples = 100000
        # Set to only use P Polarity data
        inversion_options = 'PPolarity'
        # Set the convert flag to convert the output to other source parameterisations
        convert = True
        # Set location uncertainty file path
        location_pdf_file_path = 'krafla_event.scatangle'
        # Handle location uncertainty
        # Set number of location samples to use (randomly sampled from PDF) as this
        #    reduces calculation time
        # (each location sample is equivalent to running an additional event)
        number_location_samples = 5000
        # Bin Scatangle File
        bin_scatangle = True
        # Use MTfit.__core__.MTfit function
        MTfit(data, location_pdf_file_path=location_pdf_file_path, algorithm=algorithm,
              parallel=parallel, inversion_options=inversion_options, phy_mem=phy_mem, dc=dc,
              max_samples=max_samples, convert=convert, bin_scatangle=bin_scatangle,
              number_location_samples=number_location_samples)
        # Change max_samples for MT inversion
        max_samples = 1000000
        # Create the inversion object with the set parameters.
        MTfit(data, location_pdf_file_path=location_pdf_file_path, algorithm=algorithm,
              parallel=parallel, inversion_options=inversion_options, phy_mem=phy_mem,
              dc=not dc, max_samples=max_samples, convert=convert,
              bin_scatangle=bin_scatangle, number_location_samples=number_location_samples)
        # Equivalent to pickling the data and outputting the location uncertainty:
        #  >>> from example_data import krafla_event,krafla_location
        #  >>> data=krafla_event()
        #  >>> open('krafla_event.scatangle','w').write(krafla_location())
        #  >>> import cPickle
        #  >>> cPickle.dump(data,open('krafla_event.inv','wb'))
        # And then calling from the command line
        #  MTfit --location_pdf_file_path=krafla_event.scatangle --algorithm=iterate \
        #    --pmem=1 --double-couple --max-samples=100000 \
        #    --inversion-options=PPolarity --convert --bin-scatangle krafla_event.inv
        #  MTfit --location_pdf_file_path=krafla_event.scatangle --algorithm=iterate \
        #    --pmem=1 --max-samples=10000000  --inversion-options=PPolarity --convert \
        #    --bin-scatangle krafla_event.inv
        # End
    elif case.lower() == 'ppolarityprob':
        # Polarity Probability Inversion
        # print output data
        print(data['PPolarityProb'])
        data['UID'] += '_ppolarityprob'
        # Set inversion parameters
        # Use an mcmc sampling algorithm
        algorithm = 'mcmc'
        # Set parallel to false as running McMC
        parallel = False
        # uses a soft memory limit of 0.5Gb of RAM for estimating the sample sizes
        # (This is only a soft limit, so no errors are thrown if the memory usage
        #     increases above this)
        phy_mem = 0.5
        # Run both inversions
        dc_mt = True
        # Run for one hundred thousand samples
        chain_length = 100000
        # Set to only use P Polarity data
        inversion_options = 'PPolarityProb'
        # Set the convert flag to convert the output to other source parameterisations
        convert = True
        # Set location uncertainty file path
        location_pdf_file_path = 'krafla_event.scatangle'
        # Handle location uncertainty
        # Set number of location samples to use (randomly sampled from PDF) as this
        #    reduces calculation time
        # (each location sample is equivalent to running an additional event)
        number_location_samples = 5000
        # Bin Scatangle File
        bin_scatangle = True
        # Use MTfit.__core__.MTfit function
        MTfit(data, location_pdf_file_path=location_pdf_file_path, algorithm=algorithm,
              parallel=parallel, inversion_options=inversion_options, phy_mem=phy_mem,
              chain_length=chain_length, convert=convert, bin_scatangle=bin_scatangle,
              dc_mt=dc_mt, number_location_samples=number_location_samples)
        # Trans-Dimensional inversion
        data['UID'] += '_transd'
        # Use a transdmcmc sampling algorithm
        algorithm = 'transdmcmc'
        # Use MTfit.__core__.MTfit function
        MTfit(data, location_pdf_file_path=location_pdf_file_path, algorithm=algorithm,
              parallel=parallel, inversion_options=inversion_options, phy_mem=phy_mem,
              chain_length=chain_length, convert=convert, bin_scatangle=bin_scatangle,
              number_location_samples=number_location_samples)
        # Equivalent to pickling the data and outputting the location uncertainty:
        #  >>> from example_data import krafla_event,krafla_location
        #  >>> data=krafla_event()
        #  >>> open('krafla_event.scatangle','w').write(krafla_location())
        #  >>> import cPickle
        #  >>> cPickle.dump(data,open('krafla_event_data.inv','wb'))
        # And then calling from the command line
        # MTfit --location_pdf_file_path=krafla_event.scatangle --algorithm=mcmc -b \
        #    --pmem=1 --double-couple --max-samples=100000 \
        #    --inversion-options=PPolarityProb --convert --bin-scatangle \
        #    krafla_event.inv
        # MTfit --location_pdf_file_path=krafla_event.scatangle --algorithm=transdmcmc \
        #    --pmem=1 --max-samples=100000  --inversion-options=PPolarityProb \
        #    --convert --bin-scatangle krafla_event.inv
        # End


if __name__ == "__main__":
    import sys
    case = [i for i, u in enumerate(sys.argv) if 'case=' in u]
    case = sys.argv[case[0]].split('case=')[1] if len(case) else 'ppolarity'
    parallel = '-l' not in sys.argv
    run(case, parallel)
