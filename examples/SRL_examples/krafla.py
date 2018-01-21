# Import inversion
from MTfit import MTfit
# Get Data - this requires cloning from the GitHub repository
from MTfit.examples.example_data import krafla_event, krafla_location
data = krafla_event()
# Get location pdf file
open('krafla_event.scatangle', 'w').write(krafla_location())
# P Polarity Inversion
data['UID'] += '_ppolarity'
# Set inversion parameters
# Use an iteration random sampling algorithm
algorithm = 'iterate'
# uses a soft memory limit of 1Gb of RAM for estimating the sample sizes
# (This is only a soft limit, so no errors are thrown if the
# memory usage increases above this)
phy_mem = 1
# Set to only use P Polarity data
inversion_options = 'PPolarity'
# Set location uncertainty file path
location_pdf_file_path = 'krafla_event.scatangle'
# Other parameters in the inversion:
# Convert the output to other source parameterisations
# Use 5000 location samples (randomly sampled from PDF) as this
# reduces calculation time (each location sample is equivalent to running
# an additional event) and bin the scatangle file for overlapping samples
# Run in double-couple space only for one hundred thousand samples
# Use MTfit.MTfit function to run the double-couple inverison
MTfit(data, location_pdf_file_path=location_pdf_file_path,
      algorithm=algorithm, parallel=True,
      inversion_options=inversion_options, phy_mem=phy_mem, dc=False,
      max_samples=10000, convert=True, bin_scatangle=True,
      number_location_samples=5000)
# Change max_samples for MT inversion to 100000
# Run the MT inversion object with the set parameters.
MTfit(data, location_pdf_file_path=location_pdf_file_path,
      algorithm=algorithm, parallel=True,
      inversion_options=inversion_options, phy_mem=phy_mem, dc=False,
      max_samples=100000, convert=True, bin_scatangle=True,
      number_location_samples=5000)
