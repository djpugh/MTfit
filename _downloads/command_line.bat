@echo "Command line MTfit example script\n"
@echo "Needs MTfit to have been installed to run.\n"
@echo "Making data file - csv_example_file.csv\n"
python make_csv_file.py

@echo "MTfit --version:\n"
:: Output MTfit version
MTfit --version

@echo "Running MTfit from command line:\n"
@echo "MTfit --data_file=csv_example*.csv --algorithm=iterate --max_samples=100000 -b  --inversionoptions=PPolarity"
:: Run MTfit from the command line. Options are:
::    --data_file=csv_example*.inv - use the data files matching csv_example*.inv 
::    --algorithm=iterate - use the iterative algorithm
::    --max_samples=100000 - run for 100,000 samples
::    -b - carry out the inversion for both the double couple constrained and full moment tensor spaces
::    --inversionoptions=PPolarity - carry ouy the inversion using PPolarity data only
::    --convert - convert the solution using MTconvert

MTfit --data_file=csv_example*.csv --algorithm=iterate --max_samples=100000 -b  --inversionoptions=PPolarity --convert

