@echo "Command line mtfit example script\n"
@echo "Needs mtfit to have been installed to run.\n"
@echo "Making data file - csv_example_file.csv\n"
python make_csv_file.py

@echo "mtfit --version:\n"
:: Output mtfit version
mtfit --version

@echo "Running mtfit from command line:\n"
@echo "mtfit --data_file=csv_example*.inv --algorithm=iterate --max_samples=100000 -b  --inversionoptions=PPolarity"
:: Run mtfit from the command line. Options are:
::    --data_file=csv_example*.inv - use the data files matching csv_example*.inv 
::    --algorithm=iterate - use the iterative algorithm
::    --max_samples=100000 - run for 100,000 samples
::    -b - carry out the inversion for both the double couple constrained and full moment tensor spaces
::    --inversionoptions=PPolarity - carry ouy the inversion using PPolarity data only
::    --convert - convert the solution using MTconvert

mtfit --data_file=csv_example*.csv --algorithm=iterate --max_samples=100000 -b  --inversionoptions=PPolarity --convert

