#!/usr/bin/env
"""mpi.py

Commented example script for MPI parallel inversion
"""


def run():
    # Get data:
    from example_data import mpi_data
    data = mpi_data()

    try:
        import mpi4py
    except Exception:
        print("\n\n=============================Warning=============================\n\nMPI example cannot be run without mpi4py installed ")
        return

    print("Running MPI example\n\n\tInput data dictionary:")
    # Print data
    print(data)
    # Output Data
    # pickle data using cPickle
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    # data saved to MPI_Example.inv using cPickle
    with open('MPI_Example.inv', 'wb') as f:
        pickle.dump(f, data)
    # Inversion
    # Use subprocess to call MTfit
    import subprocess
    subprocess.call(['MTfit', '-M', '--data_file=MPI_Example.inv',
                     '--algorithm=iterate', '--max_samples=100000'])
    # Equivalent to:
    #  $ MTfit -M --data_file=MPI_Example.inv --algorithm=iterate --max_samples=100000
    # End


if __name__ == "__main__":
    run()
