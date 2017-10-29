import os
import sys
import unittest
import glob
import subprocess


# this needs to be run in the repository with the examples
examples_path = '.'
sys.path.insert(0, examples_path)
try:
    from double_couple import run as double_couple_run
    from location_uncertainty import run as location_uncertainty_run
    from make_csv_file import run as make_csv_file_run
    from mpi import run as mpi_run
    from p_polarity import run as p_polarity_run
    from p_sh_amplitude_ratio import run as p_sh_amplitude_ratio_run
    from time_inversion import run as time_inversion_run
    from synthetic_event import run as synthetic_event_run
    from krafla_event import run as krafla_event_run
    from relative_event import run as relative_event_run
    in_repo = True
except ImportError:
    in_repo = False


@unittest.skipIf(not in_repo)
class ExamplesTestCase(unittest.TestCase):

    def setUp(self):
        self.csv_files = glob.glob('*.csv')
        self.inv_files = glob.glob('*.inv')
        self.mat_files = glob.glob('*.mat')
        self.log_files = glob.glob('*.log')
        self.mat_e_files = glob.glob('*.mat~')
        self.scatangle_files = glob.glob('*.scatangle')

    def tearDown(self):
        for filename in glob.glob('*.csv'):
            if filename not in self.csv_files:
                try:
                    os.remove(filename)
                except Exception:
                    pass
        for filename in glob.glob('*.inv'):
            if filename not in self.inv_files:
                try:
                    os.remove(filename)
                except Exception:
                    pass
        for filename in glob.glob('*.mat'):
            if filename not in self.mat_files:
                try:
                    os.remove(filename)
                except Exception:
                    pass
        for filename in glob.glob('*.log'):
            if filename not in self.log_files:
                try:
                    os.remove(filename)
                except Exception:
                    pass
        for filename in glob.glob('*.mat~'):
            if filename not in self.mat_e_files:
                try:
                    os.remove(filename)
                except Exception:
                    pass
        for filename in glob.glob('*.scatangle'):
            if filename not in self.scatangle_files:
                try:
                    os.remove(filename)
                except Exception:
                    pass
        del self.csv_files
        del self.inv_files
        del self.mat_files
        del self.log_files
        del self.mat_e_files

    def test_double_couple_run(self):
        double_couple_run(test=True)
        self.assertTrue(os.path.exists('./Double_Couple_Example_OutputDC.mat'))

    def test_location_uncertainty_run(self):
        location_uncertainty_run(test=True)
        self.assertTrue(os.path.exists('./Location_Uncertainty_Example_OutputMT.mat'))

    def test_make_csv_file_run(self):
        self.assertTrue(make_csv_file_run(test=True))
        self.assertTrue(os.path.exists('./csv_example_file.csv'))

    def test_mpi_run(self):
        mpi_run()

    def test_p_polarity_run(self):
        p_polarity_run(test=True)
        self.assertTrue(os.path.exists('./P_Polarity_Example_OutputMT.mat'))
        self.assertTrue(os.path.exists('./P_Polarity_Example_Dense_OutputMT.mat'))

    def test_p_sh_amplitude_ratio_run(self):
        p_sh_amplitude_ratio_run(test=True)
        self.assertTrue(os.path.exists('./P_SH_Amplitude_Ratio_Example_OutputMT.mat'))
        self.assertTrue(os.path.exists('./P_SH_Amplitude_Ratio_Example_Time_OutputMT.mat'))

    def test_time_inversion_run(self):
        time_inversion_run(test=True)
        self.assertTrue(os.path.exists('./Time_Inversion_Example_OutputMT.mat'))
        self.assertTrue(os.path.exists('./Time_Inversion_Example_OutputDC.mat'))

    def test_synthetic_event_run(self):
        # Test it runs without errors
        synthetic_event_run(test=True)
        synthetic_event_run(case='ar', test=True)

    def test_krafla_event_run(self):
        # Test it runs without errors
        krafla_event_run(test=True)
        krafla_event_run(case='ppolarityprob', test=True)

    def test_relative_event_run(self):
        relative_event_run(test=True)

    def test_command_line(self):
        script = 'command_line.sh'
        if sys.platform.startswith('win'):
            script = 'command_line.bat'
            return
        script_path = os.path.join(examples_path, script)
        os.chmod(script_path, 777)
        self.assertEqual(subprocess.call([script_path]), 0)  # Returns 0


def test_suite(verbosity=2):
    return unittest.TestSuite([unittest.TestLoader().loadTestsFromTestCase(ExamplesTestCase)])


def run_tests(verbosity=2):
    """Runs algorithm module tests."""
    suite = test_suite(verbosity)
    return unittest.TextTestRunner(verbosity=4).run(suite)


if __name__ == '__main__':
    run_tests(2)
