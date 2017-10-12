import unittest
import os

import numpy as np

from mtfit.utilities.unittest_utils import run_tests as _run_tests
from mtfit.utilities.unittest_utils import debug_tests as _debug_tests
from mtfit.utilities.file_io import pickle_output
from mtfit.plot.core import run
from mtfit.plot.core import MTplot

VERBOSITY = 2


class CoreTestCase(unittest.TestCase):

    def generate_test_file(self):
        # Breaks when MATLAB - not clear why
        pickle_output([{'MTSpace': np.matrix([[1.], [2.], [1.], [2.], [1.], [2.]])}, {}], 'plottest.out')

    def test_run(self):
        self.generate_test_file()
        run(['-d=plottest.out'])
        run(['-p=hudson', '-d=plottest.out'])
        run(['-p=lune', '-d=plottest.out'])
        run(['-p=faultplane', '-d=plottest.out'])
        try:
            os.remove('plottest.out')
        except:
            pass

    def test_plots(self):
        multiMTs = np.array([[1, 2, 0, 0, 1, 0, 2, -1],
                             [0, 0, 0, 0, 0, 0, 1, 0],
                             [-1, -1, 0, 0, -1, 0, 1, 3],
                             [0, 0, 1, 0, 0, 1, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0]])
        multiDCs = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [-1, -1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 1]])
        singleMT = np.array([1, 2, -2, 0, 1, 0])
        print("Single Beachball")
        MTplot(singleMT, nodal_line=True)
        try:
            MTplot(multiMTs, nodal_line=True)
        except ValueError:
            print('Multiple MTs not possible with beachball')

        print('Nodal line plot')
        MTplot(multiMTs, 'faultplane', nodal_line=True)
        print('Lune Plot')
        MTplot(multiMTs, 'lune', color='purple')
        print('Hudson Plot')
        MTplot(multiMTs, 'hudson', color='red')
        print('Riedesel Jordan Plot')
        MTplot(singleMT, 'riedeseljordan', color='purple')
        try:
            MTplot(multiMTs, 'riedeseljordan', color='purple')
        except ValueError:
            print('Multiple MTs not possible with beachball')
        stations = {'azimuth': np.matrix([[231.1], [42.9], [21.2], [256.4], [197.4],
                                          [229.7], [75.6], [187.5], [85.3], [307.5], [355.8], [14.7], [
                                              123.5], [231.8], [45.9], [193.8], [53.7], [218.4], [12.9],
                                          [325.5], [29.4], [278.9], [326.1], [223.7], [42.6], [253.6]]),
                    'names': ['S0271', 'S0649', 'S0484', 'S0263', 'S0142', 'S0244', 'S0415', 'S0065', 'S0362', 'S0450', 'S0534', 'S0641', 'S0155', 'S0162', 'S0650',
                              'S0195', 'S0517', 'S0004', 'S0588', 'S0377', 'S0618', 'S0347', 'S0529', 'S0083', 'S0595', 'S0236'],
                    'takeoff_angle': np.matrix([[154.7], [109.7], [145.4], [122.7], [137.6], [148.1], [122.8], [126.1], [128.2], [137.7], [138.2], [120.2],
                                                [117.], [127.5], [108.2], [147.3], [124.2], [109.8], [128.6], [165.3], [120.5], [149.5], [131.7], [118.2], [117.8], [118.6]]),
                    'polarity': [1, 0, 1, 0, -1, 1, -1, 0, 1, 1, -1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, -1, 1, 1, -1]}
        print("Fault plane")
        MTplot(singleMT, 'faultplane', stations=stations,
               fault_plane=True, color='purple')
        stations['polarity'] = [0.8, 0.5, 0.3, 0.51, -0.95, 0.67, -0.95, 0.51, 0.78, 0.84, -0.65,
                                0.83, 0.51, 0.68, 0.88, 0.91, 0.55, 0.51, 0.95, -0.66, 0.53, 0.66, -0.99, 0.99, 0.66, -0.58]
        print("Fault plane")
        MTplot(singleMT, 'faultplane', stations=stations,
               fault_plane=True, color='purple')
        print('With station_distribution')
        station_distribution = {'distribution': [
            {'azimuth': np.matrix([[231.1], [42.9], [21.2], [256.4], [197.4],
                                   [229.7], [75.6], [187.5], [85.3], [307.5], [355.8], [14.7], [
                                       123.5], [231.8], [45.9], [193.8], [53.7], [218.4], [12.9],
                                   [325.5], [29.4], [278.9], [326.1], [223.7], [42.6], [253.6]]),
             'names': ['S0271', 'S0649', 'S0484', 'S0263', 'S0142', 'S0244', 'S0415', 'S0065', 'S0362', 'S0450', 'S0534', 'S0641', 'S0155', 'S0162', 'S0650',
                       'S0195', 'S0517', 'S0004', 'S0588', 'S0377', 'S0618', 'S0347', 'S0529', 'S0083', 'S0595', 'S0236'],
             'takeoff_angle': np.matrix([[154.7], [109.7], [145.4], [122.7], [137.6], [148.1], [122.8], [126.1], [128.2], [137.7], [138.2], [120.2],
                                         [117.], [127.5], [108.2], [147.3], [124.2], [109.8], [128.6], [165.3], [120.5], [149.5], [131.7], [118.2], [117.8], [118.6]])},
            {'azimuth': np.matrix([[230.9], [43.], [21.3], [256.4], [197.3], [229.6], [75.7], [187.4], [85.3], [307.5], [355.7], [14.8],
                                   [123.5], [231.7], [45.9], [193.6], [53.7], [218.3], [13.], [325.7], [29.5], [278.9], [326.1], [223.7], [42.7], [253.5]]),
                'names': ['S0271', 'S0649', 'S0484', 'S0263', 'S0142', 'S0244', 'S0415', 'S0065', 'S0362', 'S0450', 'S0534', 'S0641', 'S0155', 'S0162', 'S0650',
                          'S0195', 'S0517', 'S0004', 'S0588', 'S0377', 'S0618', 'S0347', 'S0529', 'S0083', 'S0595', 'S0236'],
                'takeoff_angle': np.matrix([[154.8], [109.8], [145.4], [122.8], [137.6], [148.1], [122.8], [126.1], [128.2], [137.8], [138.3], [120.3],
                                            [117.1], [127.6], [108.3], [147.3], [124.2], [109.9], [128.7], [165.4], [120.5], [149.6], [131.8], [118.2], [117.9], [118.7]])}],
            'probability': [504.7, 504.7]}
        stations = {'azimuth': np.matrix([[231.1], [42.9], [21.2], [256.4], [197.4],
                                          [229.7], [75.6], [187.5], [85.3], [307.5], [355.8], [14.7], [
                                              123.5], [231.8], [45.9], [193.8], [53.7], [218.4], [12.9],
                                          [325.5], [29.4], [278.9], [326.1], [223.7], [42.6], [253.6]]),
                    'names': ['S0271', 'S0649', 'S0484', 'S0263', 'S0142', 'S0244', 'S0415', 'S0065', 'S0362', 'S0450', 'S0534', 'S0641', 'S0155', 'S0162', 'S0650',
                              'S0195', 'S0517', 'S0004', 'S0588', 'S0377', 'S0618', 'S0347', 'S0529', 'S0083', 'S0595', 'S0236'],
                    'takeoff_angle': np.matrix([[154.7], [109.7], [145.4], [122.7], [137.6], [148.1], [122.8], [126.1], [128.2], [137.7], [138.2], [120.2],
                                                [117.], [127.5], [108.2], [147.3], [124.2], [109.8], [128.6], [165.3], [120.5], [149.5], [131.7], [118.2], [117.8], [118.6]]),
                    'polarity': [1, 0, 1, 0, -1, 1, -1, 0, 1, 1, -1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, -1, 1, 1, -1]}
        print("Fault plane")
        MTplot(singleMT, 'faultplane', stations=stations,
               station_distribution=station_distribution, fault_plane=True, color='purple')
        MTplot(singleMT, 'faultplane', stations=stations,
               station_distribution=station_distribution, fault_plane=False, color='purple')
        MTplot(singleMT, 'faultplane', stations=stations, show_stations=False,
               station_distribution=station_distribution, fault_plane=True, color='purple')
        MTplot(singleMT, 'faultplane', stations=stations, show_stations=False,
               station_distribution=station_distribution, fault_plane=False, color='purple')
        MTplot(singleMT, 'faultplane', stations={
        }, station_distribution=station_distribution, fault_plane=True, color='purple')
        MTplot(singleMT, 'faultplane', stations={
        }, station_distribution=station_distribution, fault_plane=False, color='purple')
        station_distribution['probability'] = [305, 505]
        MTplot(singleMT, 'faultplane', stations=stations, show_stations=False,
               station_distribution=station_distribution, fault_plane=True, color='purple')
        MTplot(singleMT, 'faultplane', stations=stations, show_stations=False,
               station_distribution=station_distribution, fault_plane=False, color='purple')
        print('With probabilities')
        print("Fault plane")
        MTplot(multiDCs, 'faultplane', probability=np.array(
            [0.2, 0.6, 0.8, 0.1]), fault_plane=True, show_max_likelihood=True, show_mean=True, color='purple')
        MTplot(multiDCs, 'faultplane', probability=np.array(
            [0.2, 0.6, 0.8, 0.1]), fault_plane=True, max_number_planes=2, show_max_likelihood=True, show_mean=True, color='purple')
        print('Lune Plot')
        MTplot(multiMTs, 'lune', probability=np.array(
            [0.2, 0.6, 0.8, 0.1, 0.3, 0.7, 0.1, 0.8]), color='purple')
        print('Hudson Plot')
        MTplot(multiMTs, 'hudson', probability=np.array(
            [0.2, 0.6, 0.8, 0.1, 0.3, 0.7, 0.1, 0.8]), color='red')
        print('Multi plot')
        MTplot([multiMTs, multiDCs], ['hudson', 'fault_plane'], probability=[np.array([0.2, 0.6, 0.8, 0.1, 0.3,
                                                                                       0.7, 0.1, 0.8]), np.array([0.2, 0.6, 0.8, 0.1])], show_max_likelihood=True, show_mean=True, color='red')
        print('Tape hist')
        MTplot(multiMTs, 'tape', hex_bin=0)
        print('Parameter hist')
        MTplot(multiMTs, 'parameter', hex_bin=0, parameter='gamma')
        print('Parameter hist')
        MTplot(multiMTs, 'parameter', save_file='MTINV_plot_save_test.png',
               hex_bin=0, parameter='gamma')
        if not os.path.exists('MTINV_plot_save_test.png'):
            raise ValueError('save_file option has not worked')
        try:
            os.remove('MTINV_plot_save_test.png')
        except:
            pass


def test_suite(verbosity=2):
    """Returns test suite"""
    global VERBOSITY
    VERBOSITY = verbosity
    suite = [unittest.TestLoader().loadTestsFromTestCase(CoreTestCase),
             ]
    suite = unittest.TestSuite(suite)
    return suite


def run_tests(verbosity=2):
    """Run tests"""
    _run_tests(test_suite(verbosity), verbosity)


def debug_tests(verbosity=2):
    """Runs tests with debugging on errors"""
    _debug_tests(test_suite(verbosity))


if __name__ == "__main__":
    # Run tests
    run_tests(verbosity=2)
