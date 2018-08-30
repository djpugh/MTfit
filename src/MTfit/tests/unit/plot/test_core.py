import unittest
import os

import numpy as np

from MTfit.utilities.file_io import pickle_output
from MTfit.plot.core import run
from MTfit.plot.core import MTplot

VERBOSITY = 2

PLOT_DIR = os.environ.get('MTFIT_TEST_PLOT_DIR', None)


class CoreTestCase(unittest.TestCase):

    def setUp(self):
        self.multiMTs = np.array([[1, 2, 0, 0, 1, 0, 2, -1],
                                  [0, 0, 0, 0, 0, 0, 1, 0],
                                  [-1, -1, 0, 0, -1, 0, 1, 3],
                                  [0, 0, 1, 0, 0, 1, 0, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0, 0]])
        self.multiDCs = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [-1, -1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 1]])
        self.singleMT = np.array([1, 2, -2, 0, 1, 0])
        self.station_distribution = {'distribution': [
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
        self.stations = {'azimuth': np.matrix([[231.1], [42.9], [21.2], [256.4], [197.4],
                                               [229.7], [75.6], [187.5], [85.3], [307.5], [355.8], [14.7],
                                               [123.5], [231.8], [45.9], [193.8], [53.7], [218.4], [12.9],
                                               [325.5], [29.4], [278.9], [326.1], [223.7], [42.6], [253.6]]),
                         'names': ['S0271', 'S0649', 'S0484', 'S0263', 'S0142', 'S0244', 'S0415', 'S0065', 'S0362', 'S0450', 'S0534', 'S0641', 'S0155', 'S0162', 'S0650',
                                   'S0195', 'S0517', 'S0004', 'S0588', 'S0377', 'S0618', 'S0347', 'S0529', 'S0083', 'S0595', 'S0236'],
                         'takeoff_angle': np.matrix([[154.7], [109.7], [145.4], [122.7], [137.6], [148.1], [122.8], [126.1], [128.2], [137.7], [138.2], [120.2],
                                                     [117.], [127.5], [108.2], [147.3], [124.2], [109.8], [128.6], [165.3], [120.5], [149.5], [131.7], [118.2], [117.8], [118.6]]),
                         'polarity': [1, 0, 1, 0, -1, 1, -1, 0, 1, 1, -1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, -1, 1, 1, -1]}

    def generate_test_file(self):
        # Breaks when MATLAB - not clear why
        pickle_output([{'MTSpace': np.matrix([[1.], [2.], [1.], [2.], [1.], [2.]])}, {}], 'plottest.out')

    def test_run(self):
        self.generate_test_file()
        save_file, show = self.make_savefile('Run Beachball')
        if show:
            opt = []
        else:
            opt = ['-q', '--save_file='+save_file]
        run(['-d=plottest.out']+opt)
        run(['-p=hudson', '-d=plottest.out']+opt)
        run(['-p=lune', '-d=plottest.out']+opt)
        run(['-p=faultplane', '-d=plottest.out']+opt)
        try:
            os.remove('plottest.out')
        except Exception:
            pass

    def make_savefile(self, plot_name):
        if PLOT_DIR:
            if not os.path.exists(PLOT_DIR):
                os.makedirs(PLOT_DIR)
            return os.path.join(PLOT_DIR, plot_name.lower().replace(' ', '_')), False
        else:
            return '', True

    def test_single_beachball(self):
        plot_name = "Single Beachball"
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.singleMT, nodal_line=True, save_file=save_file, show=show)

    def test_multiple_mts_beachball_fail(self):
        with self.assertRaises(ValueError):
            MTplot(self.multiMTs, nodal_line=True)

    def test_nodal_line_plot(self):
        plot_name = 'Nodal line plot'
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.multiMTs, 'faultplane', nodal_line=True, save_file=save_file, show=show)

    def test_lune_plot(self):
        plot_name = 'Lune Plot'
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.multiMTs, 'lune', color='purple', save_file=save_file, show=show)

    def test_hudson_plot(self):
        plot_name = 'Hudson Plot'
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.multiMTs, 'hudson', color='red', save_file=save_file, show=show)

    def test_riedesel_jordan_plot(self):
        plot_name = 'Riedesel Jordan Plot'
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.singleMT, 'riedeseljordan', color='purple', save_file=save_file, show=show)

    def test_riedesel_jordan_plot_multiple_mts_fail(self):
        with self.assertRaises(ValueError):
            MTplot(self.multiMTs, 'riedeseljordan', color='purple')

    def test_fault_plane_with_stations(self):
        plot_name = "Fault plane with stations"
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.singleMT, 'faultplane', stations=self.stations,
               fault_plane=True, color='purple', save_file=save_file, show=show)

    def test_fault_plane_with_polarity_probability(self):
        plot_name = "Fault plane"
        self.stations['polarity'] = [0.8, 0.5, 0.3, 0.51, -0.95, 0.67, -0.95, 0.51, 0.78, 0.84, -0.65,
                                     0.83, 0.51, 0.68, 0.88, 0.91, 0.55, 0.51, 0.95, -0.66, 0.53, 0.66,
                                     -0.99, 0.99, 0.66, -0.58]

        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.singleMT, 'faultplane', stations=self.stations,
               fault_plane=True, color='purple', save_file=save_file, show=show)

    def test_fault_plane_with_stations_station_dist(self):
        plot_name = 'Fault Plane and Stations with Station Dist'
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.singleMT, 'faultplane', stations=self.stations,
               station_distribution=self.station_distribution, fault_plane=True, color='purple', save_file=save_file, show=show)

    def test_stations_station_dist(self):
        plot_name = 'Stations and Station Dist Only'
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.singleMT, 'faultplane', stations=self.stations,
               station_distribution=self.station_distribution, fault_plane=False, color='purple', save_file=save_file, show=show)

    def test_fault_plane_with_station_dist(self):
        plot_name = 'Fault Plane no Stations with Station Dist'
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.singleMT, 'faultplane', stations=self.stations, show_stations=False,
               station_distribution=self.station_distribution, fault_plane=True, color='purple', save_file=save_file, show=show)

    def test_station_dist(self):
        plot_name = 'Station Dist Only'
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.singleMT, 'faultplane', stations=self.stations, show_stations=False,
               station_distribution=self.station_distribution, fault_plane=False, color='purple', save_file=save_file, show=show)

    def test_fault_plane_with_station_dist_2(self):
        plot_name = 'Fault Plane with Station Dist Only'
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.singleMT, 'faultplane', stations={}, station_distribution=self.station_distribution, fault_plane=True,
               color='purple', save_file=save_file, show=show)

    def test_station_dist_2(self):
        plot_name = 'Station Dist Only 2'
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.singleMT, 'faultplane', stations={}, station_distribution=self.station_distribution, fault_plane=False,
               color='purple', save_file=save_file, show=show)

    def test_fault_plane_with_prob_station_dist(self):
        self.station_distribution['probability'] = [305, 505]

        plot_name = 'Fault Plane with Prob Station Dist Only'
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.singleMT, 'faultplane', stations=self.stations, show_stations=False,
               station_distribution=self.station_distribution, fault_plane=True, color='purple', save_file=save_file, show=show)

    def test_prob_station_dist(self):
        self.station_distribution['probability'] = [305, 505]
        plot_name = 'Prob Station Dist Only'
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.singleMT, 'faultplane', stations=self.stations, show_stations=False,
               station_distribution=self.station_distribution, fault_plane=False, color='purple', save_file=save_file, show=show)
        print('With probabilities')
        print("Fault plane")

    def test_fault_plane_prob_with_prob_station_dist(self):
        self.station_distribution['probability'] = [305, 505]
        plot_name = 'Fault Plane Prob with Prob Station Dist'
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.multiDCs, 'faultplane', probability=np.array([0.2, 0.6, 0.8, 0.1]), fault_plane=True, show_max_likelihood=True,
               show_mean=True, color='purple', save_file=save_file, show=show)

    def test_limited_fault_plane_prob_with_prob_station_dist(self):
        self.station_distribution['probability'] = [305, 505]
        plot_name = '2 Fault Planes with Prob Station Dist '
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.multiDCs, 'faultplane', probability=np.array([0.2, 0.6, 0.8, 0.1]), fault_plane=True, max_number_planes=2,
               show_max_likelihood=True, show_mean=True, color='purple', save_file=save_file, show=show)

    def test_lune_hist(self):
        plot_name = 'Lune Hist'
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.multiMTs, 'lune', probability=np.array(
               [0.2, 0.6, 0.8, 0.1, 0.3, 0.7, 0.1, 0.8]), color='purple', save_file=save_file, show=show)

    def test_lune(self):
        plot_name = 'Lune'
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.singleMT, 'lune', color='purple', save_file=save_file, show=show)

    def test_hudson_hist(self):
        plot_name = 'Hudson'
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.multiMTs, 'hudson', probability=np.array(
            [0.2, 0.6, 0.8, 0.1, 0.3, 0.7, 0.1, 0.8]), color='red', save_file=save_file, show=show)

    def test_hudson(self):
        plot_name = 'Hudson'
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.singleMT, 'hudson', color='blue', save_file=save_file, show=show)

    def test_hudson_fault_plane_multi(self):
        plot_name = 'Hudson and Fault Plane'
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot([self.multiMTs, self.multiDCs], ['hudson', 'fault_plane'], probability=[np.array([0.2, 0.6, 0.8, 0.1, 0.3, 0.7, 0.1, 0.8]),
                                                                                       np.array([0.2, 0.6, 0.8, 0.1])],
               show_max_likelihood=True, show_mean=True, color='red', save_file=save_file, show=show)

    def test_tape_hist(self):
        plot_name = 'Tape histogram'
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.multiMTs, 'tape', hex_bin=0, save_file=save_file, show=show)

    def test_parameter_hist(self):
        plot_name = 'Parameter'
        print(plot_name)
        save_file, show = self.make_savefile(plot_name)
        MTplot(self.multiMTs, 'parameter', hex_bin=0, parameter='gamma', save_file=save_file, show=show)

    def test_output_file(self):
        MTplot(self.multiMTs, 'parameter', save_file='MTfit_plot_save_test.png',
               hex_bin=0, parameter='gamma', show=False)
        self.assertTrue(os.path.exists('MTfit_plot_save_test.png'), os.listdir('.'))
        try:
            os.remove('MTfit_plot_save_test.png')
        except Exception:
            pass
