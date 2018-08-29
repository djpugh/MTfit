"""
test_file_io.py
***************

Tests for src/utils/file_io.py
"""

import os
import glob
import sys
import tempfile
import shutil

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np

from MTfit.utilities.unittest_utils import TestCase
from MTfit.utilities.file_io import csv2inv
from MTfit.utilities.file_io import parse_csv
from MTfit.utilities.file_io import _parse_csv_events
from MTfit.utilities.file_io import parse_hyp
from MTfit.utilities.file_io import full_pdf_output_dicts
from MTfit.utilities.file_io import hyp_output_dicts
from MTfit.utilities.file_io import hyp_output
from MTfit.utilities.file_io import read_binary_output
from MTfit.utilities.file_io import read_pickle_output
from MTfit.utilities.file_io import pickle_output
from MTfit.utilities.file_io import MATLAB_output
from MTfit.utilities.file_io import read_matlab_output
from MTfit.utilities.file_io import read_scatangle_output
from MTfit.utilities.file_io import unique_columns
from MTfit.utilities.file_io import convert_keys_to_unicode
from MTfit.utilities.file_io import convert_keys_from_unicode


class IOTestCase(TestCase):

    def setUp(self):
        self.cwd = os.getcwd()
        if sys.version_info >= (3, 0):
            self.tempdir = tempfile.TemporaryDirectory()
            os.chdir(self.tempdir.name)
        else:
            self.tempdir = tempfile.mkdtemp()
            os.chdir(self.tempdir)
        self.existing_csv_files = glob.glob('*.csv')
        self.existing_hyp_files = glob.glob('*.hyp')
        self.existing_out_files = glob.glob('*.out')
        self.existing_mat_files = glob.glob('*.mat')
        self.existing_scatangle_files = glob.glob('*.scatangle')

    def tearDown(self):
        for fname in glob.glob('*.csv'):
            if fname not in self.existing_csv_files:
                try:
                    os.remove(fname)
                except Exception:
                    print('Cannot remove ', fname)
        for fname in glob.glob('*.hyp'):
            if fname not in self.existing_hyp_files:
                try:
                    os.remove(fname)
                except Exception:
                    print('Cannot remove ', fname)
        for fname in glob.glob('*.out'):
            if fname not in self.existing_out_files:
                try:
                    os.remove(fname)
                except Exception:
                    print('Cannot remove ', fname)
        for fname in glob.glob('*.mat'):
            if fname not in self.existing_mat_files:
                try:
                    os.remove(fname)
                except Exception:
                    print('Cannot remove ', fname)
        for fname in glob.glob('*.scatangle'):
            if fname not in self.existing_scatangle_files:
                try:
                    os.remove(fname)
                except Exception:
                    print('Cannot remove ', fname)
        os.chdir(self.cwd)
        if sys.version_info >= (3, 0):
            self.tempdir.cleanup()
        else:
            try:
                shutil.rmtree(self.tempdir)
            except:
                pass

    def station_angles(self):
        out = "504.7\n"
        out += "S0271   231.1   154.7\n"
        out += "S0649   42.9    109.7\n"
        out += "S0484   21.2    145.4\n"
        out += "S0263   256.4   122.7\n"
        out += "S0142   197.4   137.6\n"
        out += "S0244   229.7   148.1\n"
        out += "S0415   75.6    122.8\n"
        out += "S0065   187.5   126.1\n"
        out += "S0362   85.3    128.2\n"
        out += "S0450   307.5   137.7\n"
        out += "S0534   355.8   138.2\n"
        out += "S0641   14.7    120.2\n"
        out += "S0155   123.5   117\n"
        out += "S0162   231.8   127.5\n"
        out += "S0650   45.9    108.2\n"
        out += "S0195   193.8   147.3\n"
        out += "S0517   53.7    124.2\n"
        out += "S0004   218.4   109.8\n"
        out += "S0588   12.9    128.6\n"
        out += "S0377   325.5   165.3\n"
        out += "S0618   29.4    120.5\n"
        out += "S0347   278.9   149.5\n"
        out += "S0529   326.1   131.7\n"
        out += "S0083   223.7   118.2\n"
        out += "S0595   42.6    117.8\n"
        out += "S0236   253.6   118.6\n"
        out += '\n'
        out += "504.7\n"
        out += "S0271   230.9   154.8\n"
        out += "S0649   43      109.8\n"
        out += "S0484   21.3    145.4\n"
        out += "S0263   256.4   122.8\n"
        out += "S0142   197.3   137.6\n"
        out += "S0244   229.6   148.1\n"
        out += "S0415   75.7    122.8\n"
        out += "S0065   187.4   126.1\n"
        out += "S0362   85.3    128.2\n"
        out += "S0450   307.5   137.8\n"
        out += "S0534   355.7   138.3\n"
        out += "S0641   14.8    120.3\n"
        out += "S0155   123.5   117.1\n"
        out += "S0162   231.7   127.6\n"
        out += "S0650   45.9    108.3\n"
        out += "S0195   193.6   147.3\n"
        out += "S0517   53.7    124.2\n"
        out += "S0004   218.3   109.9\n"
        out += "S0588   13      128.7\n"
        out += "S0377   325.7   165.4\n"
        out += "S0618   29.5    120.5\n"
        out += "S0347   278.9   149.6\n"
        out += "S0529   326.1   131.8\n"
        out += "S0083   223.7   118.2\n"
        out += "S0595   42.7    117.9\n"
        out += "S0236   253.5   118.7\n"
        out += "\n"
        return out

    def csv_file(self):
        csv = """UID=123,,,,
PPolarity,,,,
Name,Azimuth,TakeOffAngle,Measured,Error
S001,120,70,1,0.01
S002,160,60,-1,0.02
P/SHRMSAmplitudeRatio,,,,
Name,Azimuth,TakeOffAngle,Measured,Error
S003,110,10,1 2,0.05 0.04
S005,140,10,1 4,0.01 0.02
,,,,
PPolarity ,,,,
Name,Azimuth,TakeOffAngle,Measured,Error
S003,110,10,1,0.05"""
        open('csvtest.csv', 'w').write(csv)

    def hyp_file(self):
        hyp = """NLLOC "/home/djp61/NLLOC/NLLOC_TC2g/loc/obs.20150126.222324.grid0" "LOCATED" "Location completed."
SIGNATURE "   NLLoc:v6.00.0 03Feb2015 15h57m15"
COMMENT "Seiscat Location Run for Synthetics_1D_Grad_Single"
GRID  100 100 100  -5 -5 -0  0.1 0.1 0.1 PROB_DENSITY
SEARCH OCTREE nInitial 4000 nEvaluated 60000 smallestNodeSide 0.007734/0.007734/0.015625 oct_tree_integral 2.874797e-02 scatter_volume 2.867319e-02
HYPOCENTER  x 0.0312109 y 0.0234766 z 3.32031  OT 24.2751  ix -1 iy -1 iz -1
GEOGRAPHIC  OT 2015 01 26  22 23   24.2751  Lat 0.0399196 Long 0.0397218 Depth 3.32031
QUALITY  Pmax 2.85006e+141 MFmin 82.0143 MFmax 86.6127 RMS 0.00958958 Nphs 37 Gap 36.756 Dist 0.264995 Mamp  -9.9 0 Mdur  -9.9 0
VPVSRATIO  VpVsRatio 1.80732  Npair 15  Diff 0.4647
STATISTICS  ExpectX 0.0290815 Y 0.0222124 Z 3.356  CovXX 0.0084406 XY 0.000322566 XZ -0.00225914 YY 0.0102132 YZ -0.00403783 ZZ 0.046807 EllAz1  93.2974 Dip1  3.00778 Len1  0.171202 Az2  182.96 Dip2  -6.39314 Len2  0.185775 Len3  4.089675e-01
STAT_GEOG  ExpectLat 0.0399082 Long 0.0397027 Depth 3.356
TRANSFORM  LAMBERT RefEllipsoid Clarke-1880  LatOrig 0.039707  LongOrig 0.039441  FirstStdParal 0.003533  SecondStdParal 0.075882  RotCW 0.000000
QML_OriginQuality  assocPhCt 37  usedPhCt 37  assocStaCt -1  usedStaCt 22  depthPhCt -1  stdErr 0.00958958  azGap 36.756  secAzGap 58.4897  gtLevel -  minDist 0.264995 maxDist 5.47344 medDist 3.25213
QML_OriginUncertainty  horUnc -1  minHorUnc 0.138862  maxHorUnc 0.153692  azMaxHorUnc 9.99915
FOCALMECH  Hyp  0.0399196 0.0397218 3.32031 Mech  0 0 0 mf  0 nObs 0
PHASE ID Ins Cmp On Pha  FM Date     HrMn   Sec     Err  ErrMag    Coda      Amp       Per  >   TTpred    Res       Weight    StaLoc(X  Y         Z)        SDist    SAzim  RAz  RDip RQual    Tcorr 
S0325  ?    ?    ? P      U 20150126 2223        25 GAU      0.01        -1        -1        -1 >    0.7224    0.0004    1.2751   -0.1600   -0.1600    0.0000    0.2650 226.19 225.8 172.6  9     0.0000
S0273  ?    ?    ? P      U 20150126 2223     25.02 GAU      0.01        -1        -1        -1 >    0.7454   -0.0024    1.2649   -0.1600   -0.7999    0.0000    0.8453 193.07 196.4 161.8  9     0.0000
S0429  ?    ?    ? P      U 20150126 2223     25.04 GAU      0.02        -1        -1        -1 >    0.7623    0.0009    1.2266   -0.1600    1.1199    0.0000    1.1130 350.11 347.0 157.5  9     0.0000
S0374  ?    ?    ? P      U 20150126 2223     25.06 GAU      0.01        -1        -1        -1 >    0.7738    0.0096    1.2421   -1.1200    0.4800    0.0000    1.2384 291.63 295.7 154.8  9     0.0000
S0246  ?    ?    ? P      U 20150126 2223     25.05 GAU      0.02        -1        -1        -1 >    0.7742   -0.0009    1.2220   -0.4800   -1.1199    0.0000    1.2525 204.09 206.3 154.9  9     0.0000
S0246  ?    ?    ? S      ? 20150126 2223     25.64 GAU       0.1        -1        -1        -1 >    1.3626    0.0066    0.7080   -0.4800   -1.1199    0.0000    1.2525 204.09 206.3 154.9  9     0.0000
S0196  ?    ?    ? P      D 20150126 2223      25.1 GAU      0.02        -1        -1        -1 >    0.8191    0.0047    1.2020    0.1600   -1.7598    0.0000    1.7880 175.87 173.8 146.9  9     0.0000
S0196  ?    ?    ? S      ? 20150126 2223     25.74 GAU      0.02        -1        -1        -1 >    1.4416    0.0184    0.9519    0.1600   -1.7598    0.0000    1.7880 175.87 173.8 146.9  9     0.0000
S0198  ?    ?    ? P      U 20150126 2223     25.11 GAU      0.02        -1        -1        -1 >    0.8374   -0.0035    1.1959    0.8000   -1.7598    0.0000    1.9420 156.68 154.8 144.1  9     0.0000
S0198  ?    ?    ? S      ? 20150126 2223     25.77 GAU      0.02        -1        -1        -1 >    1.4739    0.0165    0.9446    0.8000   -1.7598    0.0000    1.9420 156.68 154.8 144.1  9     0.0000
S0384  ?    ?    ? P      U 20150126 2223     25.14 GAU      0.02        -1        -1        -1 >    0.8590    0.0052    1.1858    2.0800    0.4800    0.0000    2.0991  77.44  74.5 141.5  9     0.0000
S0384  ?    ?    ? S      ? 20150126 2223     25.81 GAU      0.02        -1        -1        -1 >    1.5118    0.0190    0.9286    2.0800    0.4800    0.0000    2.0991  77.44  74.5 141.5  9     0.0000
S0528  ?    ?    ? P      U 20150126 2223     25.24 GAU      0.02        -1        -1        -1 >    0.9667   -0.0015    1.1455   -1.7600    2.3998    0.0000    2.9758 322.99 323.2 130.7 10     0.0000
S0528  ?    ?    ? S      ? 20150126 2223     25.99 GAU      0.02        -1        -1        -1 >    1.7014    0.0113    0.8798   -1.7600    2.3998    0.0000    2.9758 322.99 323.2 130.7 10     0.0000
S0413  ?    ?    ? P      D 20150126 2223     25.26 GAU      0.02        -1        -1        -1 >    0.9854    0.0000    1.1383    3.0400    0.7999    0.0000    3.1074  75.53  74.1 128.3  9     0.0000
S0413  ?    ?    ? S      ? 20150126 2223     26.03 GAU      0.02        -1        -1        -1 >    1.7343    0.0188    0.8624    3.0400    0.7999    0.0000    3.1074  75.53  74.1 128.3  9     0.0000
S0527  ?    ?    ? P      U 20150126 2223     25.27 GAU      0.02        -1        -1        -1 >    0.9955   -0.0000    1.1343   -2.0800    2.3998    0.0000    3.1787 318.38 319.0 128.5 10     0.0000
S0527  ?    ?    ? S      ? 20150126 2223     26.04 GAU      0.02        -1        -1        -1 >    1.7521    0.0111    0.8650   -2.0800    2.3998    0.0000    3.1787 318.38 319.0 128.5 10     0.0000
S0465  ?    ?    ? P      U 20150126 2223     25.28 GAU      0.02        -1        -1        -1 >    1.0197   -0.0141    1.1080    3.0400    1.4399    0.0000    3.3255  64.79  64.3 125.8  9     0.0000
S0465  ?    ?    ? S      ? 20150126 2223     26.08 GAU      0.02        -1        -1        -1 >    1.7948    0.0088    0.8541    3.0400    1.4399    0.0000    3.3255  64.79  64.3 125.8  9     0.0000
S0516  ?    ?    ? P      U 20150126 2223      25.3 GAU      0.01        -1        -1        -1 >    1.0278   -0.0020    1.1450    2.7200    2.0798    0.0000    3.3850  52.59  52.3 125.8 10     0.0000
S0516  ?    ?    ? S      ? 20150126 2223     26.11 GAU      0.02        -1        -1        -1 >    1.8090    0.0249    0.8332    2.7200    2.0798    0.0000    3.3850  52.59  52.3 125.8 10     0.0000
S0579  ?    ?    ? P      U 20150126 2223     25.34 GAU      0.02        -1        -1        -1 >    1.0723   -0.0061    1.1011   -2.0800    3.0397    0.0000    3.6817 325.01 325.7 122.3 10     0.0000
S0579  ?    ?    ? S      ? 20150126 2223     26.16 GAU      0.02        -1        -1        -1 >    1.8873   -0.0029    0.8298   -2.0800    3.0397    0.0000    3.6817 325.01 325.7 122.3 10     0.0000
S0211  ?    ?    ? P      U 20150126 2223     25.33 GAU      0.02        -1        -1        -1 >    1.0709   -0.0148    1.0874   -3.3601   -1.4399    0.0000    3.6935 246.66 246.4 121.9  9     0.0000
S0211  ?    ?    ? S      ? 20150126 2223     26.16 GAU      0.02        -1        -1        -1 >    1.8848   -0.0004    0.8308   -3.3601   -1.4399    0.0000    3.6935 246.66 246.4 121.9  9     0.0000
S0207  ?    ?    ? P      D 20150126 2223     25.39 GAU      0.02        -1        -1        -1 >    1.1348   -0.0182    1.0558    3.6801   -1.7598    0.0000    4.0613 116.05 115.7 117.6  9     0.0000
S0207  ?    ?    ? S      ? 20150126 2223     26.28 GAU      0.02        -1        -1        -1 >    1.9974    0.0082    0.7992    3.6801   -1.7598    0.0000    4.0613 116.05 115.7 117.6  9     0.0000
S0287  ?    ?    ? S      ? 20150126 2223     26.27 GAU      0.02        -1        -1        -1 >    1.9840    0.0115    0.8010   -4.0000   -0.4800    0.0000    4.0626 262.88 261.7 117.6  9     0.0000
S0046  ?    ?    ? P      U 20150126 2223     25.43 GAU      0.02        -1        -1        -1 >    1.1596   -0.0025    1.0699    2.0800   -3.6796    0.0000    4.2321 151.05 150.8 116.3  9     0.0000
S0046  ?    ?    ? S      ? 20150126 2223     26.32 GAU      0.02        -1        -1        -1 >    2.0409    0.0051    0.7995    2.0800   -3.6796    0.0000    4.2321 151.05 150.8 116.3  9     0.0000
S0669  ?    ?    ? P      U 20150126 2223     25.44 GAU      0.02        -1        -1        -1 >    1.1753   -0.0081    1.0596    1.7600    3.9996    0.0000    4.3357  23.50  23.4 115.2  9     0.0000
S0208  ?    ?    ? P      U 20150126 2223     25.44 GAU      0.02        -1        -1        -1 >    1.1777   -0.0105    1.0556    4.0001   -1.7598    0.0000    4.3511 114.20 114.0 115.1  9     0.0000
S0208  ?    ?    ? S      ? 20150126 2223     26.35 GAU       0.1        -1        -1        -1 >    2.0729    0.0034    0.6289    4.0001   -1.7598    0.0000    4.3511 114.20 114.0 115.1  9     0.0000
S0675  ?    ?    ? P      U 20150126 2223     25.61 GAU      0.02        -1        -1        -1 >    1.3584   -0.0195    0.9755    3.6801    3.9996    0.0000    5.3967  42.54  42.2 107.0  9     0.0000
S0675  ?    ?    ? S      ? 20150126 2223     26.67 GAU      0.02        -1        -1        -1 >    2.3909    0.0086    0.7983    3.6801    3.9996    0.0000    5.3967  42.54  42.2 107.0  9     0.0000
S0002  ?    ?    ? S      ? 20150126 2223     26.66 GAU      0.02        -1        -1        -1 >    2.4029   -0.0135    0.7945   -3.6800   -3.9996    0.0000    5.4734 222.69 222.2 106.8  9     0.0000
END_PHASE
END_NLLOC
"""
        open('hyptest.hyp', 'w').write(hyp)

    def csv_events(self):
        csv = [['UID=123,,,,', 'PPolarity,,,,', 'Name,Azimuth,TakeOffAngle,Measured,Error', 'S001,120,70,1,0.01',
                'S002,160,60,-1,0.02', 'P/SHRMSAmplitudeRatio,,,,', 'Name,Azimuth,TakeOffAngle,Measured,Error',
                'S003,110,10,1 2,0.05 0.04', 'S005,140,10,1 4,0.01 0.02'], ['PPolarity ,,,,', 'Name,Azimuth,TakeOffAngle,Measured,Error', 'S003,110,10,1,0.05']]
        return csv

    def test_csv2inv(self):
        self.csv_file()
        self.assertTrue(os.path.exists('csvtest.csv'))
        csv2inv('csvtest.csv')
        self.assertTrue(os.path.exists('csvtest.inv'))
        d = pickle.load(open('csvtest.inv', 'rb'))
        self.assertEqual(len(d), 2)
        self.assertEqual(d[0]['UID'], '123')
        self.assertEqual(d[0]['PPolarity']['Stations']['Name'], ['S001', 'S002'])
        self.assertEqual(d[0]['PPolarity']['Measured'][0, 0], 1)
        self.assertEqual(d[0]['PPolarity']['Measured'][1, 0], -1)
        self.assertEqual(sorted(d[0].keys()), ['P/SHRMSAmplitudeRatio', 'PPolarity', 'UID'])
        self.assertEqual(d[0]['P/SHRMSAmplitudeRatio']['Error'][0, 0], 0.05)
        self.assertEqual(d[0]['P/SHRMSAmplitudeRatio']['Error'][0, 1], 0.04)
        self.assertEqual(d[0]['P/SHRMSAmplitudeRatio']['Error'][1, 0], 0.01)
        self.assertEqual(d[0]['P/SHRMSAmplitudeRatio']['Error'][1, 1], 0.02)
        self.assertEqual(d[1]['UID'], '2')
        os.remove('csvtest.csv')
        os.remove('csvtest.inv')

    def test_parse_csv(self):
        self.csv_file()
        d = parse_csv('csvtest.csv')
        self.assertEqual(len(d), 2)
        self.assertEqual(d[0]['UID'], '123')
        self.assertEqual(d[0]['PPolarity']['Stations']['Name'], ['S001', 'S002'])
        self.assertEqual(d[0]['PPolarity']['Measured'][0, 0], 1)
        self.assertEqual(d[0]['PPolarity']['Measured'][1, 0], -1)
        self.assertEqual(sorted(d[0].keys()), ['P/SHRMSAmplitudeRatio', 'PPolarity', 'UID'])
        self.assertEqual(d[0]['P/SHRMSAmplitudeRatio']['Error'][0, 0], 0.05)
        self.assertEqual(d[0]['P/SHRMSAmplitudeRatio']['Error'][0, 1], 0.04)
        self.assertEqual(d[0]['P/SHRMSAmplitudeRatio']['Error'][1, 0], 0.01)
        self.assertEqual(d[0]['P/SHRMSAmplitudeRatio']['Error'][1, 1], 0.02)
        self.assertEqual(d[1]['UID'], '2')

    def test__parse_csv_events(self):
        d = _parse_csv_events(self.csv_events())
        self.assertEqual(len(d), 2)
        self.assertEqual(d[0]['UID'], '123')
        self.assertEqual(d[0]['PPolarity']['Stations']['Name'], ['S001', 'S002'])
        self.assertEqual(d[0]['PPolarity']['Measured'][0, 0], 1)
        self.assertEqual(d[0]['PPolarity']['Measured'][1, 0], -1)
        self.assertEqual(sorted(d[0].keys()), ['P/SHRMSAmplitudeRatio', 'PPolarity', 'UID'])
        self.assertEqual(d[0]['P/SHRMSAmplitudeRatio']['Error'][0, 0], 0.05)
        self.assertEqual(d[0]['P/SHRMSAmplitudeRatio']['Error'][0, 1], 0.04)
        self.assertEqual(d[0]['P/SHRMSAmplitudeRatio']['Error'][1, 0], 0.01)
        self.assertEqual(d[0]['P/SHRMSAmplitudeRatio']['Error'][1, 1], 0.02)
        self.assertEqual(d[1]['UID'], '2')

    def test_parse_hyp(self):
        self.hyp_file()
        events = parse_hyp('hyptest.hyp')
        self.assertEqual(len(events), 1)
        self.assertTrue('PPolarity' in events[0].keys())
        self.assertTrue('UID' in events[0].keys())
        self.assertTrue('hyp_file' in events[0].keys())
        self.assertEqual(events[0]['UID'], "20150126222324275")
        self.assertEqual(len(events[0].keys()), 3)

    def test_full_pdf_output_dicts(self):
        self.hyp_file()
        events = parse_hyp('hyptest.hyp')
        event = events[0]
        mdict, sdict = full_pdf_output_dicts(event, False, {'moment_tensor_space': np.matrix([[1., 2.], [2., 1.], [1., 2.], [2., 1.], [
                                             1., 2.], [2., 1.]]), 'ln_pdf': np.matrix([0, 0.7]), 'dV': 1, 'probability': np.matrix([[1., 2.]]), 'total_number_samples': 400})
        self.assertTrue('Other' in mdict.keys())
        self.assertTrue('Events' in mdict.keys())
        self.assertTrue('Stations' in mdict.keys())
        self.assertTrue('MTSpace' in mdict['Events'].keys())
        self.assertTrue('Probability' in mdict['Events'].keys())
        self.assertTrue('ln_pdf' in mdict['Events'].keys())
        self.assertTrue('UID' in mdict['Events'].keys())
        self.assertTrue('NSamples' in mdict['Events'].keys())
        self.assertEqual(mdict['Stations'].shape, (20, 4))
        event['PPolarityProbability'] = event['PPolarity'].copy()
        event['PPolarityProbability']['Measured'] = np.matrix([[0.6, 0.4],
                                                               [0.7, 0.3],
                                                               [0.8, 0.2],
                                                               [0.67, 0.33],
                                                               [0.94, 0.06],
                                                               [0.32, 0.68],
                                                               [0.96, 0.04],
                                                               [0.76, 0.24],
                                                               [0.82, 0.18],
                                                               [0.12, 0.88],
                                                               [0.57, 0.43],
                                                               [0.68, 0.32],
                                                               [0.51, 0.49],
                                                               [0.68, 0.32],
                                                               [0.50, 0.50],
                                                               [0.02, 0.98],
                                                               [0.6, 0.4],
                                                               [0.7, 0.3],
                                                               [0.8, 0.2],
                                                               [0.67, 0.33]
                                                               ])
        event.pop('PPolarity')
        mdict, sdict = full_pdf_output_dicts(event, ['PPolarityProbability'], {'moment_tensor_space': np.matrix([[1., 2.], [2., 1.], [1., 2.], [
                                             2., 1.], [1., 2.], [2., 1.]]), 'ln_pdf': np.matrix([0, 0.7]), 'dV': 1, 'probability': np.matrix([[1., 2.]]), 'total_number_samples': 400})
        self.assertTrue('Other' in mdict.keys())
        self.assertTrue('Events' in mdict.keys())
        self.assertTrue('Stations' in mdict.keys())
        self.assertTrue('MTSpace' in mdict['Events'].keys())
        self.assertTrue('Probability' in mdict['Events'].keys())
        self.assertTrue('ln_pdf' in mdict['Events'].keys())
        self.assertTrue('UID' in mdict['Events'].keys())
        self.assertTrue('NSamples' in mdict['Events'].keys())
        self.assertEqual(mdict['Stations'].shape, (20, 4))
        self.assertTrue(all(mdict['Stations'][:, 3] == np.array(
            [0.6, 0.7, 0.8, 0.67, 0.94, -0.68, 0.96, 0.76, 0.82, -0.88, 0.57, 0.68, 0.51, 0.68, 0.5, -0.98, 0.6, 0.7, 0.8, 0.67])))
        mdict, sdict = full_pdf_output_dicts(event, False, {'moment_tensor_space': np.matrix([[1., 2.], [2., 1.], [1., 2.], [2., 1.], [
                                             1., 2.], [2., 1.]]), 'ln_pdf': np.matrix([0, 0.7]), 'dV': 1, 'probability': np.matrix([[1., 2.]]), 'total_number_samples': 400})
        self.assertTrue('Other' in mdict.keys())
        self.assertTrue('Events' in mdict.keys())
        self.assertTrue('Stations' in mdict.keys())
        self.assertTrue('MTSpace' in mdict['Events'].keys())
        self.assertTrue('Probability' in mdict['Events'].keys())
        self.assertTrue('ln_pdf' in mdict['Events'].keys())
        self.assertTrue('UID' in mdict['Events'].keys())
        self.assertTrue('NSamples' in mdict['Events'].keys())
        self.assertEqual(mdict['Stations'].shape, (20, 4))

    def test_hyp_output_dicts(self):
        self.hyp_file()
        events = parse_hyp('hyptest.hyp')
        event = events[0]
        a, b, c = hyp_output_dicts(event, False, {'moment_tensor_space': np.matrix([[1., 2.], [2., 1.], [1., 2.], [2., 1.], [
                                   1., 2.], [2., 1.]]), 'ln_pdf': np.matrix([0, 0.7]), 'probability': np.matrix([[1., 2.]]), 'total_number_samples': 400})
        self.assertEqual(len(a.split('\n')), len(event['hyp_file'])+1)
        self.assertTrue('MOMENTTENSOR' in a)
        self.assertEqual(float(a.split()[a.split().index('MTNN')+1]), 2.0)
        self.assertEqual(len(b), 169)
        events = parse_hyp('hyptest.hyp')
        event = events[0]
        a, b, c = hyp_output_dicts(event, False, {'moment_tensor_space': np.matrix([[1., 2., 3.], [2., 1., 3.], [1., 2., 3.], [2., 1., 3.], [
                                   1., 2., 3.], [2., 1., 3.]]), 'ln_pdf': np.matrix([0, 0.7, 0]), 'probability': np.matrix([[1., 2., 1.]]), 'total_number_samples': 400})
        self.assertEqual(len(a.split('\n')), len(event['hyp_file'])+1)
        self.assertTrue('MOMENTTENSOR' in a)
        self.assertEqual(float(a.split()[a.split().index('MTNN')+1]), 2.0)
        self.assertEqual(len(b), 233)
        events = parse_hyp('hyptest.hyp')
        event = events[0]
        event['hyp_file'].pop(14)
        a, b, c = hyp_output_dicts(event, False, {'moment_tensor_space': np.matrix([[1., 2., 3.], [2., 1., 3.], [1., 2., 3.], [2., 1., 3.], [
                                   1., 2., 3.], [2., 1., 3.]]), 'ln_pdf': np.matrix([0, 0.7, 0]), 'probability': np.matrix([[1., 2., 1.]]), 'total_number_samples': 400})
        self.assertEqual(len(a.split('\n')), len(event['hyp_file'])+2)
        self.assertTrue('MOMENTTENSOR' in a)
        self.assertEqual(float(a.split()[a.split().index('MTNN')+1]), 2.0)
        self.assertEqual(len(b), 233)
        event.pop('hyp_file')
        a, b, c = hyp_output_dicts(event, False, {'moment_tensor_space': np.matrix([[1., 2., 3.], [2., 1., 3.], [1., 2., 3.], [2., 1., 3.], [
                                   1., 2., 3.], [2., 1., 3.]]), 'ln_pdf': np.matrix([0, 0.7, 0]), 'probability': np.matrix([[1., 2., 1.]]), 'total_number_samples': 400})
        self.assertEqual(len(a.split('\n')), 36)
        self.assertTrue('MOMENTTENSOR' in a)
        self.assertEqual(float(a.split()[a.split().index('MTNN')+1]), 2.0)
        self.assertEqual(len(b), 233)
        events = parse_hyp('hyptest.hyp')
        event = events[0]
        event['hyp_file'].pop(14)
        a, b, c = hyp_output_dicts(event, False, {'moment_tensor_space': np.matrix([[1., -0.51969334, 3.], [2., 0.22610635, 3.], [1., 0.29358698, 3.], [2., 0.58532165, 3.], [
                                   1., -0.27015115, 3.], [2., -0.42073549, 3.]]), 'ln_pdf': np.matrix([0, 0.7, 0]), 'probability': np.matrix([[1., 2., 1.]]), 'total_number_samples': 400})
        self.assertEqual(len(a.split('\n')), len(event['hyp_file'])+2)
        self.assertFalse('MOMENTTENSOR' in a)
        self.assertTrue('FOCALMECH' in a)
        self.assertAlmostEqual(
            float(a.split()[a.split().index('Mech')+1]), 0.5*180/np.pi, 5)
        self.assertEqual(len(b), 233)
        a, b, c = hyp_output_dicts(event, False, {'moment_tensor_space': np.matrix([[1., -0.51969334, 3.], [2., 0.22610635, 3.], [1., 0.29358698, 3.], [2., 0.58532165, 3.], [1., -0.27015115, 3.], [2., -0.42073549, 3.]]),
                                                  'probability': np.matrix([[1., 2., 1.]]), 'ln_pdf': np.matrix([0, 0.7, 0]), 'total_number_samples': 400, 'g': np.array([0.2, 0.2, 0.2]), 'd': np.array([0.2, 0.2, 0.2]), 'k': np.array([0.2, 0.2, 0.2]),
                                                  'h': np.array([0.2, 0.2, 0.2]), 's': np.array([0.2, 0.2, 0.2]), 'S1': np.array([0.2, 0.2, 0.2]), 'D1': np.array([0.2, 0.2, 0.2]), 'R1': np.array([0.2, 0.2, 0.2]), 'u': np.array([0.2, 0.2, 0.2]),
                                                  'v': np.array([0.2, 0.2, 0.2]), 'S2': np.array([0.2, 0.2, 0.2]), 'D2': np.array([0.2, 0.2, 0.2]), 'R2': np.array([0.2, 0.2, 0.2]), 'ln_bayesian_evidence': (1, 10)})
        self.assertEqual(len(a.split('\n')), len(event['hyp_file'])+2)
        self.assertFalse('MOMENTTENSOR' in a)
        self.assertTrue('FOCALMECH' in a)
        self.assertAlmostEqual(
            float(a.split()[a.split().index('Mech')+1]), 0.5*180/np.pi, 5)
        self.assertEqual(len(b), 545)

    def test_hyp_output(self):
        self.hyp_file()
        events = parse_hyp('hyptest.hyp')
        event = events[0]
        a, b, c = hyp_output_dicts(event, False, {'moment_tensor_space': np.matrix([[1., 2.], [2., 1.], [1., 2.], [2., 1.], [
                                   1., 2.], [2., 1.]]), 'ln_pdf': np.matrix([0, 0.7, 0]), 'probability': np.matrix([[1., 2.]]), 'total_number_samples': 400})
        try:
            os.remove('MTfitOUTPUTTEST.hyp')
        except Exception:
            pass
        try:
            os.remove('MTfitOUTPUTTEST.mt')
        except Exception:
            pass
        self.assertFalse(os.path.exists('MTfitOUTPUTTEST.hyp'))
        self.assertFalse(os.path.exists('MTfitOUTPUTTEST.mt'))
        fid, out_str = hyp_output([a, b], fid='MTfitOUTPUTTEST.hyp')
        self.assertTrue(os.path.exists('MTfitOUTPUTTEST.hyp'))
        self.assertTrue(os.path.exists('MTfitOUTPUTTEST.mt'))
        try:
            os.remove('MTfitOUTPUTTEST.hyp')
        except Exception:
            pass
        try:
            os.remove('MTfitOUTPUTTEST.mt')
        except Exception:
            pass
        event['hyp_file'].pop(14)
        a, b, c = hyp_output_dicts(event, False, {'moment_tensor_space': np.matrix([[1., -0.51969334, 3.], [2., 0.22610635, 3.], [1., 0.29358698, 3.], [2., 0.58532165, 3.], [1., -0.27015115, 3.], [2., -0.42073549, 3.]]),
                                                  'probability': np.matrix([[1., 2., 1.]]), 'ln_pdf': np.matrix([0, 0.7, 0]), 'total_number_samples': 400, 'g': np.array([0.2, 0.2, 0.2]), 'd': np.array([0.2, 0.2, 0.2]), 'k': np.array([0.2, 0.2, 0.2]),
                                                  'h': np.array([0.2, 0.2, 0.2]), 's': np.array([0.2, 0.2, 0.2]), 'S1': np.array([0.2, 0.2, 0.2]), 'D1': np.array([0.2, 0.2, 0.2]), 'R1': np.array([0.2, 0.2, 0.2]), 'u': np.array([0.2, 0.2, 0.2]),
                                                  'v': np.array([0.2, 0.2, 0.2]), 'S2': np.array([0.2, 0.2, 0.2]), 'D2': np.array([0.2, 0.2, 0.2]), 'R2': np.array([0.2, 0.2, 0.2]), 'ln_bayesian_evidence': 1.+10})
        try:
            os.remove('MTfitOUTPUTTEST.hyp')
        except Exception:
            pass
        try:
            os.remove('MTfitOUTPUTTEST.mt')
        except Exception:
            pass
        self.assertFalse(os.path.exists('MTfitOUTPUTTEST.hyp'))
        self.assertFalse(os.path.exists('MTfitOUTPUTTEST.mt'))
        fid, out_str = hyp_output([a, b], fid='MTfitOUTPUTTEST.hyp')
        self.assertTrue(os.path.exists('MTfitOUTPUTTEST.hyp'))
        self.assertTrue(os.path.exists('MTfitOUTPUTTEST.mt'))
        try:
            os.remove('MTfitOUTPUTTEST.hyp')
        except Exception:
            pass
        try:
            os.remove('MTfitOUTPUTTEST.mt')
        except Exception:
            pass

    def test_read_binary_output(self):
        self.hyp_file()
        events = parse_hyp('hyptest.hyp')
        event = events[0]
        x = {'moment_tensor_space': np.matrix([[1., 2.], [2., 1.], [1., 2.], [2., 1.], [1., 2.], [2., 1.]]), 'dkl': 1.2, 'ln_pdf': np.matrix(
            [0, 0.7]), 'probability': np.matrix([[1., 2.]]), 'total_number_samples': 400}
        a, b, c = hyp_output_dicts(event, False, x)
        try:
            os.remove('MTfitOUTPUTTEST.hyp')
        except Exception:
            pass
        try:
            os.remove('MTfitOUTPUTTEST.mt')
        except Exception:
            pass
        self.assertFalse(os.path.exists('MTfitOUTPUTTEST.hyp'))
        self.assertFalse(os.path.exists('MTfitOUTPUTTEST.mt'))
        fid, out_str = hyp_output([a, b], fid='MTfitOUTPUTTEST.hyp')
        y = read_binary_output('MTfitOUTPUTTEST.mt')
        self.assertTrue(os.path.exists('MTfitOUTPUTTEST.hyp'))
        self.assertTrue(os.path.exists('MTfitOUTPUTTEST.mt'))
        self.assertEqual(sorted(y[0].keys()), sorted(x.keys()))
        self.assertTrue(
            (y[0]['moment_tensor_space'] == x['moment_tensor_space']).all())
        self.assertEqual(y[0]['dkl'], x['dkl'])
        try:
            os.remove('MTfitOUTPUTTEST.hyp')
        except Exception:
            pass
        try:
            os.remove('MTfitOUTPUTTEST.mt')
        except Exception:
            pass
        x = {'moment_tensor_space': np.matrix([[1., -0.51969334, 3.], [2., 0.22610635, 3.], [1., 0.29358698, 3.], [2., 0.58532165, 3.], [1., -0.27015115, 3.], [2., -0.42073549, 3.]]),
             'probability': np.matrix([[1., 2., 1.]]), 'dkl': 2.4, 'ln_pdf': np.matrix([0, 0.7, 0]), 'total_number_samples': 400, 'g': np.array([0.1, 0.2, 0.3]), 'd': np.array([0.2, 0.2, 0.2]), 'k': np.array([0.2, 0.2, 0.2]),
             'h': np.array([0.2, 0.2, 0.2]), 's': np.array([0.2, 0.2, 0.2]), 'S1': np.array([0.2, 0.2, 0.2]), 'D1': np.array([0.2, 0.2, 0.2]), 'R1': np.array([0.2, 0.2, 0.2]), 'u': np.array([0.2, 0.2, 0.2]),
             'v': np.array([0.2, 0.2, 0.2]), 'S2': np.array([0.2, 0.2, 0.2]), 'D2': np.array([0.2, 0.2, 0.2]), 'R2': np.array([0.2, 0.2, 0.2]), 'ln_bayesian_evidence': 1.+10.}
        a, b, c = hyp_output_dicts(event, False, x)
        try:
            os.remove('MTfitOUTPUTTEST.hyp')
        except Exception:
            pass
        try:
            os.remove('MTfitOUTPUTTEST.mt')
        except Exception:
            pass
        self.assertFalse(os.path.exists('MTfitOUTPUTTEST.hyp'))
        self.assertFalse(os.path.exists('MTfitOUTPUTTEST.mt'))
        fid, out_str = hyp_output([a, b], fid='MTfitOUTPUTTEST.hyp')
        y = read_binary_output('MTfitOUTPUTTEST.mt')
        self.assertTrue(os.path.exists('MTfitOUTPUTTEST.hyp'))
        self.assertTrue(os.path.exists('MTfitOUTPUTTEST.mt'))
        self.assertEqual(sorted(y[0].keys()), sorted(x.keys()))
        self.assertTrue((y[0]['g'] == x['g']).all())
        self.assertEqual(
            y[0]['ln_bayesian_evidence'], x['ln_bayesian_evidence'])
        self.assertEqual(y[0]['dkl'], x['dkl'])
        try:
            os.remove('MTfitOUTPUTTEST.hyp')
        except Exception:
            pass
        try:
            os.remove('MTfitOUTPUTTEST.mt')
        except Exception:
            pass

    def test_read_pickle_output(self):
        self.hyp_file()
        events = parse_hyp('hyptest.hyp')
        event = events[0]
        mdict, sdict = full_pdf_output_dicts(event, False, {'moment_tensor_space': np.matrix([[1., 2.], [2., 1.], [1., 2.], [2., 1.], [
                                             1., 2.], [2., 1.]]), 'ln_pdf': np.matrix([0, 0.7]), 'dV': 1, 'probability': np.matrix([[1., 2.]]), 'total_number_samples': 400})
        pickle_output([mdict, sdict], 'hyptest.out')
        event, stations = read_pickle_output('hyptest.out')
        self.assertAlmostEqual(event, mdict['Events'], 10)
        self.assertAlmostEqual(
            stations['azimuth'], mdict['Stations'][:, 1], 10)
        self.assertAlmostEqual(
            stations['takeoff_angle'], mdict['Stations'][:, 2], 10)
        self.assertAlmostEqual(
            stations['polarity'], mdict['Stations'][:, 3], 10)

    def test_read_matlab_output(self):
        self.hyp_file()
        events = parse_hyp('hyptest.hyp')
        event = events[0]
        mdict, sdict = full_pdf_output_dicts(event, False, {'moment_tensor_space': np.matrix([[1., 2.], [2., 1.], [1., 2.], [2., 1.], [
                                             1., 2.], [2., 1.]]), 'ln_pdf': np.matrix([0, 0.7]), 'dV': 1, 'probability': np.matrix([[1., 2.]]), 'total_number_samples': 400})
        MATLAB_output([mdict, sdict], 'hyptest.mat')
        event, stations = read_matlab_output('hyptest.mat')
        self.assertAlmostEqual(event, mdict['Events'], 10)
        self.assertAlmostEqual(
            stations['azimuth'], mdict['Stations'][:, 1], 10)
        self.assertAlmostEqual(
            stations['takeoff_angle'], mdict['Stations'][:, 2], 10)
        self.assertAlmostEqual(
            stations['polarity'], mdict['Stations'][:, 3], 10)

    def test_read_scatangle_output(self):
        open('test_scatangle_file.scatangle', 'w').write(self.station_angles())
        data = read_scatangle_output('test_scatangle_file.scatangle')
        self.assertTrue('distribution' in data.keys())
        self.assertTrue('probability' in data.keys())
        self.assertEqual(len(data['probability']), 2)
        self.assertEqual(len(data['distribution']), 2)
        self.assertTrue('takeoff_angle' in data['distribution'][0].keys())
        self.assertTrue('azimuth' in data['distribution'][0].keys())
        self.assertTrue('names' in data['distribution'][0].keys())


class UtilsTestCase(TestCase):

    def test_unique_columns(self):
        data = np.matrix([[1, 0, 0, 2, 0, 1, 0],
                          [0, 0, 0, -1, 0, 0, 0],
                          [-1, 0, 0, -1, 0, -1, 0],
                          [0, 0, 1, 0, 1, 0, 1],
                          [0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]])
        unique = unique_columns(data)
        self.assertEqual(unique.shape, (6, 4))
        unique, counts = unique_columns(data, counts=True)
        self.assertEqual(unique.shape, (6, 4))
        self.assertEqual(set(counts), set([1, 2, 3]))
        unique, counts, index = unique_columns(data, counts=True, index=True)
        self.assertEqual(unique.shape, (6, 4))
        self.assertEqual(set(counts), set([1, 2, 3]))
        self.assertEqual(set(index), set([3, 0, 2, 1]))
        unique, index = unique_columns(data, counts=False, index=True)
        self.assertEqual(unique.shape, (6, 4))
        self.assertEqual(set(index), set([3, 0, 2, 1]))

    def test_convert_keys_to_unicode(self):
        test = {'a': 1, 'b': {'c': 2}}
        result = convert_keys_to_unicode(test)
        if sys.version_info.major > 2:
            self.assertTrue(all([isinstance(u, str) for u in result.keys()]))
            self.assertTrue(all([isinstance(u, str) for u in result['b'].keys()]))
        else:
            self.assertTrue(all([isinstance(u, unicode) for u in result.keys()]))
            self.assertTrue(all([isinstance(u, unicode) for u in result['b'].keys()]))

    def test_convert_keys_from_unicode(self):
        test = {'a': 1, 'b': {'c': 2}}
        result = convert_keys_to_unicode(test)
        if sys.version_info.major > 2:
            self.assertTrue(all([isinstance(u, str) for u in result.keys()]))
            self.assertTrue(all([isinstance(u, str) for u in result['b'].keys()]))
        else:
            self.assertTrue(all([isinstance(u, unicode) for u in result.keys()]))
            self.assertTrue(all([isinstance(u, unicode) for u in result['b'].keys()]))
        result = convert_keys_from_unicode(result)
        self.assertTrue(all([isinstance(u, str) for u in result.keys()]))
        self.assertTrue(all([isinstance(u, str) for u in result['b'].keys()]))
