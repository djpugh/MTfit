import unittest

import numpy as np

from mtfit.plot.plot_classes import MTData
from mtfit.utilities.unittest_utils import run_tests as _run_tests
from mtfit.utilities.unittest_utils import debug_tests as _debug_tests
from mtfit.utilities.unittest_utils import TestCase


class MTDataTestCase(TestCase):

    def setUp(self):
        self.MTData = MTData(np.array([[1, 2, 0, 0, 1, 0, 2, -1],
                                       [0, 0, 0, 0, 0, 0, 1, 0],
                                       [-1, -1, 0, 0, -1,
                                           0, 1, 3],
                                       [0, 0, 1, 0, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 1, 0, 0, 0, 0]]))

    def tearDown(self):
        del self.MTData

    def test__set_probability(self):
        self.MTData._set_probability([1])
        self.assertEqual(len(self.MTData.probability), 1)
        with self.assertRaises(ValueError):
            self.MTData._set_probability([1, 2])
        self.MTData._set_probability([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(len(self.MTData.probability), 8)

    def test__check(self):
        new_mts = self.MTData._check(
            np.matrix([[1, 0, 0], [0, 0, 0], [0, 0, -1]]))
        self.assertEqual(new_mts.shape[0], 6)
        self.assertEqual(new_mts.shape[1], 1)
        new_mts = self.MTData._check(np.array([[[1, 0, 0], [0, 0, 0], [
                                     0, 0, -1]], [[1, 0, 0], [0, 0, 0], [0, 0, -1]], [[1, 0, 0], [0, 0, 0], [0, 0, -1]]]))
        self.assertEqual(new_mts.shape[0], 6)
        self.assertEqual(new_mts.shape[1], 3)

    def test___getitem__(self):
        newMTData = self.MTData[:, 0:3]
        self.assertTrue(isinstance(newMTData, MTData))
        self.assertEqual(len(newMTData), 3)
        self.assertEqual(len(self.MTData), 8)
        self.assertEqual(newMTData.strike[0], self.MTData.strike[0])
        self.MTData.u[0]
        newMTData = self.MTData[:, 0:4]
        self.assertTrue(isinstance(newMTData, MTData))
        self.assertEqual(len(newMTData), 4)
        self.assertTrue('u' in newMTData.__dict__.keys() and 'strike' in newMTData.__dict__.keys())
        self.MTData.mean_orientation
        newMTData = self.MTData[:, 0:4]
        self.assertFalse('mean_strike' in newMTData.__dict__.keys() or 'mean_dip' in newMTData.__dict__.keys() or 'mean_rake' in newMTData.__dict__.keys() or
                         'mean_normal' in newMTData.__dict__.keys() or 'var_clustered_rake1' in newMTData.__dict__.keys() or 'cov_clustered_N1' in newMTData.__dict__.keys())

    def test___setitem__(self):
        og = self.MTData.gamma[0]
        self.MTData[:, 0] = np.array([0, 1, 0, -1, -2, 0])
        self.assertEqual(len(self.MTData), 8)
        self.assertNotEqual(self.MTData.gamma[0], og)
        self.assertTrue((self.MTData[:, 0] == np.array([[0, 1, 0, -1, -2, 0]]).T))

    def test___delitem__(self):
        with self.assertRaises(ValueError):
            del self.MTData[:, 0]
        return
        # Not carried out
        og = self.MTData.gamma[0]
        del self.MTData[:, 0]
        self.assertEqual(len(self.MTData), 7)
        self.assertNotEqual(self.MTData.gamma[0], og)
        self.assertTrue(
            (self.MTData[:, 0] == np.array([[2, 0, -1, 0, 0, 0]]).T).all())

    def test___len__(self):
        self.assertEqual(len(self.MTData), 8)
        self.assertEqual(len(self.MTData), self.MTData.MTs.shape[1])

    def test___str__(self):
        self.assertEqual(self.MTData.__str__(), self.MTData.MTs.__str__())

    def test___repr__(self):
        self.assertEqual(self.MTData.__repr__(), self.MTData.MTs.__repr__())

    def test__getattr__(self):
        with self.assertRaises(AttributeError):
            self.MTData.test
        self.assertEqual(self.MTData.gamma[0], 0)
        self.assertEqual(self.MTData.u[0], 0)
        try:
            self.assertEqual(self.MTData.strike[0], 90)
        except Exception:
            self.assertEqual(self.MTData.strike[0], 270)

    def test__get_converted(self):
        self.assertEqual(self.MTData._get_converted(), {})
        self.MTData.gamma
        self.assertEqual(tuple(sorted(self.MTData._get_converted().keys())),
                         ('E', 'N', 'P', 'T', 'delta', 'gamma'))
        self.assertEqual(tuple(sorted(self.MTData._get_converted((slice(None, None, None), 2)).keys())),
                         ('E', 'N', 'P', 'T', 'delta', 'gamma'))
        self.assertAlmostEqual(self.MTData._get_converted((slice(None, None, None), 2))['delta'], 0.0)
        self.assertEqual(self.MTData._get_converted((slice(None, None, None), 2))['E'].shape, (3,))
        self.assertEqual(self.MTData._get_converted((slice(None, None, None), 2))['T'].shape, (3,))

    def test__convert(self):
        test_data = [('gamma', 0.), ('delta', 0.), ('E', np.array([[1.], [0], [-1.]])), ('T', np.matrix([[1], [0], [0]])),
                     ('N', np.matrix([[0], [1], [0]])), ('P', np.matrix(
                         [[0], [0], [1]])), ('u', 0.), ('v', 0.),
                     ('tau', 0.), ('k', 0.), ('strike1', 90), ('dip1',
                                                               45), ('rake1', -90), ('strike2', 270), ('dip2', 45),
                     ('rake2', -90), ('strike', 90), ('dip', 45), ('rake', -
                                                                   90), ('N1', np.matrix([[1/np.sqrt(2)], [0], [1/np.sqrt(2)]])),
                     ('N2', np.matrix([[1/np.sqrt(2)], [0], [-1/np.sqrt(2)]])), ('kappa', np.pi/2), ('h', np.cos(np.pi/4)), ('sigma', -np.pi/2), ('phi1', np.array([[1/np.sqrt(2)], [0.], [1/np.sqrt(2)]])), ('phi2', np.array([[1/np.sqrt(2)], [0.], [-1/np.sqrt(2)]])), ('area_displacement', 1.0), ('explosion', 0.0)]
        for attr, value in test_data:
            if attr in ['T', 'N', 'P', 'E', 'N1', 'N2']:
                self.assertTrue((self.MTData[:, 0]._convert(attr) == value).all())
            else:
                if attr in ['strike1', 'strike2', 'dip1', 'dip2', 'rake1', 'rake2', 'strike', 'dip', 'rake']:
                    try:
                        self.assertAlmostEquals(self.MTData[:, 0]._convert(attr), value, 10)
                    except AssertionError:
                        if '1' in attr:
                            alt_attr = attr.replace('1', '2')
                        else:
                            alt_attr = attr.replace('2', '1')
                        if alt_attr[-1] not in ['1', '2']:
                            alt_attr += '2'
                        value = dict(test_data)[alt_attr]
                        self.assertAlmostEquals(self.MTData[:, 0]._convert(attr), value, 10)
                else:
                    self.assertAlmostEquals(self.MTData[:, 0]._convert(attr), value, 10)
        self.MTData._set_ln_pdf([0.1, 0.2, 0.3, 0.2, 0.1, 0.4, 0.6, 0.8])
        self.MTData.total_number_samples = 10000
        self.assertAlmostEquals(self.MTData.ln_bayesian_evidence, -6.764305312122505)

    def test_cluster_normals(self):
        self.tearDown()
        self.MTData = MTData(np.array([[1, 0.9,   1,   1, 0.8, 1.1, 0.9,   1],
                                       [0,   0,   0,   0,
                                           0,  0,  0,   0],
                                       [-1,  -1, -0.9, -0.8,  -
                                           1, -1, -1, -0.9],
                                       [0,   0, 0.1,
                                           0, 0.2,  0,  0,   0],
                                       [0, 0.1,   0,   0,
                                           0,  0,  0,   0],
                                       [0,   0,   0, 0.2,   0,  0,  0,   0]]))
        self.MTData.cluster_normals()
        self.assertTrue((self.MTData.clustered_N1[2, :] < 0).all())
        self.assertTrue((self.MTData.clustered_N2[2, :] > 0).all())

    def test_mean_orientation(self):
        self.test_cluster_normals()
        [s, d, r] = self.MTData.mean_orientation
        self.assertAlmostEquals(s, 272.91522545)
        self.assertAlmostEquals(d, 45.2947782)
        self.assertAlmostEquals(r, -88.295161590065447)
        self.assertAlmostEquals(self.MTData.var_clustered_rake1, 23.251792031905531)
        self.assertTrue(self.MTData.cov_clustered_N1.max() < 0.005)
        self.MTData._set_probability([0.05, 0.15, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1])
        [s, d, r] = self.MTData.mean_orientation
        self.assertAlmostEquals(s, 274.25373043)
        self.assertAlmostEquals(d, 45.47487435)
        self.assertAlmostEquals(r, -85.908387816157074)
        self.assertAlmostEquals(self.MTData.var_clustered_rake1, 46.782048639043481)
        self.assertTrue(self.MTData.cov_clustered_N1.max() < 0.007)

    def test_mean(self):
        self.test_cluster_normals()
        mts = self.MTData.mean
        mean = np.array([[0.9625],
                         [0.],
                         [-0.95],
                         [0.0375],
                         [0.0125],
                         [0.025]])
        covariance = np.array([[0.00839286,  0.,  0.00214286, -0.00410714, -0.00089286,
                                0.00107143],
                               [0.,  0.,  0.,  0.,  0.,
                                0.],
                               [0.00214286,  0.,  0.00571429, -0.00071429, -0.00071429,
                                0.00428571],
                               [-0.00410714,  0., -0.00071429,  0.00553571, -0.00053571,
                                -0.00107143],
                               [-0.00089286,  0., -0.00071429, -0.00053571,  0.00125,
                                -0.00035714],
                               [0.00107143,  0.,  0.00428571, -0.00107143, -0.00035714,
                                0.005]])
        self.assertAlmostEquals(mts, mean)
        cov = self.MTData.covariance
        self.assertAlmostEquals(cov, covariance)
        self.MTData._set_probability(
            [0.05, 0.15, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1])
        mean = np.array([[0.965],
                         [0.],
                         [-0.92],
                         [0.03],
                         [0.015],
                         [0.06]])
        mts = self.MTData.mean
        self.assertAlmostEquals(mts, mean)
        covariance = np.matrix([[0.00751497,  0.,  0.00335329, -0.00353293, -0.00116766,
                                 0.00251497],
                                [0.,  0.,  0.,  0.,  0.,
                                 0.],
                                [0.00335329,  0.,  0.0091018, -0.00167665, -0.00143713,
                                 0.00862275],
                                [-0.00353293,  0., -0.00167665,  0.00491018, -0.00053892,
                                 -0.00215569],
                                [-0.00116766,  0., -0.00143713, -0.00053892,  0.00152695,
                                 -0.00107784],
                                [0.00251497,  0.,  0.00862275, -0.00215569, -0.00107784,
                                 0.01005988]])
        cov = self.MTData.covariance
        self.assertAlmostEquals(cov, covariance)

    def test_max_probability(self):
        self.test_cluster_normals()
        self.MTData._set_probability(
            [0.05, 0.15, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1])
        max_prob = np.matrix([[1],
                              [0],
                              [-0.8],
                              [0],
                              [0],
                              [0.2]])
        max_p = self.MTData.max_probability
        self.assertAlmostEquals(max_p.MTs, max_prob)
        self.assertAlmostEquals(max_p.MTs, self.MTData.MTs[
                                :, self.MTData.probability == self.MTData.probability.max()])
        self.MTData._set_probability(
            [0.05, 0.15, 0.1, 0.3, 0.1, 0.3, 0.1, 0.1])
        max_prob = np.matrix([[1, 1.1],
                              [0,  0],
                              [-0.8, -1],
                              [0,  0],
                              [0,  0],
                              [0.2,  0]])
        max_p = self.MTData.max_probability
        self.assertAlmostEquals(max_p.MTs, max_prob)
        self.assertAlmostEquals(max_p.MTs, self.MTData.MTs[
                                :, self.MTData.probability == self.MTData.probability.max()])
        max_prob = np.matrix([[1],
                              [0],
                              [-0.8],
                              [0],
                              [0],
                              [0.2]])
        max_p = self.MTData.get_max_probability(single=True)
        self.assertAlmostEquals(max_p.MTs, max_prob)

    def test_get_unique_McMC(self):
        self.tearDown()
        self.MTData = MTData(np.array([[1, 2, 0, 0, 1, 0, 2, -1],
                                       [0, 0, 0, 0, 0, 0, 0, 0],
                                       [-1, -1, 0, 0, -1,
                                       0, -1, 3],
                                       [0, 0, 1, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0]]))
        newMTData = self.MTData.get_unique_McMC()
        self.assertEqual(len(newMTData), 4)
        self.assertEqual(set(newMTData.probability), set([2, 2, 3, 1]))
        self.tearDown()
        self.MTData = MTData(np.array([[1, 2, 0, 0, 1, 0, 1, -1],
                                       [0, 0, 0, 0, 0, 0, 0, 0],
                                       [-1, -1, 0, 0, -1,
                                       0, -1, 3],
                                       [0, 0, 1, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0]]))
        newMTData = self.MTData.get_unique_McMC()
        self.assertEqual(len(newMTData), 4)
        self.assertEqual(set(newMTData.probability), set([3, 1, 3, 1]))
        self.MTData.gamma
        newMTData = self.MTData.get_unique_McMC()
        self.assertEqual(len(newMTData), 4)
        self.assertEqual(set(newMTData.probability), set([3, 1, 3, 1]))
        self.assertEqual(len(newMTData.gamma), 4)
        self.assertIn(0., newMTData.gamma)

    def test_MTindices(self):
        self.assertAlmostEquals(self.MTData.xx, np.array([1, 2, 0, 0, 1, 0, 2, -1]))
        self.assertAlmostEquals(self.MTData.yy, np.array([0, 0, 0, 0, 0, 0, 1, 0]))
        self.assertAlmostEquals(self.MTData.zz, np.array([-1, -1, 0, 0, -1, 0, 1, 3]))
        self.assertAlmostEquals(self.MTData.xy, np.array([0, 0, 1, 0, 0, 1, 0, 1]))
        self.assertAlmostEquals(self.MTData.xz, np.array([0, 0, 0, 0, 0, 0, 0, 0]))
        self.assertAlmostEquals(self.MTData.yz, np.array([0, 0, 0, 1, 0, 0, 0, 0]))


def test_suite(verbosity=2):
    """Returns test suite"""
    global VERBOSITY
    VERBOSITY = verbosity
    suite = [unittest.TestLoader().loadTestsFromTestCase(MTDataTestCase),
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
