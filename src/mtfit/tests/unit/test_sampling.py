import unittest
import os
import glob
import sys
import tempfile
import shutil

import numpy as np

from MTfit.sampling import Sample
from MTfit.sampling import FileSample
from MTfit.probability.probability import LnPDF


class SampleTestCase(unittest.TestCase):

    def setUp(self):
        self.Sample = Sample()

    def tearDown(self):
        del self.Sample

    def test_append(self):
        moment_tensors = np.matrix([[2], [1], [2], [1], [2], [1]])
        ln_pdf = LnPDF(np.matrix([[1], [2]]))
        self.Sample.append(moment_tensors, ln_pdf, 1)
        self.assertTrue((self.Sample.moment_tensors[:, 0] == np.matrix([[2], [1], [2], [1], [2], [1]])).all())
        self.assertEqual(self.Sample.n, 1)
        self.assertEqual(self.Sample.ln_pdf[0, 0], 1.)
        moment_tensors = np.matrix([[2, 3], [1, 5], [2, 1], [1, 1], [2, 2], [1, 1]])
        ln_pdf = LnPDF(np.matrix([[1, 0], [2, 1]]))
        self.Sample.append(moment_tensors, ln_pdf, 2)
        self.assertTrue((self.Sample.moment_tensors[:, 0:3] == np.matrix([[2, 2, 3], [1, 1, 5], [2, 2, 1], [1, 1, 1], [2, 2, 2], [1, 1, 1]])).all())
        moment_tensors = np.matrix([[2], [1], [2], [1], [2], [1]])
        ln_pdf = LnPDF(np.matrix([[0], [0]]))
        self.Sample.append(moment_tensors, ln_pdf, 1)
        self.assertTrue((self.Sample.moment_tensors[:, 0:3] == np.matrix([[2, 2, 3], [1, 1, 5], [2, 2, 1], [1, 1, 1], [2, 2, 2], [1, 1, 1]])).all())
        self.Sample.append(np.matrix([[], [], [], [], [], []]), np.matrix([[]]), 1)
        self.assertTrue((self.Sample.moment_tensors[:, 0:3] == np.matrix([[2, 2, 3], [1, 1, 5], [2, 2, 1], [1, 1, 1], [2, 2, 2], [1, 1, 1]])).all())
        self.assertEqual(self.Sample.n, 5)
        # test multiple events
        self.tearDown()
        self.Sample = Sample(initial_sample_size=1, number_events=2)
        moment_tensors = [np.matrix([[2], [1], [2], [1], [2], [1]]), np.matrix([[2], [1], [2], [1], [2], [1]])]
        ln_pdf = LnPDF(np.matrix([[1]]))
        self.Sample.append(moment_tensors, ln_pdf, 1)
        self.assertTrue((self.Sample.moment_tensors[:, 0] == np.matrix([[2], [1], [2], [1], [2], [1], [2], [1], [2], [1], [2], [1]])).all())
        # test scale_factor - multiple events
        self.tearDown()
        self.Sample = Sample(initial_sample_size=1, number_events=2)
        moment_tensors = [np.matrix([[2], [1], [2], [1], [2], [1]]), np.matrix([[2], [1], [2], [1], [2], [1]])]
        ln_pdf = LnPDF(np.matrix([[1]]))
        scale_factor = np.array([{'mu': np.array([[[2, 3], [3, 2]]]), 'sigma': np.array([[1, 1], [1, 1]])}])
        self.assertFalse(self.Sample.__dict__.__contains__('scale_factor'))
        self.Sample.append(moment_tensors, ln_pdf, 1, scale_factor)
        self.assertTrue((self.Sample.moment_tensors[:, 0] == np.matrix([[2], [1], [2], [1], [2], [1], [2], [1], [2], [1], [2], [1]])).all())
        self.assertTrue(self.Sample.__dict__.__contains__('scale_factor'))
        self.assertTrue((self.Sample.scale_factor[0]['mu'] == np.array([[2, 3], [3, 2]])).all())
        self.tearDown()
        self.Sample = Sample(initial_sample_size=3)
        # test initial samples
        moment_tensors = np.matrix([[2], [1], [2], [1], [2], [1]])
        ln_pdf = LnPDF(np.matrix([[1], [2]]))
        self.Sample.append(moment_tensors, ln_pdf, 1)
        self.assertTrue((self.Sample.moment_tensors[:, 0] == np.matrix([[2], [1], [2], [1], [2], [1]])).all())
        self.assertEqual(self.Sample.n, 1)
        self.assertEqual(self.Sample.ln_pdf[0, 0], 1.)
        moment_tensors = np.matrix([[2, 3], [1, 5], [2, 1], [1, 1], [2, 2], [1, 1]])
        ln_pdf = LnPDF(np.matrix([[1, 0], [2, 1]]))
        self.Sample.append(moment_tensors, ln_pdf, 2)
        self.assertTrue((self.Sample.moment_tensors[:, 0:3] == np.matrix([[2, 2, 3], [1, 1, 5], [2, 2, 1], [1, 1, 1], [2, 2, 2], [1, 1, 1]])).all())

        moment_tensors = np.matrix([[2], [1], [2], [1], [2], [1]])
        ln_pdf = LnPDF(np.matrix([[1], [2]]))
        self.Sample.append(moment_tensors, ln_pdf, 1)
        self.assertTrue((self.Sample.moment_tensors[:, 0:4] == np.matrix([[2, 2, 3, 2], [1, 1, 5, 1], [2, 2, 1, 2], [1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1]])).all())

    def test_output(self):
        self.test_append()
        self.assertTrue((self.Sample.output()[0]['moment_tensor_space'] == np.matrix([[2, 2, 3, 2], [1, 1, 5, 1], [2, 2, 1, 2], [1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1]])).all())
        output_ln_pdf = np.sum(np.exp(np.matrix([[1., 1., 0., 1.], [2., 2., 1., 2.]])), 0)
        output_ln_pdf /= np.sum(output_ln_pdf)
        self.assertAlmostEqual(self.Sample.output()[0]['probability'][0], output_ln_pdf[0, 0])
        self.assertAlmostEqual(self.Sample.output()[0]['probability'][1], output_ln_pdf[0, 1])
        self.assertAlmostEqual(self.Sample.output()[0]['probability'][2], output_ln_pdf[0, 2])
        self.assertAlmostEqual(self.Sample.output()[0]['probability'][3], output_ln_pdf[0, 3])
        self.assertEqual(self.Sample.output()[0]['dV'], 1)
        output_ln_pdf = np.sum(np.exp(np.matrix([[1., 1., 0., 1.], [2., 2., 1., 2.]])), 0)
        self.assertAlmostEqual(self.Sample.output(False)[0]['probability'][0], output_ln_pdf[0, 0])
        self.assertAlmostEqual(self.Sample.output(False)[0]['probability'][1], output_ln_pdf[0, 1])
        self.assertAlmostEqual(self.Sample.output(False)[0]['probability'][2], output_ln_pdf[0, 2])
        self.assertAlmostEqual(self.Sample.output(False)[0]['probability'][3], output_ln_pdf[0, 3])
        self.assertEqual(self.Sample.output(False)[0]['dV'], 1)
        self.tearDown()
        self.Sample = Sample(initial_sample_size=1, number_events=2)
        moment_tensors = [np.matrix([[2], [1], [2], [1], [2], [1]]), np.matrix([[3], [1], [2], [1], [2], [1]])]
        ln_pdf = LnPDF(np.matrix([[1]]))
        self.Sample.append(moment_tensors, ln_pdf, 1)
        self.assertTrue((self.Sample.moment_tensors[:, 0] == np.matrix([[2], [1], [2], [1], [2], [1], [3], [1], [2], [1], [2], [1]])).all())
        self.assertTrue((self.Sample.output()[0]['moment_tensor_space_1'] == np.matrix([[2], [1], [2], [1], [2], [1]])).all())
        self.assertTrue((self.Sample.output()[0]['moment_tensor_space_2'] == np.matrix([[3], [1], [2], [1], [2], [1]])).all())
        self.assertTrue((self.Sample.output()[0]['probability'] == (np.array([1]))).all(), str(self.Sample.output()[0]['probability']))
        self.assertEqual(self.Sample.output()[0]['dV'], 1)
        self.tearDown()
        self.Sample = Sample(initial_sample_size=1, number_events=2)
        moment_tensors = [np.matrix([[2], [1], [2], [1], [2], [1]]), np.matrix([[2], [1], [2], [1], [2], [1]])]
        ln_pdf = LnPDF(np.matrix([[1.]]))
        scale_factor = np.array([{'mu': np.array([[[2, 3], [3, 2]]]), 'sigma': np.array([[1, 1], [1, 1]])}])
        self.assertFalse(self.Sample.__dict__.__contains__('scale_factor'))
        self.Sample.append(moment_tensors, ln_pdf, 1, scale_factor)
        self.assertTrue((self.Sample.moment_tensors[:, 0] == np.matrix([[2], [1], [2], [1], [2], [1], [2], [1], [2], [1], [2], [1]])).all())
        self.assertTrue(self.Sample.__dict__.__contains__('scale_factor'))
        self.assertTrue((self.Sample.scale_factor[0]['mu'] == np.array([[2, 3], [3, 2]])).all())
        self.assertTrue((self.Sample.output()[0]['scale_factors'][0]['mu'] == np.array([[[2, 3], [3, 2]]])).all(), str(self.Sample.output()[0]['scale_factors'][0]['mu']))
        self.tearDown()
        self.setUp()
        self.test_append()
        out, out_str = self.Sample.output(convert=True)
        self.assertTrue('dV' in out.keys())
        self.assertTrue('moment_tensor_space' in out.keys())
        self.assertTrue('probability' in out.keys())
        self.assertTrue('ln_pdf' in out.keys())
        self.assertTrue('g' in out.keys())


class FileSampleTestCase(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        if sys.version_info >= (3, 0):
            self.tempdir = tempfile.TemporaryDirectory()
            os.chdir(self.tempdir.name)
        else:
            self.tempdir = tempfile.mkdtemp()
            os.chdir(self.tempdir)
        self.FileSample = FileSample('test')

    def tearDown(self):
        output_files = glob.glob('*in_progress*.mat*')
        for filename in output_files:
            try:
                os.remove(filename)
            except Exception:
                pass
        output_files = glob.glob('*.mat*')
        for filename in output_files:
            try:
                os.remove(filename)
            except Exception:
                pass
        os.chdir(self.cwd)
        if sys.version_info >= (3, 0):
            self.tempdir.cleanup()
        else:
            try:
                shutil.rmtree(self.tempdir)
            except:
                pass

    def test___init__(self):
        self.assertEqual(self.FileSample.fname, 'test_in_progress.mat')
        self.assertEqual(self.FileSample.n, 0)
        self.assertEqual(self.FileSample._i, 1)

    def test_recover(self):
        self.test_append()
        self.__setattr__('FileSample', FileSample('test'))
        self.assertEqual(self.FileSample._i, 3)
        self.assertEqual(self.FileSample.n, 50)
        self.assertEqual(self.FileSample.non_zero_samples, 2)

    def test_append(self):
        self.assertEqual(self.FileSample.fname, 'test_in_progress.mat')
        self.assertEqual(self.FileSample.n, 0)
        self.FileSample.append(np.matrix([[1], [2], [1], [2], [1], [1]]), np.matrix([1]), 10)
        self.assertEqual(self.FileSample.n, 10)
        self.assertTrue(os.path.exists(self.FileSample.fname))
        from hdf5storage import loadmat
        x = loadmat(self.FileSample.fname)
        self.assertEqual(sorted(x.keys()), ['LnPDF_1', 'MTSpace_1', 'i', 'n', 'non_zero_samples'])
        self.assertEqual(x['n'], 10)
        self.FileSample.append(np.matrix([[1], [2], [1], [2], [1], [1]]), np.matrix([1]), 20)
        self.assertEqual(self.FileSample.n, 30)
        self.assertTrue(os.path.exists(self.FileSample.fname))
        x = loadmat(self.FileSample.fname)
        self.assertEqual(sorted(x.keys()), ['LnPDF_1', 'LnPDF_2', 'MTSpace_1', 'MTSpace_2', 'i', 'n', 'non_zero_samples'])
        self.assertEqual(x['n'], 30)
        self.FileSample.append(np.matrix([[1], [2], [1], [2], [1], [1]]), np.matrix([-np.inf]), 20)
        self.assertEqual(self.FileSample.n, 50)
        x = loadmat(self.FileSample.fname)
        self.assertEqual(sorted(x.keys()), ['LnPDF_1', 'LnPDF_2', 'MTSpace_1', 'MTSpace_2', 'i', 'n', 'non_zero_samples'])
        self.assertEqual(x['n'], 50)
        self.tearDown()
        self.setUp()
        self.assertEqual(self.FileSample.fname, 'test_in_progress.mat')
        self.assertEqual(self.FileSample.n, 0)
        self.FileSample.append(np.matrix([[1], [2], [1], [2], [1], [1]]), np.matrix([1]), 10, [{'mu': np.array([[1, 2], [2, 1]]), 'sigma': np.array([[0, 0.1], [0.1, 0]])}])
        self.assertEqual(self.FileSample.n, 10)
        self.assertTrue(os.path.exists(self.FileSample.fname))
        from hdf5storage import loadmat
        x = loadmat(self.FileSample.fname)
        self.assertEqual(sorted(x.keys()), ['LnPDF_1', 'MTSpace_1', 'i', 'n', 'non_zero_samples', 'scale_factor_1'])
        self.assertEqual(x['n'], 10)
        self.FileSample.append(np.matrix([[1], [2], [1], [2], [1], [1]]), np.matrix([1]), 20, [{'mu': np.array([[1, 2], [2, 1]]), 'sigma': np.array([[0, 0.1], [0.1, 0]])}])
        self.assertEqual(self.FileSample.n, 30)
        self.assertTrue(os.path.exists(self.FileSample.fname))
        x = loadmat(self.FileSample.fname)
        self.assertEqual(sorted(x.keys()), ['LnPDF_1', 'LnPDF_2', 'MTSpace_1', 'MTSpace_2', 'i', 'n', 'non_zero_samples', 'scale_factor_1', 'scale_factor_2'])
        self.assertEqual(x['n'], 30)
        self.FileSample.append(np.matrix([[1], [2], [1], [2], [1], [1]]), np.matrix([-np.inf]), 20, [{'mu': np.array([[1, 2], [2, 1]]), 'sigma': np.array([[0, 0.1], [0.1, 0]])}])
        self.assertEqual(self.FileSample.n, 50)
        x = loadmat(self.FileSample.fname)
        self.assertEqual(sorted(x.keys()), ['LnPDF_1', 'LnPDF_2', 'MTSpace_1', 'MTSpace_2', 'i', 'n', 'non_zero_samples', 'scale_factor_1', 'scale_factor_2'])
        self.assertEqual(x['n'], 50)

    def test_output(self):
        self.test_append()
        out, out_str = self.FileSample.output(convert=True)
        self.assertTrue('dV' in out.keys())
        self.assertTrue('moment_tensor_space' in out.keys())
        self.assertTrue('probability' in out.keys())
        self.assertTrue('ln_pdf' in out.keys())
        self.assertTrue('g' in out.keys())
        self.assertEqual(out['moment_tensor_space'].shape, (6, 2))
        self.assertEqual(out['probability'].shape, (2, ))
