import unittest
import time

import numpy as np

from MTfit.algorithms.monte_carlo import BaseMonteCarloRandomSample
from MTfit.algorithms.monte_carlo import IterationSample
from MTfit.algorithms.monte_carlo import TimeSample


class BaseMonteCarloRandomSampleTestCase(unittest.TestCase):

    def setUp(self):
        self.monte_carlo_random_sample = BaseMonteCarloRandomSample()

    def tearDown(self):
        del self.monte_carlo_random_sample

    @unittest.expectedFailure
    def test___init__(self):
        raise NotImplementedError()

    @unittest.expectedFailure
    def test_total_number_samples(self):
        raise NotImplementedError()

    @unittest.expectedFailure
    def test_check_finished(self):
        raise NotImplementedError()

    def test_initialise(self):
        self.assertTrue(self.monte_carlo_random_sample.initialise()[0].shape,
                        (6, self.monte_carlo_random_sample.number_samples))

    def test_iterate(self):
        moment_tensors, end = self.monte_carlo_random_sample.iterate({'moment_tensors': self.monte_carlo_random_sample.random_mt(),
                                                                      'ln_pdf': np.ones((1, self.monte_carlo_random_sample.number_samples)),
                                                                      'n': self.monte_carlo_random_sample.number_samples})
        self.assertEqual(moment_tensors.shape, (6, self.monte_carlo_random_sample.number_samples))
        self.assertFalse(end)

    def test_iterate_quality_check(self):
        self.monte_carlo_random_sample._initialising = True
        self.monte_carlo_random_sample.quality_check = 1
        self.monte_carlo_random_sample._number_check_samples = 30000
        self.monte_carlo_random_sample.pdf_sample.append(np.ones((6, 20000)),
                                                         1.0*np.ones((1, 20000)), 40000)
        MTs, end = self.monte_carlo_random_sample.iterate({'moment_tensors': self.monte_carlo_random_sample.random_mt(),
                                                           'ln_pdf': 1.0*np.ones((1, self.monte_carlo_random_sample.number_samples)), 'n': 1})
        self.assertEqual(MTs, [])
        self.assertTrue(end)
        self.monte_carlo_random_sample._initialising = True
        self.monte_carlo_random_sample.quality_check = 80.0
        MTs, end = self.monte_carlo_random_sample.iterate({'moment_tensors': self.monte_carlo_random_sample.random_mt(),
                                                           'ln_pdf': -np.inf*np.ones((1, self.monte_carlo_random_sample.number_samples)),
                                                           'n': 1})
        self.assertEqual(MTs.shape, (6, self.monte_carlo_random_sample.number_samples))
        self.assertEqual(MTs.shape, (6, 10000))
        self.assertFalse(end)

    def test_random_sample(self):
        self.assertEqual(self.monte_carlo_random_sample.random_sample().shape[0], 6)
        self.tearDown()
        self.monte_carlo_random_sample = BaseMonteCarloRandomSample(number_events=3)
        self.assertEqual(self.monte_carlo_random_sample.number_events, 3)
        self.assertIsInstance(self.monte_carlo_random_sample.random_sample(), list)
        self.assertEqual(len(self.monte_carlo_random_sample.random_sample()), 3)
        self.assertEqual(self.monte_carlo_random_sample.random_sample()[0].shape[0], 6)


class TimeSampleTestCase(unittest.TestCase):

    def setUp(self):
        self.time_sample = TimeSample()

    def tearDown(self):
        del self.time_sample

    @unittest.expectedFailure
    def test___init__(self):
        raise NotImplementedError()

    @unittest.expectedFailure
    def test_check_finished(self):
        raise NotImplementedError()

    @unittest.expectedFailure
    def test_max_value(self):
        raise NotImplementedError()

    def test_initialise(self):
        task, end = self.time_sample.initialise()
        self.assertTrue(task.shape, (6, self.time_sample.number_samples))
        self.assertEqual(self.time_sample.iteration, 0)
        self.assertTrue(self.time_sample.start_time)

    def test_iterate(self):
        self.time_sample.initialise()
        moment_tensors, end = self.time_sample.iterate({'moment_tensors': self.time_sample.random_mt(),
                                                        'ln_pdf': np.ones((1, self.time_sample.number_samples)),
                                                        'n': self.time_sample.number_samples})
        self.assertEqual(moment_tensors.shape, (6, self.time_sample.number_samples))
        self.assertFalse(end)
        self.time_sample.max_time = 2
        runStart = time.time()
        while time.time()-runStart < 2.:
            pass
        moment_tensors, end = self.time_sample.iterate({'moment_tensors': self.time_sample.random_mt(),
                                                        'ln_pdf': np.ones((1, self.time_sample.number_samples)),
                                                        'n': self.time_sample.number_samples})
        self.assertEqual(moment_tensors.shape, (6, self.time_sample.number_samples))
        self.assertTrue(end)
        self.assertEqual(self.time_sample.iteration, 2)


class IterationSampleTestCase(unittest.TestCase):

    def setUp(self):
        self.iteration_sample = IterationSample()

    def tearDown(self):
        del self.iteration_sample

    @unittest.expectedFailure
    def test___init__(self):
        raise NotImplementedError()

    @unittest.expectedFailure
    def test_check_finished(self):
        raise NotImplementedError()

    @unittest.expectedFailure
    def test_max_value(self):
        raise NotImplementedError()

    def test_initialise(self):
        task, end = self.iteration_sample.initialise()
        self.assertTrue(task.shape, (6, self.iteration_sample.number_samples))
        self.assertEqual(self.iteration_sample.iteration, 0)
        self.assertTrue(self.iteration_sample.start_time)

    def test_iterate(self):
        self.iteration_sample.initialise()
        self.iteration_sample.max_samples = 6*self.iteration_sample.number_samples - 2
        moment_tensors, end = self.iteration_sample.iterate({'moment_tensors': self.iteration_sample.random_mt(),
                                                             'ln_pdf': np.ones((1, self.iteration_sample.number_samples)),
                                                             'n': self.iteration_sample.number_samples})
        self.assertEqual(moment_tensors.shape, (6, self.iteration_sample.number_samples))
        self.assertFalse(end)
        i = 0
        while i < 5:
            moment_tensors, end = self.iteration_sample.iterate({'moment_tensors': self.iteration_sample.random_mt(),
                                                                 'ln_pdf': np.ones((1, self.iteration_sample.number_samples)),
                                                                 'n': self.iteration_sample.number_samples})
            i += 1

        self.assertEqual(moment_tensors.shape, (6, self.iteration_sample.number_samples))
        self.assertTrue(end)
        self.assertEqual(self.iteration_sample.iteration, 6)
