import unittest
import copy

import numpy as np

from mtfit.utilities.unittest_utils import run_tests as _run_tests
from mtfit.utilities.unittest_utils import debug_tests as _debug_tests

from mtfit.algorithms.markov_chain_monte_carlo import IterativeMetropolisHastingsGaussianTape
from mtfit.algorithms.markov_chain_monte_carlo import IterativeTransDMetropolisHastingsGaussianTape
from mtfit.algorithms.markov_chain_monte_carlo import IterativeMultipleTryMetropolisHastingsGaussianTape
from mtfit.algorithms.markov_chain_monte_carlo import MarginalisedMarkovChainMonteCarlo
from mtfit.algorithms.markov_chain_monte_carlo import MarginalisedMetropolisHastings
from mtfit.algorithms.markov_chain_monte_carlo import MarginalisedMetropolisHastingsGaussianTape
from mtfit.algorithms.markov_chain_monte_carlo import IterativeMultipleTryTransDMetropolisHastingsGaussianTape
from mtfit.algorithms.markov_chain_monte_carlo import McMCAlgorithmCreator


try:
    from mtfit.algorithms import cmarkov_chain_monte_carlo
except ImportError:
    cmarkov_chain_monte_carlo = False

VERBOSITY = 2


class IterativeMultipleTryMetropolisHastingsGaussianTapeTestCase(unittest.TestCase):

    def setUp(self):
        self.mcmc_algorithm = IterativeMultipleTryMetropolisHastingsGaussianTape()
        self.mcmc_algorithm.xi = self.mcmc_algorithm.random_mt()

    def tearDown(self):
        del self.mcmc_algorithm

    def test_new_sample(self):
        self.assertEqual(self.mcmc_algorithm._number_samples, int(1./self.mcmc_algorithm.min_acceptance_rate))
        self.mcmc_algorithm._number_samples = 100
        x0 = self.mcmc_algorithm.new_sample()
        if cmarkov_chain_monte_carlo:
            self.assertEqual(x0.shape[0], 6)
            self.assertEqual(x0.shape[1], 100)
        else:
            self.assertEqual(x0.shape[0], 6)
            self.assertEqual(x0.shape[1], 1)

    def test__acceptance_check(self):
        self.mcmc_algorithm.new_sample()
        self.mcmc_algorithm.ln_likelihood_xi = - np.inf
        if isinstance(self.mcmc_algorithm.xi_1, dict):
            n = 1
        else:
            n = len(self.mcmc_algorithm.xi_1)
        xi_1, ln_pi1, sf1, index = self.mcmc_algorithm._acceptance_check(self.mcmc_algorithm.xi_1,
                                                                         1.0*np.ones((n)))
        self.assertEqual(index, 0)


class MarginalisedMarkovChainMonteCarloTestCase(unittest.TestCase):

    def setUp(self):
        self.mcmc_algorithm = MarginalisedMarkovChainMonteCarlo(learning_length=0)

    def tearDown(self):
        del self.mcmc_algorithm

    def test_initialise(self):
        self.assertTrue(self.mcmc_algorithm.initialise()[0].shape, (6, self.mcmc_algorithm.number_samples))
        self.assertTrue(self.mcmc_algorithm.initialise()[0].shape, (6, 1))
        self.tearDown()
        self.mcmc_algorithm = MarginalisedMarkovChainMonteCarlo(learning_length=0, initial_sample='grid')
        self.assertTrue(self.mcmc_algorithm.initialise()[0].shape,
                        (6, self.mcmc_algorithm._initialiser.number_samples))
        self.assertTrue(self.mcmc_algorithm.initialise()[0].shape, (6, 50000))

    def test_iterate(self):
        self.mcmc_algorithm.ln_likelihood_xi = 1000000000000.0
        self.mcmc_algorithm.scale_factor_i = False
        self.mcmc_algorithm.xi = self.mcmc_algorithm.random_mt()
        MTs, end = self.mcmc_algorithm.iterate({'moment_tensors': self.mcmc_algorithm.random_mt(),
                                                'ln_pdf': 1.0*np.ones((1, self.mcmc_algorithm.number_samples)),
                                                'n': 1})
        self.assertEqual(MTs.shape, (6, self.mcmc_algorithm.number_samples))
        self.assertEqual(MTs.shape, (6, 1))
        self.assertFalse(end)

    def test_acceptance_rate(self):
        self.assertEqual(self.mcmc_algorithm._accepted, -1)
        self.test_iterate()
        self.test_iterate()
        self.test_iterate()
        self.assertEqual(self.mcmc_algorithm._tried, 3)
        self.mcmc_algorithm._accepted = 0
        self.assertEqual(self.mcmc_algorithm.acceptance_rate(), 0/3.0)
        self.mcmc_algorithm._accepted = 1
        self.assertEqual(self.mcmc_algorithm.acceptance_rate(), 1.0/3.0)

    def test__modify_acceptance_rate(self):
        self.mcmc_algorithm.learning_length = 1000
        for i in range(100):
            if np.random.rand() < 0.12:
                self.mcmc_algorithm._learning_accepted.append(1)
            else:
                self.mcmc_algorithm._learning_accepted.append(0)
        self.mcmc_algorithm.alpha = 1
        self.mcmc_algorithm._modify_acceptance_rate()
        self.assertTrue(self.mcmc_algorithm.alpha < 1)
        self.mcmc_algorithm._learning_accepted = []
        for i in range(100):
            if np.random.rand() < 0.05:
                self.mcmc_algorithm._learning_accepted.append(1)
            else:
                self.mcmc_algorithm._learning_accepted.append(0)
        self.mcmc_algorithm.alpha = 1.
        self.mcmc_algorithm._modify_acceptance_rate()
        self.assertTrue(self.mcmc_algorithm.alpha < 1)
        self.mcmc_algorithm._learning_accepted = []
        self.mcmc_algorithm._old_rate = self.mcmc_algorithm.min_acceptance_rate
        for i in range(100):
            if np.random.rand() < 0.85:
                self.mcmc_algorithm._learning_accepted.append(1)
            else:
                self.mcmc_algorithm._learning_accepted.append(0)
        self.mcmc_algorithm.alpha = 1.
        self.mcmc_algorithm._modify_acceptance_rate()
        self.assertTrue(self.mcmc_algorithm.alpha > 1)

    def test_quality_check(self):
        self.mcmc_algorithm._initialising = True
        self.mcmc_algorithm.quality_check = 0
        self.mcmc_algorithm._number_initialisation_samples = 40000
        self.mcmc_algorithm._init_nonzero = 20000
        self.mcmc_algorithm._init_max_p = 1.3
        MTs, end = self.mcmc_algorithm.iterate({'moment_tensors': self.mcmc_algorithm.random_mt(),
                                                'ln_pdf': 1.0*np.ones((1, self.mcmc_algorithm.number_samples)),
                                                'n': 1})
        self.assertEqual(MTs, [])
        self.assertTrue(end)
        self.mcmc_algorithm._initialising = True
        self.mcmc_algorithm.quality_check = 40
        self.mcmc_algorithm._number_initialisation_samples = 40000
        self.mcmc_algorithm._init_nonzero = 1000
        self.mcmc_algorithm._init_max_p = 1.3
        self.mcmc_algorithm._init_max_mt = self.mcmc_algorithm.random_mt()
        MTs, end = self.mcmc_algorithm.iterate({'moment_tensors': self.mcmc_algorithm.random_mt(),
                                                'ln_pdf': 1.0*np.ones((1, self.mcmc_algorithm.number_samples)),
                                                'n': 1})
        self.assertEqual(MTs.shape, (6, self.mcmc_algorithm.number_samples))
        self.assertEqual(MTs.shape, (6, 1))
        self.assertFalse(end)


class MarginalisedMetropolisHastingsTestCase(unittest.TestCase):

    def setUp(self):
        self.mcmc_algorithm = MarginalisedMetropolisHastings()

    def tearDown(self):
        del self.mcmc_algorithm

    def test_acceptance(self):
        self.mcmc_algorithm.initialise()
        self.assertEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), 1.0)


class MarginalisedMetropolisHastingsGaussianTapeTestCase(unittest.TestCase):

    def setUp(self):
        self.mcmc_algorithm = MarginalisedMetropolisHastingsGaussianTape()

    def tearDown(self):
        del self.mcmc_algorithm

    def test_random_mt(self):
        self.assertEqual(sorted(self.mcmc_algorithm.random_mt().keys()),
                         ['delta', 'gamma', 'h', 'kappa', 'sigma'])
        self.assertNotEqual(self.mcmc_algorithm.random_mt()['delta'], 0)
        self.assertNotEqual(self.mcmc_algorithm.random_mt()['gamma'], 0)

    def test_random_dc(self):
        self.assertEqual(sorted(self.mcmc_algorithm.random_dc().keys()),
                         ['delta', 'gamma', 'h', 'kappa', 'sigma'])
        self.assertEqual(self.mcmc_algorithm.random_dc()['delta'], 0)
        self.assertEqual(self.mcmc_algorithm.random_dc()['gamma'], 0)

    def test_new_sample(self):
        self.mcmc_algorithm.initialise()
        self.assertEqual(sorted(self.mcmc_algorithm.convert_sample(self.mcmc_algorithm.new_sample()).keys()),
                         ['delta', 'gamma', 'h', 'kappa', 'sigma'])

    def test_transition_pdf(self):
        self.mcmc_algorithm.initialise()
        x1 = self.mcmc_algorithm.convert_sample(self.mcmc_algorithm.new_sample())
        x0 = self.mcmc_algorithm.convert_sample(self.mcmc_algorithm.new_sample())
        self.assertIsInstance(self.mcmc_algorithm.transition_pdf(x1, x0), float)
        x1 = {'gamma': 0, 'delta': 0, 'kappa': np.pi, 'h': 0.5, 'sigma': 0}
        x0 = {'gamma': 0, 'delta': 0, 'kappa': 0.5+np.pi, 'h': 0.7, 'sigma': 0.2}
        self.mcmc_algorithm.alpha = {'gamma': 0.2, 'delta': 0.4, 'kappa': 1, 'h': 0.2, 'sigma': 0.4}
        self.assertAlmostEqual(self.mcmc_algorithm.transition_pdf(x1, x0), 2.291888412753515)
        self.assertAlmostEqual(self.mcmc_algorithm.transition_pdf(x0, x1), 2.1646452165457575)
        x1 = {'gamma': np.pi/5, 'delta': -np.pi/2, 'kappa': 0, 'h': 0.9, 'sigma': np.pi-0.4}
        x0 = {'gamma': np.pi/6, 'delta': -np.pi/2, 'kappa': 0.2, 'h': 0.98, 'sigma': np.pi-0.2}
        old_alpha = self.mcmc_algorithm.alpha
        self.mcmc_algorithm.alpha = {'gamma': 0.2, 'delta': 0.4, 'kappa': 1, 'h': 0.2, 'sigma': 0.4}
        self.assertAlmostEqual(self.mcmc_algorithm.transition_pdf(x1, x0), 68262.57402456368)
        self.assertAlmostEqual(self.mcmc_algorithm.transition_pdf(x0, x1), 15823.834783183942)
        if cmarkov_chain_monte_carlo:
            x1 = self.mcmc_algorithm.convert_sample(self.mcmc_algorithm.new_sample())
            x0 = self.mcmc_algorithm.convert_sample(self.mcmc_algorithm.new_sample())
            self.mcmc_algorithm.alpha = old_alpha
            self.assertIsInstance(cmarkov_chain_monte_carlo._gaussian_transition_ratio_test(x1,
                                                                                            x0,
                                                                                            self.mcmc_algorithm.alpha),
                                  float)
            x1 = {'gamma': 0, 'delta': 0, 'kappa': np.pi, 'h': 0.5, 'sigma': 0}
            x0 = {'gamma': 0, 'delta': 0, 'kappa': 0.5+np.pi, 'h': 0.7, 'sigma': 0.2}
            self.mcmc_algorithm.alpha = {'gamma': 0.2, 'delta': 0.4, 'kappa': 1, 'h': 0.2, 'sigma': 0.4}
            # DC
            self.assertAlmostEqual(cmarkov_chain_monte_carlo._gaussian_transition_ratio_test(x1, x0,
                                                                                             self.mcmc_algorithm.alpha),
                                   2.1646452165457575/2.291888412753515, 4)
            x1 = {'gamma': np.pi/5, 'delta': -np.pi/2, 'kappa': 0, 'h': 0.9, 'sigma': np.pi-0.4}
            x0 = {'gamma': np.pi/6, 'delta': -np.pi/2, 'kappa': 0.2, 'h': 0.98, 'sigma': np.pi-0.2}
            self.mcmc_algorithm.alpha = {'gamma': 0.2, 'delta': 0.4, 'kappa': 1, 'h': 0.2, 'sigma': 0.4}
            self.assertAlmostEqual(cmarkov_chain_monte_carlo._gaussian_transition_ratio_test(x1, x0, self.mcmc_algorithm.alpha),
                                   15823.834783183942/68262.57402456368, 4)

    def test_acceptance(self):
        self.mcmc_algorithm.initialise()
        self.mcmc_algorithm.xi = {'h': 0.7972501404226121, 'sigma': 0.3034672053613414,
                                  'kappa': 5.562484106876691, 'gamma': -0.4613373190200656,
                                  'delta': -0.3559523484353853}
        self.mcmc_algorithm.xi_1 = {'h': np.array([0.62021735]), 'sigma': np.array([1.08204009]),
                                    'kappa': np.array([5.51100894]), 'gamma': np.array([-0.45280446]),
                                    'delta': np.array([-0.43080595])}
        if cmarkov_chain_monte_carlo:
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                True)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                True)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                True)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                True)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            self.mcmc_algorithm.new_sample()
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                True)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                True)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                True)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1,
                                                                  np.log(1.8)),
                                   acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                True)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1,
                                                                  np.log(1.4)),
                                   acc, 4)
            self.mcmc_algorithm.new_sample()
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                True)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                True)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                True)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                True)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)


class IterativeMetropolisHastingsGaussianTapeTestCase(unittest.TestCase):

    def setUp(self):
        self.mcmc_algorithm = IterativeMetropolisHastingsGaussianTape()

    def tearDown(self):
        del self.mcmc_algorithm

    def test___iterate__(self):
        self.mcmc_algorithm.ln_likelihood_xi = 1
        self.mcmc_algorithm.xi = self.mcmc_algorithm.random_mt()
        self.mcmc_algorithm.chain_length = 1
        self.mcmc_algorithm._tried = 2
        x1 = self.mcmc_algorithm.random_mt()
        x0 = self.mcmc_algorithm.random_mt()
        self.mcmc_algorithm.xi_1 = x1
        self.mcmc_algorithm.x = x0
        self.mcmc_algorithm.ln_likelihood_xi = 1.0
        self.assertIsInstance(self.mcmc_algorithm.xi, dict)
        MTs, End = self.mcmc_algorithm.iterate({'moment_tensors': np.array([[1], [0], [0], [0], [0], [0]]),
                                                'ln_pdf': 1.0, 'N': 1})
        self.assertIsInstance(self.mcmc_algorithm.xi, dict)
        self.assertTrue(End)
        self.mcmc_algorithm.chain_length = 100
        # import ipdb;ipdb.set_trace()
        MTs, End = self.mcmc_algorithm.iterate({'moment_tensors': np.array([[1], [0], [0], [0], [0], [0]]),
                                                'ln_pdf': 1.0, 'N': 1})
        self.assertIsInstance(self.mcmc_algorithm.xi, dict)
        self.assertFalse(End)
        self.tearDown()
        self.setUp()
        self.mcmc_algorithm.xi = self.mcmc_algorithm.random_mt()
        self.mcmc_algorithm._initialising = False
        self.mcmc_algorithm._number_learning_accepted = self.mcmc_algorithm.learning_length + 100
        self.assertEqual(self.mcmc_algorithm.pdf_sample.ln_pdf.shape, (1, 0))
        self.mcmc_algorithm.ln_likelihood_xi = 0.0
        self.mcmc_algorithm.xi_1 = self.mcmc_algorithm.convert_sample(np.array([[1], [0],
                                                                               [0], [0],
                                                                               [0], [0]]))
        MTs, End = self.mcmc_algorithm.iterate({'moment_tensors': np.array([[1], [0], [0], [0], [0], [0]]),
                                                'ln_pdf': 1.0, 'N': 1})
        self.assertEqual(self.mcmc_algorithm.pdf_sample.ln_pdf.shape, (1, 1))
        self.mcmc_algorithm.ln_likelihood_xi = 0.0
        MTs, End = self.mcmc_algorithm.iterate({'moment_tensors': np.array([[1], [0], [0], [0], [0], [0]]),
                                                'ln_pdf': 1.0, 'N': 1})
        self.assertEqual(self.mcmc_algorithm.pdf_sample.ln_pdf.shape, (1, 2))


class IterativeTransDMetropolisHastingsGaussianTapeTestCase(unittest.TestCase):

    def setUp(self):
        self.mcmc_algorithm = IterativeTransDMetropolisHastingsGaussianTape()

    def tearDown(self):
        del self.mcmc_algorithm

    def test_new_sample(self):
        self.mcmc_algorithm.dimension_jump_prob = 1
        self.mcmc_algorithm.dc = False
        self.mcmc_algorithm.xi = self.mcmc_algorithm.random_mt()
        self.mcmc_algorithm.convert_sample(
            self.mcmc_algorithm.new_sample())
        self.assertTrue(
            self.mcmc_algorithm.jump)
        self.assertEqual(
            self.mcmc_algorithm.xi_1['gamma'], 0)
        self.assertEqual(
            self.mcmc_algorithm.xi_1['delta'], 0)
        self.mcmc_algorithm.dc = False
        self.mcmc_algorithm.dimension_jump_prob = 0
        self.mcmc_algorithm.convert_sample(self.mcmc_algorithm.new_sample())
        self.assertNotEqual(self.mcmc_algorithm.xi_1['gamma'], 0)
        self.assertNotEqual(self.mcmc_algorithm.xi_1['delta'], 0)
        self.assertFalse(self.mcmc_algorithm.jump)

    def test_jump_params(self):
        self.mcmc_algorithm.gaussian_jump_params = False
        self.assertEqual(len(self.mcmc_algorithm.jump_params()), 2)
        self.assertEqual(self.mcmc_algorithm.jump_params(self.mcmc_algorithm.random_mt()),
                         3./(2*np.pi))

    def test_acceptance(self):
        self.mcmc_algorithm.ln_likelihood_xi = 0.5
        x0 = {'gamma': 0, 'delta': 0, 'kappa': np.pi, 'h': 0.5, 'sigma': 0}
        x1 = copy.copy(x0)
        x1['gamma'], x1['delta'] = (np.pi/12, np.pi/4)
        likelihoodx1 = 0.2
        # print self.mcmc_algorithm.jump
        self.mcmc_algorithm.jump = True
        self.mcmc_algorithm.xi = x0
        self.mcmc_algorithm.gaussian_jump_params = False
        a = self.mcmc_algorithm.acceptance(x1, likelihoodx1)
        self.assertAlmostEqual(a, 0.39102020357073225)
        self.mcmc_algorithm.gaussian_jump_params = True
        a = self.mcmc_algorithm.acceptance(x1, likelihoodx1)
        self.assertAlmostEqual(a, 1)
        if not cmarkov_chain_monte_carlo:
            return
        i = 0
        while i < 10:
            i += 1
            self.mcmc_algorithm.jump = False
            self.mcmc_algorithm.dimension_jump_prob = 0.6
            self.mcmc_algorithm.initialise()
            self.mcmc_algorithm.new_sample()
            self.mcmc_algorithm.xi = self.mcmc_algorithm.xi_1
            if self.mcmc_algorithm.xi['gamma'] == 0.0 and self.mcmc_algorithm.xi['delta'] == 0.0:
                self.mcmc_algorithm.dc = True
                try:
                    self.mcmc_algorithm.xi.pop('g0')
                    self.mcmc_algorithm.xi.pop('d0')
                except Exception:
                    pass
            else:
                self.mcmc_algorithm.dc = False
            self.mcmc_algorithm.new_sample()
            self.mcmc_algorithm.ln_likelihood_xi = np.log(0.3)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)

            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            self.mcmc_algorithm.new_sample()
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            # import ipdb;ipdb.set_trace()
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            self.mcmc_algorithm.new_sample()
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])), self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.         acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            self.mcmc_algorithm.new_sample()
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            self.mcmc_algorithm.new_sample()
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            # Flat jump
            self.mcmc_algorithm.gaussian_jump_params = False
            self.mcmc_algorithm.ln_likelihood_xi = np.log(0.3)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            self.mcmc_algorithm.new_sample()
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            self.mcmc_algorithm.new_sample()
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            # if self.mcmc_algorithm.dc:
            #     import ipdb;ipdb.set_trace()
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)

            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            self.mcmc_algorithm.new_sample()
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            self.mcmc_algorithm.new_sample()
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)

            # Other tests
            while not (self.mcmc_algorithm.dc != (self.mcmc_algorithm.xi['gamma'] == 0.0 and self.mcmc_algorithm.xi['delta'] == 0.0)):
                self.mcmc_algorithm.new_sample()
                self.mcmc_algorithm.xi = self.mcmc_algorithm.xi_1
            self.mcmc_algorithm.dc = not self.mcmc_algorithm.dc
            if self.mcmc_algorithm.dc:
                try:
                    self.mcmc_algorithm.xi.pop('g0')
                    self.mcmc_algorithm.xi.pop('d0')
                except Exception:
                    pass
            self.mcmc_algorithm.gaussian_jump_params = True
            self.mcmc_algorithm.new_sample()
            self.mcmc_algorithm.ln_likelihood_xi = np.log(0.3)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)

            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            self.mcmc_algorithm.new_sample()
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            self.mcmc_algorithm.new_sample()
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(0.3)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)

            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            self.mcmc_algorithm.new_sample()
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            self.mcmc_algorithm.new_sample()
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            self.mcmc_algorithm.gaussian_jump_params = False

            self.mcmc_algorithm.ln_likelihood_xi = np.log(0.3)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)

            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            self.mcmc_algorithm.new_sample()
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            self.mcmc_algorithm.new_sample()
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(0.3)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)

            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            self.mcmc_algorithm.new_sample()
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)
            self.mcmc_algorithm.new_sample()
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.array([1.0]),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, 1.0), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(2.2)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([0.5])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(0.5)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.5)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.8])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.8)), acc, 4)
            self.mcmc_algorithm.ln_likelihood_xi = np.log(1.7)
            acc = cmarkov_chain_monte_carlo._acceptance_test_fn(self.mcmc_algorithm.xi_1,
                                                                self.mcmc_algorithm.xi,
                                                                self.mcmc_algorithm.alpha,
                                                                np.log(np.array([1.4])),
                                                                self.mcmc_algorithm.ln_likelihood_xi,
                                                                True,
                                                                self.mcmc_algorithm.gaussian_jump_params)
            self.assertAlmostEqual(self.mcmc_algorithm.acceptance(self.mcmc_algorithm.xi_1, np.log(1.4)), acc, 4)


class McMCAlgorithmCreatorTestCase(unittest.TestCase):

    def test___new__(self):
        obj = McMCAlgorithmCreator()
        self.assertIsInstance(obj, IterativeMultipleTryMetropolisHastingsGaussianTape)
        obj = McMCAlgorithmCreator(trans_dimensional=True)
        self.assertIsInstance(obj, IterativeMultipleTryTransDMetropolisHastingsGaussianTape)
        obj = McMCAlgorithmCreator(multiple_events=True)
        self.assertIsInstance(obj, IterativeMultipleTryMetropolisHastingsGaussianTape)
        obj = McMCAlgorithmCreator(mode='asafa')
        self.assertIsInstance(obj, IterativeMetropolisHastingsGaussianTape)


def test_suite(verbosity=2):
    global VERBOSITY
    VERBOSITY = verbosity
    suite = [unittest.TestLoader().loadTestsFromTestCase(MarginalisedMarkovChainMonteCarloTestCase),
             unittest.TestLoader().loadTestsFromTestCase(MarginalisedMetropolisHastingsTestCase),
             unittest.TestLoader().loadTestsFromTestCase(MarginalisedMetropolisHastingsGaussianTapeTestCase),
             unittest.TestLoader().loadTestsFromTestCase(IterativeMetropolisHastingsGaussianTapeTestCase),
             unittest.TestLoader().loadTestsFromTestCase(IterativeTransDMetropolisHastingsGaussianTapeTestCase),
             unittest.TestLoader().loadTestsFromTestCase(McMCAlgorithmCreatorTestCase),
             unittest.TestLoader().loadTestsFromTestCase(IterativeMultipleTryMetropolisHastingsGaussianTapeTestCase)]
    suite = unittest.TestSuite(suite)
    try:
        from mtfit.algorithms.cmarkov_chain_monte_carlo import test_suite as cmarkov_chain_monte_carlo_test_suite
        suite = unittest.TestSuite([suite, cmarkov_chain_monte_carlo_test_suite(verbosity)])
    except ImportError:
        pass
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
