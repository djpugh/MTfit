"""
test_multiprocessing_utils.py
*****************************

Tests for src/utils/multiprocessing_utils.py
"""

import unittest
import multiprocessing
import multiprocessing.queues

from mtfit.utilities.unittest_utils import TestCase
from mtfit.utilities.unittest_utils import run_tests as _run_tests
from mtfit.utilities.unittest_utils import debug_tests as _debug_tests
from mtfit.utilities.multiprocessing_helper import Worker
from mtfit.utilities.multiprocessing_helper import JobPool
from mtfit.tests.unit.utilities.multiprocessing_test_classes import TestTask
from mtfit.tests.unit.utilities.multiprocessing_test_classes import TestTask2


class WorkerTestCase(TestCase):

    def setUp(self):
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.worker = Worker(self.task_queue, self.result_queue)
        self.worker.start()

    def tearDown(self):
        self.task_queue.put(None)
        if self.worker.is_alive():
            self.worker.join()
        del self.worker
        self.task_queue.close()
        del self.task_queue
        self.result_queue.close()
        del self.result_queue

    def test___init__(self):
        self.assertEqual(self.worker.task_queue, self.task_queue)
        self.assertEqual(self.worker.result_queue, self.result_queue)
        self.assertTrue(
            isinstance(self.worker.task_queue, multiprocessing.queues.Queue))
        self.assertTrue(
            isinstance(self.worker.result_queue, multiprocessing.queues.Queue))

    def test_run(self):
        self.test___init__()
        self.assertTrue(self.result_queue.empty())
        self.task_queue.put(TestTask(1, 2))
        r = self.result_queue.get(timeout=10)
        self.assertTrue(self.result_queue.empty())
        self.assertEqual(r, 2)
        self.task_queue.put(TestTask(1, 5))
        r = self.result_queue.get(timeout=10)
        self.assertTrue(self.result_queue.empty())
        self.assertEqual(r, 5)


class JobPoolTestCase(TestCase):

    def setUp(self, task=TestTask, single_life=False):
        self.job_pool = JobPool(2, task=task, single_life=single_life)

    def tearDown(self):
        self.job_pool.close()
        del self.job_pool

    def test_close(self):
        for w in self.job_pool.workers:
            self.assertTrue(w.is_alive())
        self.job_pool.close()
        for w in self.job_pool.workers:
            self.assertFalse(w.is_alive())

    def test_task(self):
        self.job_pool.task(1, 2)
        self.assertEqual(self.job_pool.number_jobs, 1)

    def test_custom_task(self):
        self.job_pool.custom_task(TestTask2, 4, 2)
        self.assertEqual(self.job_pool.number_jobs, 1)
        self.assertEqual(self.job_pool.result(), 6)

    def test_result(self):
        self.job_pool.task(1, 2)
        result = self.job_pool.result()
        self.assertEqual(result, 2)

    def test_all_results(self):
        self.job_pool.task(1, 2)
        self.job_pool.task(1, 3)
        self.assertEqual(sorted(self.job_pool.all_results()), [2, 3])

    def test_parallel(self):
        self.job_pool.custom_task(TestTask2, 2, 4)
        self.job_pool.custom_task(TestTask2, 4, 1)
        self.job_pool.custom_task(TestTask2, 4, 0)
        self.assertNotEqual(self.job_pool.all_results(), [6, 5, 4])

    def test_single_life(self):
        self.tearDown()
        self.setUp(single_life=True)
        self.assertEqual(self.job_pool.number_workers, 2)
        self.job_pool.custom_task(TestTask2, 2, 4)
        self.job_pool.custom_task(TestTask2, 4, 1)
        self.job_pool.custom_task(TestTask2, 4, 0)
        self.job_pool.custom_task(TestTask2, 8, 1)
        self.assertNotEqual(self.job_pool.all_results(), [6, 5, 4, 9])

    def test_exceptionBreak(self):
        self.job_pool.custom_task(TestTask2, 2, 4)
        self.job_pool.custom_task(TestTask2, "4", 1)
        self.job_pool.custom_task(TestTask2, "4", 0)
        self.job_pool.custom_task(TestTask2, "8", 1)
        results = self.job_pool.all_results()
        self.assertTrue(6 in results)
        results.pop(results.index(6))
        self.assertTrue(isinstance(results[0], TypeError))


def test_suite(verbosity=2):
    global VERBOSITY
    VERBOSITY = verbosity
    suite = [unittest.TestLoader().loadTestsFromTestCase(JobPoolTestCase),
             unittest.TestLoader().loadTestsFromTestCase(WorkerTestCase)]
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
