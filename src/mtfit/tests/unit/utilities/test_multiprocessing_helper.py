"""
test_multiprocessing_utils.py
*****************************

Tests for src/utils/multiprocessing_utils.py
"""
import sys
import unittest
import multiprocessing
import multiprocessing.queues


from MTfit.utilities.unittest_utils import TestCase
from MTfit.utilities.multiprocessing_helper import Worker
from MTfit.utilities.multiprocessing_helper import JobPool
from MTfit.utilities.multiprocessing_helper import PoisonPill
from MTfit.tests.unit.utilities.multiprocessing_test_classes import TaskTest
from MTfit.tests.unit.utilities.multiprocessing_test_classes import TaskTest2


class WorkerTestCase(TestCase):

    def setUp(self):
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.worker = Worker(self.task_queue, self.result_queue)
        self.worker.start()

    def tearDown(self):
        self.task_queue.put(PoisonPill())
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
        self.task_queue.put(TaskTest(1, 2))
        r = self.result_queue.get(timeout=10)
        self.assertTrue(self.result_queue.empty())
        self.assertEqual(r, 2)
        self.task_queue.put(TaskTest(1, 5))
        r = self.result_queue.get(timeout=10)
        self.assertTrue(self.result_queue.empty())
        self.assertEqual(r, 5)


class JobPoolTestCase(TestCase):

    def setUp(self, task=TaskTest, single_life=False):
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
        self.job_pool.custom_task(TaskTest2, 4, 2)
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
        self.job_pool.custom_task(TaskTest2, 2, 4)
        self.job_pool.custom_task(TaskTest2, 4, 1)
        self.job_pool.custom_task(TaskTest2, 4, 0)
        self.assertNotEqual(self.job_pool.all_results(), [6, 5, 4])

    def test_single_life(self):
        self.tearDown()
        self.setUp(single_life=True)
        self.assertEqual(self.job_pool.number_workers, 2)
        self.job_pool.custom_task(TaskTest2, 2, 4)
        self.job_pool.custom_task(TaskTest2, 4, 1)
        self.job_pool.custom_task(TaskTest2, 4, 0)
        self.job_pool.custom_task(TaskTest2, 8, 1)
        self.assertNotEqual(self.job_pool.all_results(), [6, 5, 4, 9])

    def test_exceptionBreak(self):
        self.job_pool.custom_task(TaskTest2, 2, 4)
        self.job_pool.custom_task(TaskTest2, "4", 1)
        self.job_pool.custom_task(TaskTest2, "4", 0)
        self.job_pool.custom_task(TaskTest2, "8", 1)
        results = self.job_pool.all_results()
        self.assertTrue(6 in results)
        results.pop(results.index(6))
        self.assertTrue(isinstance(results[0], TypeError))
