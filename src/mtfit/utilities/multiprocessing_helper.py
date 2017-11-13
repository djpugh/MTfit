"""
multiprocessing_utils.py
************************

Handles multiprocessing queue and worker behaviour, including return codes
"""

# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.

import multiprocessing
import gc
import os


from ..probability import LnPDF

#
# Job return codes
#
RETURN_CODES = [10, 20]


class PoisonPill(object):
    pass


#
# Worker
#


class Worker(multiprocessing.Process):

    """
    Worker object for multiprocessing

    The worker object is a multiprocessing process subclass and is used
    for running tasks. It can be killed after a single task or allowed
    to run multiple tasks. If running multiple tasks, uses a poison pill
    to exit.

    Initialisation
        Args
            task_queue: multiprocessing.Queue object for storing tasks
            result_queue: multiprocessing.Queue object for storing results
            single_life:[False] Boolean flag for killing worker after single job.

    """

    def __init__(self, task_queue, result_queue, single_life=False):
        """
        Worker initialisation

        Args
            task_queue: multiprocessing.Queue object for storing tasks
            result_queue: multiprocessing.Queue object for storing results
            single_life:[False] Boolean flag for killing worker after single job.
        """
        super(Worker, self).__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.single_life = single_life
        self.__closed__ = False
        print('{} Initialised'.format(self.name))
        print(self._Popen)

    def run(self):
        """
        Enable the worker

        Enables the worker to collect tasks and return results to their
        respective queues in the JobPool. The worker is killed if a
        poison pill is provided, or single_life is enabled.

        """
        while True:
            try:
                next_task = self.task_queue.get()
                if isinstance(next_task, PoisonPill):
                    break
                answer = next_task()
                if isinstance(answer, dict):
                    for key, value in answer.items():
                        if isinstance(value, LnPDF):
                            answer[key] = {'ln_pdf_obj': value.__getstate__()}
                self.result_queue.put(answer)
                if self.single_life:
                    break
                gc.collect()
            except Exception as e:
                self.result_queue.put(e)
                gc.collect()
                break
        return

    def start(self):
        print(self.name, 'Starting')
        super(Worker, self).start()
        print(self.name, 'Started', self._popen.pid)

    def join(self):
        print(self.name, 'Ending')
        super(Worker, self).join()
        print(self.name, 'Ended')


# Job Pool
class JobPool(object):

    """
    Manages workers and  queues for multiprocessing

    Simple object to manage the Worker objects and the task and result
    queues for multiprocessing. Has methods for adding a task and getting
    a result.
    """

    def __init__(self, number_workers=0, task=None, single_life=False):
        """
        Initialisation of JobPool

        Initialises the workers and the queues, and starts the workers.

        Args
            number_workers:[0] Number of Workers to initialise, default
                           is to use the number of workers given by
                           multiprocessing.cpu_count()
            task: [ForwardTask] Default task to use when adding new task
                  (self.custom_task() can be used for other tasks).
            single_life:[False] Boolean flag, sets whether each worker
                        has a single life or multiple lives, i.e. runs
                        multiple tasks or a single task before dying.

        """
        print('Initialising Job Pool')
        self.tasks = multiprocessing.Queue()
        self.results = multiprocessing.Queue()
        self.workers = []
        # Set default task
        self.task_class = task
        if not number_workers:
            number_workers = multiprocessing.cpu_count()
            if len([u for u in os.environ.keys() if 'PBS_' in u]):
                try:
                    number_workers = int(os.environ['PBS_NUM_PPN'])
                except Exception:
                    pass
        self.number_workers = number_workers
        self.single_life = single_life
        self.number_jobs = 0
        # Add new workers until the number of workers is correct
        self.clean_workers()
        print('Job Pool Initialised')

    def __len__(self):
        """
        Gets number of workers

        Returns
            integer number of workers.

        """
        return self.number_workers

    def task(self, *args):
        """
        Adds default task based on args

        Creates new default task (set during initialisation) using *args and adds it to the task queue

        Args
            args*: arguments for the custom task.

        """
        self.clean_workers()
        task = self.task_class(*args)
        self.tasks.put(task)
        self.number_jobs += 1

    def clean_workers(self):
        """
        Cleans workers

        Removes killed workers and adds new ones until the number of workers is correct.

        """
        for i, worker in enumerate(self.workers):
            if not worker.is_alive():
                if not self.single_life:
                    print(str(worker.name)+' Dead')
                self.workers.pop(i)
                del worker
        while len(self.workers) < self.number_workers:
            w = Worker(self.tasks, self.results, self.single_life)
            self.workers.append(w)
            w.start()
        gc.collect()

    def custom_task(self, custom_task, *args):
        """
        Adds a custom task(*args) to the queue.

        Args
            custom_task: Callable task object taking *args on initialisation.
            args*: arguments for the custom task.
        """
        self.clean_workers()
        task = custom_task(*args)
        self.tasks.put(task)
        self.number_jobs += 1

    def result(self):
        """

        Returns a result object (skipping tasks that return results in the RETURN_CODES)

        Returns
            Task result
        """
        # clean up dead workers
        self.clean_workers()
        # Get result
        result = self.results.get()
        self.number_jobs -= 1
        # Check if result is in return codes (errors or otherwise)
        if isinstance(result, int) and result in RETURN_CODES:
            result = self.result()
        return self.rebuild_ln_pdf(result)

    def rebuild_ln_pdf(self, result):
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, dict):
                    if list(value.keys()) == ['ln_pdf_obj']:
                        # This is a ln_pdf object so rebuild
                        result[key] = LnPDF(value['ln_pdf_obj'][0], dV=value['ln_pdf_obj'][1])
                    else:
                        result[key] = self.rebuild_ln_pdf(value)
        elif isinstance(result, list):
            for i, res in enumerate(result):
                result[i] = self.rebuild_ln_pdf(res)
        return result

    def all_results(self):
        """
        Returns all the results

        Blocking job to return all results from tasks, waits until all tasks in the queue are finished.
        Can cause issues with single_life - blocks until finished -

        Returns
            List of task results
        """
        results = []
        # Loop over all un-returned jobs
        while self.number_jobs:
            results.append(self.result())
        return results

    def close(self):
        """
        Closes worker pool

        Adds poison pill to tasks and then joins workers to close them.
        """
        self.clean_workers()
        for i in range(self.number_workers):
            # Add poison pill
            self.tasks.put(PoisonPill())
        # Join workers to main thread
        for w in self.workers:
            w.join()
