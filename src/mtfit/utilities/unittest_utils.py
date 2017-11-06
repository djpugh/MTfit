"""unittest_utils
******************
Provides test functions for running and debugging unit tests.
"""

# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.

import unittest
import sys
import time
from unittest.signals import registerResult
try:
    import ipdb
except Exception:
    import pdb as ipdb

import numpy as np


class TestError(RuntimeError):
    pass


class TextTestResult(unittest.TextTestResult):

    def __init__(self, *args, **kwargs):
        super(TextTestResult, self).__init__(*args, **kwargs)
        self.not_implemented = []
        self.__doc__ = super(TextTestResult, self).__doc__

    def addError(self, test, err):
        # Add NotImplementedError handling
        if isinstance(err[1], NotImplementedError):
            self.not_implemented.append(
                (test, self._exc_info_to_string(err, test)))
            self._mirrorOutput = True
            if self.showAll:
                self.stream.writeln("not implemented")
            elif self.dots:
                self.stream.write('n')
                self.stream.flush()
        else:
            super(TextTestResult, self).addError(test, err)


class TextTestRunner(unittest.TextTestRunner):
    resultclass = TextTestResult

    def run(self, test, test_result=None):
        "Run the given test case or test suite."
        # Monkey patch on unittest class to add not implemented errors into run
        if test_result is None:
            test_result = self._makeResult()
        registerResult(test_result)
        test_result.failfast = self.failfast
        test_result.buffer = self.buffer
        startTime = time.time()
        startTestRun = getattr(test_result, 'startTestRun', None)
        if startTestRun is not None:
            startTestRun()
        try:
            test(test_result)
        finally:
            stopTestRun = getattr(test_result, 'stopTestRun', None)
            if stopTestRun is not None:
                stopTestRun()
        stopTime = time.time()
        timeTaken = stopTime - startTime
        test_result.printErrors()
        if hasattr(test_result, 'separator2'):
            self.stream.writeln(test_result.separator2)
        run = test_result.testsRun
        self.stream.writeln("Ran %d test%s in %.3fs" %
                            (run, run != 1 and "s" or "", timeTaken))
        self.stream.writeln()

        expectedFails = unexpectedSuccesses = skipped = 0
        try:
            results = map(len, (test_result.expectedFailures,
                                test_result.unexpectedSuccesses,
                                test_result.skipped,
                                test_result.not_implemented))
        except AttributeError:
            pass
        else:
            expectedFails, unexpectedSuccesses, skipped, not_implemented = results

        infos = []
        if not test_result.wasSuccessful():
            self.stream.write("FAILED")
            failed, errored = map(len, (test_result.failures, test_result.errors))
            if failed:
                infos.append("failures=%d" % failed)
            if errored:
                infos.append("errors=%d" % errored)
        else:
            self.stream.write("OK")
        if skipped:
            infos.append("skipped=%d" % skipped)
        if expectedFails:
            infos.append("expected failures=%d" % expectedFails)
        if unexpectedSuccesses:
            infos.append("unexpected successes=%d" % unexpectedSuccesses)
        if not_implemented:
            infos.append("not implemented=%d" % not_implemented)

        if infos:
            self.stream.writeln(" (%s)" % (", ".join(infos),))
        else:
            self.stream.write("\n")
        return test_result


class TestCase(unittest.TestCase):

    def assertAlmostEqual(self, first, second, places=None, msg=None, delta=None):

        if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
            # Handle array and vectors
            if isinstance(second, (list, float, int)):
                second = np.array(second)
            if isinstance(first, (list, float, int)):
                first = np.array(first)
            if len([u for u in second.shape if u != 1]) == len([u for u in first.shape if u != 1]):
                if places is not None:
                    np.testing.assert_array_almost_equal(np.array(first).squeeze(), np.array(second).squeeze(), places)
                else:
                    np.testing.assert_array_almost_equal(np.array(first).squeeze(), np.array(second).squeeze())
            else:
                if places is not None:
                    np.testing.assert_array_almost_equal(first, second, places)
                else:
                    np.testing.assert_array_almost_equal(first, second)
            return
        elif isinstance(first, dict) and isinstance(second, dict):
            self.assertEqual(sorted(first.keys()), sorted(second.keys()), 'Dictionary keys do not match')
            for key in first.keys():
                self.assertAlmostEqual(first[key], second[key])
            return
        elif isinstance(first, list) and isinstance(second, list):
            super(TestCase, self).assertAlmostEqual(set(first), set(second), msg, delta)
        else:
            super(TestCase, self).assertAlmostEqual(first, second, places, msg, delta)

    def assertAlmostEquals(self, *args, **kwargs):
        self.assertAlmostEqual(*args, **kwargs)

    def assertEqual(self, first, second, msg=None):
        if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
            # Handle matrix and vector

            np.testing.assert_array_equal(first, second)
            return
        elif isinstance(first, dict) and isinstance(second, dict):
            self.assertEqual(
                sorted(first.keys()), sorted(second.keys()), 'Dictionary keys do not match')
            for key in first.keys():
                self.assertEqual(first[key], second[key])
            return
        elif isinstance(first, list) and isinstance(second, list):
            super(TestCase, self).assertEqual(first, second, msg)

        else:
            super(TestCase, self).assertAlmostEqual(first, second, msg)

    def assertEquals(self, *args, **kwargs):
        self.assertEqual(*args, **kwargs)

    def assertVectorEquals(self, first, second, *args):
        try:
            first_norm = np.sqrt(np.sum(np.multiply(first, first)))
            second_norm = np.sqrt(np.sum(np.multiply(second, second)))
            return self.assertAlmostEquals(first/first_norm, second/second_norm, *args)
        except AssertionError as e1:
            try:
                return self.assertAlmostEquals(-first/first_norm, second/second_norm, *args)
            except AssertionError as e2:
                if sys.version_info.major > 2:
                    raise AssertionError('{} or {}'.format(e1.args, e2.args))
                else:
                    raise AssertionError(e1.message+' or '+e2.message)


def run_tests(suite, verbosity=2, test_result=None):
    """Run the test suite tests
    Args
        suite: unittest.TestSuite - input test suite.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    test_runner = TextTestRunner(verbosity=verbosity)
    test_result = test_runner.run(suite, test_result)
    test_runner.stream.flush()
    try:
        test_runner.stream.stream.flush()
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    return test_result


def debug_tests(suite, _print=True):
    """Run the test suite tests, with a debugger for errors.
    Args
        suite: unittest.TestSuite - input test suite.
    _print is a recursion specific keyword.
    """
    if _print:
        print('Running tests in debugger.')
    expected = 0
    exceptions = 0
    fails = 0
    not_implemented = 0
    skipped = 0
    n = 0
    start_time = time.time()
    for test in suite:
        if not isinstance(test, unittest.TestCase):
            n_, expected_, exceptions_, fails_, skipped_, not_implemented_ = debug_tests(
                test, False)
            n += n_
            expected += expected_
            exceptions += exceptions_
            fails += fails_
            skipped += skipped_
            not_implemented += not_implemented_
        else:
            n += 1
            try:
                test.debug()
                print(str(test)+' ... ok')
            except Exception as e:
                if isinstance(e, unittest.case._ExpectedFailure):
                    expected += 1
                    print(str(test)+' ... expected failure')
                if isinstance(e, unittest.case.SkipTest):
                    skipped += 1
                    print(str(test)+' ... skipped')
                elif isinstance(e, AssertionError):
                    fails += 1
                    print(str(test))
                    ipdb.post_mortem(sys.exc_info()[2])
                elif isinstance(e, NotImplementedError):
                    not_implemented += 1
                    print(str(test)+' ... not implemented')
                else:
                    exceptions += 1
                    print(test._testMethodName)
                    try:
                        from IPython.core.ultratb import VerboseTB
                        vtb = VerboseTB(call_pdb=1)
                        vtb(*sys.exc_info())
                    except Exception:
                        import traceback
                        print('\n')
                        traceback.print_exc()
                    ipdb.post_mortem(sys.exc_info()[2])
    if _print:
        end_time = time.time()
        run_time = end_time-start_time
        print(
            '----------------------------------------------------------------------')
        print('Ran '+str(n)+' tests in {:.3f}s\n'.format(run_time))
        result_str = ''
        if fails <= 0 and exceptions <= 0:
            result_str += 'OK '
        else:
            result_str += 'FAILED '
        end_bracket = False
        comma = False
        results = [(exceptions, 'exceptions'), (fails, 'failures'),
                   (expected, 'expected failures'), (skipped, 'skipped'), (not_implemented, 'not implemented')]
        if fails > 0 or exceptions > 0 or expected > 0 or skipped > 0 or not_implemented > 0:
            result_str += '('
            end_bracket = True
            for result_count, result_label in results:
                if result_count > 0:
                    if comma:
                        result_str += ',  '
                    result_str += result_label+'='+str(result_count)
                    comma = True
        if end_bracket:
            result_str += ')'
        print(result_str)

    return n, expected, exceptions, fails, skipped, not_implemented
