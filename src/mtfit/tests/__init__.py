from MTfit.tests.unit import run_tests as run_unit_tests
from MTfit.tests.unit import test_suite as unit_test_suite


def test_suite():
    return unit_test_suite(2)


if __name__ == "__main__":
    results = run_unit_tests(1)
