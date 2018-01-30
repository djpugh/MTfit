from MTfit.tests.unit import run_tests as run_unit_tests
# from MTfit.tests.functional import run_tests as run_functional_tests

if __name__ == "__main__":
    results = run_unit_tests(1)
    if not results.wasSuccessful():
        raise ValueError('Tests failed')
