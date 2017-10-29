from mtfit.tests.unit import run_tests as run_unit_tests
# from mtfit.tests.functional import run_tests as run_functional_tests

if __name__ == "__main__":
    results = run_unit_tests(2)

    # if passed:
    #     run_functional_tests(1)
    # else:
    #     raise Exception('Tests failed')
