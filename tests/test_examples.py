import pytest
from pathlib import Path

# this is a workaround to print text, which is printed during import time, before tests are run
# to guarantee independence of test run order

from abcvoting.output import WARNING
from abcvoting.output import output


@pytest.fixture
def check_output(capfd, request):
    """Pytest fixture to compare output (stdout) with stored text file. Output might depend on
    installed packages, might need to be adjusted to make test work on all platforms.

    If a test fails, the actual output is copied to a file called <testname>.new, so it should
    be easy to accept changes by `mv expected_output/<testnaem>.new survey_output/<testnaem>`.
    """
    # reset verbosity, because might have been modified, this is just paranoia
    output.set_verbosity(WARNING)

    yield

    # reset verbosity, examples modify the verbosity level
    output.set_verbosity(WARNING)

    stdout = capfd.readouterr().out
    test_name = request.node.name
    fname = Path(__file__).parent / "expected_output" / test_name
    try:
        with open(fname, "r", encoding="utf8") as file:
            expected_output = file.read()
    except FileNotFoundError:
        expected_output = None

    if expected_output != stdout:
        with open(f"{fname}.new", "w", encoding="utf8") as file:
            file.write(stdout)

    assert expected_output == stdout, f"Unexpected output, output written to {fname}.new"


def test_abcsurvey_example01_py(check_output):
    from examples.abcsurvey import example01

    example01.print_profile()


# noinspection PyUnresolvedReferences
def test_abcsurvey_example02_py(check_output):
    from examples.abcsurvey import example02


# noinspection PyUnresolvedReferences
def test_abcsurvey_example03_py(check_output):
    from examples.abcsurvey import example03


# noinspection PyUnresolvedReferences
def test_abcsurvey_example04_py(check_output):
    from examples.abcsurvey import example04


# noinspection PyUnresolvedReferences
def test_abcsurvey_example05_py(check_output):
    from examples.abcsurvey import example05


# noinspection PyUnresolvedReferences
def test_abcsurvey_example06_py(check_output):
    from examples.abcsurvey import example06


# noinspection PyUnresolvedReferences
def test_abcsurvey_example07_py(check_output):
    from examples.abcsurvey import example07


# noinspection PyUnresolvedReferences
def test_abcsurvey_example08_py(check_output):
    from examples.abcsurvey import example08


# noinspection PyUnresolvedReferences
@pytest.mark.gurobi
def test_abcsurvey_example09_py():
    # TODO check_ouput test does not work here because of
    #      undeterministic Gurobi output
    from examples.abcsurvey import example09


# noinspection PyUnresolvedReferences
def test_abcsurvey_example10_py(check_output):
    from examples.abcsurvey import example10


# noinspection PyUnresolvedReferences
def test_abcsurvey_example11_py(check_output):
    from examples.abcsurvey import example11


# noinspection PyUnresolvedReferences
def test_abcsurvey_example12_py(check_output):
    from examples.abcsurvey import example12


# noinspection PyUnresolvedReferences
def test_abcsurvey_example13_py(check_output):
    from examples.abcsurvey import example13


# noinspection PyUnresolvedReferences
def test_abcsurvey_remark02_py(check_output):
    from examples.abcsurvey import remark02


# noinspection PyUnresolvedReferences
def test_abcsurvey_remark03_py(check_output):
    from examples.abcsurvey import remark03


# noinspection PyUnresolvedReferences
@pytest.mark.gurobi
def test_abcsurvey_propositionA2_py(check_output):
    from examples.abcsurvey import propositionA2


# noinspection PyUnresolvedReferences
@pytest.mark.gurobi
def test_abcsurvey_propositionA3_py(check_output):
    from examples.abcsurvey import propositionA3


# noinspection PyUnresolvedReferences
def test_abcsurvey_propositionA4_py(check_output):
    from examples.abcsurvey import propositionA4


# noinspection PyUnresolvedReferences
def test_simple_py(check_output):
    from examples import simple


# noinspection PyUnresolvedReferences
@pytest.mark.gurobi
def test_allrules_py(check_output):
    from examples import allrules


# noinspection PyUnresolvedReferences
def test_preflib_py(check_output):
    from examples import handling_preflib_files


# noinspection PyUnresolvedReferences
def test_random_profiles_py(check_output):
    from examples import random_profiles
