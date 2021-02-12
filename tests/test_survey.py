"""
Test if examples in directory examples/ work
"""
import pytest
from pathlib import Path

# this is a workaround to print text, which is printed during import time, before tests are run
# to guarantee independence of test run order
import abcvoting.abcrules
import abcvoting.abcrules_gurobi


@pytest.fixture
def check_output(capfd, request):
    """Pytest fixture to compare output (stdout) with stored text file. Output might depend on
    installed packages, might need to be adjusted to make test work on all platforms.

    If a test fails, the actual output is copied to a file called <testname>.new, so it should
    be easy to accept changes by `mv survey_output/<testnaem>.new survey_output/<testnaem>`.
    """
    yield
    output = capfd.readouterr().out
    test_name = request.node.name
    fname = Path(__file__).parent / "survey_output" / test_name
    try:
        with open(fname, "r", encoding="utf8") as file:
            expected_output = file.read()
    except FileNotFoundError:
        expected_output = None

    if expected_output != output:
        with open(f"{fname}.new", "w", encoding="utf8") as file:
            file.write(output)

    assert expected_output == output, f"Unexpected output, output written to {fname}.new"


# noinspection PyUnresolvedReferences
def test_example01_py(check_output):
    # Note: this test does not output anything if imported, prints only when run as script.
    from survey import example01


# noinspection PyUnresolvedReferences
def test_example02_py(check_output):
    from survey import example02


# noinspection PyUnresolvedReferences
def test_example03_py(check_output):
    from survey import example03


# noinspection PyUnresolvedReferences
def test_example04_py(check_output):
    from survey import example04


# noinspection PyUnresolvedReferences
def test_example05_py(check_output):
    from survey import example05


# noinspection PyUnresolvedReferences
def test_example06_py(check_output):
    from survey import example06


# noinspection PyUnresolvedReferences
def test_example07_py(check_output):
    from survey import example07


# noinspection PyUnresolvedReferences
def test_example08_py(check_output):
    from survey import example08


# noinspection PyUnresolvedReferences
@pytest.mark.gurobi
def test_example09_py(check_output):
    from survey import example09


# noinspection PyUnresolvedReferences
def test_example10_py(check_output):
    from survey import example10


# noinspection PyUnresolvedReferences
def test_example11_py(check_output):
    from survey import example11


# noinspection PyUnresolvedReferences
def test_example12_py(check_output):
    from survey import example12


# noinspection PyUnresolvedReferences
def test_example13_py(check_output):
    from survey import example13


# noinspection PyUnresolvedReferences
def test_remark02_py(check_output):
    from survey import remark02


# noinspection PyUnresolvedReferences
def test_remark03_py(check_output):
    from survey import remark03


# noinspection PyUnresolvedReferences
@pytest.mark.gurobi
def test_propositionA2_py(check_output):
    from survey import propositionA2


# noinspection PyUnresolvedReferences
@pytest.mark.gurobi
def test_propositionA3_py(check_output):
    from survey import propositionA3


# noinspection PyUnresolvedReferences
def test_propositionA4_py(check_output):
    from survey import propositionA4
