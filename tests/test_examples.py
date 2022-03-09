"""
Unit tests for examples/*.
"""

import pytest
from pathlib import Path

# this is a workaround to print text, which is printed during import time, before tests are run
# to guarantee independence of test run order

from abcvoting.output import WARNING
from abcvoting.output import output
from test_abcrules import remove_solver_output
import re


def remove_algorithm_info(out):
    """Remove information about algorithms which may differ from system to system or are random."""
    filter_patterns = (
        "Algorithm: .*\n",
        "----------------------\nRandom Serial Dictator\n----------------------"
        + "\n\n1 winning committee:\n {., ., ., .}",
    )

    for filter_pattern in filter_patterns:
        out = re.sub(filter_pattern, "", out)

    assert "Random Serial Dictator" not in out
    return out


@pytest.fixture
def check_output(capfd, request):
    """
    Pytest fixture to compare output (stdout) with stored text file.

    Output might depend on
    installed packages, might need to be adjusted to make test work on all platforms.
    If a test fails, the actual output is copied to a file called <testname>.new, so it should
    be easy to accept changes by `mv expected_output/<testname>.new expected_output/<testname>`.
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
        expected_output = remove_algorithm_info(str(expected_output))
    except FileNotFoundError:
        expected_output = None

    stdout = remove_solver_output(str(stdout))
    stdout = remove_algorithm_info(stdout)

    if expected_output != stdout:
        with open(f"{fname}.new", "w", encoding="utf8") as file:
            file.write(stdout)

    assert expected_output == stdout, f"Unexpected output, output written to {fname}.new"


# noinspection PyUnresolvedReferences
def test_abcbook_example01_py(check_output):
    from examples.abcbook import example201  # noqa: F401


# noinspection PyUnresolvedReferences
def test_abcbook_example02_py(check_output):
    from examples.abcbook import example202  # noqa: F401


# noinspection PyUnresolvedReferences
def test_abcbook_example03_py(check_output):
    from examples.abcbook import example203  # noqa: F401


# noinspection PyUnresolvedReferences
def test_abcbook_example04_py(check_output):
    from examples.abcbook import example204  # noqa: F401


# noinspection PyUnresolvedReferences
def test_abcbook_example05_py(check_output):
    from examples.abcbook import example205  # noqa: F401


# noinspection PyUnresolvedReferences
def test_abcbook_example06_py(check_output):
    from examples.abcbook import example206  # noqa: F401


# noinspection PyUnresolvedReferences
def test_abcbook_example07_py(check_output):
    from examples.abcbook import example207  # noqa: F401


# noinspection PyUnresolvedReferences
def test_abcbook_example08_py(check_output):
    from examples.abcbook import example208  # noqa: F401


# noinspection PyUnresolvedReferences
def test_abcbook_example09_py(check_output):
    from examples.abcbook import example209  # noqa: F401


# noinspection PyUnresolvedReferences
def test_abcbook_example10_py(check_output):
    from examples.abcbook import example210  # noqa: F401


# noinspection PyUnresolvedReferences
def test_abcbook_example11_py(check_output):
    from examples.abcbook import example211  # noqa: F401


# noinspection PyUnresolvedReferences
def test_abcbook_example12_py(check_output):
    from examples.abcbook import example212  # noqa: F401


# noinspection PyUnresolvedReferences
def test_abcbook_example13_py(check_output):
    from examples.abcbook import example213  # noqa: F401


# noinspection PyUnresolvedReferences
def test_abcbook_example14_py(check_output):
    from examples.abcbook import example214  # noqa: F401


# noinspection PyUnresolvedReferences
def test_abcbook_remark02_py(check_output):
    from examples.abcbook import remark02  # noqa: F401


# noinspection PyUnresolvedReferences
def test_abcbook_propositionA2_py(check_output):
    from examples.abcbook import propositionA2  # noqa: F401


# noinspection PyUnresolvedReferences
def test_abcbook_propositionA3_py(check_output):
    from examples.abcbook import propositionA3  # noqa: F401


@pytest.mark.gurobi
def test_abcbook_propositionA4_py(check_output):
    from examples.abcbook import propositionA4  # noqa: F401


# noinspection PyUnresolvedReferences
def test_simple_py(check_output):
    from examples import simple  # noqa: F401


@pytest.mark.gurobi
def test_allrules_py(check_output):
    from examples import allrules  # noqa: F401


# noinspection PyUnresolvedReferences
def test_preflib_py(check_output):
    from examples import handling_preflib_files  # noqa: F401
