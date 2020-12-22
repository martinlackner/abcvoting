"""
Test if examples in directory examples/ work
"""

import pytest
from sys import modules


# noinspection PyUnresolvedReferences
def test_example01_py():
    from survey import example01


# noinspection PyUnresolvedReferences
def test_example02_py():
    from survey import example02


# noinspection PyUnresolvedReferences
def test_example03_py():
    from survey import example03


# noinspection PyUnresolvedReferences
def test_example04_py():
    from survey import example04


# noinspection PyUnresolvedReferences
def test_example05_py():
    from survey import example05


# noinspection PyUnresolvedReferences
def test_example06_py():
    from survey import example06


# noinspection PyUnresolvedReferences
def test_example07_py():
    from survey import example07


# noinspection PyUnresolvedReferences
def test_example08_py():
    from survey import example08 


# noinspection PyUnresolvedReferences
def test_example09_py():
    pytest.importorskip("gurobipy")
    from survey import example09


# noinspection PyUnresolvedReferences
def test_example10_py():
    from survey import example10


# noinspection PyUnresolvedReferences
def test_example11_py():
    from survey import example11


# noinspection PyUnresolvedReferences
def test_example12_py():
    from survey import example12


# noinspection PyUnresolvedReferences
def test_example13_py():
    from survey import example13 


# noinspection PyUnresolvedReferences
def test_remark02_py():
    from survey import remark02


# noinspection PyUnresolvedReferences
def test_remark03_py():
    from survey import remark03


# noinspection PyUnresolvedReferences
def test_propositionA2_py():
    pytest.importorskip("gurobipy")
    from survey import propositionA2


# noinspection PyUnresolvedReferences
def test_propositionA3_py():
    pytest.importorskip("gurobipy")
    from survey import propositionA3


# noinspection PyUnresolvedReferences
def test_propositionA4_py():
    from survey import propositionA4
