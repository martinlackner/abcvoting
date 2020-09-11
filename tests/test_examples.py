"""
Test if examples in directory examples/ work
"""

import pytest


# noinspection PyUnresolvedReferences
def test_simple_py():
    from examples import simple


# noinspection PyUnresolvedReferences
def test_allrules_py():
    pytest.importorskip("gurobipy")
    from examples import allrules


# noinspection PyUnresolvedReferences
def test_preflib_py():
    from examples import handling_preflib_files


# noinspection PyUnresolvedReferences
def test_random_profiles_py():
    from examples import random_profiles
