"""
Test if examples in directory examples/ work
"""

import pytest


def test_simple_py():
    from examples import simple


def test_allrules_py():
    pytest.importorskip("gurobipy")
    from examples import allrules


def test_preflib_py():
    from examples import handling_preflib_files


def test_random_profiles_py():
    from examples import random_profiles
