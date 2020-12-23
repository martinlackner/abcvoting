"""
Unit tests for scoring_functions.py
"""

import pytest

from abcvoting.preferences import Profile
from abcvoting.scores import monroescore_flowbased
from abcvoting.scores import monroescore_matching
from abcvoting import scores
try:
    from gmpy2 import mpq as Fraction
except ImportError:
    from fractions import Fraction


@pytest.mark.parametrize(
    "committee,score",
    [([1, 3, 2], 5), ([2, 1, 5], 4), ([2, 5, 4], 3),
     ([1, 2, 5, 4], 5), ([0, 2, 4, 5], 4),
     ([0, 1, 3, 4, 5], 5), ([0, 1, 2, 4, 5], 6),
     ([0, 1, 2, 3, 4, 5], 6)]
)
@pytest.mark.parametrize(
    "num_cand",
    [6, 7, 8]
)
def test_monroescore_flowbased(committee, score, num_cand):
    profile = Profile(num_cand)
    preflist = [[0, 1], [1], [1, 3], [4], [2], [1, 5, 3]]
    profile.add_voters(preflist)

    assert monroescore_flowbased(profile, committee) == score


@pytest.mark.parametrize(
    "committee,score",
    [([1, 3, 2], 5), ([2, 1, 5], 4), ([2, 5, 4], 3),
     ([0, 1, 2, 3, 4, 5], 6)]
)
@pytest.mark.parametrize(
    "num_cand", [6, 7, 8]
)
def test_monroescore_matching(committee, score, num_cand):
    profile = Profile(num_cand)
    preflist = [[0, 1], [1], [1, 3], [4], [2], [1, 5, 3]]
    profile.add_voters(preflist)

    assert monroescore_matching(profile, committee) == score


@pytest.mark.parametrize(
    "scorefct_str,score", [("pav", Fraction(119, 12)), ("av", 14), ("slav", Fraction(932, 105)),
                           ("cc", 7), ("geom2", Fraction(77, 8))]
)
@pytest.mark.parametrize(
    "num_cand", [8, 9]
)
def test_thiele_scores(scorefct_str, score, num_cand):
    profile = Profile(num_cand)
    preflist = [[0, 1], [1], [1, 3], [4], [1, 2, 3, 4, 5], [1, 5, 3], [0, 1, 2, 4, 5]]
    profile.add_voters(preflist)
    committee = [6, 7]
    assert scores.thiele_score(scorefct_str, profile, committee) == 0
    committee = [1, 2, 3, 4]
    assert scores.thiele_score(scorefct_str, profile, committee) == score
