"""
Unit tests for preferences.py
"""

import pytest
from abcvoting.preferences import Profile
from abcvoting.preferences import DichotomousPreferences


def test_invalidpreferences():
    with pytest.raises(ValueError):
        DichotomousPreferences([-1])

    with pytest.raises(ValueError):
        DichotomousPreferences([1]).check_valid(1)

    with pytest.raises(TypeError):
        DichotomousPreferences([0.42])

    with pytest.raises(ValueError):
        DichotomousPreferences([1, 1, 2, 42])


@pytest.mark.parametrize(
    "num_cand", [6, 8]
)
def test_invalidprofiles(num_cand):
    with pytest.raises(ValueError):
        Profile(0)
    with pytest.raises(ValueError):
        Profile(-8)
    with pytest.raises(ValueError):
        Profile(4, ["a", "b", "c"])
    Profile(4, ["a", 3, "b", "c"])
    profile = Profile(num_cand, "abcdefgh")
    pref = DichotomousPreferences([num_cand])
    with pytest.raises(ValueError):
        profile.add_preference(pref)
    with pytest.raises(TypeError):
        profile.add_preference([0, 4, 5, "1"])
    with pytest.raises(TypeError):
        profile.add_preference(["1", 0, 4, 5])

    with pytest.raises(TypeError):
        # note: this raises a TypeError because a list of lists can't be converted to a set,
        # but that's fine too
        profile.add_preference([[0, 4, 5]])


@pytest.mark.parametrize(
    "num_cand", [6, 7]
)
def test_unitweights(num_cand):
    profile = Profile(num_cand)
    profile.add_preferences([])
    profile.add_preference(DichotomousPreferences([0, 4, 5]))
    profile.add_preference([0, 4, 5])
    p1 = DichotomousPreferences([0, 4, 5])
    p2 = DichotomousPreferences([1, 2])
    profile.add_preferences([p1, p2])
    assert profile.has_unit_weights()

    profile.add_preference(DichotomousPreferences([0, 4, 5], 2.4))
    assert not profile.has_unit_weights()

    assert profile.totalweight() == 6.4


@pytest.mark.parametrize(
    "num_cand", [6, 7]
)
def test_iterate(num_cand):
    profile = Profile(num_cand)
    profile.add_preference(DichotomousPreferences([1, 3, 5], 3))
    profile.add_preference([0, 4, 5])
    assert len(profile) == 2
    for p in profile:
        assert isinstance(p, DichotomousPreferences)


@pytest.mark.parametrize(
    "add_pref",
    [[3], [1, 5], [0], [2], [2, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
)
@pytest.mark.parametrize(
    "num_cand", [8, 9]
)
def test_party_list(num_cand, add_pref):
    profile = Profile(num_cand)
    profile.add_preference(DichotomousPreferences([1, 3, 5], 3))
    profile.add_preference([0, 4, 6])
    profile.add_preference([0, 4, 6])
    profile.add_preference([2, 7])
    profile.add_preference(DichotomousPreferences([1, 3, 5], 3))
    assert profile.party_list()
    profile.add_preference(add_pref)
    assert not profile.party_list()
