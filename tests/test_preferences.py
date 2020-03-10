"""
Unit tests for preferences.py
"""

import pytest
from abcvoting.preferences import Profile
from abcvoting.preferences import DichotomousPreferences


def test_invalidpreferences():
    with pytest.raises(ValueError):
        DichotomousPreferences([-1])


@pytest.mark.parametrize(
    "num_cand", [6, 7, 8]
)
def test_invalidprofiles(num_cand):
    with pytest.raises(ValueError):
        Profile(0)
    with pytest.raises(ValueError):
        Profile(-8)
    profile = Profile(num_cand)
    pref = DichotomousPreferences([num_cand])
    with pytest.raises(ValueError):
        profile.add_preferences(pref)
    with pytest.raises(TypeError):
        profile.add_preferences([0, 4, 5, "1"])
    with pytest.raises(TypeError):
        profile.add_preferences(["1", 0, 4, 5])
    with pytest.raises(TypeError):
        profile.add_preferences({1: [1, 2]})


@pytest.mark.parametrize(
    "num_cand", [6, 7, 8]
)
def test_unitweights(num_cand):
    profile = Profile(num_cand)
    profile.add_preferences([])
    profile.add_preferences(DichotomousPreferences([0, 4, 5]))
    profile.add_preferences([0, 4, 5])
    p1 = DichotomousPreferences([0, 4, 5])
    p2 = DichotomousPreferences([1, 2])
    profile.add_preferences([p1, p2])
    assert profile.has_unit_weights()

    profile.add_preferences(DichotomousPreferences([0, 4, 5], 2.4))
    assert not profile.has_unit_weights()

    assert profile.totalweight() == 6.4

@pytest.mark.parametrize(
    "num_cand", [6, 7, 8]
)
def test_iterate(num_cand):
    profile = Profile(num_cand)
    profile.add_preferences(DichotomousPreferences([1, 3, 5], 3))
    profile.add_preferences([0, 4, 5])
    assert len(profile) == 2
    for p in profile:
        assert type(p) is DichotomousPreferences
