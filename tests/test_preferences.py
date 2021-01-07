"""
Unit tests for preferences.py
"""

import pytest
from abcvoting.preferences import Profile
from abcvoting.preferences import ApprovalSet


def test_invalid_approvalsets():
    with pytest.raises(ValueError):
        ApprovalSet([-1])

    with pytest.raises(ValueError):
        ApprovalSet([1]).check_valid(1)

    with pytest.raises(TypeError):
        ApprovalSet([0.42])

    with pytest.raises(ValueError):
        ApprovalSet([1, 1, 2, 42])


@pytest.mark.parametrize("num_cand", [6, 8])
def test_invalidprofiles(num_cand):
    with pytest.raises(ValueError):
        Profile(0)
    with pytest.raises(ValueError):
        Profile(-8)
    with pytest.raises(ValueError):
        Profile(4, ["a", "b", "c"])
    Profile(4, ["a", 3, "b", "c"])
    profile = Profile(num_cand, "abcdefgh")
    pref = ApprovalSet([num_cand])
    with pytest.raises(ValueError):
        profile.add_voter(pref)
    with pytest.raises(TypeError):
        profile.add_voter([0, 4, 5, "1"])
    with pytest.raises(TypeError):
        profile.add_voter(["1", 0, 4, 5])

    with pytest.raises(TypeError):
        # note: this raises a TypeError because a list of lists can't be converted to a set,
        # but that's fine too
        profile.add_voter([[0, 4, 5]])


@pytest.mark.parametrize("num_cand", [6, 7])
def test_empty_approval(num_cand):
    profile = Profile(num_cand)
    profile.add_voter([])
    profile.add_voters([[]])
    profile.add_voters([[], [0, 3], []])

    profile = Profile(num_cand)
    profile.add_voters([[]])


@pytest.mark.parametrize("num_cand", [6, 7])
def test_unitweights(num_cand):
    profile = Profile(num_cand)
    profile.add_voters([])
    profile.add_voter(ApprovalSet([0, 4, 5]))
    profile.add_voter([0, 4, 5])
    p1 = ApprovalSet([0, 4, 5])
    p2 = ApprovalSet([1, 2])
    profile.add_voters([p1, p2])
    assert profile.has_unit_weights()

    profile.add_voter(ApprovalSet([0, 4, 5], 2.4))
    assert not profile.has_unit_weights()

    assert profile.totalweight() == 6.4


@pytest.mark.parametrize("num_cand", [6, 7])
def test_iterate(num_cand):
    profile = Profile(num_cand)
    profile.add_voter(ApprovalSet([1, 3, 5], 3))
    profile.add_voter([0, 4, 5])
    assert len(profile) == 2
    for p in profile:
        assert isinstance(p, ApprovalSet)


@pytest.mark.parametrize(
    "additional_apprset", [[3], [1, 5], [0], [2], [2, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
)
@pytest.mark.parametrize("num_cand", [8, 9])
def test_party_list(num_cand, additional_apprset):
    profile = Profile(num_cand)
    profile.add_voter(ApprovalSet([1, 3, 5], 3))
    profile.add_voter([0, 4, 6])
    profile.add_voter([0, 4, 6])
    profile.add_voter([2, 7])
    profile.add_voter(ApprovalSet([1, 3, 5], 3))
    assert profile.party_list()
    profile.add_voter(additional_apprset)
    assert not profile.party_list()
