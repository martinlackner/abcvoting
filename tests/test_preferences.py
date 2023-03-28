"""
Unit tests for abcvoting/preferences.py.
"""

import pytest
from abcvoting.preferences import Profile, Voter
from abcvoting.misc import CandidateSet


def test_invalid_approvalsets():
    with pytest.raises(ValueError):
        Voter([-1])

    with pytest.raises(ValueError):
        Voter([1], num_cand=1)

    with pytest.raises(TypeError):
        Voter([0.42])

    with pytest.raises(ValueError):
        Voter([1, 1, 2, 42])

    with pytest.raises(ValueError):
        CandidateSet([-1])

    with pytest.raises(ValueError):
        CandidateSet([1], num_cand=1)

    with pytest.raises(TypeError):
        CandidateSet([0.42])

    with pytest.raises(ValueError):
        CandidateSet([1, 1, 2, 42])


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
    voter = Voter([num_cand])
    with pytest.raises(ValueError):
        profile.add_voter(voter)
    with pytest.raises(TypeError):
        profile.add_voter([0, 4, 5, "1"])
    with pytest.raises(TypeError):
        profile.add_voter(["1", 0, 4, 5])

    with pytest.raises(TypeError):
        # note: this raises a TypeError because a list of lists can't be converted to a set,
        # but that's fine too
        profile.add_voter([[0, 4, 5]])


def test_invalidweights():
    with pytest.raises(ValueError):
        Voter([0, 4, 5], weight=-1)
    with pytest.raises(ValueError):
        Voter([0, 4, 5], weight=0)


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
    profile.add_voter(Voter([0, 4, 5]))
    profile.add_voter([0, 4, 5])
    p1 = Voter([0, 4, 5])
    p2 = Voter([1, 2])
    profile.add_voters([p1, p2])
    assert profile.has_unit_weights()

    assert len(profile) == 4
    assert profile.total_weight() == 4

    profile.convert_to_weighted()
    assert len(profile) == 2
    assert profile.total_weight() == 4
    assert not profile.has_unit_weights()

    profile.convert_to_unit_weights()
    assert len(profile) == 4
    assert profile.total_weight() == 4
    assert profile.has_unit_weights()

    profile.add_voter(Voter([0, 4, 5], 2.4))
    assert not profile.has_unit_weights()
    assert profile.total_weight() == 6.4

    with pytest.raises(TypeError):
        # requires integer weights
        profile.convert_to_unit_weights()



@pytest.mark.parametrize("num_cand", [6, 7])
def test_iterate(num_cand):
    profile = Profile(num_cand)
    profile.add_voter(Voter([1, 3, 5], 3))
    profile.add_voter([0, 4, 5])
    assert len(profile) == 2
    for p in profile:
        assert isinstance(p, Voter)


@pytest.mark.parametrize(
    "additional_approval_set", [[3], [1, 5], [0], [2], [2, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
)
@pytest.mark.parametrize("num_cand", [8, 9])
def test_party_list(num_cand, additional_approval_set):
    profile = Profile(num_cand)
    profile.add_voter(Voter([1, 3, 5], 3))
    profile.add_voter([0, 4, 6])
    profile.add_voter([0, 4, 6])
    profile.add_voter([2, 7])
    profile.add_voter(Voter([1, 3, 5], 3))
    assert profile.is_party_list()
    profile.add_voter(additional_approval_set)
    assert not profile.is_party_list()


def test_approved_candidates():
    profile = Profile(10)
    profile.add_voter(Voter([1, 3, 5], 3))
    profile.add_voter([0])
    assert profile.approved_candidates() == {0, 1, 3, 5}
    profile.add_voter([4])
    assert profile.approved_candidates() == {0, 1, 3, 4, 5}
    profile.add_voters([[7], [1, 8]])
    assert profile.approved_candidates() == {0, 1, 3, 4, 5, 7, 8}
    profile[0].approved = [1, 5]
    assert profile.approved_candidates() == {0, 1, 4, 5, 7, 8}

def test_copy_profile():
    profile = Profile(10)
    profile.add_voter(Voter([1, 3, 5], 3))
    profile.add_voter(Voter([2]))
    copy = profile.copy()
    assert profile.num_cand == copy.num_cand
    for voter, voter_copy in zip(profile, copy):
        assert voter.approved == voter_copy.approved
        assert voter.weight == voter_copy.weight
        assert voter is not voter_copy
    copy.add_voter(Voter([4], 2))
    assert len(profile) != len(copy)

def test_voter_str():
    v = Voter({0,1})
    assert str(v) == '{0, 1}'
    assert v.str_with_names() == '{0, 1}'
    assert v.str_with_names({0: 'hello', 1: 'world'}) == '{hello, world}'