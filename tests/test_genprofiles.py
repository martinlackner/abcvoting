"""
Unit tests for genprofiles.py
"""

import pytest
from abcvoting import genprofiles
import random


@pytest.mark.parametrize("num_cand", [6, 7, 8])
@pytest.mark.parametrize("num_voters", [10, 20, 30])
@pytest.mark.parametrize("setsize", [1, 2, 3])
@pytest.mark.parametrize("replace", [0.0, 0.5, 1.0])
def test_urn(num_cand, num_voters, setsize, replace):
    random.seed(0)
    profile = genprofiles.random_urn_profile(num_cand, num_voters, setsize, replace)
    assert len(profile) == num_voters
    assert profile.num_cand == num_cand
    for voter in profile:
        assert len(voter.approved) == setsize


@pytest.mark.parametrize("num_cand", [12])
@pytest.mark.parametrize("num_voters", [10, 20, 30])
@pytest.mark.parametrize("replace", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("num_parties,uniform", [(3, True), (3, False), (4, True), (5, False)])
def test_urn_party_list(num_cand, num_voters, num_parties, replace, uniform):
    random.seed(0)
    profile = genprofiles.random_urn_party_list_profile(
        num_cand, num_voters, num_parties, replace, uniform
    )
    assert len(profile) == num_voters
    assert profile.num_cand == num_cand
    assert profile.party_list()
    for voter in profile:
        assert len(voter.approved) > 0


@pytest.mark.parametrize("num_cand", [6, 7, 8])
@pytest.mark.parametrize("num_voters", [10, 20, 30])
@pytest.mark.parametrize("setsize", [1, 2, 3])
def test_IC(num_cand, num_voters, setsize):
    random.seed(0)
    profile = genprofiles.random_IC_profile(num_cand, num_voters, setsize)
    assert len(profile) == num_voters
    assert profile.num_cand == num_cand
    for voter in profile:
        assert len(voter.approved) == setsize


@pytest.mark.parametrize("num_cand", [10])
@pytest.mark.parametrize("num_voters", [10, 20, 30])
@pytest.mark.parametrize("num_parties,uniform", [(2, True), (2, False), (5, True), (3, False)])
def test_IC_party_list(num_cand, num_voters, num_parties, uniform):
    random.seed(0)
    profile = genprofiles.random_IC_party_list_profile(num_cand, num_voters, num_parties, uniform)
    assert len(profile) == num_voters
    assert profile.num_cand == num_cand
    assert profile.party_list()
    for voter in profile:
        assert len(voter.approved) > 0


@pytest.mark.parametrize("num_cand", [6, 7, 8])
@pytest.mark.parametrize("num_voters", [10, 20, 30])
@pytest.mark.parametrize("setsize", [1, 2, 3])
@pytest.mark.parametrize("dispersion", [0.1, 0.5, 0.9])
def test_mallows(num_cand, num_voters, setsize, dispersion):
    random.seed(0)
    profile = genprofiles.random_mallows_profile(num_cand, num_voters, setsize, dispersion)
    assert len(profile) == num_voters
    assert profile.num_cand == num_cand
    for voter in profile:
        assert len(voter.approved) == setsize
    for voter in profile:
        assert len(voter.approved) > 0


@pytest.mark.parametrize("points,distance", [([-1, 1, 3, 1], 4), ([2, 3, -2, 0], 5)])
def test_euclidean(points, distance):
    assert genprofiles.__euclidean(points[:2], points[2:]) == distance


@pytest.mark.parametrize("num_cand", [6, 7, 8])
@pytest.mark.parametrize("num_voters", [10, 20, 30])
@pytest.mark.parametrize("sigma", [0.1, 1, 3])
def test_2d(num_cand, num_voters, sigma):
    random.seed(0)
    profile = genprofiles.random_2d_points_profile(
        num_cand, num_voters, "normal", "twogroups", sigma, 1.1
    )
    assert len(profile) == num_voters
    assert profile.num_cand == num_cand
    for voter in profile:
        assert len(voter.approved) > 0


@pytest.mark.parametrize("num_cand", [10])
@pytest.mark.parametrize("num_voters", [10, 20, 30])
@pytest.mark.parametrize("num_parties,uniform", [(2, True), (2, False), (5, True), (4, False)])
@pytest.mark.parametrize("sigma", [0.1, 1, 3])
def test_2d_party_list(num_cand, num_voters, num_parties, uniform, sigma):
    random.seed(0)
    profile = genprofiles.random_2d_points_party_list_profile(
        num_cand, num_voters, num_parties, "uniform_square", "normal", sigma, uniform
    )
    assert len(profile) == num_voters
    assert profile.num_cand == num_cand
    assert profile.party_list()
    for voter in profile:
        assert len(voter.approved) > 0
