"""
Unit tests for abcvoting/generate.py.
"""

import pytest
from abcvoting import generate
from abcvoting.preferences import Profile
import random


@pytest.mark.parametrize("num_voters", [10, 20, 30])
@pytest.mark.parametrize("num_cand", [6, 7, 8])
@pytest.mark.parametrize("prob_model", generate.PROBABILITY_MODELS_IDS)
def test_random_profile(num_voters, num_cand, prob_model):
    kwargs = {}
    if prob_model in ["IC fixed-size"]:
        kwargs = {"setsize": 3}
    if prob_model in ["Truncated Mallows"]:
        kwargs = {"setsize": 3, "dispersion": 0.2}
    if prob_model in ["Truncated Urn", "Urn fixed-size"]:
        kwargs = {"setsize": 3, "replace": 0.2}
    if prob_model in ["Urn"]:
        kwargs = {"p": 0.3, "replace": 0.2}
    if prob_model in ["IC", "Disjoint Resampling"]:
        kwargs = {"p": 0.2}
    if prob_model in ["Resampling", "Noise"]:
        kwargs = {"p": 0.2, "phi": 0.5}
    if prob_model in ["Euclidean VCR"]:
        kwargs = {
            "voter_probdist": generate.PointProbabilityDistribution(name="2d_disc"),
            "candidate_probdist": generate.PointProbabilityDistribution(name="2d_disc"),
            "voter_radius": 0.1,
            "candidate_radius": 0.1,
        }
    if prob_model in ["Euclidean fixed-size"]:
        kwargs = {
            "voter_probdist": generate.PointProbabilityDistribution(name="2d_disc"),
            "candidate_probdist": generate.PointProbabilityDistribution(name="2d_disc"),
            "setsize": 3,
        }
    if prob_model in ["Euclidean Threshold"]:
        kwargs = {
            "voter_probdist": generate.PointProbabilityDistribution(name="2d_disc"),
            "candidate_probdist": generate.PointProbabilityDistribution(name="2d_disc"),
            "threshold": 1.2,
        }
    profile = generate.random_profile(prob_model, num_voters, num_cand, **kwargs)
    assert isinstance(profile, Profile)
    assert len(profile) == num_voters, "wrong number of voters"
    if "setsize" in kwargs:
        for voter in profile:
            assert len(voter.approved) == kwargs["setsize"]
    if "threshold" in kwargs:
        for voter in profile:
            assert len(voter.approved) >= 1


@pytest.mark.parametrize("num_cand", [6, 7, 8])
@pytest.mark.parametrize("num_voters", [10, 20, 30])
@pytest.mark.parametrize("setsize", [1, 2, 3])
@pytest.mark.parametrize("replace", [0.0, 0.5, 1.0])
def test_urn_fixed_size(num_voters, num_cand, setsize, replace):
    random.seed(0)
    profile = generate.random_urn_fixed_size_profile(num_voters, num_cand, setsize, replace)
    assert len(profile) == num_voters
    assert profile.num_cand == num_cand
    for voter in profile:
        assert len(voter.approved) == setsize


@pytest.mark.parametrize("num_cand", [6, 7, 8])
@pytest.mark.parametrize("num_voters", [10, 20, 30])
@pytest.mark.parametrize("setsize", [1, 2, 3])
def test_ic_fixed_size(num_voters, num_cand, setsize):
    random.seed(0)
    profile = generate.random_ic_fixed_size_profile(num_voters, num_cand, setsize)
    assert len(profile) == num_voters
    assert profile.num_cand == num_cand
    for voter in profile:
        assert len(voter.approved) == setsize


@pytest.mark.parametrize("num_cand", [6, 7, 8])
@pytest.mark.parametrize("num_voters", [10, 20, 30])
@pytest.mark.parametrize("setsize", [1, 2, 3])
@pytest.mark.parametrize("dispersion", [0.1, 0.5, 0.9])
def test_mallows(num_voters, num_cand, setsize, dispersion):
    random.seed(0)
    profile = generate.random_mallows_profile(num_voters, num_cand, setsize, dispersion)
    assert len(profile) == num_voters
    assert profile.num_cand == num_cand
    for voter in profile:
        assert len(voter.approved) == setsize
    for voter in profile:
        assert len(voter.approved) > 0


some_pointprobabilitydistributions = [
    generate.PointProbabilityDistribution(name="2d_disc", width=3, center_point=[-1, -2]),
    generate.PointProbabilityDistribution("1d_interval", center_point=[0]),
    generate.PointProbabilityDistribution("1d_gaussian", center_point=[4]),
    generate.PointProbabilityDistribution("1d_gaussian_interval", center_point=[6], width=0.5),
    generate.PointProbabilityDistribution("2d_square", center_point=[0, 2]),
    generate.PointProbabilityDistribution("2d_gaussian", center_point=[4, 2], sigma=0.25),
    generate.PointProbabilityDistribution("2d_gaussian_disc", center_point=[6, 2], sigma=0.25),
    generate.PointProbabilityDistribution("3d_cube", center_point=[0, 0, 0], width=3),
    None,
]


@pytest.mark.parametrize("num_voters", [10])
@pytest.mark.parametrize("num_cand", [10, 20])
@pytest.mark.parametrize("point_distribution", some_pointprobabilitydistributions)
@pytest.mark.parametrize(
    "prob_model, kwargs",
    [
        ("Euclidean VCR", {"voter_radius": 0.1, "candidate_radius": 0.1}),
        ("Euclidean VCR", {"voter_radius": [0.1, 0.2] * 5, "candidate_radius": 0.1}),
        ("Euclidean Threshold", {"threshold": 1.1}),
        ("Euclidean fixed-size", {"setsize": 4}),
    ],
)
def test_euclidean(num_voters, num_cand, prob_model, point_distribution, kwargs):
    kwargs["voter_probdist"] = point_distribution
    kwargs["candidate_probdist"] = point_distribution
    if point_distribution is None:
        kwargs["candidate_points"] = [[0, 2, 3], [1, 2, -1]] * (num_cand // 2)
        kwargs["voter_points"] = [[0, 1, 3], [1, 2, 3]] * (num_voters // 2)
    profile = generate.random_profile(prob_model, num_voters, num_cand, **kwargs)
    assert isinstance(profile, Profile)
    assert len(profile) == num_voters, "wrong number of voters"
    if "setsize" in kwargs:
        for voter in profile:
            assert len(voter.approved) == kwargs["setsize"]
    if "threshold" in kwargs:
        for voter in profile:
            assert len(voter.approved) >= 1


@pytest.mark.parametrize(
    "prob_model, kwargs",
    [
        ("Euclidean VCR", {"voter_radius": 0.1, "candidate_radius": 0.1}),
        ("Euclidean Threshold", {"threshold": 1.1}),
        ("Euclidean fixed-size", {"setsize": 4}),
    ],
)
def test_euclidean_errors(prob_model, kwargs):
    kwargs["voter_probdist"] = generate.PointProbabilityDistribution(
        "1d_interval", center_point=[0]
    )
    kwargs["candidate_probdist"] = generate.PointProbabilityDistribution(
        "2d_square", center_point=[0, 2]
    )
    with pytest.raises(ValueError):
        generate.random_profile(prob_model, num_voters=10, num_cand=20, **kwargs)

    kwargs["voter_probdist"] = generate.PointProbabilityDistribution(
        "3d_cube", center_point=[0, 0, 0], width=3
    )
    kwargs["candidate_probdist"] = generate.PointProbabilityDistribution(
        "2d_disc", center_point=[0, 2]
    )
    with pytest.raises(ValueError):
        generate.random_profile(prob_model, num_voters=10, num_cand=20, **kwargs)

    kwargs["candidate_probdist"] = None
    kwargs["candidate_points"] = [[0, 2, 3], [1, 2]]
    with pytest.raises(ValueError):
        generate.random_profile(prob_model, num_voters=10, num_cand=20, **kwargs)

    with pytest.raises(ValueError):
        generate.random_point(
            generate.PointProbabilityDistribution("3d_object", center_point=[0, 0], width=1)
        )

    with pytest.raises(ValueError):
        generate.PointProbabilityDistribution("3dcube", center_point=[0, 0], width=1)

    with pytest.raises(ValueError):
        generate.PointProbabilityDistribution("threed_cube", center_point=[0, 0], width=1)

    with pytest.raises(ValueError):
        generate.random_profile(prob_model, num_voters=10, num_cand=20, **kwargs)
