"""
Unit tests for abcvoting/generate.py.
"""

import pytest
from abcvoting import generate
from abcvoting.generate import PointProbabilityDistribution
from abcvoting.preferences import Profile
import numpy as np


@pytest.mark.parametrize("num_voters", [10, 20, 30])
@pytest.mark.parametrize("num_cand", [6, 7, 8])
@pytest.mark.parametrize("prob_dist_id", generate.PROBABILITY_DISTRIBUTION_IDS)
def test_random_profile(num_voters, num_cand, prob_dist_id):
    prob_distribution = {"id": prob_dist_id}
    if prob_dist_id in ["IC fixed-size"]:
        prob_distribution.update({"setsize": 3})
    if prob_dist_id in ["Truncated Mallows"]:
        prob_distribution.update({"setsize": 3, "dispersion": 0.2})
    if prob_dist_id in ["Truncated Urn", "Urn fixed-size"]:
        prob_distribution.update({"setsize": 3, "replace": 0.2})
    if prob_dist_id in ["Urn"]:
        prob_distribution.update({"p": 0.3, "replace": 0.2})
    if prob_dist_id in ["IC", "Disjoint Resampling"]:
        prob_distribution.update({"p": 0.2})
    if prob_dist_id in ["Resampling", "Noise"]:
        prob_distribution.update({"p": 0.2, "phi": 0.5})
    if prob_dist_id in ["Euclidean VCR"]:
        prob_distribution.update(
            {
                "voter_prob_distribution": PointProbabilityDistribution(name="2d_disc"),
                "candidate_prob_distribution": PointProbabilityDistribution(name="2d_disc"),
                "voter_radius": 0.1,
                "candidate_radius": 0.1,
            }
        )
    if prob_dist_id in ["Euclidean fixed-size"]:
        prob_distribution.update(
            {
                "voter_prob_distribution": PointProbabilityDistribution(name="2d_disc"),
                "candidate_prob_distribution": PointProbabilityDistribution(name="2d_disc"),
                "setsize": 3,
            }
        )
    if prob_dist_id in ["Euclidean Threshold"]:
        prob_distribution.update(
            {
                "voter_prob_distribution": PointProbabilityDistribution(name="2d_disc"),
                "candidate_prob_distribution": PointProbabilityDistribution(name="2d_disc"),
                "threshold": 1.2,
            }
        )
    profile = generate.random_profile(num_voters, num_cand, prob_distribution)
    assert isinstance(profile, Profile)
    assert len(profile) == num_voters, "wrong number of voters"
    if "setsize" in prob_distribution:
        for voter in profile:
            assert len(voter.approved) == prob_distribution["setsize"]
    if "threshold" in prob_distribution:
        for voter in profile:
            assert len(voter.approved) >= 1


@pytest.mark.parametrize("num_cand", [6, 7, 8])
@pytest.mark.parametrize("num_voters", [10, 20, 30])
@pytest.mark.parametrize("setsize", [1, 2, 3])
@pytest.mark.parametrize("replace", [0.0, 0.5, 1.0])
def test_urn_fixed_size(num_voters, num_cand, setsize, replace):
    generate.rng = np.random.default_rng(0)  # seed for numpy RNG
    profile = generate.random_urn_fixed_size_profile(num_voters, num_cand, setsize, replace)
    assert len(profile) == num_voters
    assert profile.num_cand == num_cand
    for voter in profile:
        assert len(voter.approved) == setsize


@pytest.mark.parametrize("num_cand", [6, 7, 8])
@pytest.mark.parametrize("num_voters", [10, 20, 30])
@pytest.mark.parametrize("setsize", [1, 2, 3])
def test_ic_fixed_size(num_voters, num_cand, setsize):
    generate.rng = np.random.default_rng(0)  # seed for numpy RNG
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
    generate.rng = np.random.default_rng(0)  # seed for numpy RNG
    profile = generate.random_mallows_profile(num_voters, num_cand, setsize, dispersion)
    assert len(profile) == num_voters
    assert profile.num_cand == num_cand
    for voter in profile:
        assert len(voter.approved) == setsize
    for voter in profile:
        assert len(voter.approved) > 0


some_pointprobabilitydistributions = [
    PointProbabilityDistribution(name="2d_disc", width=3, center_point=[-1, -2]),
    PointProbabilityDistribution("1d_interval", center_point=[0]),
    PointProbabilityDistribution("1d_gaussian", center_point=[4]),
    PointProbabilityDistribution("1d_gaussian_interval", center_point=[6], width=0.5),
    PointProbabilityDistribution("2d_square", center_point=[0, 2]),
    PointProbabilityDistribution("2d_gaussian", center_point=[4, 2], sigma=0.25),
    PointProbabilityDistribution("2d_gaussian_disc", center_point=[6, 2], sigma=0.25),
    PointProbabilityDistribution("3d_cube", center_point=[0, 0, 0], width=3),
    None,
]


@pytest.mark.parametrize("num_voters", [10])
@pytest.mark.parametrize("num_cand", [10, 20])
@pytest.mark.parametrize("point_distribution", some_pointprobabilitydistributions)
@pytest.mark.parametrize(
    "prob_distribution",
    [
        {"id": "Euclidean VCR", "voter_radius": 0.1, "candidate_radius": 0.1},
        {"id": "Euclidean VCR", "voter_radius": [0.1, 0.2] * 5, "candidate_radius": 0.1},
        {"id": "Euclidean Threshold", "threshold": 1.1},
        {"id": "Euclidean fixed-size", "setsize": 4},
    ],
)
def test_euclidean(num_voters, num_cand, prob_distribution, point_distribution):
    print(prob_distribution)
    assert "id" in prob_distribution.keys()
    print("passed")
    prob_distribution["voter_prob_distribution"] = point_distribution
    prob_distribution["candidate_prob_distribution"] = point_distribution
    if point_distribution is None:
        prob_distribution["candidate_points"] = [[0, 2, 3], [1, 2, -1]] * (num_cand // 2)
        prob_distribution["voter_points"] = [[0, 1, 3], [1, 2, 3]] * (num_voters // 2)
    print(prob_distribution)
    profile = generate.random_profile(num_voters, num_cand, prob_distribution)
    assert isinstance(profile, Profile)
    assert len(profile) == num_voters, "wrong number of voters"
    if "setsize" in prob_distribution:
        for voter in profile:
            assert len(voter.approved) == prob_distribution["setsize"]
    if "threshold" in prob_distribution:
        for voter in profile:
            assert len(voter.approved) >= 1


@pytest.mark.parametrize(
    "prob_distribution",
    [
        {"id": "Euclidean VCR", "voter_radius": 0.1, "candidate_radius": 0.1},
        {"id": "Euclidean Threshold", "threshold": 1.1},
        {"id": "Euclidean fixed-size", "setsize": 4},
    ],
)
def test_euclidean_errors(prob_distribution):
    prob_distribution["voter_prob_distribution"] = PointProbabilityDistribution(
        "1d_interval", center_point=[0]
    )
    prob_distribution["candidate_prob_distribution"] = PointProbabilityDistribution(
        "2d_square", center_point=[0, 2]
    )
    with pytest.raises(ValueError):
        generate.random_profile(num_voters=10, num_cand=20, prob_distribution=prob_distribution)

    prob_distribution["voter_prob_distribution"] = PointProbabilityDistribution(
        "3d_cube", center_point=[0, 0, 0], width=3
    )
    prob_distribution["candidate_prob_distribution"] = PointProbabilityDistribution(
        "2d_disc", center_point=[0, 2]
    )
    with pytest.raises(ValueError):
        generate.random_profile(num_voters=10, num_cand=20, prob_distribution=prob_distribution)

    prob_distribution["candidate_prob_distribution"] = None
    prob_distribution["candidate_points"] = [[0, 2, 3], [1, 2]]
    with pytest.raises(ValueError):
        generate.random_profile(num_voters=10, num_cand=20, prob_distribution=prob_distribution)

    with pytest.raises(ValueError):
        generate.random_point(
            PointProbabilityDistribution("3d_object", center_point=[0, 0], width=1)
        )

    with pytest.raises(ValueError):
        PointProbabilityDistribution("3dcube", center_point=[0, 0], width=1)

    with pytest.raises(ValueError):
        PointProbabilityDistribution("threed_cube", center_point=[0, 0], width=1)

    with pytest.raises(ValueError):
        generate.random_profile(num_voters=10, num_cand=20, prob_distribution=prob_distribution)
