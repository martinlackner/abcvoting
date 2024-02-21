"""
Random generation of approval profiles.

This module is based on the paper on the package
`prefsampling <https://github.com/COMSOC-Community/prefsampling>`_.
"""

import math
import numpy as np
import prefsampling.approval as app_samplers
import prefsampling.ordinal as ord_samplers
from numpy.random import default_rng
from abcvoting.preferences import Profile, Voter


def prefsampling_wrapper(sampler, sampler_params):
    samples = sampler(**sampler_params)
    num_candidates = sampler_params["num_candidates"]
    profile = Profile(num_candidates)
    for sample in samples:
        profile.add_voter(Voter(sample, num_cand=num_candidates))
    return profile


def random_profile(num_voters, num_cand, prob_distribution):
    """
    Generate a random profile using the probability distribution `prob_distribution`.

    The following probability distributions are supported:

    .. doctest::

        >>> PROBABILITY_DISTRIBUTION_IDS  # doctest: +NORMALIZE_WHITESPACE
        ('IC fixed-size', 'IC', 'Truncated Mallows', 'Urn fixed-size', 'Urn',
        'Truncated Urn', 'Euclidean VCR', 'Euclidean fixed-size', 'Euclidean Threshold',
        'Resampling', 'Disjoint Resampling', 'Noise')

    Parameters
    ----------
        num_voters : int
            The desired number of voters in the profile.

        num_cand : int
            The desired number of candidates in the profile.

        prob_distribution : dict
            Specification of the probability distribution.

    Returns
    -------
        abcvoting.preferences.Profile

    Examples
    --------
    Generate a profile via the Independent Culture (IC) distribution with a probability of `0.5`.

    .. testsetup::

        generate.rng = np.random.default_rng(24121838)

    .. doctest::

        >>> prob_distribution = {"id": "IC", "p": 0.5}
        >>> profile = random_profile(num_voters=5, num_cand=5, prob_distribution=prob_distribution)
        >>> print(profile)
        profile with 5 voters and 5 candidates:
         voter 0:   {0, 1, 2, 3},
         voter 1:   {0, 2, 4},
         voter 2:   {1, 3},
         voter 3:   {0, 3, 4},
         voter 4:   {0}
    """
    if "id" not in prob_distribution:
        raise KeyError('Probability distribution requires key "id".')
    if prob_distribution["id"] not in PROBABILITY_DISTRIBUTION_IDS:
        raise ValueError(f"Probability distribution id {prob_distribution} unknown.")
    kwargs = {key: value for key, value in prob_distribution.items() if key != "id"}
    return PROBABILITY_DISTRIBUTIONS[prob_distribution["id"]](num_voters, num_cand, **kwargs)


def random_urn_fixed_size_profile(num_voters, num_cand, setsize, replace):
    """
    Generate a random profile using the *Polya Urn with fixed-size approval sets* distribution.

    Parameters
    ----------
        num_voters : int
            The desired number of voters in the profile.

        num_cand : int
            The desired number of candidates in the profile.

        setsize : int
            Number of candidates that each voter approves.

        replace : float
            New balls added to the urn in each iteration, relative to the original number.

            The urn starts with (`num_cand` choose `setsize`) balls, each representing a set
            of candidates with size `setsize`. This quantity is normalized to `1.0`.
            The `replace` value is a float that indicates how many balls are added using this
            normalization. Specifically, `replace` *  (`num_cand` choose `setsize`) are added
            in each iteration.

    Returns
    -------
        abcvoting.preferences.Profile
    """
    currsize = 1.0
    approval_sets = []
    replacedsets = {}

    for _ in range(num_voters):
        r = rng.random() * currsize
        if r < 1.0:
            # base case: sample uniformly at random
            randset = list(range(num_cand))
            rng.shuffle(randset)
            randset = randset[:setsize]
            approval_sets.append(randset)
            key = tuple(set(randset))
            if key in replacedsets:
                replacedsets[key] += 1
            else:
                replacedsets[key] = 1
            currsize += replace
        else:
            # sample from one of the replaced ballots
            r = rng.integers(0, sum(replacedsets.values()))
            for approval_set, count in replacedsets.items():
                if r <= count:
                    approval_sets.append(list(approval_set))
                    break
                r -= count
    profile = Profile(num_cand)
    profile.add_voters(approval_sets)
    return profile


def random_ic_fixed_size_profile(num_voters, num_cand, setsize):
    """
    Generate a random profile using the *IC with fixed-size approval sets* distribution.

    Parameters
    ----------
        num_voters : int
            The desired number of voters in the profile.

        num_cand : int
            The desired number of candidates in the profile.

        setsize : int
            Number of candidates that each voter approves.

    Returns
    -------
        abcvoting.preferences.Profile
    """
    return prefsampling_wrapper(
        app_samplers.impartial_constant_size,
        {
            "num_voters": num_voters,
            "num_candidates": num_cand,
            "rel_num_approvals": setsize / num_cand,
        },
    )


def random_mallows_profile(num_voters, num_cand, setsize, dispersion):
    """
    Generate a random profile using the *Truncated Mallows* probability distribution.

    Parameters
    ----------
        num_voters : int
            The desired number of voters in the profile.

        num_cand : int
            The desired number of candidates in the profile.

        setsize : int
            Number of candidates that each voter approves.

        dispersion : float in [0, 1]
            Dispersion parameter of the Mallows model.

    Returns
    -------
        abcvoting.preferences.Profile
    """

    return prefsampling_wrapper(
        app_samplers.truncated_ordinal,
        {
            "num_voters": num_voters,
            "num_candidates": num_cand,
            "rel_num_approvals": setsize / num_cand,
            "ordinal_sampler": ord_samplers.mallows,
            "ordinal_sampler_parameters": {"phi": dispersion},
        },
    )


# Impartial Culture
def random_ic_profile(num_voters, num_cand, p=0.5):
    """
    Generate a random profile using the *Independent Culture (IC)* probability distribution.

    Parameters
    ----------
        num_voters : int
            The desired number of voters in the profile.

        num_cand : int
            The desired number of candidates in the profile.

        p : float in [0, 1]
            Probability of approving a candidate.

    Returns
    -------
        abcvoting.preferences.Profile

    References
    ----------
    Corresponds to *p-IC* in:

    *How to Sample Approval Elections?*
    Stanisław Szufa, Piotr Faliszewski, Łukasz Janeczko, Martin Lackner, Arkadii Slinko,
    Krzysztof Sornat, Nimrod Talmon.
    https://arxiv.org/abs/2207.01140
    """
    return prefsampling_wrapper(
        app_samplers.impartial, {"num_voters": num_voters, "num_candidates": num_cand, "p": p}
    )


def _ordinal_urn_profile(num_voters, num_cand, replace):
    """
    Generate rankings according to the Urn probability distribution.
    """
    rankings = []
    urn_size = 1.0
    for j in range(num_voters):
        rho = rng.uniform(0, urn_size)
        if rho <= 1.0:
            rankings.append(rng.permutation(num_cand))
        else:
            rankings.append(rankings[rng.integers(0, j)])
        urn_size += replace
    return rankings


def random_urn_profile(num_voters, num_cand, p, replace):
    """
    Generate a random profile using the *Polya Urn* probability distribution.

    Parameters
    ----------
        num_voters : int
            The desired number of voters in the profile.

        num_cand : int
            The desired number of candidates in the profile.

        p : float in [0, 1]
            If a new vote is generated, each candidate is approved with likelihood `p`.

        replace : float
            New balls added to the urn in each iteration, relative to the original number.

            A value of `1.0` means that in the second iteration, there is a chance of 0.5 that
            the ballot of the first iteration is chosen and a chance of 0.5 that a new ballot is
            drawn from p-IC.

    Returns
    -------
        abcvoting.preferences.Profile
    """
    approval_sets = []
    urn_size = 1.0
    for j in range(num_voters):
        rho = rng.uniform(0, urn_size)
        if rho <= 1.0:
            vote = set()
            for c in range(num_cand):
                if rng.random() <= p:
                    vote.add(c)
            approval_sets.append(vote)
        else:
            approval_sets.append(approval_sets[rng.integers(0, j)])
        urn_size += replace
    profile = Profile(num_cand)
    profile.add_voters(approval_sets)
    return profile


def random_truncated_urn_profile(num_voters, num_cand, setsize, replace):
    """
    Generate a random profile using the *Truncated Polya Urn* probability distribution.

    Parameters
    ----------
        num_voters : int
            The desired number of voters in the profile.

        num_cand : int
            The desired number of candidates in the profile.

        setsize : int
            Number of candidates that each voter approves (top entries from a ranking).

        replace : float
            New balls added to the urn in each iteration, relative to the original number.

            The urn starts with `num_cand` factorial balls, each representing a ranking
            of candidates. This quantity is normalized to `1.0`.
            The `replace` value is a float that indicates how many balls are added using this
            normalization. Specifically, `replace` *  (`num_cand` factorial) are added
            in each iteration.

    Returns
    -------
        abcvoting.preferences.Profile

    References
    ----------
    *How to Sample Approval Elections?*
    Stanisław Szufa, Piotr Faliszewski, Łukasz Janeczko, Martin Lackner, Arkadii Slinko,
    Krzysztof Sornat, Nimrod Talmon.
    https://arxiv.org/abs/2207.01140
    """

    return prefsampling_wrapper(
        app_samplers.truncated_ordinal,
        {
            "num_voters": num_voters,
            "num_candidates": num_cand,
            "rel_num_approvals": setsize / num_cand,
            "ordinal_sampler": ord_samplers.urn,
            "ordinal_sampler_parameters": {"alpha": replace},
        },
    )


# Resampling
def random_resampling_profile(num_voters, num_cand, p, phi):
    """
    Generate a random profile using the *Resampling* probability distribution.

    Parameters
    ----------
        num_voters : int
            The desired number of voters in the profile.

        num_cand : int
            The desired number of candidates in the profile.

        p : float in [0, 1]
            Fraction of candidates that will be approved by the central ballot.

            Precisely, the central ballot will approve the first floor(`p` * `num_cand`)
            candidates.

        phi : float in [0, 1]
            Probability to resample an approval.

            For each generated vote, we go through all candidates. For each candidate, we
            copy the approval of the central ballot with probability 1 - `phi`. Otherwise,
            with probability `phi`, we resample the approval of the candidate (so that the
            generated vote approves that candidate with probability `p`).

    Returns
    -------
        abcvoting.preferences.Profile

    References
    ----------
    *How to Sample Approval Elections?*
    Stanisław Szufa, Piotr Faliszewski, Łukasz Janeczko, Martin Lackner, Arkadii Slinko,
    Krzysztof Sornat, Nimrod Talmon.
    https://arxiv.org/abs/2207.01140
    """
    return prefsampling_wrapper(
        app_samplers.resampling,
        {"num_voters": num_voters, "num_candidates": num_cand, "phi": phi, "p": p},
    )


def random_disjoint_resampling_profile(num_voters, num_cand, p, phi=None, num_groups=2):
    """
    Generate a random profile using the *(p,phi,g)-Disjoint Resampling* probability distribution.

    Parameters
    ----------
        num_voters : int
            The desired number of voters in the profile.

        num_cand : int
            The desired number of candidates in the profile.

        p : float in [0, 1]
            Probability of approving a candidate in case that the model resamples.

        phi : float in [0, 1]
            Probability to resample an approval.

            For each generated vote, we go through all candidates. For each candidate, we
            copy the approval of the central ballot with probability 1 - `phi`. Otherwise,
            with probability `phi`, we resample the approval of the candidate (so that the
            generated vote approves that candidate with probability `p`).

        num_groups : int, optional
            Corresponds to the parameter g in (p,phi,g)-Disjoint Resampling.

            The model randomly partitions the candidates into `num_groups` groups. To generate
            a vote, the model first randomly selects a group. Then, it samples a vote from a
            (p,phi)-resampling model, where the central ballot approves exactly all candidates
            in the selected group. Thus, the parameter `p` is not used to obtain the central
            ballot, but is used in case of resampling.

    Returns
    -------
        abcvoting.preferences.Profile

    References
    ----------
    *How to Sample Approval Elections?*
    Stanisław Szufa, Piotr Faliszewski, Łukasz Janeczko, Martin Lackner, Arkadii Slinko,
    Krzysztof Sornat, Nimrod Talmon.
    https://arxiv.org/abs/2207.01140
    """

    if phi is None:
        phi = rng.random()

    return prefsampling_wrapper(
        app_samplers.disjoint_resampling,
        {
            "num_voters": num_voters,
            "num_candidates": num_cand,
            "phi": phi,
            "p": p,
            "g": num_groups,
        },
    )


def random_noise_model_profile(num_voters, num_cand, p, phi, distance="hamming"):
    """
    Generate a random profile using the *Random Noise* probability distribution.

    Parameters
    ----------
        num_voters : int
            The desired number of voters in the profile.

        num_cand : int
            The desired number of candidates in the profile.

        p : float in [0, 1]
            Fraction of candidates that will be approved by the central ballot.

            Precisely, the central ballot will approve the first floor(`p` * `num_cand`)
            candidates.

        phi : float in [0, 1]
            Probability to resample an approval.

            For each generated vote, we go through all candidates. For each candidate, we
            copy the approval of the central ballot with probability 1 - `phi`. Otherwise,
            with probability `phi`, we resample the approval of the candidate (so that the
            generated vote approves that candidate with probability `p`).

        distance : str, optional
            The used distance measure.

            The default is Hamming distance ("hamming"). Other possibilities are "jaccard",
            "zelinka", and "bunke-shearer".

    Returns
    -------
        abcvoting.preferences.Profile

    References
    ----------
    *How to Sample Approval Elections?*
    Stanisław Szufa, Piotr Faliszewski, Łukasz Janeczko, Martin Lackner, Arkadii Slinko,
    Krzysztof Sornat, Nimrod Talmon.
    https://arxiv.org/abs/2207.01140
    """

    if distance == "hamming":
        noise_type = app_samplers.NoiseType.HAMMING
    elif distance == "jaccard":
        noise_type = app_samplers.NoiseType.JACCARD
    elif distance == "zelinka":
        noise_type = app_samplers.NoiseType.ZELINKA
    elif distance == "bunke-shearer":
        noise_type = app_samplers.NoiseType.BUNKE_SHEARER
    return prefsampling_wrapper(
        app_samplers.noise,
        {
            "num_voters": num_voters,
            "num_candidates": num_cand,
            "phi": phi,
            "p": p,
            "noise_type": noise_type,
        },
    )


def random_euclidean_fixed_size_profile(
    num_voters,
    num_cand,
    voter_prob_distribution,
    candidate_prob_distribution,
    setsize,
    voter_points=None,
    candidate_points=None,
    return_points=False,
):
    """
    Generate a random profile using the *Euclidean with fixed-size approval sets* distribution.

    Parameters
    ----------
        num_voters : int
            The desired number of voters in the profile.

        num_cand : int
            The desired number of candidates in the profile.

        voter_prob_distribution : PointProbabilityDistribution or None
            A probability distribution used to generate voter points.

        candidate_prob_distribution : PointProbabilityDistribution or None
            A probability distribution used to generate candidate points.

        setsize : int
            Number of candidates that each voter approves.

        voter_points : iterable
            A list of points.

            The length of this list must be `num_voters`. The dimension of the points must be
            the same as the points in `candiddate_points` or as specified by
            `candidate_prob_distribution`. This parameter is only used if `voter_prob_distribution`
            is `None`.

        candidate_points : iterable
            A list of points.

            The length of this list must be `num_cand`. The dimension of the points must be
            the same as the points in `voter_points` or as specified by `voter_prob_distribution`.
            This parameter is only used if `candidate_prob_distribution` is `None`.

        return_points : bool, optional
            If true, also return the list of voter points and a list of candidate points.

    Returns
    -------
        abcvoting.preferences.Profile
    """

    voter_points, candidate_points = _voter_and_candidate_points(
        num_voters,
        num_cand,
        voter_prob_distribution,
        candidate_prob_distribution,
        voter_points,
        candidate_points,
    )
    profile = Profile(num_cand)
    approval_sets = []
    for voterpoint in voter_points:
        distances = {
            cand: np.linalg.norm(voterpoint - candidate_points[cand])
            for cand in profile.candidates
        }
        cands_sorted = sorted(distances, key=distances.get)
        approval_sets.append(cands_sorted[:setsize])
    profile.add_voters(approval_sets)

    if return_points:
        return profile, voter_points, candidate_points

    return profile


def random_euclidean_threshold_profile(
    num_voters,
    num_cand,
    voter_prob_distribution,
    candidate_prob_distribution,
    threshold,
    voter_points=None,
    candidate_points=None,
    return_points=False,
):
    """
    Generate a random profile using the *Euclidean Threshold* probability distribution.

    Parameters
    ----------
        num_voters : int
            The desired number of voters in the profile.

        num_cand : int
            The desired number of candidates in the profile.

        voter_prob_distribution : PointProbabilityDistribution or None
            A probability distribution used to generate voter points.

        candidate_prob_distribution : PointProbabilityDistribution or None
            A probability distribution used to generate candidate points.

        threshold : float
            Voters' tolerance for approving candidates. This is a float >= 1.

            A voter approves all candidates that have at a distance of at most `threshold` * `d`,
            where `d` is the minimum distance between this voter and any candidate.
            Setting `threshold` to 1 means that only the closest candidate is approved (there
            might be more than one).

        voter_points : iterable
            A list of points.

            The length of this list must be `num_voters`. The dimension of the points must be
            the same as the points in `candidate_points` or as specified by
            `candidate_prob_distribution`. This parameter is only used if `voter_prob_distribution`
            is `None`.

        candidate_points : iterable
            A list of points.

            The length of this list must be `num_cand`. The dimension of the points must be
            the same as the points in `voter_points` or as specified by `voter_prob_distribution`.
            This parameter is only used if `candidate_prob_distribution` is `None`.

        return_points : bool, optional
            If true, also return the list of voter points and a list of candidate points.

    Returns
    -------
        abcvoting.preferences.Profile
    """

    if threshold < 1:
        raise ValueError("threshold must be >= 1.")

    voter_points, candidate_points = _voter_and_candidate_points(
        num_voters,
        num_cand,
        voter_prob_distribution,
        candidate_prob_distribution,
        voter_points,
        candidate_points,
    )

    profile = Profile(num_cand)
    approval_sets = []
    for voterpoint in voter_points:
        distances = {
            cand: np.linalg.norm(voterpoint - candidate_points[cand])
            for cand in profile.candidates
        }
        mindist = min(distances.values())
        approval_sets.append(
            [cand for cand in profile.candidates if distances[cand] <= mindist * threshold]
        )
    profile.add_voters(approval_sets)

    if return_points:
        return profile, voter_points, candidate_points

    return profile


def random_euclidean_vcr_profile(
    num_voters,
    num_cand,
    voter_prob_distribution,
    candidate_prob_distribution,
    voter_radius,
    candidate_radius,
    voter_points=None,
    candidate_points=None,
    return_points=False,
):
    """
    Generate a random profile using the *Euclidean VCR (Voter Candidate Range)* distribution.

    Parameters
    ----------
        num_voters : int
            The desired number of voters in the profile.

        num_cand : int
            The desired number of candidates in the profile.

        voter_prob_distribution : PointProbabilityDistribution or None
            A probability distribution used to generate voter points.

        candidate_prob_distribution : PointProbabilityDistribution or None
            A probability distribution used to generate candidate points.

        voter_radius, candidate_radius : float or list of float
            Radius of candidates and voters to determine approval ballots.

            If a float is given, this radius applies to all voters/candidates.
            If a list of floats is given, this specifies the radius for each voter/candidate
            individually. In this case, the length of `voter_radius`/`candidate_radius` must
            be `num_voters`/`num_cand`.

            A voter approves a candidate if their distance is <= the voter's radius + the
            candidate's radius.

        voter_points : iterable
            A list of points.

            The length of this list must be `num_voters`. The dimension of the points must be
            the same as the points in `candiddate_points` or as specified by
            `candidate_prob_distribution`.
            This parameter is only used if `voter_prob_distribution` is `None`.

        candidate_points : iterable
            A list of points.

            The length of this list must be `num_cand`. The dimension of the points must be
            the same as the points in `voter_points` or as specified by `voter_prob_distribution`.
            This parameter is only used if `candidate_prob_distribution` is `None`.

        return_points : bool, optional
            If true, also return the list of voter points and a list of candidate points.

    Returns
    -------
        abcvoting.preferences.Profile

    References
    ----------
    *How to Sample Approval Elections?*
    Stanisław Szufa, Piotr Faliszewski, Łukasz Janeczko, Martin Lackner, Arkadii Slinko,
    Krzysztof Sornat, Nimrod Talmon.
    https://arxiv.org/abs/2207.01140
    """
    voter_points, candidate_points = _voter_and_candidate_points(
        num_voters,
        num_cand,
        voter_prob_distribution,
        candidate_prob_distribution,
        voter_points,
        candidate_points,
    )
    try:
        if len(voter_radius) != num_voters:
            raise ValueError("Length of `voter_radius` must be equal to `num_voters`.")
        voter_range = voter_radius
    except TypeError:
        voter_range = np.array([voter_radius for _ in range(num_voters)])
    try:
        if len(candidate_radius) != num_cand:
            raise ValueError("Length of `candidate_radius` must be equal to `num_cand`.")
        cand_range = candidate_radius
    except TypeError:
        cand_range = np.array([candidate_radius for _ in range(num_cand)])

    approval_sets = [set() for _ in range(num_voters)]
    for v in range(num_voters):
        for c in range(num_cand):
            if voter_range[v] + cand_range[c] >= np.linalg.norm(
                voter_points[v] - candidate_points[c]
            ):
                approval_sets[v].add(c)
    profile = Profile(num_cand)
    profile.add_voters(approval_sets)

    if return_points:
        return profile, voter_points, candidate_points

    return profile


def _voter_and_candidate_points(
    num_voters,
    num_cand,
    voter_prob_distribution,
    candidate_prob_distribution,
    voter_points,
    candidate_points,
):
    if voter_prob_distribution is not None:
        voter_points = np.array([random_point(voter_prob_distribution) for _ in range(num_voters)])
        voter_dimension = voter_prob_distribution.dimension
    else:
        voter_dimension = {len(p) for p in voter_points}
        if len(voter_dimension) != 1:
            raise ValueError("Voter points have different dimensions.")
        voter_points = np.array(voter_points)
        if len(voter_points) != num_voters:
            raise ValueError("Length of `voters` is not the same as `num_voters`.")
        voter_dimension = min(voter_dimension)
    if candidate_prob_distribution is not None:
        candidate_points = np.array(
            [random_point(candidate_prob_distribution) for _ in range(num_cand)]
        )
        candidate_dimension = candidate_prob_distribution.dimension
    else:
        candidate_dimension = {len(p) for p in candidate_points}
        if len(candidate_dimension) != 1:
            raise ValueError("Candidate points have different dimensions.")
        candidate_points = np.array(candidate_points)
        if len(candidate_points) != num_cand:
            raise ValueError("Length of `candidates` is not the same as `num_cand`.")
        candidate_dimension = min(candidate_dimension)
    if voter_dimension != candidate_dimension:
        raise ValueError("Voter points and candidate points have a different dimension.")

    return voter_points, candidate_points


class PointProbabilityDistribution:
    r"""
    Class for specifying a probability distribution generating points.

    Parameters
    ----------
        name : str
            Name (identifier) of the probability distribution. See example below.

        center_point : tuple or float, optional
            Center point of the distribution.

            This can be either a point (tuple) or a float. If it is a float, it is assumed
            that the point has this value in all coordinates.

        sigma : float, optional
            Standard deviation (only required for Gaussian distributions).

        width : float
            Width of the geometric shape that constrains the probability distribution.

            This parameter is used for `"1d_interval"`, `"2d_square"`, `"2d_disc"`, and
            `"2d_gaussian_disc"`.

    Examples
    --------
    Here is a visual representation of the different probability distributions available.

    .. testsetup::

        from abcvoting import generate
        from abcvoting.generate import PointProbabilityDistribution
        import matplotlib.pyplot as plt

    .. testcode::

        # distributions to generate points in 1- and 2-dimensional space
        distributions = [
            PointProbabilityDistribution("1d_interval", center_point=[0]),
            PointProbabilityDistribution("1d_gaussian", center_point=[4]),
            PointProbabilityDistribution("1d_gaussian_interval", center_point=[6], width=0.5),
            PointProbabilityDistribution("2d_square", center_point=[0, 2]),
            PointProbabilityDistribution("2d_disc", center_point=[2, 2]),
            PointProbabilityDistribution("2d_gaussian", center_point=[4, 2], sigma=0.25),
            PointProbabilityDistribution("2d_gaussian_disc", center_point=[6, 2], sigma=0.25),
        ]

    .. plot::

        from abcvoting import generate
        from abcvoting.generate import PointProbabilityDistribution
        import matplotlib.pyplot as plt

        # distributions to generate points in 1- and 2-dimensional space
        distributions = [
            PointProbabilityDistribution("1d_interval", center_point=[0]),
            PointProbabilityDistribution("1d_gaussian", center_point=[4]),
            PointProbabilityDistribution("1d_gaussian_interval", center_point=[6], width=0.5),
            PointProbabilityDistribution("2d_square", center_point=[0, 2]),
            PointProbabilityDistribution("2d_disc", center_point=[2, 2]),
            PointProbabilityDistribution("2d_gaussian", center_point=[4, 2], sigma=0.25),
            PointProbabilityDistribution("2d_gaussian_disc", center_point=[6, 2], sigma=0.25),
        ]

        fig, ax = plt.subplots(dpi=600, figsize=(7, 3))
        points = []
        for dist in distributions:
            if dist.name.startswith("2d"):
                for _ in range(1000):
                    points.append(generate.random_point(dist))
                title_coord = [dist.center_point[0], dist.center_point[1] + 0.6]
            else:
                for _ in range(100):
                    points.append([generate.random_point(dist), 0])
                title_coord = [dist.center_point[0], 0.2]
            title = dist.name + "\n"
            if dist.width != 1.0:
                title += f"(width={dist.width})"
            plt.annotate(title, title_coord, ha="center")

        ax.scatter([x for x, y in points], [y for x, y in points], alpha=0.5, s=5)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlim(-0.8, 7.3)
        plt.ylim(-0.2, 3.2)
        fig.tight_layout()
        plt.show()
    """

    def __init__(self, name, center_point=0.5, sigma=0.15, width=1.0):
        self.name = name
        self.sigma = sigma  # for Gaussian
        self.width = width

        try:
            self.dimension = int(name.split("d_")[0])
        except ValueError as error:
            raise ValueError(
                f"Could not extract dimension from probability distribution {name}."
            ) from error

        try:
            len(center_point)
            self.center_point = np.array(center_point)
            if len(self.center_point) != self.dimension:
                raise ValueError("Center point has a wrong dimension.")
        except TypeError:
            self.center_point = np.array([center_point] * self.dimension)


def random_point(prob_distribution):
    """
    Generate a point in space according to a given probability distribution.

    Parameters
    ----------
        prob_distribution : PointProbabilityDistribution
            A probability distribution (see :class:`PointProbabilityDistribution`).

    Returns
    -------
        abcvoting.preferences.Profile
    """
    if prob_distribution.name == "1d_interval":
        offset = prob_distribution.center_point[0] - prob_distribution.width / 2
        point = rng.random() * prob_distribution.width + offset
    elif prob_distribution.name == "1d_gaussian":
        point = rng.normal(prob_distribution.center_point[0], prob_distribution.sigma)
    elif prob_distribution.name == "1d_gaussian_interval":
        while True:
            point = rng.normal(prob_distribution.center_point[0], prob_distribution.sigma)
            if (
                prob_distribution.center_point[0] - prob_distribution.width / 2
                <= point
                <= prob_distribution.center_point[0] + prob_distribution.width / 2
            ):
                break
    elif prob_distribution.name == "2d_disc":
        phi = 2.0 * 180.0 * rng.random()
        radius = math.sqrt(rng.random()) * prob_distribution.width / 2
        point = [
            prob_distribution.center_point[0] + radius * math.cos(phi),
            prob_distribution.center_point[1] + radius * math.sin(phi),
        ]
    elif prob_distribution.name == "2d_square":
        offset = prob_distribution.center_point - prob_distribution.width / 2
        point = rng.random(2) * prob_distribution.width + offset
    elif prob_distribution.name == "2d_gaussian":
        point = [
            rng.normal(prob_distribution.center_point[0], prob_distribution.sigma),
            rng.normal(prob_distribution.center_point[1], prob_distribution.sigma),
        ]
    elif prob_distribution.name == "2d_gaussian_disc":
        while True:
            point = [
                rng.normal(prob_distribution.center_point[0], prob_distribution.sigma),
                rng.normal(prob_distribution.center_point[1], prob_distribution.sigma),
            ]
            if (
                np.linalg.norm(point - prob_distribution.center_point)
                <= prob_distribution.width / 2
            ):
                break
    elif prob_distribution.name == "3d_cube":
        point = rng.random(3) * prob_distribution.width + prob_distribution.center_point
    else:
        raise ValueError(f"unknown name of point distribution: {prob_distribution.name}")

    return point


PROBABILITY_DISTRIBUTIONS = {
    "IC fixed-size": random_ic_fixed_size_profile,
    "IC": random_ic_profile,
    "Truncated Mallows": random_mallows_profile,
    "Urn fixed-size": random_urn_fixed_size_profile,
    "Urn": random_urn_profile,
    "Truncated Urn": random_truncated_urn_profile,
    "Euclidean VCR": random_euclidean_vcr_profile,
    "Euclidean fixed-size": random_euclidean_fixed_size_profile,
    "Euclidean Threshold": random_euclidean_threshold_profile,
    "Resampling": random_resampling_profile,
    "Disjoint Resampling": random_disjoint_resampling_profile,
    "Noise": random_noise_model_profile,
}
PROBABILITY_DISTRIBUTION_IDS = tuple(PROBABILITY_DISTRIBUTIONS.keys())

rng = default_rng()  # random number generator
