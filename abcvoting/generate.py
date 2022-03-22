# -*- coding: utf-8 -*-
"""
Random generation of approval profiles.

This module is based on the paper
    *How to Sample Approval Elections?*
    Stanisław Szufa, Piotr Faliszewski, Łukasz Janeczko, Martin Lackner, Arkadii Slinko,
    Krzysztof Sornat, Nimrod Talmon.
    TODO: URL
"""


import random
from abcvoting.preferences import Profile
from abcvoting import misc
import numpy as np
import math


def random_profile(prob_model, num_voters, num_cand, **kwargs):
    """
    Generate a random profile using the probability model `prob_model`.

    The following probability models are supported:

    .. doctest::

        >>> PROBABILITY_MODELS_IDS  # doctest: +NORMALIZE_WHITESPACE
        ('IC fixed-size', 'IC', 'Truncated Mallows', 'Urn fixed-size', 'Urn',
        'Truncated Urn', 'Euclidean VCR', 'Euclidean fixed-size', 'Euclidean Threshold',
        'Resampling', 'Disjoint Resampling', 'Noise')

    Parameters
    ----------
        prob_model : str
            A probability model.

        num_voters : int
            The desired number of voters in the profile.

        num_cand : int
            The desired number of candidates in the profile.

        **kwargs : dict
            Further arguments for the probability model `prob_model`.

    Returns
    -------
        abcvoting.preferences.Profile

    Examples
    --------
    Generate a profile via the Independent Culture (IC) distribution with a probability of `0.5`.

    .. testsetup::

        np.random.seed(24121838)

    .. doctest::

        >>> profile = random_profile(prob_model="IC", num_voters=5, num_cand=5, p=0.5)
        >>> print(profile)
        profile with 5 votes and 5 candidates:
         {1, 2},
         {0, 1, 2},
         {3, 4},
         {0, 4},
         {2, 4}
    """
    if prob_model not in PROBABILITY_MODELS_IDS:
        raise ValueError(f"Probability model {prob_model} unknown.")
    return _probability_models[prob_model](num_voters, num_cand, **kwargs)


def random_urn_fixed_size_profile(num_voters, num_cand, setsize, replace):
    """
    Generate a random profile using the *Polya Urn with fixed-size approval sets* model.

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
        r = random.random() * currsize
        if r < 1.0:
            # base case: sample uniformly at random
            randset = random.sample(range(num_cand), setsize)
            approval_sets.append(randset)
            key = tuple(set(randset))
            if key in replacedsets:
                replacedsets[key] += 1
            else:
                replacedsets[key] = 1
            currsize += replace
        else:
            # sample from one of the replaced ballots
            r = random.randint(0, sum(replacedsets.values()))
            for approval_set in replacedsets:
                count = replacedsets[approval_set]
                if r <= count:
                    approval_sets.append(list(approval_set))
                    break
                else:
                    r -= count
    profile = Profile(num_cand)
    profile.add_voters(approval_sets)
    return profile


def random_ic_fixed_size_profile(num_voters, num_cand, setsize):
    """
    Generate a random profile using the *IC with fixed-size approval sets* model.

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
    approval_sets = []
    for _ in range(num_voters):
        randset = random.sample(range(num_cand), setsize)
        approval_sets.append(randset)
    profile = Profile(num_cand)
    profile.add_voters(approval_sets)
    return profile


def random_mallows_profile(num_voters, num_cand, setsize, dispersion):
    """
    Generate a random profile using the *Truncated Mallows* probability model.

    After the definition for
    repeated insertion  mode (RIM) in
    https://icml.cc/2011/papers/135_icmlpaper.pdf

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

    def _select_pos(distribution):
        """Returns a randomly selected value with the help of the distribution."""
        if round(sum(distribution), 10) != 1.0:
            raise Exception("Invalid Distribution", distribution, "sum:", sum(distribution))
        r = round(random.random(), 10)  # or random.uniform(0, 1)
        pos = -1
        s = 0
        for p in distribution:
            pos += 1
            s += p
            if s >= r:
                return pos

        return pos  # in case of rounding errors

    if not (0 < dispersion <= 1):
        raise Exception("Invalid dispersion, needs to be in (0, 1].")
    reference_ranking = list(range(num_cand))
    random.shuffle(reference_ranking)
    insert_dist = _compute_mallows_insert_distributions(num_cand, dispersion)
    approval_sets = []
    for _ in range(num_voters):
        vote = []
        for i, distribution in enumerate(insert_dist):
            pos = _select_pos(distribution)
            vote.insert(pos, reference_ranking[i])

        approval_sets.append(vote[:setsize])
    profile = Profile(num_cand)
    profile.add_voters(approval_sets)
    return profile


def _compute_mallows_insert_distributions(num_cand, dispersion):
    """
    Compute the insertion probability vectors for the dispersion and a given number of candidates.
    """
    distributions = []
    denominator = 0
    for i in range(num_cand):
        # compute the denominator = dispersion^0 + dispersion^1
        # + ... dispersion^(i-1)
        denominator += pow(dispersion, i)
        dist = []
        for j in range(i + 1):  # 0..i
            dist.append(pow(dispersion, i - j) / denominator)
        distributions.append(dist)
    return distributions


# Impartial Culture
def random_ic_profile(num_voters, num_cand, p=0.5):
    """
    Generate a random profile using the *Independent Culture (IC)* probability model.

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
    TODO: URL
    """
    approval_sets = [set() for _ in range(num_voters)]
    for i in range(num_voters):
        for j in range(num_cand):
            if np.random.random() <= p:
                approval_sets[i].add(j)
    profile = Profile(num_cand)
    profile.add_voters(approval_sets)
    return profile


def _ordinal_urn_profile(num_voters, num_cand, replace):
    """
    Generate rankings according to the Urn probability model.
    """
    rankings = []
    urn_size = 1.0
    for j in range(num_voters):
        rho = np.random.uniform(0, urn_size)
        if rho <= 1.0:
            rankings.append(np.random.permutation(num_cand))
        else:
            rankings.append(rankings[np.random.randint(0, j)])
        urn_size += replace
    print(rankings)
    return rankings


def random_urn_profile(num_voters, num_cand, p, replace):
    """
    Generate a random profile using the *Polya Urn* probability model.

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
        rho = np.random.uniform(0, urn_size)
        if rho <= 1.0:
            vote = set()
            for c in range(num_cand):
                if np.random.random() <= p:
                    vote.add(c)
            approval_sets.append(vote)
        else:
            approval_sets.append(approval_sets[np.random.randint(0, j)])
        urn_size += replace
    profile = Profile(num_cand)
    profile.add_voters(approval_sets)
    return profile


def random_truncated_urn_profile(num_voters, num_cand, setsize, replace):
    """
    Generate a random profile using the *Truncated Polya Urn* probability model.

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
    TODO: URL
    """
    ordinal_votes = _ordinal_urn_profile(num_voters, num_cand, replace)
    approval_sets = []
    for v in range(num_voters):
        approval_sets.append(set(ordinal_votes[v][:setsize]))
    profile = Profile(num_cand)
    profile.add_voters(approval_sets)
    return profile


# Resampling
def random_resampling_profile(num_voters, num_cand, p, phi):
    """
    Generate a random profile using the *Resampling* probability model.

    Parameters
    ----------
        num_voters : int
            The desired number of voters in the profile.

        num_cand : int
            The desired number of candidates in the profile.

        p, phi : float in [0, 1]
            Parameters of (p,phi)-Resampling.

    Returns
    -------
        abcvoting.preferences.Profile

    References
    ----------
    *How to Sample Approval Elections?*
    Stanisław Szufa, Piotr Faliszewski, Łukasz Janeczko, Martin Lackner, Arkadii Slinko,
    Krzysztof Sornat, Nimrod Talmon.
    TODO: URL
    """
    k = int(p * num_cand)
    central_vote = {i for i in range(k)}

    approval_sets = [set() for _ in range(num_voters)]
    for v in range(num_voters):
        vote = set()
        for c in range(num_cand):
            if np.random.random() < phi:
                if np.random.random() < p:
                    vote.add(c)
            else:
                if c in central_vote:
                    vote.add(c)
        approval_sets[v] = vote
    profile = Profile(num_cand)
    profile.add_voters(approval_sets)
    return profile


def random_disjoint_resampling_profile(num_voters, num_cand, p, phi=None, num_groups=2):
    """
    Generate a random profile using the *(p,phi,g)-Disjoint Resampling* probability model.

    Parameters
    ----------
        num_voters : int
            The desired number of voters in the profile.

        num_cand : int
            The desired number of candidates in the profile.

        p, phi : float in [0, 1]
            Parameters of (p,phi,g)-Disjoint Resampling.

        num_groups : int, optional
            Corresponds to the parameter g in (p,phi,g)-Disjoint Resampling.

    Returns
    -------
        abcvoting.preferences.Profile

    References
    ----------
    *How to Sample Approval Elections?*
    Stanisław Szufa, Piotr Faliszewski, Łukasz Janeczko, Martin Lackner, Arkadii Slinko,
    Krzysztof Sornat, Nimrod Talmon.
    TODO: URL
    """

    def _uniform_in_simplex(n):
        """Return uniformly random vector in the n-simplex."""
        k = np.random.exponential(scale=1.0, size=n)
        return k / sum(k)

    if phi is None:
        phi = np.random.random()
    k = int(p * num_cand)

    sizes = _uniform_in_simplex(num_groups)
    sizes = np.cumsum(np.concatenate(([0], sizes)))

    approval_sets = [set() for _ in range(num_voters)]

    for g in range(num_groups):

        central_vote = {g * k + i for i in range(k)}

        for v in range(int(sizes[g] * num_voters), int(sizes[g + 1] * num_voters)):
            vote = set()
            for c in range(num_cand):
                if np.random.random() <= phi:
                    if np.random.random() <= p:
                        vote.add(c)
                else:
                    if c in central_vote:
                        vote.add(c)
            approval_sets[v] = vote
    profile = Profile(num_cand)
    profile.add_voters(approval_sets)
    return profile


# Noise model
def random_noise_model_profile(num_voters, num_cand, p, phi, distance="hamming"):
    """
    Generate a random profile using the *Random Noise* probability model.

    Parameters
    ----------
        num_voters : int
            The desired number of voters in the profile.

        num_cand : int
            The desired number of candidates in the profile.

        p, phi : float in [0, 1]
            Parameters.

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
    TODO: URL
    """
    k = int(p * num_cand)
    A = {i for i in range(k)}
    B = set(range(num_cand)) - A

    choices = []
    probabilites = []

    # PREPARE BUCKETS
    for x in range(len(A) + 1):
        num_options_in = misc.binom(len(A), x)
        for y in range(len(B) + 1):
            num_options_out = misc.binom(len(B), y)

            if distance == "hamming":
                factor = phi ** (len(A) - x + y)  # Hamming
            elif distance == "jaccard":
                factor = phi ** ((len(A) - x + y) / (len(A) + y))  # Jaccard
            elif distance == "zelinka":
                factor = phi ** max(len(A) - x, y)  # Zelinka
            elif distance == "bunke-shearer":
                factor = phi ** (max(len(A) - x, y) / max(len(A), x + y))  # Bunke-Shearer
            else:
                raise ValueError(f"Distance {distance} not known.")

            num_options = num_options_in * num_options_out * factor

            choices.append((x, y))
            probabilites.append(num_options)

    denominator = sum(probabilites)
    probabilites = [p / denominator for p in probabilites]

    # SAMPLE VOTES
    approval_sets = []
    for _ in range(num_voters):
        _id = np.random.choice(range(len(choices)), 1, p=probabilites)[0]
        x, y = choices[_id]
        vote = set(np.random.choice(list(A), x, replace=False))
        vote = vote.union(set(np.random.choice(list(B), y, replace=False)))
        approval_sets.append(vote)
    profile = Profile(num_cand)
    profile.add_voters(approval_sets)
    return profile


def random_euclidean_fixed_size_profile(
    num_voters,
    num_cand,
    voter_probdist,
    candidate_probdist,
    setsize,
    voter_points=None,
    candidate_points=None,
    return_points=False,
):
    """
    Generate a random profile using the *Euclidean VCR (Voter Candidate Range)* probability model.

    Parameters
    ----------
        num_voters : int
            The desired number of voters in the profile.

        num_cand : int
            The desired number of candidates in the profile.

        voter_probdist : PointProbabilityDistribution or None
            A probability distribution used to generate voter points.

        candidate_probdist : PointProbabilityDistribution or None
            A probability distribution used to generate candidate points.

        setsize : int
            Number of candidates that each voter approves.

        voter_points : iterable
            A list of points.

            The length of this list must be `num_voters`. The dimension of the points must be
            the same as the points in `candiddate_points` or as specified by `candidate_probdist`.
            This parameter is only used if `voter_probdist` is `None`.

        candidate_points : iterable
            A list of points.

            The length of this list must be `num_cand`. The dimension of the points must be
            the same as the points in `voter_points` or as specified by `voter_probdist`.
            This parameter is only used if `candidate_probdist` is `None`.

        return_points : bool, optional
            If true, also return the list of voter points and a list of candidate points.

    Returns
    -------
        abcvoting.preferences.Profile
    """

    voter_points, candidate_points = _voter_and_candidate_points(
        num_voters,
        num_cand,
        voter_probdist,
        candidate_probdist,
        voter_points,
        candidate_points,
    )
    profile = Profile(num_cand)
    approval_sets = []
    for v, voterpoint in enumerate(voter_points):
        distances = {
            cand: np.linalg.norm(voterpoint - candidate_points[cand])
            for cand in profile.candidates
        }
        cands_sorted = sorted(distances, key=distances.get)
        approval_sets.append(cands_sorted[:setsize])
    profile.add_voters(approval_sets)

    if return_points:
        return profile, voter_points, candidate_points
    else:
        return profile


def random_euclidean_threshold_profile(
    num_voters,
    num_cand,
    voter_probdist,
    candidate_probdist,
    threshold,
    voter_points=None,
    candidate_points=None,
    return_points=False,
):
    """
    Generate a random profile using the *Euclidean Threshold* probability model.

    Parameters
    ----------
        num_voters : int
            The desired number of voters in the profile.

        num_cand : int
            The desired number of candidates in the profile.

        voter_probdist : PointProbabilityDistribution or None
            A probability distribution used to generate voter points.

        candidate_probdist : PointProbabilityDistribution or None
            A probability distribution used to generate candidate points.

        threshold : float
            Voters' tolerance for approving candidates. This is a float >= 1.

            A voter approves all candididates that have at a distance of at most `threshold` * `d`,
            where `d` is the minimum distance between this voter and any candidate.
            Setting `threshold` to 1 means that only the closest candidate is approved (there
            might be more than one).

        voter_points : iterable
            A list of points.

            The length of this list must be `num_voters`. The dimension of the points must be
            the same as the points in `candiddate_points` or as specified by `candidate_probdist`.
            This parameter is only used if `voter_probdist` is `None`.

        candidate_points : iterable
            A list of points.

            The length of this list must be `num_cand`. The dimension of the points must be
            the same as the points in `voter_points` or as specified by `voter_probdist`.
            This parameter is only used if `candidate_probdist` is `None`.

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
        voter_probdist,
        candidate_probdist,
        voter_points,
        candidate_points,
    )

    profile = Profile(num_cand)
    approval_sets = []
    for v, voterpoint in enumerate(voter_points):
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
    else:
        return profile


def random_euclidean_vcr_profile(
    num_voters,
    num_cand,
    voter_probdist,
    candidate_probdist,
    voter_radius,
    candidate_radius,
    voter_points=None,
    candidate_points=None,
    return_points=False,
):
    """
    Generate a random profile using the *Euclidean VCR (Voter Candidate Range)* probability model.

    Parameters
    ----------
        num_voters : int
            The desired number of voters in the profile.

        num_cand : int
            The desired number of candidates in the profile.

        voter_probdist : PointProbabilityDistribution or None
            A probability distribution used to generate voter points.

        candidate_probdist : PointProbabilityDistribution or None
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
            the same as the points in `candiddate_points` or as specified by `candidate_probdist`.
            This parameter is only used if `voter_probdist` is `None`.

        candidate_points : iterable
            A list of points.

            The length of this list must be `num_cand`. The dimension of the points must be
            the same as the points in `voter_points` or as specified by `voter_probdist`.
            This parameter is only used if `candidate_probdist` is `None`.

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
    TODO: URL
    """
    voter_points, candidate_points = _voter_and_candidate_points(
        num_voters,
        num_cand,
        voter_probdist,
        candidate_probdist,
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
            print(voter_points[v], candidate_points[c])
            if voter_range[v] + cand_range[c] >= np.linalg.norm(
                voter_points[v] - candidate_points[c]
            ):
                approval_sets[v].add(c)
    profile = Profile(num_cand)
    profile.add_voters(approval_sets)

    if return_points:
        return profile, voter_points, candidate_points
    else:
        return profile


def _voter_and_candidate_points(
    num_voters,
    num_cand,
    voter_probdist,
    candidate_probdist,
    voter_points,
    candidate_points,
):
    if voter_probdist is not None:
        voter_points = np.array([random_point(voter_probdist) for _ in range(num_voters)])
        voter_dimension = voter_probdist.dimension
    else:
        voter_dimension = set([len(p) for p in voter_points])
        if len(voter_dimension) != 1:
            raise ValueError("Voter points have different dimensions.")
        voter_points = np.array(voter_points)
        if len(voter_points) != num_voters:
            raise ValueError("Length of `voters` is not the same as `num_voters`.")
        voter_dimension = min(voter_dimension)
    if candidate_probdist is not None:
        candidate_points = np.array([random_point(candidate_probdist) for _ in range(num_cand)])
        candidate_dimension = candidate_probdist.dimension
    else:
        candidate_dimension = set([len(p) for p in candidate_points])
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
        except ValueError:
            raise ValueError(f"Could not extract dimension from probability distribution {name}.")

        try:
            len(center_point)
            self.center_point = np.array(center_point)
            if len(self.center_point) != self.dimension:
                raise ValueError("Center point has a wrong dimension.")
        except TypeError:
            self.center_point = np.array([center_point] * self.dimension)


def random_point(probdist):
    """
    Generate a point in space according to a given probability distribution.

    Parameters
    ----------
        probdist : PointProbabilityDistribution
            A probability distribution (see :class:`PointProbabilityDistribution`).

    Returns
    -------
        abcvoting.preferences.Profile
    """
    if probdist.name == "1d_interval":
        return np.random.rand() * probdist.width + (probdist.center_point[0] - probdist.width / 2)
    elif probdist.name == "1d_gaussian":
        point = np.random.normal(probdist.center_point[0], probdist.sigma)
    elif probdist.name == "1d_gaussian_interval":
        while True:
            point = np.random.normal(probdist.center_point[0], probdist.sigma)
            if (
                probdist.center_point[0] - probdist.width / 2
                <= point
                <= probdist.center_point[0] + probdist.width / 2
            ):
                break
    elif probdist.name == "2d_disc":
        phi = 2.0 * 180.0 * np.random.random()
        radius = math.sqrt(np.random.random()) * probdist.width / 2
        point = [
            probdist.center_point[0] + radius * math.cos(phi),
            probdist.center_point[1] + radius * math.sin(phi),
        ]
    elif probdist.name == "2d_square":
        point = np.random.random(2) * probdist.width + (probdist.center_point - probdist.width / 2)
    elif probdist.name == "2d_gaussian":
        point = [
            np.random.normal(probdist.center_point[0], probdist.sigma),
            np.random.normal(probdist.center_point[1], probdist.sigma),
        ]
    elif probdist.name == "2d_gaussian_disc":
        while True:
            point = [
                np.random.normal(probdist.center_point[0], probdist.sigma),
                np.random.normal(probdist.center_point[1], probdist.sigma),
            ]
            if np.linalg.norm(point - probdist.center_point) <= probdist.width / 2:
                break
    elif probdist.name == "3d_cube":
        point = np.random.random(3) * probdist.width + probdist.center_point
    else:
        raise ValueError(f"unknown name of point distribution: {probdist.name}")
    return point


_probability_models = {
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
PROBABILITY_MODELS_IDS = tuple(_probability_models.keys())
