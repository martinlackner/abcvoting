"""Calculating scores related to the computation of ABC voting rules."""

try:
    from gmpy2 import mpq as Fraction
except ImportError:
    from fractions import Fraction
import functools
import networkx as nx
from abcvoting.misc import hamming


class UnknownScoreFunctionError(ValueError):
    """
    Exception raised if unknown rule id is used.

    Parameters
    ----------
        scorefct_id : str
            The score function id that is not known.
    """

    def __init__(self, scorefct_id):
        message = 'Thiele score function "' + str(scorefct_id) + '" is not known.'
        super().__init__(message)


def get_marginal_scorefct(scorefct_id, committeesize=None):
    """
    Return marginal score function (for a Thiele method) given its name.

    Parameters
    ----------
        scorefct_id : str
            A string identifying the score function.

        committeesize : int, optional
            Committee size.

            Some marginal score functions require fixing the size of committees.

    Returns
    -------
        function
            The corresponding marginal score function.
    """
    if scorefct_id == "pav":
        return pav_score_fct
    if scorefct_id == "slav":
        return slav_score_fct
    if scorefct_id == "cc":
        return cc_score_fct
    if scorefct_id == "av":
        return av_score_fct
    if scorefct_id[:4] == "geom":
        base = Fraction(scorefct_id[4:])
        return functools.partial(geometric_marginal_score_fct, base=base)
    if scorefct_id[:7] == "atleast":
        param = int(scorefct_id[7:])
        return functools.partial(at_least_ell_fct, ell=param)

    raise UnknownScoreFunctionError(scorefct_id)


def thiele_score(scorefct_id, profile, committee):
    """
    Compute Thiele score of a committee subject to a given scorefct_id.

    Parameters
    ----------
        scorefct_id : str
            Identifies the score function to be used.

            `scorefct_id` has to be recognized by `abcvoting.scores.get_scorefct`.

        profile : abcvoting.preferences.Profile
            A profile.

        committee : set or tuple or list
            A committee.

    Returns
    -------
        int or Fraction
            The Thiele score using the score function given by `scorefct_id`.
    """
    marginal_scorefct = get_marginal_scorefct(scorefct_id, len(committee))
    score = 0
    for vote in profile:
        cand_in_com = 0
        for _ in set(committee) & vote.approved:
            cand_in_com += 1
            score += vote.weight * marginal_scorefct(cand_in_com)
    return score


#
# Thiele marginal score functions:
#


def geometric_marginal_score_fct(i, base):
    """
    Geometric marginal score functions.

    This is the additional (marginal) score from a voter for the `i`-th approved candidate
    in the committee.

    For example, the 2-Geometric marginal scoring function (`base=2`) is

    .. math::

       f(i) = 1 / 2^{i-1}.

    For a mathematical description of Geometric score functions, see e.g.
    Martin Lackner and Piotr Skowron
    Utilitarian Welfare and Representation Guarantees of Approval-Based Multiwinner Rules
    In Artificial Intelligence, 288: 103366, 2020.
    https://arxiv.org/abs/1801.01527

    Parameters
    ----------
        i : int
            We are calculating the score for the `i`-th approved candidate in the committee.

        base : float or int or Fraction
            The base for the geometric function `1 / base ** (i-1)`.

    Returns
    -------
        Fraction
            The corresponding marginal score.
    """
    if i == 0:
        return 0
    return Fraction(1, base ** (i - 1))


def pav_score_fct(i):
    """
    PAV marginal score function.

    This is the additional (marginal) score from a voter for the `i`-th approved candidate
    in the committee.

    Parameters
    ----------
        i : int
            We are calculating the score for the `i`-th approved candidate in the committee.

    Returns
    -------
        Fraction
            The corresponding marginal score.
    """
    if i == 0:
        return 0
    return Fraction(1, i)


def slav_score_fct(i):
    """
    SLAV (Sainte-Lague Approval Voting) marginal score function.

    This is the additional (marginal) score from a voter for the `i`-th approved candidate
    in the committee.

    For a mathematical description of this score function, see e.g.
    Martin Lackner and Piotr Skowron
    Utilitarian Welfare and Representation Guarantees of Approval-Based Multiwinner Rules
    In Artificial Intelligence, 288: 103366, 2020.
    https://arxiv.org/abs/1801.01527

    Parameters
    ----------
        i : int
            We are calculating the score for the `i`-th approved candidate in the committee.

    Returns
    -------
        Fraction
            The corresponding marginal score.
    """
    if i == 0:
        return 0
    return Fraction(1, 2 * i - 1)


def av_score_fct(i):
    """
    AV marginal score function.

    This is the additional (marginal) score from a voter for the `i`-th approved candidate
    in the committee.

    Note: this is used only for unit tests at the moment,
    because AV is separable anyway and therefore not implemented as optimization problem.

    Parameters
    ----------
        i : int
            We are calculating the score for the `i`-th approved candidate in the committee.

    Returns
    -------
        Fraction
            The corresponding marginal score.
    """
    if i >= 1:
        return Fraction(1)
    return Fraction(0)


def cc_score_fct(i):
    """
    CC (Chamberlin-Courant) marginal score function.

    This is the additional (marginal) score from a voter for the `i`-th approved candidate
    in the committee.

    Parameters
    ----------
        i : int
            We are calculating the score for the `i`-th approved candidate in the committee.

    Returns
    -------
        Fraction
            The corresponding marginal score.
    """
    if i == 1:
        return Fraction(1)
    return Fraction(0)


def at_least_ell_fct(i, ell):
    """
    At-least-ell marginal score function.

    This is the additional (marginal) score from a voter for the `i`-th approved candidate
    in the committee.

    Gives a score of 1 if `ell` approved candidates are in the committee.
    The CC score function is equivalent to the `at_least_ell_fct` score function for `ell=1`.

    Parameters
    ----------
        i : int
            We are calculating the score for the `i`-th approved candidate in the committee.

        ell : int
            Gives a score of 1 if `ell` approved candidates are in the committee.

    Returns
    -------
        Fraction
            The corresponding marginal score.
    """
    if i == ell:
        return Fraction(1)
    return Fraction(0)


#
# end of Thiele marginal score functions:
#


def cumulative_score(marginal_scorefct, cand_in_com):
    """
    Return cumulative score using the marginal score function `marginal_scorefct` for a voter.

    A cumulative score function f(i) returns the total score for having
    i candidates in the committee (as opposed to score functions that return the score increase
    when adding the i-th candidate).

    Parameters
    ----------
        marginal_scorefct : func
            The marginal score function.

        cand_in_com : int
            The number of approved candidates in the committee.

    Returns
    -------
        Fraction
            Score of a voter with `cand_in_com` many approved candidates in the committee.
    """
    return sum(marginal_scorefct(i + 1) for i in range(cand_in_com))


def marginal_thiele_scores_add(marginal_scorefct, profile, committee):
    """
    Return marginal score increases from adding one candidate to the committee.

    The function returns a list of length `num_cand` where the i-th entry contains the
    marginal score increase when adding candidate i.
    Candidates that are already in the committee receive a small value (-1).

    Parameters
    ----------
        marginal_scorefct : func
            The marginal score function to be used.

        profile : abcvoting.preferences.Profile
            A profile.

        committee : iterable of int
            A committee.

    Returns
    -------
        list
            Marginal score increases from adding candidates to the committee.
    """
    marginal = [0] * profile.num_cand
    for voter in profile:
        intersectionsize = len(voter.approved.intersection(committee))
        for cand in voter.approved:
            marginal[cand] += voter.weight * marginal_scorefct(intersectionsize + 1)
    for cand in committee:
        marginal[cand] = -1
    return marginal


def marginal_thiele_scores_remove(marginal_scorefct, profile, committee):
    """
    Return marginal score decreases from removing one candidate from the committee.

    The function returns a list of length `num_cand` where the i-th entry contains the
    marginal score decrease when removing candidate i.
    Candidates that are not in the committee receive a large value (max(marg_util_cand) + 1).

    Parameters
    ----------
        marginal_scorefct : func
            The marginal score function to be used.

        profile : abcvoting.preferences.Profile
            A profile.

        committee : set
            A committee.

    Returns
    -------
        list
            Marginal score decreases from removing candidates from the committee.
    """
    marg_util_cand = [0] * profile.num_cand
    #  marginal utility gained by adding candidate to the committee
    for voter in profile:
        satisfaction = len(voter.approved.intersection(committee))
        for cand in voter.approved:
            marg_util_cand[cand] += voter.weight * marginal_scorefct(satisfaction)
    for cand in profile.candidates:
        if cand not in committee:
            # do not choose candidates that already have been removed
            marg_util_cand[cand] = max(marg_util_cand) + 1
    return marg_util_cand


def monroescore(profile, committee):
    """
    Return Monroe score of a given committee.

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committee : iterable of int
            A committee.

    Returns
    -------
        int
            The Monroe score.
    """
    if len(profile) % len(committee) == 0:
        # faster
        return monroescore_matching(profile, committee)

    return monroescore_flowbased(profile, committee)


def monroescore_matching(profile, committee):
    """
    Return Monroe score of a given committee.

    Uses a matching-based algorithm that works only if
    the committee size divides the number of voters.

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committee : set
            A committee.

    Returns
    -------
        int
            The Monroe score.
    """
    if len(profile) % len(committee) != 0:
        raise ValueError(
            "monroescore_matching() works only if "
            + "the committee size divides the number of voters "
        )
    graph = nx.Graph()
    sizeofdistricts = len(profile) // len(committee)
    voter_nodes = range(len(profile))
    graph.add_nodes_from(voter_nodes, bipartite=0)
    graph.add_nodes_from(
        [f"{cand}/{j}" for cand in committee for j in range(sizeofdistricts)], bipartite=1
    )  # candidates with multiplicities

    for cand in committee:
        interestedvoters = [
            i for i in voter_nodes if cand in profile[i].approved
        ]  # voters that approve `cand`
        graph.add_edges_from(
            [(i, f"{cand}/{j}") for j in range(sizeofdistricts) for i in interestedvoters]
        )  # edge from all interested voters to all nodes corresponding to `cand`
    matching = nx.bipartite.maximum_matching(graph, top_nodes=voter_nodes)
    return len(matching) // 2  # `matching` is a dictionary representing *directed* edges


def monroescore_flowbased(profile, committee):
    """
    Return Monroe score of a given committee.

    Uses a flow-based algorithm that works even if
    `committeesize` does not divide the number of voters.
    Slower than monroescore_matching().

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committee : set
            A committee.

    Returns
    -------
        int
            The Monroe score.
    """
    graph = nx.DiGraph()
    committeesize = len(committee)
    # the lower bound of the size of districts
    lower_bound = len(profile) // committeesize
    # number of voters that will be contribute to the excess
    # of the lower bounds of districts
    overflow = len(profile) - committeesize * lower_bound
    # add a sink node for the overflow
    graph.add_node("sink", demand=overflow)
    for i in committee:
        graph.add_node(i, demand=lower_bound)
        graph.add_edge(i, "sink", weight=0, capacity=1)
    for i, voter in enumerate(profile):
        voter_name = "v" + str(i)
        graph.add_node(voter_name, demand=-1)
        for cand in committee:
            if cand in voter.approved:
                graph.add_edge(voter_name, cand, weight=0, capacity=1)
            else:
                graph.add_edge(voter_name, cand, weight=1, capacity=1)
    # compute the minimal cost assignment of voters to candidates,
    # i.e. the unrepresented voters, and subtract it from the total number
    # of voters
    return len(profile) - nx.capacity_scaling(graph)[0]


def minimaxav_score(profile, committee):
    """
    Return the Minimax AV (MAV) score of a committee.

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committee : iterable of int
            A committee.

    Returns
    -------
        int
            The Minimax AV score of `committee`.
    """
    score = 0
    for voter in profile:
        hamdistance = hamming(voter.approved, committee)
        if hamdistance > score:
            score = hamdistance
    return score


def num_voters_with_upper_bounded_hamming_distance(upperbound, profile, committee):
    """
    Return the number of voters having a Hamming distance <= `upperbound` to the given committee.

    Parameters
    ----------
        upperbound : int
            The Hamming distance upper bound.

        profile : abcvoting.preferences.Profile
            A profile.

        committee : set
            A committee.

    Returns
    -------
        int
            The number of voters having a Hamming distance <= `upperbound`.
    """
    return len([voter for voter in profile if hamming(voter.approved, committee) <= upperbound])
