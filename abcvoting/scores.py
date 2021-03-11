"""
Calculating all kinds of scores
"""


try:
    from gmpy2 import mpq as Fraction
except ImportError:
    from fractions import Fraction
import functools
from abcvoting.bipartite_matching import matching
import networkx as nx
from abcvoting.misc import hamming


THIELE_SCOREFCTS_STRINGS = ["pav", "slav", "cc", "av"] + [f"geom{param}" for param in [1.5, 2, 5]]


class UnknownScoreFunctionError(ValueError):
    """Exception raised if unknown rule id is used"""

    def __init__(self, scorefct_id):
        message = 'Thiele score function "' + str(scorefct_id) + '" is not known.'
        super(ValueError, self).__init__(message)


def get_scorefct(scorefct_id, committeesize=None):
    """returns score function (for a Thiele method) given its name"""
    if scorefct_id not in THIELE_SCOREFCTS_STRINGS and not scorefct_id.startswith("geom"):
        raise UnknownScoreFunctionError(scorefct_id)

    if scorefct_id == "pav":
        return pav_score_fct
    elif scorefct_id == "slav":
        return slav_score_fct
    elif scorefct_id == "cc":
        return cc_score_fct
    elif scorefct_id == "av":
        return av_score_fct
    elif scorefct_id[:4] == "geom":
        base = Fraction(scorefct_id[4:])
        return functools.partial(geometric_score_fct, base=base)

    raise RuntimeError(
        f"Score function {scorefct_id} is valid but not handled correctly in get_scorefct()."
    )


def thiele_score(scorefct_id, profile, committee):
    """computes the Thiele score of a committee subject to
    a given score function (scorefct_id)
    """
    scorefct = get_scorefct(scorefct_id, len(committee))
    score = 0
    for vote in profile:
        cand_in_com = 0
        for _ in set(committee) & vote.approved:
            cand_in_com += 1
            score += vote.weight * scorefct(cand_in_com)
    return score


### Thiele score functions
def geometric_score_fct(i, base):
    if i == 0:
        return 0
    else:
        return Fraction(1, base ** (i - 1))


def pav_score_fct(i):
    if i == 0:
        return 0
    else:
        return Fraction(1, i)


def slav_score_fct(i):
    if i == 0:
        return 0
    else:
        return Fraction(1, 2 * i - 1)


def av_score_fct(i):
    # note: this is used only for unit tests atm, because AV is separable anyway and therefore not
    # implemented as optimization problem
    if i >= 1:
        return 1
    else:
        return 0


def cc_score_fct(i):
    if i == 1:
        return 1
    else:
        return 0


### end of Thiele score functions


def cumulative_score_fct(scorefct, cand_in_com):
    return sum(scorefct(i + 1) for i in range(cand_in_com))


# returns a list of length num_cand
# the i-th entry contains the marginal score increase
#  gained by adding candidate i
def marginal_thiele_scores_add(scorefct, profile, committee):
    marg = [0] * profile.num_cand
    for voter in profile:
        for cand in voter.approved:
            if voter.approved & set(committee):
                marg[cand] += voter.weight * scorefct(len(voter.approved & set(committee)) + 1)
            else:
                marg[cand] += voter.weight * scorefct(1)
    for cand in committee:
        marg[cand] = -1
    return marg


def marginal_thiele_scores_remove(scorefct, profile, committee):
    marg_util_cand = [0] * profile.num_cand
    #  marginal utility gained by adding candidate to the committee
    for voter in profile:
        for cand in voter.approved:
            satisfaction = len(voter.approved.intersection(committee))
            marg_util_cand[cand] += voter.weight * scorefct(satisfaction)
    for cand in profile.candidates:
        if cand not in committee:
            # do not choose candidates that already have been removed
            marg_util_cand[cand] = max(marg_util_cand) + 1
    return marg_util_cand


def monroescore(profile, committee):
    """Returns Monroe score of a given committee."""
    if len(profile) % len(committee) == 0:
        # faster
        return monroescore_matching(profile, committee)
    else:
        return monroescore_flowbased(profile, committee)


def monroescore_matching(profile, committee):
    """Returns Monroe score of a given committee.
    Uses a matching-based algorithm that works only if
    the committee size divides the number of voters"""
    if len(profile) % len(committee) != 0:
        raise ValueError(
            "monroescore_matching() works only if "
            + "the committee size divides the number of voters "
        )
    graph = {}
    sizeofdistricts = len(profile) // len(committee)
    for cand in committee:
        interestedvoters = []
        for i in range(len(profile)):
            if cand in profile[i].approved:
                interestedvoters.append(i)
        for j in range(sizeofdistricts):
            graph[str(cand) + "/" + str(j)] = interestedvoters
    m, _, _ = matching.bipartiteMatch(graph)
    return len(m)


def monroescore_flowbased(profile, committee):
    """Returns Monroe score of a given committee.
    Uses a flow-based algorithm that works even if
    committeesize does not divide the number of voters"""
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


def mavscore(profile, committee):
    """Returns the Minimax AV (MAV) score of a committee."""
    score = 0
    for voter in profile:
        hamdistance = hamming(voter.approved, committee)
        if hamdistance > score:
            score = hamdistance
    return score
