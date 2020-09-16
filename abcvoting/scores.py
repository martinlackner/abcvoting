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


# returns score function given its name
def get_scorefct(scorefct_str, committeesize):
    if scorefct_str == 'pav':
        return __pav_score_fct
    elif scorefct_str == 'slav':
        return __slav_score_fct
    elif scorefct_str == 'cc':
        return __cc_score_fct
    elif scorefct_str == 'av':
        return __av_score_fct
    elif scorefct_str[:4] == 'geom':
        base = Fraction(scorefct_str[4:])
        return functools.partial(__geom_score_fct, base=base)
    else:
        raise Exception("Score function", scorefct_str, "does not exist.")


def thiele_score(scorefct_str, profile, committee):
    """ computes the Thiele score of a committee subject to
    a given score function (scorefct_str)
    """
    scorefct = get_scorefct(scorefct_str, len(committee))
    score = 0
    for vote in profile:
        cand_in_com = 0
        for _ in set(committee) & vote.approved:
            cand_in_com += 1
            score += vote.weight * scorefct(cand_in_com)
    return score


def __geom_score_fct(i, base):
    if i == 0:
        return 0
    else:
        return Fraction(1, base ** (i - 1))


def __pav_score_fct(i):
    if i == 0:
        return 0
    else:
        return Fraction(1, i)


def __slav_score_fct(i):
    if i == 0:
        return 0
    else:
        return Fraction(1, 2 * i - 1)


def __av_score_fct(i):
    if i >= 1:
        return 1
    else:
        return 0


def __cc_score_fct(i):
    if i == 1:
        return 1
    else:
        return 0


def cumulative_score_fct(scorefct, cand_in_com):
    return sum(scorefct(i + 1) for i in range(cand_in_com))


# returns a list of length num_cand
# the i-th entry contains the marginal score increase
#  gained by adding candidate i
def marginal_thiele_scores_add(scorefct, profile, committee):
    marg = [0] * profile.num_cand
    for pref in profile:
        for c in pref:
            if pref.approved & set(committee):
                marg[c] += pref.weight * scorefct(len(pref.approved &
                                                      set(committee)) + 1)
            else:
                marg[c] += pref.weight * scorefct(1)
    for c in committee:
        marg[c] = -1
    return marg


def marginal_thiele_scores_remove(scorefct, profile, committee):
    marg_util_cand = [0] * profile.num_cand
    #  marginal utility gained by adding candidate to the committee
    for pref in profile:
        for c in pref:
            satisfaction = len(pref.approved.intersection(committee))
            marg_util_cand[c] += pref.weight * scorefct(satisfaction)
    for c in range(profile.num_cand):
        if c not in committee:
            # do not choose candidates that already have been removed
            marg_util_cand[c] = max(marg_util_cand) + 1
    return marg_util_cand


def monroescore(profile, committee):
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
        raise ValueError
    graph = {}
    sizeofdistricts = len(profile) // len(committee)
    for cand in committee:
        interestedvoters = []
        for i in range(len(profile)):
            if cand in profile[i]:
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
    graph.add_node('sink', demand=overflow)
    for i in committee:
        graph.add_node(i, demand=lower_bound)
        graph.add_edge(i, 'sink', weight=0, capacity=1)
    for i, vote in enumerate(profile):
        voter_name = 'v' + str(i)
        graph.add_node(voter_name, demand=-1)
        for cand in committee:
            if cand in vote:
                graph.add_edge(voter_name, cand, weight=0, capacity=1)
            else:
                graph.add_edge(voter_name, cand, weight=1, capacity=1)
    # compute the minimal cost assignment of voters to candidates,
    # i.e. the unrepresented voters, and subtract it from the total number
    # of voters
    return len(profile) - nx.capacity_scaling(graph)[0]


def mavscore(profile, committee):
    score = 0
    for pref in profile:
        hamdistance = hamming(pref, committee)
        if hamdistance > score:
            score = hamdistance
    return score
