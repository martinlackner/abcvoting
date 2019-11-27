# Calculating scores


try:
    from gmpy2 import mpq as Fraction
except ImportError:
    from fractions import Fraction
import functools
from bipartite_matching import matching
import networkx as nx


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
    elif scorefct_str.startswith('generalizedcc'):
        param = Fraction(scorefct_str[13:])
        return functools.partial(__generalizedcc_score_fct, ell=param,
                                 committeesize=committeesize)
    elif scorefct_str.startswith('lp-av'):
        param = Fraction(scorefct_str[5:])
        return functools.partial(__lp_av_score_fct, ell=param)
    else:
        raise Exception("Scoring function", scorefct_str, "does not exist.")


# computes the Thiele score of a committee subject to
# a given score function (scorefct_str)
def thiele_score(profile, committee, scorefct_str="pav"):
    scorefct = get_scorefct(scorefct_str, len(committee))
    score = 0
    for pref in profile.preferences:
        cand_in_com = 0
        for _ in set(committee) & pref.approved:
            cand_in_com += 1
            score += pref.weight * scorefct(cand_in_com)
    return score


def __generalizedcc_score_fct(i, ell, committeesize):
    # corresponds to (1,1,1,..,1,0,..0) of length *committeesize*
    #  with *ell* zeros
    # e.g.:
    # ell = committeesize - 1 ... Chamberlin-Courant
    # ell = 0 ... Approval Voting
    if i > committeesize - ell:
        return 0
    if i > 0:
        return 1
    else:
        return 0


def __lp_av_score_fct(i, ell):
    # l-th root of i
    # l=1 ... Approval Voting
    # l=\infty ... Chamberlin-Courant
    if i == 1:
        return 1
    else:
        return i ** Fraction(1, ell) - (i - 1) ** Fraction(1, ell)


def __geom_score_fct(i, base):
    if i == 0:
        return 0
    else:
        return Fraction(1, base**i)


def __pav_score_fct(i):
    if i == 0:
        return 0
    else:
        return Fraction(1, i)


def __slav_score_fct(i):
    if i == 0:
        return 0
    else:
        return Fraction(1, 2*i - 1)


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
def additional_thiele_scores(profile, committee, scorefct):
    marg = [0] * profile.num_cand
    for pref in profile.preferences:
        for c in pref.approved:
            if pref.approved & set(committee):
                marg[c] += pref.weight * scorefct(len(pref.approved &
                                                      set(committee)) + 1)
            else:
                marg[c] += pref.weight * scorefct(1)
    for c in committee:
        marg[c] = -1
    return marg


def monroescore_matching(profile, committee):
    """Returns Monroe score of a given committee.
    Uses a matching-based algorithm that works only if
    committeesize divides the number of voters"""
    graph = {}
    sizeofdistricts = len(profile.preferences) / len(committee)
    for cand in committee:
        interestedvoters = []
        for i in range(len(profile.preferences)):
            if cand in profile.preferences[i].approved:
                interestedvoters.append(i)
        for j in range(sizeofdistricts):
            graph[str(cand) + "/" + str(j)] = interestedvoters
    m, _, _ = matching.bipartiteMatch(graph)
    return len(m)


def monroescore_flowbased(profile, committee, committeesize=0):
    """Returns Monroe score of a given committee.
    Uses a flow-based algorithm that works even if
    committeesize does not divide the number of voters"""
    G = nx.DiGraph()
    voters = profile.preferences
    if committeesize == 0:
        committeesize = len(committee)
    # the lower bound of the size of districts
    lower_bound = len(profile.preferences) // committeesize
    # number of voters that will be contribute to the excess
    # of the lower bounds of districts
    overflow = len(voters) - committeesize * lower_bound
    # add a sink node for the overflow
    G.add_node('sink', demand=overflow)
    for i in committee:
        G.add_node(i, demand=lower_bound)
        G.add_edge(i, 'sink', weight=0, capacity=1)
    for i in range(len(voters)):
        voter_name = 'v' + str(i)
        G.add_node(voter_name, demand=-1)
        for cand in committee:
            if cand in voters[i].approved:
                G.add_edge(voter_name, cand, weight=0, capacity=1)
            else:
                G.add_edge(voter_name, cand, weight=1, capacity=1)
    # compute the minimal cost assignment of voters to candidates,
    # i.e. the unrepresented voters, and subtract it from the total number
    # of voters
    return len(voters) - nx.capacity_scaling(G)[0]
