# Implementations of many approval-based multi-winner voting rules

# Authors: Martin Lackner, Stefan Forster


import sys
from itertools import combinations
try:
    from gmpy2 import mpq as Fraction
except ImportError:
    from fractions import Fraction
from rules_approval_ilp import compute_monroe_ilp, compute_thiele_methods_ilp
import networkx as nx
from committees import sort_committees,\
                       enough_approved_candidates,\
                       print_committees
import score_functions as sf


########################################################################
# Collection of approval-based multi-winner rules #######################
########################################################################

MWRULES = {
    "av": "Approval Voting",
    "sav": "Satisfaction Approval Voting",
    "pav-ilp": "Proportional Approval Voting (PAV) via ILP",
    "pav-noilp": "Proportional Approval Voting (PAV) via branch-and-bound",
    "seqpav": "Sequential Proportional Approval Voting (seq-PAV)",
    "revseqpav": "Reverse Sequential Prop. Approval Voting (revseq-PAV)",
    "phrag": "Phragmen's sequential rule (seq-Phragmen)",
    "monroe-ilp": "Monroe's rule via ILP",
    "monroe-noilp": "Monroe's rule via flow algorithm",
    "cc-ilp": "Chamberlin-Courant (CC) via ILP",
    "cc-noilp": "Chamberlin-Courant (CC) via branch-and-bound",
    "seqcc": "Sequential Chamberlin-Courant (seq-CC)",
    "revseqcc": "Reverse Sequential Chamberlin-Courant (revseq-CC)",
    "mav-noilp": "Maximin Approval Voting via brute-force",
}


def compute_rule(name, profile, committeesize, resolute=False):
    """Returns the list of winning committees according to the named rule"""
    if name == "seqpav":
        return compute_seqpav(profile, committeesize, resolute=resolute)
    elif name == "revseqpav":
        return compute_revseqpav(profile, committeesize, resolute=resolute)
    elif name == "av":
        return compute_av(profile, committeesize, resolute=resolute)
    elif name == "sav":
        return compute_sav(profile, committeesize, resolute=resolute)
    elif name == "pav-ilp":
        return compute_pav(profile, committeesize,
                           ilp=True, resolute=resolute)
    elif name == "pav-noilp":
        return compute_pav(profile, committeesize,
                           ilp=False, resolute=resolute)
    elif name == "phrag":
        return compute_seqphragmen(profile, committeesize, resolute=resolute)
    elif name == "monroe-ilp":
        return compute_monroe(profile, committeesize,
                              ilp=True, resolute=resolute)
    elif name == "monroe-noilp":
        return compute_monroe(profile, committeesize,
                              ilp=False, resolute=resolute)
    elif name == "cc-ilp":
        return compute_cc(profile, committeesize,
                          ilp=True, resolute=resolute)
    elif name == "cc-noilp":
        return compute_cc(profile, committeesize,
                          ilp=False, resolute=resolute)
    if name == "seqcc":
        return compute_seqcc(profile, committeesize, resolute=resolute)
    elif name == "revseqcc":
        return compute_revseqcc(profile, committeesize, resolute=resolute)
    elif name == "mav-noilp":
        return compute_mav(profile, committeesize,
                           ilp=False, resolute=resolute)
    else:
        raise NotImplementedError("voting method " + str(name) + " not known")


def allrules(profile, committeesize, ilp=True, include_resolute=False):
    """Prints the winning committees for all implemented rules"""
    for rule in MWRULES.keys():
        if not ilp and "-ilp" in rule:
            continue
        print MWRULES[rule] + ":"
        com = compute_rule(rule, profile, committeesize)
        print_committees(com)

        if include_resolute:
            print MWRULES[rule] + " (with tie-breaking):"
            com = compute_rule(rule, profile, committeesize, resolute=True)
            print_committees(com)


########################################################################


# computes arbitrary Thiele methods via branch-and-bound
def compute_thiele_methods_branchandbound(profile, committeesize,
                                          scorefct_str, resolute=False):
    enough_approved_candidates(profile, committeesize)
    scorefct = sf.get_scorefct(scorefct_str, committeesize)

    best_committees = []
    init_com = compute_seq_thiele_resolute(profile, committeesize,
                                           scorefct_str)
    best_score = sf.thiele_score(profile, init_com[0], scorefct_str)
    part_coms = [[]]
    while part_coms:
        part_com = part_coms.pop(0)
        # potential committee, check if at least as good
        # as previous best committee
        if len(part_com) == committeesize:
            score = sf.thiele_score(profile, part_com, scorefct_str)
            if score == best_score:
                best_committees.append(part_com)
            elif score > best_score:
                best_committees = [part_com]
                best_score = score
        else:
            if len(part_com) > 0:
                largest_cand = part_com[-1]
            else:
                largest_cand = -1
            missing = committeesize - len(part_com)
            marg_util_cand = sf.additional_thiele_scores(profile, part_com,
                                                         scorefct)
            upper_bound = (
                sum(sorted(marg_util_cand[largest_cand + 1:])[-missing:])
                + sf.thiele_score(profile, part_com, scorefct_str))
            if upper_bound >= best_score:
                for c in range(largest_cand + 1,
                               profile.num_cand - missing + 1):
                    part_coms.insert(0, part_com + [c])

    committees = sort_committees(best_committees)
    if resolute:
        return [committees[0]]
    else:
        return committees


# Sequential PAV
def compute_seqpav(profile, committeesize, resolute=False):
    """Returns the list of winning committees according sequential PAV"""
    if resolute:
        return compute_seq_thiele_resolute(profile, committeesize, 'pav')
    else:
        return compute_seq_thiele_methods(profile, committeesize, 'pav')


# Reverse Sequential PAV
def compute_revseqpav(profile, committeesize, resolute=False):
    if resolute:
        return compute_revseq_thiele_methods_resolute(profile,
                                                      committeesize, 'pav')
    else:
        return compute_revseq_thiele_methods(profile, committeesize, 'pav')


# Sequential Chamberlin-Courant
def compute_seqcc(profile, committeesize, resolute=False):
    """Returns the list of winning committees according to sequential CC"""
    if resolute:
        return compute_seq_thiele_resolute(profile, committeesize, 'cc')
    else:
        return compute_seq_thiele_methods(profile, committeesize, 'cc')


# Reverse Sequential Chamberlin-Courant
def compute_revseqcc(profile, committeesize, resolute=False):
    if resolute:
        return compute_revseq_thiele_methods_resolute(profile, committeesize,
                                                      'cc')
    else:
        return compute_revseq_thiele_methods(profile, committeesize, 'cc')


# Satisfaction Approval Voting (SAV)
def compute_sav(profile, committeesize, resolute=False):
    return compute_av(profile, committeesize, resolute, sav=True)


# Approval Voting (AV)
def compute_av(profile, committeesize, resolute=False, sav=False):
    """Returns the list of winning committees according to Approval Voting"""
    enough_approved_candidates(profile, committeesize)

    appr_scores = [0] * profile.num_cand
    for pref in profile.preferences:
        for cand in pref.approved:
            if sav:
                # Satisfaction Approval Voting
                appr_scores[cand] += Fraction(pref.weight, len(pref.approved))
            else:
                # (Classic) Approval Voting
                appr_scores[cand] += pref.weight

    # smallest score to be in the committee
    cutoff = sorted(appr_scores)[-committeesize]

    certain_cand = [c for c in range(profile.num_cand)
                    if appr_scores[c] > cutoff]
    possible_cand = [c for c in range(profile.num_cand)
                     if appr_scores[c] == cutoff]
    missing = committeesize - len(certain_cand)
    if resolute:
        return sort_committees([(certain_cand + possible_cand[:missing])])
    else:
        return sort_committees([(certain_cand + list(selection))
                                for selection
                                in combinations(possible_cand, missing)])


# Sequential Thiele methods (resolute)
def compute_seq_thiele_methods(profile, committeesize, scorefct_str):
    enough_approved_candidates(profile, committeesize)
    scorefct = sf.get_scorefct(scorefct_str, committeesize)

    comm_scores = {(): 0}

    # build committees starting with the empty set
    for _ in range(0, committeesize):
        comm_scores_next = {}
        for committee, score in comm_scores.iteritems():
            # marginal utility gained by adding candidate to the committee
            additional_score_cand = sf.additional_thiele_scores(
                                        profile, committee, scorefct)

            for c in range(profile.num_cand):
                if additional_score_cand[c] >= max(additional_score_cand):
                    next_comm = tuple(sorted(committee + (c,)))
                    comm_scores_next[next_comm] = (comm_scores[committee]
                                                   + additional_score_cand[c])
        # remove suboptimal committees
        comm_scores = {}
        cutoff = max(comm_scores_next.values())
        for com, score in comm_scores_next.iteritems():
            if score >= cutoff:
                comm_scores[com] = score
    return sort_committees(comm_scores.keys())


# Sequential Thiele methods with resolute
def compute_seq_thiele_resolute(profile, committeesize, scorefct_str):
    enough_approved_candidates(profile, committeesize)
    scorefct = sf.get_scorefct(scorefct_str, committeesize)

    committee = []

    # build committees starting with the empty set
    for _ in range(0, committeesize):
        additional_score_cand = sf.additional_thiele_scores(
                                    profile, committee, scorefct)
        next_cand = additional_score_cand.index(max(additional_score_cand))
        committee.append(next_cand)
    return [sorted(committee)]


# required for computing Reverse Sequential Thiele methods
def __least_relevant_cands(profile, comm, utilityfct):
    # marginal utility gained by adding candidate to the committee
    marg_util_cand = [0] * profile.num_cand

    for pref in profile.preferences:
        for c in pref.approved:
            satisfaction = len(pref.approved.intersection(comm))
            marg_util_cand[c] += pref.weight * utilityfct(satisfaction)
    for c in range(profile.num_cand):
        if c not in comm:
            # do not choose candidates that already have been removed
            marg_util_cand[c] = max(marg_util_cand) + 1
    # find smallest elements in marg_util_cand and return indices
    return ([cand for cand in range(profile.num_cand)
             if marg_util_cand[cand] == min(marg_util_cand)],
            min(marg_util_cand))


# Reverse Sequential Thiele methods without resolute
def compute_revseq_thiele_methods(profile, committeesize, scorefct_str):
    enough_approved_candidates(profile, committeesize)
    scorefct = sf.get_scorefct(scorefct_str, committeesize)

    allcandcomm = tuple(range(profile.num_cand))
    comm_scores = {allcandcomm: sf.thiele_score(profile, allcandcomm,
                                                scorefct_str)}

    for _ in range(0, profile.num_cand - committeesize):
        comm_scores_next = {}
        for committee, score in comm_scores.iteritems():
            cands_to_remove, score_reduction = \
                __least_relevant_cands(profile, committee, scorefct)
            for c in cands_to_remove:
                next_comm = tuple(set(committee) - set([c]))
                comm_scores_next[next_comm] = score - score_reduction
        # remove suboptimal committees
        comm_scores = {}
        cutoff = max(comm_scores_next.values())
        for com, score in comm_scores_next.iteritems():
            if score >= cutoff:
                comm_scores[com] = score
    return sort_committees(comm_scores.keys())


# Reverse Sequential Thiele methods with resolute
def compute_revseq_thiele_methods_resolute(profile, committeesize,
                                           scorefct_str):
    enough_approved_candidates(profile, committeesize)
    scorefct = sf.get_scorefct(scorefct_str, committeesize)

    committee = set(range(profile.num_cand))

    for _ in range(0, profile.num_cand - committeesize):
        cands_to_remove, _ = __least_relevant_cands(profile, committee,
                                                    scorefct)
        committee.remove(cands_to_remove[0])
    return [sorted(list(committee))]


# Phragmen's Sequential Rule
def compute_seqphragmen(profile, committeesize, resolute=False):
    """Returns the list of winning committees
    according to sequential Phragmen"""
    enough_approved_candidates(profile, committeesize)

    load = {v: 0 for v in profile.preferences}
    com_loads = {(): load}

    approvers_weight = {}
    for c in range(profile.num_cand):
        approvers_weight[c] = sum(v.weight
                                  for v in profile.preferences
                                  if c in v.approved)

    # build committees starting with the empty set
    for _ in range(0, committeesize):
        com_loads_next = {}
        for committee, load in com_loads.iteritems():
            approvers_load = {}
            for c in range(profile.num_cand):
                approvers_load[c] = sum(v.weight * load[v]
                                        for v in profile.preferences
                                        if c in v.approved)
            new_maxload = [Fraction(approvers_load[c] + 1, approvers_weight[c])
                           if approvers_weight[c] > 0 else committeesize + 1
                           for c in range(profile.num_cand)]
            for c in range(profile.num_cand):
                if c in committee:
                    new_maxload[c] = sys.maxint
            for c in range(profile.num_cand):
                if new_maxload[c] <= min(new_maxload):
                    new_load = {}
                    for v in profile.preferences:
                        if c in v.approved:
                            new_load[v] = new_maxload[c]
                        else:
                            new_load[v] = load[v]
                    com_loads_next[tuple(sorted(committee + (c,)))] = new_load
        # remove suboptimal committees
        com_loads = {}
        cutoff = min([max(load) for load in com_loads_next.values()])
        for com, load in com_loads_next.iteritems():
            if max(load) <= cutoff:
                com_loads[com] = load

    committees = sort_committees(com_loads.keys())
    if resolute:
        return [committees[0]]
    else:
        return committees


# Maximin Approval Voting
def compute_mav(profile, committeesize, ilp=False, resolute=False):
    """Returns the list of winning committees according to Maximin AV"""

    if ilp:
        raise NotImplementedError("MAV is not implemented as an ILP.")

    def hamming(a, b, elements):
        diffs = 0
        for x in elements:
            if (x in a and x not in b) or (x in b and x not in a):
                diffs += 1
        return diffs

    def mavscore(committee, profile):
        score = 0
        for vote in profile.preferences:
            hamdistance = hamming(vote.approved, committee,
                                  range(profile.num_cand))
            if hamdistance > score:
                score = hamdistance
        return score

    enough_approved_candidates(profile, committeesize)

    opt_committees = []
    opt_mavscore = profile.num_cand + 1
    for comm in combinations(range(profile.num_cand), committeesize):
        score = mavscore(comm, profile)
        if score < opt_mavscore:
            opt_committees = [comm]
            opt_mavscore = score
        elif mavscore(comm, profile) == opt_mavscore:
            opt_committees.append(comm)

    opt_committees = sort_committees(opt_committees)
    if resolute:
        return [opt_committees[0]]
    else:
        return sort_committees(opt_committees)


# Proportional Approval Voting
def compute_pav(profile, committeesize, ilp=True, resolute=False):
    """Returns the list of winning committees according to Proportional AV"""
    if ilp:
        return compute_thiele_methods_ilp(profile, committeesize,
                                          'pav', resolute)
    else:
        return compute_thiele_methods_branchandbound(profile, committeesize,
                                                     'pav', resolute)


# Chamberlin-Courant
def compute_cc(profile, committeesize, ilp=True, resolute=False):
    """Returns the list of winning committees
    according to Chamblerlin-Courant"""
    if ilp:
        return compute_thiele_methods_ilp(profile, committeesize,
                                          'cc', resolute)
    else:
        return compute_thiele_methods_branchandbound(profile, committeesize,
                                                     'cc', resolute)


# Monroe's rule
def compute_monroe(profile, committeesize, ilp=True, resolute=False):
    """Returns the list of winning committees according to Monroe's rule"""
    if ilp:
        return compute_monroe_ilp(profile, committeesize, resolute)
    else:
        return compute_monroe_bruteforce(profile, committeesize, resolute)


def __monroescore(profile, committee, comm_size):
    """Returns Monroe score of a given committee"""
    G = nx.DiGraph()
    voters = profile.preferences
    # the lower bound of the size of districts
    lower_bound = len(profile.preferences) // comm_size
    # number of voters that will be contribute to the excess
    # of the lower bounds of districts
    overflow = len(voters) - comm_size * lower_bound
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


# Monroe's rule, computed via (brute-force) matching
def compute_monroe_bruteforce(profile, committeesize, resolute=False):
    """Returns the list of winning committees via brute-force Monroe's rule"""
    enough_approved_candidates(profile, committeesize)

    if not profile.has_unit_weights():
        raise Exception("Monroe is only defined for unit weights (weight=1)")
    opt_committees = []
    opt_monroescore = -1
    for comm in combinations(range(profile.num_cand), committeesize):
        score = __monroescore(profile, comm, committeesize)
        if score > opt_monroescore:
            opt_committees = [comm]
            opt_monroescore = score
        elif __monroescore(profile, comm, committeesize) == opt_monroescore:
            opt_committees.append(comm)

    opt_committees = sort_committees(opt_committees)
    if resolute:
        return [opt_committees[0]]
    else:
        return opt_committees
