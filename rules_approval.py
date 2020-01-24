from __future__ import print_function
# Implementations of approval-based multi-winner voting rules


import sys
from itertools import combinations
try:
    from gmpy2 import mpq as Fraction
except ImportError:
    from fractions import Fraction
from rules_approval_ilp import compute_monroe_ilp, compute_thiele_methods_ilp,\
                               compute_optphragmen_ilp, compute_minimaxav_ilp
from committees import sort_committees,\
                       enough_approved_candidates,\
                       print_committees
import score_functions as sf


########################################################################


MWRULES = {
    "av": "Approval Voting",
    "sav": "Satisfaction Approval Voting",
    "pav-ilp": "Proportional Approval Voting (PAV) via ILP",
    "pav-noilp": "Proportional Approval Voting (PAV) via branch-and-bound",
    "seqpav": "Sequential Proportional Approval Voting (seq-PAV)",
    "revseqpav": "Reverse Sequential Prop. Approval Voting (revseq-PAV)",
    "slav-ilp": "Sainte-Lague Approval Voting (SLAV) via ILP",
    "slav-noilp": "Sainte-Lague Approval Voting (SLAV) via branch-and-bound",
    "seqslav": "Sequential Sainte-Lague Approval Voting (seq-SLAV)",
    "phrag": "Phragmen's sequential rule (seq-Phragmen)",
    "optphrag": "Phragmen's optimization rule (opt-Phragmen)",
    "monroe-ilp": "Monroe's rule via ILP",
    "monroe-noilp": "Monroe's rule via flow algorithm",
    "gmmonroe": "Greedy Marginal Monroe's rule",
    "cc-ilp": "Chamberlin-Courant (CC) via ILP",
    "cc-noilp": "Chamberlin-Courant (CC) via branch-and-bound",
    "seqcc": "Sequential Chamberlin-Courant (seq-CC)",
    "revseqcc": "Reverse Sequential Chamberlin-Courant (revseq-CC)",
    "minimaxav-noilp": "Minimax Approval Voting via brute-force",
    "minimaxav-ilp": "Minimax Approval Voting via ILP",
    "rule_x": "Rule X",
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
    elif name == "seqslav":
        return compute_seqslav(profile, committeesize, resolute=resolute)
    elif name == "slav-ilp":
        return compute_slav(profile, committeesize,
                            ilp=True, resolute=resolute)
    elif name == "slav-noilp":
        return compute_slav(profile, committeesize,
                            ilp=False, resolute=resolute)
    elif name == "phrag":
        return compute_seqphragmen(profile, committeesize, resolute=resolute)
    elif name == "monroe-ilp":
        return compute_monroe(profile, committeesize,
                              ilp=True, resolute=resolute)
    elif name == "monroe-noilp":
        return compute_monroe(profile, committeesize,
                              ilp=False, resolute=resolute)
    elif name == "gmmonroe":
        return compute_gmmonroe(profile, committeesize, resolute=resolute)
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
    elif name == "minimaxav-noilp":
        return compute_minimaxav(profile, committeesize,
                                 ilp=False, resolute=resolute)
    elif name == "minimaxav-ilp":
        return compute_minimaxav(profile, committeesize,
                                 ilp=True, resolute=resolute)
    elif name == "optphrag":
        return compute_optphragmen_ilp(profile, committeesize,
                                       resolute=resolute)
    elif name == "rule_x":
        return compute_rule_x(profile, committeesize, resolute=resolute)
    else:
        raise NotImplementedError("voting method " + str(name) + " not known")


def allrules(profile, committeesize, ilp=True, include_resolute=False):
    """Prints the winning committees for all implemented rules"""
    for rule in list(MWRULES.keys()):
        if not ilp and "-ilp" in rule:
            continue
        print(MWRULES[rule] + ":")
        com = compute_rule(rule, profile, committeesize)
        print_committees(com)

        if include_resolute:
            print(MWRULES[rule] + " (with tie-breaking):")
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


# Sequential SLAV
def compute_seqslav(profile, committeesize, resolute=False):
    """Returns the list of winning committees according sequential SLAV"""
    if resolute:
        return compute_seq_thiele_resolute(profile, committeesize, 'slav')
    else:
        return compute_seq_thiele_methods(profile, committeesize, 'slav')


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
        for committee, score in comm_scores.items():
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
        for com, score in comm_scores_next.items():
            if score >= cutoff:
                comm_scores[com] = score
    return sort_committees(list(comm_scores.keys()))


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
        for committee, score in comm_scores.items():
            cands_to_remove, score_reduction = \
                __least_relevant_cands(profile, committee, scorefct)
            for c in cands_to_remove:
                next_comm = tuple(set(committee) - set([c]))
                comm_scores_next[next_comm] = score - score_reduction
        # remove suboptimal committees
        comm_scores = {}
        cutoff = max(comm_scores_next.values())
        for com, score in comm_scores_next.items():
            if score >= cutoff:
                comm_scores[com] = score
    return sort_committees(list(comm_scores.keys()))


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
    comm_loads = {(): load}

    approvers_weight = {}
    for c in range(profile.num_cand):
        approvers_weight[c] = sum(v.weight
                                  for v in profile.preferences
                                  if c in v.approved)

    # build committees starting with the empty set
    for _ in range(0, committeesize):
        comm_loads_next = {}
        for committee, load in comm_loads.items():
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
                    new_maxload[c] = sys.maxsize
            for c in range(profile.num_cand):
                if new_maxload[c] <= min(new_maxload):
                    new_load = {}
                    for v in profile.preferences:
                        if c in v.approved:
                            new_load[v] = new_maxload[c]
                        else:
                            new_load[v] = load[v]
                    comm_loads_next[tuple(sorted(committee + (c,)))] = new_load
        # remove suboptimal committees
        comm_loads = {}
        cutoff = min([max(load.values()) for load in comm_loads_next.values()])
        for com, load in comm_loads_next.items():
            if max(load.values()) <= cutoff:
                comm_loads[com] = load
        if resolute:
            committees = sort_committees(list(comm_loads.keys()))
            comm = tuple(committees[0])
            comm_loads = {comm: comm_loads[comm]}

    committees = sort_committees(list(comm_loads.keys()))
    if resolute:
        return [committees[0]]
    else:
        return committees


# Minimax Approval Voting
def compute_minimaxav(profile, committeesize, ilp=True, resolute=False):
    """Returns the list of winning committees according to Minimax AV"""

    if ilp:
        return compute_minimaxav_ilp(profile, committeesize, resolute)

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
                                  list(range(profile.num_cand)))
            if hamdistance > score:
                score = hamdistance
        return score

    enough_approved_candidates(profile, committeesize)

    opt_committees = []
    opt_mavscore = profile.num_cand + 1
    for comm in combinations(list(range(profile.num_cand)), committeesize):
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


# Sainte-Lague Approval Voting
def compute_slav(profile, committeesize, ilp=True, resolute=False):
    """Returns the list of winning committees according to Proportional AV"""
    if ilp:
        return compute_thiele_methods_ilp(profile, committeesize,
                                          'slav', resolute)
    else:
        return compute_thiele_methods_branchandbound(profile, committeesize,
                                                     'slav', resolute)


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


# Monroe's rule, computed via (brute-force) matching
def compute_monroe_bruteforce(profile, committeesize,
                              resolute=False, flowbased=True):
    """Returns the list of winning committees via brute-force Monroe's rule"""
    enough_approved_candidates(profile, committeesize)

    if not profile.has_unit_weights():
        raise Exception("Monroe is only defined for unit weights (weight=1)")

    if profile.totalweight() % committeesize != 0 or flowbased:
        monroescore = sf.monroescore_flowbased
    else:
        monroescore = sf.monroescore_matching

    opt_committees = []
    opt_monroescore = -1
    for comm in combinations(list(range(profile.num_cand)), committeesize):
        score = monroescore(profile, comm)
        if score > opt_monroescore:
            opt_committees = [comm]
            opt_monroescore = score
        elif monroescore(profile, comm) == opt_monroescore:
            opt_committees.append(comm)

    opt_committees = sort_committees(opt_committees)
    if resolute:
        return [opt_committees[0]]
    else:
        return opt_committees


def compute_gmmonroe(profile, committeesize, resolute=False):
    enough_approved_candidates(profile, committeesize)
    if not profile.has_unit_weights():
        raise Exception("Greedy Marginal Monroe is only defined \
                        for unit weights (weight=1)")
    if profile.totalweight() % committeesize != 0:
        raise Exception("Greedy Marginal Monroe is only defined \
                        when number of voters is divisible by committee size")

    # start with an empty committee
    opt_committees = [set()]

    for i in range(committeesize):
        possible_extensions = []
        for comm in opt_committees:
            opt_monroescore = -1
            possible_candidates = []
            # extend the already found committees in a greedy way
            for element in set(range(profile.num_cand)) - comm:
                score = sf.monroescore_flowbased(profile, comm | {element},
                                                 committeesize)
                if score >= opt_monroescore:
                    possible_candidates.append(element)
                if score > opt_monroescore:
                    possible_candidates = [element]
                    opt_monroescore = score
            # extemd the committee by their respective extension elements
            for candidate in possible_candidates:
                new_comm = comm | {candidate}
                possible_extensions.append(new_comm)
        # replace the committees of the smaller size by their extensions
        opt_committees = [set(c) for c in set(map(tuple, possible_extensions))]

    opt_committees = sort_committees(opt_committees)
    if resolute:
        return [opt_committees[0]]
    else:
        return opt_committees


def compute_rule_x(profile, committeesize, resolute=False):
    """Returns the list of winning candidates according to rule x.
    But rule x does stop if not enough budget is there to finance a
    candidate. As this is not optimal the committee is filled with the
    candidates that have the most remaining budget as support."""
    enough_approved_candidates(profile, committeesize)
    if not profile.has_unit_weights():
        raise Exception("Rule X is only defined \
                            for unit weights (weight=1)")
    num_voters = len(profile.preferences)
    price = Fraction(num_voters, committeesize)


    start_budget = {v: Fraction(1, 1) for v in range(num_voters)}
    cands = range(profile.num_cand)
    committees = [(start_budget, set())]
    final_committees = []

    for i in range(committeesize):
        next_committees = []
        for committee in committees:
            budget = committee[0]
            q_affordability = {}
            curr_cands = set(cands) - committee[1]
            for c in curr_cands:
                approved_by = set()
                for v, vote in enumerate(profile.preferences):
                    if c in vote.approved and budget[v] > 0.0:
                        approved_by.add(v)
                too_poor = set()
                already_available = Fraction(0)
                rich = set(approved_by)
                q = 0.0
                while already_available < price and q == 0.0 and len(rich) > 0:
                    fair_split = Fraction(price-already_available, len(rich))
                    still_rich = set()
                    for v in rich:
                        if budget[v] <= fair_split:
                            too_poor.add(v)
                            already_available += budget[v]
                        else:
                            still_rich.add(v)
                    if len(still_rich) == len(rich):
                        q = fair_split
                        q_affordability[c] = q
                    elif already_available == price:
                        q = fair_split
                        q_affordability[c] = q
                    else:
                        rich = still_rich

            if len(q_affordability) > 0:
                min_q = min(q_affordability.values())
                cheapest_split = [c for c in q_affordability
                                  if q_affordability[c] == min_q]

                for c in cheapest_split:
                    b = dict(committee[0])
                    for v, vote in enumerate(profile.preferences):
                        if c in vote.approved:
                            b[v] -= min(budget[v], min_q)
                    comm = set(committee[1])
                    comm.add(c)
                    next_committees.append((b, comm))

            else:  # no affordable candidate remains
                comms = fill_remaining_committee(committee, curr_cands,
                                                 committeesize, profile)
                for b, comm in comms:
                    final_committees.append(comm)

        committees = next_committees
    for comm in committees:
        # remove duplicates
        if comm[1] not in final_committees:
            final_committees.append(comm[1])

    committees = sort_committees(final_committees)
    if resolute:
        if len(committees) > 0:
            return [committees[0]]
        else:
            return []
    else:
        return committees


# Rule X has no definition of how to fill remaining committee spots
# this function takes the candidates with the most remaining budget
# selecting one candidate depletes all budgets of the voters that
# approve that candidate.
# This can produce multiple possible committees.
def fill_remaining_committee(committee, curr_cands, committee_size,
                             profile):
    missing = committee_size - len(committee[1])
    committees = [committee]
    for _ in range(missing):
        next_comms = []
        for comm in committees:
            budget, appr_set = comm
            remaining_cands = curr_cands - appr_set
            budget_support = {}
            for cand in remaining_cands:
                budget_support[cand] = 0
                for v, vote in enumerate(profile.preferences):
                    if cand in vote.approved:
                        budget_support[cand] += budget[v]

            max_support = max(budget_support.values())
            winners = [c for c in remaining_cands
                       if budget_support[c] == max_support]
            for c in winners:
                budget_c = {}
                for voter, value in budget.items():
                    if c in profile.preferences[voter].approved:
                        budget_c[voter] = 0
                    else:
                        budget_c[voter] = value
                next_comms.append((budget_c, appr_set.union([c])))

        committees = next_comms

    return committees




