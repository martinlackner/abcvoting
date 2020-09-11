# -*- coding: utf-8 -*-
"""Approval-based committee (ABC) voting rules"""


from __future__ import print_function
import sys
import functools
from itertools import combinations
try:
    from gmpy2 import mpq as Fraction
except ImportError:
    print("Warning: module gmpy2 not found, "
          + "resorting to Python's fractions.Fraction")
    from fractions import Fraction
from abcvoting import abcrules_gurobi
from abcvoting.misc import sort_committees
from abcvoting.misc import hamming
from abcvoting.misc import enough_approved_candidates
from abcvoting.misc import str_committees_header
from abcvoting.misc import str_candset, str_candsets
from abcvoting.misc import header
from abcvoting import scores


########################################################################


class UnknownRuleIDError(ValueError):
    """Exception raised if unknown rule id is used"""

    def __init__(self, rule_id):
        message = "Rule ID \"" + str(rule_id) + "\" is not known."
        super(ValueError, self).__init__(message)


class ABCRule:
    """Class for ABC rules containing basic information and function call"""
    def __init__(self, rule_id, shortname, longname, fct,
                 algorithms=("standard"), resolute=(True, False)):
        self.rule_id = rule_id
        self.shortname = shortname
        self.longname = longname
        self.fct = fct
        self.algorithms = algorithms
        # algorithms should be sorted by speed (fastest first)
        self.resolute = resolute

        assert len(resolute) > 0
        assert len(algorithms) > 0

    def compute(self, profile, committeesize, **kwargs):
        return self.fct(profile, committeesize, **kwargs)

    def fastest_algo(self):
        for algo in self.algorithms:
            if algo == "gurobi" and not abcrules_gurobi.available:
                continue
            return algo


########################################################################


def compute(rule_id, profile, committeesize, **kwargs):
    try:
        return rules[rule_id].compute(profile, committeesize, **kwargs)
    except KeyError:
        raise UnknownRuleIDError(rule_id)


# computes arbitrary Thiele methods via branch-and-bound
def compute_thiele_method(scorefct_str, profile, committeesize,
                          algorithm="branch-and-bound",
                          resolute=False, verbose=0):
    """Thiele methods

    Compute winning committees of the Thiele method specified
    by the score function (scorefct_str)
    """
    enough_approved_candidates(profile, committeesize)
    scorefct = scores.get_scorefct(scorefct_str, committeesize)

    # optional output
    if verbose:
        print(header(rules[scorefct_str].longname))
        if resolute:
            print("Computing only one winning committee (resolute=True)\n")
    if verbose >= 3:
        if algorithm == "gurobi":
            print("Using the Gurobi ILP solver\n")
        if algorithm == "branch-and-bound":
            print("Using a branch-and-bound algorithm\n")
    # end of optional output

    if algorithm == "gurobi":
        committees = abcrules_gurobi.__gurobi_thiele_methods(
            profile, committeesize, scorefct, resolute)

        committees = sort_committees(committees)
    elif algorithm == "branch-and-bound":
        committees = __thiele_methods_branchandbound(
            profile, committeesize, scorefct_str, resolute)
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm)
            + " not specified for compute_thiele_method")

    # optional output
    if verbose >= 2:
        print("Optimal " + scorefct_str.upper() + "-score: "
              + str(scores.thiele_score(scorefct_str, profile, committees[0])))
        print()
    if verbose:
        print(str_committees_header(committees, winning=True))
        print(str_candsets(committees, names=profile.names))
    # end of optional output

    return committees


# computes arbitrary Thiele methods via branch-and-bound
def __thiele_methods_branchandbound(profile, committeesize,
                                    scorefct_str, resolute):
    """Branch-and-bound algorithm to compute winning committees
    for Thiele methods"""
    enough_approved_candidates(profile, committeesize)
    scorefct = scores.get_scorefct(scorefct_str, committeesize)

    best_committees = []
    init_com = compute_seq_thiele_method(
        profile, committeesize, scorefct_str, resolute=True)[0]
    best_score = scores.thiele_score(scorefct_str, profile, init_com)
    part_coms = [[]]
    while part_coms:
        part_com = part_coms.pop(0)
        # potential committee, check if at least as good
        # as previous best committee
        if len(part_com) == committeesize:
            score = scores.thiele_score(scorefct_str, profile, part_com)
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
            marg_util_cand = scores.marginal_thiele_scores_add(
                scorefct, profile, part_com)
            upper_bound = (
                sum(sorted(marg_util_cand[largest_cand + 1:])[-missing:])
                + scores.thiele_score(scorefct_str, profile, part_com))
            if upper_bound >= best_score:
                for c in range(largest_cand + 1,
                               profile.num_cand - missing + 1):
                    part_coms.insert(0, part_com + [c])

    committees = sort_committees(best_committees)
    if resolute:
        committees = [committees[0]]

    return committees


# Sequential PAV
def compute_seqpav(profile, committeesize, algorithm="standard",
                   resolute=True, verbose=0):
    """Sequential PAV (seq-PAV)"""
    return compute_seq_thiele_method(
        profile, committeesize, 'pav', algorithm=algorithm,
        resolute=resolute, verbose=verbose)


def compute_seqslav(profile, committeesize, algorithm="standard",
                    resolute=True, verbose=0):
    """Sequential Sainte-Lague Approval Voting (SLAV)"""
    return compute_seq_thiele_method(
        profile, committeesize, "slav", algorithm=algorithm,
        resolute=resolute, verbose=verbose)


# Reverse Sequential PAV
def compute_revseqpav(profile, committeesize, algorithm="standard",
                      resolute=True, verbose=0):
    """Reverse sequential PAV (revseq-PAV)"""
    return compute_revseq_thiele_method(
        profile, committeesize, 'pav', algorithm=algorithm,
        resolute=resolute, verbose=verbose)


def compute_seqcc(profile, committeesize, algorithm="standard",
                  resolute=True, verbose=0):
    """Sequential Chamberlin-Courant (seq-CC)"""
    return compute_seq_thiele_method(
        profile, committeesize, 'cc', algorithm=algorithm,
        resolute=resolute, verbose=verbose)


def compute_sav(profile, committeesize, algorithm="standard",
                resolute=False, verbose=0):
    """Satisfaction Approval Voting (SAV)"""
    if algorithm == "standard":
        return __separable("sav", profile, committeesize, resolute, verbose)
    else:
            raise NotImplementedError(
                "Algorithm " + str(algorithm)
                + " not specified for compute_sav")


# Approval Voting (AV)
def compute_av(profile, committeesize, algorithm="standard",
               resolute=False, verbose=0):
    """Approval Voting"""
    if algorithm == "standard":
        return __separable("av", profile, committeesize, resolute, verbose)
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm)
            + " not specified for compute_av")


def __separable(rule_id, profile, committeesize, resolute, verbose):
    enough_approved_candidates(profile, committeesize)

    appr_scores = [0] * profile.num_cand
    for pref in profile:
        for cand in pref:
            if rule_id == "sav":
                # Satisfaction Approval Voting
                appr_scores[cand] += Fraction(pref.weight, len(pref))
            elif rule_id == "av":
                # (Classic) Approval Voting
                appr_scores[cand] += pref.weight
            else:
                raise UnknownRuleIDError(rule_id)

    # smallest score to be in the committee
    cutoff = sorted(appr_scores)[-committeesize]

    certain_cands = [c for c in range(profile.num_cand)
                     if appr_scores[c] > cutoff]
    possible_cands = [c for c in range(profile.num_cand)
                      if appr_scores[c] == cutoff]
    missing = committeesize - len(certain_cands)
    if len(possible_cands) == missing:
        # candidates with appr_scores[c] == cutoff
        # are also certain candidates because all these candidates
        # are required to fill the committee
        certain_cands = sorted(certain_cands + possible_cands)
        possible_cands = []
        missing = 0

    if resolute:
        committees = sort_committees(
            [(certain_cands + possible_cands[:missing])])
    else:
        committees = sort_committees(
            [(certain_cands + list(selection))
             for selection
             in combinations(possible_cands, missing)])

    # optional output
    if verbose:
        print(header(rules[rule_id].longname))
        if resolute:
            print("Computing only one winning committee (resolute=True)\n")
    if verbose >= 2:
        print("Scores of candidates:")
        for c in range(profile.num_cand):
            print(profile.names[c] + ": " + str(appr_scores[c]))

        print("\nCandidates are contained in winning committees")
        print("if their score is >= " + str(cutoff) + ".")

        if len(certain_cands) > 0:
            print("\nThe following candidates are contained in")
            print("every winning committee:")
            namedset = [profile.names[c] for c in certain_cands]
            print(" " + ", ".join(map(str, namedset)))
            print()

        if len(possible_cands) > 0:
            print("The following candidates are contained in")
            print("some of the winning committees:")
            namedset = [profile.names[c] for c in possible_cands]
            print(" " + ", ".join(map(str, namedset)))
            print("(" + str(missing) + " of those candidates is contained\n"
                  + " in every winning committee.)\n")
    if verbose:
        print(str_committees_header(committees, winning=True))
        print(str_candsets(committees, names=profile.names))
    # end of optional output

    return committees


def __seq_thiele_resolute(profile, committeesize, scorefct_str, verbose):
    """Compute a *resolute* reverse sequential Thiele method

    Tiebreaking between candidates in favor of candidate with smaller
    number/index (candidates with larger numbers get deleted first).
    """
    committee = []

    scorefct = scores.get_scorefct(scorefct_str, committeesize)

    # optional output
    if verbose >= 2:
        output = "starting with the empty committee (score = "
        output += str(scores.thiele_score(
            scorefct_str, profile, committee)) + ")"
        print(output + "\n")
    # end of optional output

    # build a committee starting with the empty set
    for _ in range(committeesize):
        additional_score_cand = scores.marginal_thiele_scores_add(
            scorefct, profile, committee)
        next_cand = additional_score_cand.index(max(additional_score_cand))
        committee.append(next_cand)
        # optional output
        if verbose >= 2:
            output = "adding candidate number "
            output += str(len(committee)) + ": "
            output += profile.names[next_cand] + "\n"
            output += " score increases by "
            output += str(max(additional_score_cand))
            output += " to a total of "
            output += str(scores.thiele_score(
                scorefct_str, profile, committee))
            tied_cands = [c for c in range(len(additional_score_cand))
                          if (c > next_cand and
                              (additional_score_cand[c]
                               == max(additional_score_cand)))]
            if len(tied_cands) > 0:
                output += " tie broken in favor of " + str(next_cand)
                output += " candidates " + str_candset(tied_cands)
                output += " would increase the score by the same amount ("
                output += str(max(additional_score_cand)) + ")"
            print(output + "\n")
        # end of optional output
    return [sorted(committee)]


def __seq_thiele_irresolute(profile, committeesize, scorefct_str):
    """Compute an *irresolute* reverse sequential Thiele method

    Consider all possible ways to break ties between candidates
    (aka parallel universe tiebreaking)
    """
    scorefct = scores.get_scorefct(scorefct_str, committeesize)

    comm_scores = {(): 0}
    # build committees starting with the empty set
    for _ in range(committeesize):
        comm_scores_next = {}
        for committee, score in comm_scores.items():
            # marginal utility gained by adding candidate to the committee
            additional_score_cand = scores.marginal_thiele_scores_add(
                scorefct, profile, committee)
            for c in range(profile.num_cand):
                if additional_score_cand[c] >= max(additional_score_cand):
                    next_comm = tuple(sorted(committee + (c,)))
                    comm_scores_next[next_comm] = (
                        score + additional_score_cand[c])
        comm_scores = comm_scores_next
    return sort_committees(list(comm_scores.keys()))


def compute_seq_thiele_method(profile, committeesize, scorefct_str,
                              algorithm="standard", resolute=True, verbose=0):
    """Sequential Thiele methods"""

    enough_approved_candidates(profile, committeesize)

    if algorithm != "standard":
        raise NotImplementedError(
            "Algorithm " + str(algorithm)
            + " not specified for compute_seq_thiele_method")

    # optional output
    if verbose:
        print(header(rules["seq" + scorefct_str].longname))
        if resolute:
            print("Computing only one winning committee (resolute=True)\n")
    # end of optional output

    if resolute:
        committees = __seq_thiele_resolute(
            profile, committeesize, scorefct_str, verbose=verbose)
    else:
        committees = __seq_thiele_irresolute(
            profile, committeesize, scorefct_str)

    # optional output
    if verbose:
        print(str_committees_header(committees, winning=True))
        print(str_candsets(committees, names=profile.names))
        if resolute or len(committees) == 1:
            print(scorefct_str.upper() + "-score of winning committee:",
                  end="")
        else:
            print(scorefct_str.upper() + "-score of winning committees:")
        for comm in committees:
            print(" " + str(scores.thiele_score(scorefct_str, profile, comm)))
        print()
    # end of optional output

    return committees


def __revseq_thiele_irresolute(profile, committeesize, scorefct_str):
    """Compute an *irresolute* sequential Thiele method

    Consider all possible ways to break ties between candidates
    (aka parallel universe tiebreaking)
    """
    scorefct = scores.get_scorefct(scorefct_str, committeesize)

    allcandcomm = tuple(range(profile.num_cand))
    comm_scores = {allcandcomm: scores.thiele_score(
        scorefct_str, profile, allcandcomm)}

    for _ in range(profile.num_cand - committeesize):
        comm_scores_next = {}
        for committee, score in comm_scores.items():
            marg_util_cand = scores.marginal_thiele_scores_remove(
                scorefct, profile, committee)
            score_reduction = min(marg_util_cand)
            # find smallest elements in marg_util_cand and return indices
            cands_to_remove = [cand for cand in range(profile.num_cand)
                               if marg_util_cand[cand] == min(marg_util_cand)]
            for c in cands_to_remove:
                next_comm = tuple(set(committee) - {c})
                comm_scores_next[next_comm] = score - score_reduction
            comm_scores = comm_scores_next
    return sort_committees(list(comm_scores.keys()))


def __revseq_thiele_resolute(profile, committeesize, scorefct_str, verbose):
    """Compute a *resolute* reverse sequential Thiele method

    Tiebreaking between candidates in favor of candidate with smaller
    number/index (candidates with smaller numbers are added first).
    """
    scorefct = scores.get_scorefct(scorefct_str, committeesize)

    committee = set(range(profile.num_cand))

    # optional output
    if verbose >= 2:
        output = "full committee (" + str(len(committee))
        output += " candidates) has a total score of "
        output += str(scores.thiele_score(
            scorefct_str, profile, committee))
        print(output + "\n")
    # end of optional output

    for _ in range(profile.num_cand - committeesize):
        marg_util_cand = scores.marginal_thiele_scores_remove(
            scorefct, profile, committee)
        score_reduction = min(marg_util_cand)
        # find smallest elements in marg_util_cand and return indices
        cands_to_remove = [cand for cand in range(profile.num_cand)
                           if marg_util_cand[cand] == min(marg_util_cand)]
        committee.remove(cands_to_remove[-1])

        # optional output
        if verbose >= 2:
            rem_cand = cands_to_remove[-1]
            output = "removing candidate number "
            output += str(profile.num_cand - len(committee)) + ": "
            output += profile.names[rem_cand] + "\n"
            output += " score decreases by "
            output += str(score_reduction)
            output += " to a total of "
            output += str(scores.thiele_score(
                scorefct_str, profile, committee))
            if len(cands_to_remove) > 1:
                output += " (tie between candidates "
                output += str_candset(cands_to_remove) + ")\n"
            print(output + "\n")
        # end of optional output

    return [sorted(list(committee))]


def compute_revseq_thiele_method(profile, committeesize,
                                 scorefct_str, algorithm="standard",
                                 resolute=True, verbose=0):
    """Reverse sequential Thiele methods"""
    enough_approved_candidates(profile, committeesize)

    if algorithm != "standard":
        raise NotImplementedError(
            "Algorithm " + str(algorithm)
            + " not specified for compute_revseq_thiele_method")

    # optional output
    if verbose:
        print(header(rules["revseq" + scorefct_str].longname))
        if resolute:
            print("Computing only one winning committee (resolute=True)\n")
    # end of optional output

    if resolute:
        committees = __revseq_thiele_resolute(
            profile, committeesize, scorefct_str, verbose=verbose)
    else:
        committees = __revseq_thiele_irresolute(
            profile, committeesize, scorefct_str)

    # optional output
    if verbose:
        print(str_committees_header(committees, winning=True))
        print(str_candsets(committees, names=profile.names))
    if verbose >= 2:
        if resolute or len(committees) == 1:
            print("PAV-score of winning committee:", end="")
        else:
            print("PAV-score of winning committees:")
        for comm in committees:
            print(" " + str(scores.thiele_score(scorefct_str, profile, comm)))
        print()
    # end of optional output

    return committees


def __minimaxav_bruteforce(profile, committeesize):
    """Brute-force algorithm for computing Minimax AV (MAV)"""
    opt_committees = []
    opt_mavscore = profile.num_cand + 1
    for comm in combinations(list(range(profile.num_cand)), committeesize):
        score = scores.mavscore(profile, comm)
        if score < opt_mavscore:
            opt_committees = [comm]
            opt_mavscore = score
        elif scores.mavscore(profile, comm) == opt_mavscore:
            opt_committees.append(comm)

    committees = sort_committees(opt_committees)

    return committees


# Minimax Approval Voting
def compute_mav(profile, committeesize, algorithm="brute-force",
                resolute=False, verbose=0):
    """Minimax AV (MAV)"""
    enough_approved_candidates(profile, committeesize)

    # optional output
    if verbose:
        print(header(rules["mav"].longname))
        if resolute:
            print("Computing only one winning committee (resolute=True)\n")
    if verbose >= 3:
        if algorithm == "gurobi":
            print("Using the Gurobi ILP solver\n")
        if algorithm == "brute-force":
            print("Using a brute-force algorithm\n")
    # end of optional output

    if algorithm == "gurobi":
        committees = abcrules_gurobi.__gurobi_minimaxav(
            profile, committeesize, resolute)
        committees = sort_committees(committees)
    elif algorithm == "brute-force":
        committees = __minimaxav_bruteforce(profile, committeesize)
        if resolute:
            committees = [committees[0]]
    else:
        raise NotImplementedError("Algorithm " + str(algorithm)
                                  + " not specified for compute_mav")

    opt_mavscore = scores.mavscore(profile, committees[0])

    # optional output
    if verbose:
        print("Minimum maximal distance: " + str(opt_mavscore))

        print(str_committees_header(committees, winning=True))
        print(str_candsets(committees, names=profile.names))

        print("Corresponding distances to voters:")
        for comm in committees:
            print([hamming(pref, comm) for pref in profile])
        print()
    # end of optional output

    return committees


# Lexicographic Minimax Approval Voting
def compute_lexmav(profile, committeesize, algorithm="brute-force",
                   resolute=False, verbose=0):
    """Lexicographic Minimax AV"""
    enough_approved_candidates(profile, committeesize)

    if not profile.has_unit_weights():
        raise ValueError(rules["lexmav"].shortname +
                         " is only defined for unit weights (weight=1)")

    if algorithm != "brute-force":
        raise NotImplementedError(
            "Algorithm " + str(algorithm)
            + " not specified for compute_lexmav")

    opt_committees = []
    opt_distances = [profile.num_cand + 1] * len(profile)
    for comm in combinations(list(range(profile.num_cand)), committeesize):
        distances = sorted([hamming(pref, comm)
                            for pref in profile],
                           reverse=True)
        for i in range(len(distances)):
            if opt_distances[i] < distances[i]:
                break
            if opt_distances[i] > distances[i]:
                opt_distances = distances
                opt_committees = [comm]
                break
        else:
            opt_committees.append(comm)

    committees = sort_committees(opt_committees)
    if resolute:
        committees = [committees[0]]

    # optional output
    if verbose:
        print(header(rules["lexmav"].longname))
        if resolute:
            print("Computing only one winning committee (resolute=True)\n")

        print("Minimum maximal distance: " + str(max(opt_distances)))

        print(str_committees_header(committees, winning=True))
        print(str_candsets(committees, names=profile.names))

        print("Corresponding distances to voters:")
        for comm in committees:
            print([hamming(pref, comm) for pref in profile])
        print()
    # end of optional output

    return committees


# Proportional Approval Voting
def compute_pav(profile, committeesize, algorithm="branch-and-bound",
                resolute=False, verbose=0):
    """Proportional Approval Voting (PAV)"""
    return compute_thiele_method(
        'pav', profile, committeesize, algorithm=algorithm,
        resolute=resolute, verbose=verbose)


# Sainte-Lague Approval Voting
def compute_slav(profile, committeesize, algorithm="branch-and-bound",
                 resolute=False, verbose=0):
    """Sainte-Lague Approval Voting (SLAV)"""
    return compute_thiele_method(
        'slav', profile, committeesize, algorithm=algorithm,
        resolute=resolute, verbose=verbose)


# Chamberlin-Courant
def compute_cc(profile, committeesize, algorithm="branch-and-bound",
               resolute=False, verbose=0):
    """Approval Chamberlin-Courant (CC)"""
    return compute_thiele_method(
        'cc', profile, committeesize, algorithm=algorithm,
        resolute=resolute, verbose=verbose)


def compute_monroe(profile, committeesize, algorithm="brute-force",
                   resolute=False, verbose=0):
    """Monroe's rule"""
    enough_approved_candidates(profile, committeesize)

    # optional output
    if verbose:
        print(header(rules["monroe"].longname))
        if resolute:
            print("Computing only one winning committee (resolute=True)\n")
    if verbose >= 3:
        if algorithm == "gurobi":
            print("Using the Gurobi ILP solver\n")
        if algorithm == "brute-force":
            print("Using a brute-force algorithm\n")
    # end of optional output

    if not profile.has_unit_weights():
        raise ValueError(rules["monroe"].shortname +
                         " is only defined for unit weights (weight=1)")

    if algorithm == "gurobi":
        committees = abcrules_gurobi.__gurobi_monroe(
            profile, committeesize, resolute)
        committees = sort_committees(committees)
    elif algorithm == "brute-force":
        committees = __monroe_bruteforce(
            profile, committeesize, resolute)
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm)
            + " not specified for compute_monroe")

    # optional output
    if verbose:
        print("Optimal Monroe score: "
              + str(scores.monroescore(profile, committees[0])) + "\n")

        print(str_committees_header(committees, winning=True))
        print(str_candsets(committees, names=profile.names))
    # end of optional output

    return committees


# Monroe's rule, computed via (brute-force) matching
def __monroe_bruteforce(profile, committeesize, resolute):
    """Brute-force computation of Monroe's rule"""
    opt_committees = []
    opt_monroescore = -1
    for comm in combinations(list(range(profile.num_cand)), committeesize):
        score = scores.monroescore(profile, comm)
        if score > opt_monroescore:
            opt_committees = [comm]
            opt_monroescore = score
        elif scores.monroescore(profile, comm) == opt_monroescore:
            opt_committees.append(comm)

    committees = sort_committees(opt_committees)
    if resolute:
        committees = [committees[0]]

    return committees


def compute_greedy_monroe(profile, committeesize,
                          algorithm="standard", resolute=True, verbose=0):
    """"Greedy Monroe"""
    enough_approved_candidates(profile, committeesize)
    if not profile.has_unit_weights():
        raise ValueError(rules["greedy-monroe"].shortname +
                         " is only defined for unit weights (weight=1)")

    if not resolute:
        raise NotImplementedError(
            "compute_greedy_monroe does not support resolute=False.")

    if algorithm != "standard":
        raise NotImplementedError(
            "Algorithm " + str(algorithm)
            + " not specified for compute_greedy_monroe")

    num_voters = len(profile)
    committee = []

    # remaining voters
    remaining_voters = list(range(num_voters))
    remaining_cands = set(range(profile.num_cand))

    assignment = []
    for t in range(committeesize):
        maxapprovals = -1
        selected = None
        for c in remaining_cands:
            approvals = len([i for i in remaining_voters
                             if c in profile[i]])
            if approvals > maxapprovals:
                maxapprovals = approvals
                selected = c

        # determine how many voters are removed (at most)
        if t < num_voters - committeesize * (num_voters // committeesize):
            num_remove = num_voters // committeesize + 1
        else:
            num_remove = num_voters // committeesize

        # only voters that approve the chosen candidate
        # are removed
        to_remove = [i for i in remaining_voters
                     if selected in profile[i]]
        if len(to_remove) > num_remove:
            to_remove = to_remove[:num_remove]
        assignment.append((selected, to_remove))
        remaining_voters = [i for i in remaining_voters
                            if i not in to_remove]
        committee.append(selected)
        remaining_cands.remove(selected)

    committees = sort_committees([committee])

    # optional output
    if verbose:
        print(header(rules["greedy-monroe"].longname))

    if verbose >= 2:
        score1 = scores.monroescore(profile, committees[0])

        score2 = len(profile) - len(remaining_voters)
        print("The Monroe assignment computed by Greedy Monroe")
        print("has a Monroe score of " + str(score2) + ".")

        if score1 > score2:
            print("Monroe assignment found by Greedy Monroe is not "
                  + "optimal for the winning committee,")
            print("i.e., by redistributing voters to candidates a higher "
                  + "satisfaction is possible "
                  + "(without changing the committee).")
            print("Optimal Monroe score of the winning committee is "
                  + str(score1) + ".")

        # build actual Monroe assignment for winning committee
        for t, district in enumerate(assignment):
            cand, voters = district
            if t < num_voters - committeesize * (num_voters // committeesize):
                missing = num_voters // committeesize + 1 - len(voters)
            else:
                missing = num_voters // committeesize - len(voters)
            for _ in range(missing):
                v = remaining_voters.pop()
                voters.append(v)

        print("Assignment (unsatisfatied voters marked with *):\n")
        for cand, voters in assignment:
            print(" candidate " + profile.names[cand]
                  + " assigned to: ", end="")
            output = ""
            for v in sorted(voters):
                output += str(v)
                if cand not in profile[v].approved:
                    output += "*"
                output += ", "
            print(output[:-2])
        print()

    if verbose:
        print(str_committees_header(committees, winning=True))
        print(str_candsets(committees, names=profile.names))
    # end of optional output

    return committees


def __seqphragmen_resolute(profile, committeesize, verbose,
                           start_load=None, partial_committee=None):
    """Algorithm for computing resolute seq-Phragmen  (1 winning committee)"""
    approvers_weight = {}
    for c in range(profile.num_cand):
        approvers_weight[c] = sum(pref.weight for pref in profile if c in pref)

    load = start_load
    if load is None:
        load = {v: 0 for v, _ in enumerate(profile)}

    committee = partial_committee
    if partial_committee is None:
        committee = []  # build committees starting with the empty set

    for _ in range(len(committee), committeesize):
        approvers_load = {}
        for c in range(profile.num_cand):
            approvers_load[c] = sum(pref.weight * load[v]
                                    for v, pref in enumerate(profile)
                                    if c in pref)
        new_maxload = [Fraction(approvers_load[c] + 1, approvers_weight[c])
                       if approvers_weight[c] > 0 else committeesize + 1
                       for c in range(profile.num_cand)]
        # exclude committees already in the committee
        large = max(new_maxload) + 1
        for c in range(profile.num_cand):
            if c in committee:
                new_maxload[c] = large
        # find smallest maxload
        opt = min(new_maxload)
        next_cand = new_maxload.index(opt)
        # compute new loads and add new candidate
        for v, pref in enumerate(profile):
            if next_cand in pref:
                load[v] = new_maxload[next_cand]
            else:
                load[v] = load[v]
        committee = sorted(committee + [next_cand])

        # optional output
        if verbose >= 2:
            output = "adding candidate number "
            output += str(len(committee)) + ": "
            output += profile.names[next_cand] + "\n"
            output += " maximum load increased to "
            output += str(opt)
            print(output)
            print(" load distribution:")
            output = "  ("
            for v, _ in enumerate(profile):
                output += str(load[v]) + ", "
            print(output[:-2] + ")")
            tied_cands = [c for c in range(profile.num_cand)
                          if (c > next_cand and
                              (new_maxload[c] == new_maxload))]
            if len(tied_cands) > 0:
                output = " tie broken in favor of " + profile.names[next_cand]
                output += ",\n candidates " + str_candset(tied_cands)
                output += " would increase the load to the same amount ("
                output += str(new_maxload) + ")"
                print(output)
            print()
        # end of optional output

    comm_loads = {tuple(committee): load}
    return [committee], comm_loads


def __seqphragmen_irresolute(profile, committeesize,
                             start_load=None, partial_committee=None):
    """Algorithm for computing irresolute seq-Phragmen (>=1 winning committees)
    """
    approvers_weight = {}
    for c in range(profile.num_cand):
        approvers_weight[c] = sum(pref.weight for pref in profile if c in pref)

    load = start_load
    if load is None:
        load = {v: 0 for v, _ in enumerate(profile)}

    if partial_committee is None:
        partial_committee = []  # build committees starting with the empty set
    comm_loads = {tuple(partial_committee): load}

    for _ in range(len(partial_committee), committeesize):
        comm_loads_next = {}
        for committee, load in comm_loads.items():
            approvers_load = {}
            for c in range(profile.num_cand):
                approvers_load[c] = sum(pref.weight * load[v]
                                        for v, pref in enumerate(profile)
                                        if c in pref)
            new_maxload = [
                Fraction(approvers_load[c] + 1, approvers_weight[c])
                if approvers_weight[c] > 0 else committeesize + 1
                for c in range(profile.num_cand)]
            # exclude committees already in the committee
            for c in range(profile.num_cand):
                if c in committee:
                    new_maxload[c] = sys.maxsize
            # compute new loads
            # and add new committees
            for c in range(profile.num_cand):
                if new_maxload[c] <= min(new_maxload):
                    new_load = {}
                    for v, pref in enumerate(profile):
                        if c in pref:
                            new_load[v] = new_maxload[c]
                        else:
                            new_load[v] = load[v]
                    new_comm = tuple(sorted(committee + (c,)))
                    comm_loads_next[new_comm] = new_load
        comm_loads = comm_loads_next

    committees = sort_committees(list(comm_loads.keys()))
    return committees, comm_loads


def compute_seqphragmen(profile, committeesize, algorithm="standard",
                        resolute=True, verbose=False):
    """Phragmen's sequential rule (seq-Phragmen)"""
    enough_approved_candidates(profile, committeesize)

    if algorithm != "standard":
        raise NotImplementedError(
            "Algorithm " + str(algorithm)
            + " not specified for compute_seqphragmen")

    # optional output
    if verbose:
        print(header(rules["seqphrag"].longname))
        if resolute:
            print("Computing only one winning committee (resolute=True)\n")
    # end of optional output

    if resolute:
        committees, comm_loads = __seqphragmen_resolute(
            profile, committeesize, verbose)
    else:
        committees, comm_loads = __seqphragmen_irresolute(
            profile, committeesize)

    # optional output
    if verbose:
        print(str_committees_header(committees, winning=True))
        print(str_candsets(committees, names=profile.names))
    if verbose >= 2:
        if resolute or len(committees) == 1:
            print("corresponding load distribution:")
        else:
            print("corresponding load distributions:")
        for comm in committees:
            output = "("
            for v, _ in enumerate(profile):
                output += str(comm_loads[tuple(comm)][v]) + ", "
            print(output[:-2] + ")")
    # end of optional output

    return committees


def __rule_x_get_min_q(profile, budget, cand):
    rich = set([v for v, pref in enumerate(profile)
                if cand in pref])
    poor = set()

    while len(rich) > 0:
        poor_budget = sum(budget[v] for v in poor)
        q = Fraction(1 - poor_budget, len(rich))
        new_poor = set([v for v in rich
                        if budget[v] < q])
        if len(new_poor) == 0:
            return q
        rich -= new_poor
        poor.update(new_poor)

    return None  # not sufficient budget available


def compute_rule_x(profile, committeesize, algorithm="standard",
                   resolute=True, verbose=0, skip_phragmen_phase=False):
    """Rule X

    See https://arxiv.org/pdf/1911.11747.pdf, page 7
    
    skip_phragmen_phase : bool, optional
        omit the second phase (that uses seq-Phragmen)
        may result in a committee that is too small
    """
    enough_approved_candidates(profile, committeesize)
    if not profile.has_unit_weights():
        raise ValueError(rules["rule-x"].shortname +
                         " is only defined for unit weights (weight=1)")

    if algorithm != "standard":
        raise NotImplementedError(
            "Algorithm " + str(algorithm)
            + " not specified for compute_rule_x")

    # optional output
    if verbose:
        print(header(rules["rule-x"].longname))
        if resolute:
            print("Computing only one winning committee (resolute=True)\n")
    # end of optional output

    start_budget = {v: Fraction(committeesize, len(profile))
                    for v, _ in enumerate(profile)}
    cands = range(profile.num_cand)
    commbugdets = [(set(), start_budget)]
    final_committees = set()

    # optional output
    if resolute and verbose >= 2:
        print("Phase 1:\n")
        print("starting budget:")
        output = "  ("
        for v, _ in enumerate(profile):
            output += str(start_budget[v]) + ", "
        print(output[:-2] + ")\n")
    # end of optional output

    for _ in range(committeesize):
        next_commbudgets = []
        for committee, budget in commbugdets:

            curr_cands = set(cands) - committee
            min_q = {}
            for c in curr_cands:
                q = __rule_x_get_min_q(profile, budget, c)
                if q is not None:
                    min_q[c] = q

            if len(min_q) > 0:  # one or more candidates are affordable
                next_cands = [c for c in min_q.keys()
                              if min_q[c] == min(min_q.values())]
                for next_cand in next_cands:
                    new_budget = dict(budget)
                    for v, pref in enumerate(profile):
                        if next_cand in pref:
                            new_budget[v] -= min(budget[v], min_q[next_cand])
                    new_comm = set(committee)
                    new_comm.add(next_cand)
                    next_commbudgets.append((new_comm, new_budget))

                    # optional output
                    if resolute and verbose >= 2:
                        output = "adding candidate number "
                        output += str(len(committee)) + ": "
                        output += profile.names[next_cand] + "\n"
                        output += " with maxmimum cost per voter q = "
                        output += str(min(min_q.values()))
                        print(output)
                        print(" remaining budget:")
                        output = "  ("
                        for v, _ in enumerate(profile):
                            output += str(new_budget[v]) + ", "
                        print(output[:-2] + ")")
                        if len(next_cands) > 1:
                            output = " tie broken in favor of "
                            output += profile.names[next_cand] + ","
                            output += "\n candidates "
                            output += str_candset(next_cands[1:])
                            output += " are tied"
                            print(output)
                        print()
                    # end of optional output

                    if resolute:
                        break

            else:  # no affordable candidates remain
              if skip_phragmen_phase:
                  final_committees.add(tuple(committee))
              else:
                # fill committee via seq-Phragmen

                # optional output
                if resolute and verbose >= 2:
                    print("Phase 2 (seq-Phragmén):\n")
                # end of optional output

                start_load = {}
                # translate budget to loads
                for v in range(len(profile)):
                    start_load[v] = (Fraction(committeesize, len(profile))
                                     - budget[v])

                # optional output
                if resolute and verbose >= 2:
                    print("starting loads (= budget spent):")
                    output = "  ("
                    for v, _ in enumerate(profile):
                        output += str(start_load[v]) + ", "
                    print(output[:-2] + ")\n")
                # end of optional output

                if resolute:
                    committees, _ = __seqphragmen_resolute(
                        profile, committeesize, verbose=verbose,
                        partial_committee=list(committee),
                        start_load=start_load)
                else:
                    committees, _ = __seqphragmen_irresolute(
                        profile, committeesize,
                        partial_committee=list(committee),
                        start_load=start_load)
                final_committees.update([tuple(comm) for comm in committees])
                # after filling the remaining spots these committees
                # have size committeesize

            commbugdets = next_commbudgets

    final_committees.update([tuple(comm) for comm, _ in commbugdets])

    committees = sort_committees(final_committees)
    if resolute:
        committees = committees[:1]

    # optional output
    if verbose:
        print(str_committees_header(committees, winning=True))
        print(str_candsets(committees, names=profile.names))
    # end of optional output

    return committees


def compute_optphragmen(profile, committeesize,
                        algorithm="gurobi", resolute=False, verbose=0):
    enough_approved_candidates(profile, committeesize)

    # optional output
    if verbose:
        print(header(rules["optphrag"].longname))
        if resolute:
            print("Computing only one winning committee (resolute=True)\n")
    if verbose >= 3:
        if algorithm == "gurobi":
            print("Using the Gurobi ILP solver")
    # end of optional output

    if algorithm != "gurobi":
        raise NotImplementedError("Algorithm " + str(algorithm)
                                  + " not specified for compute_optphragmen")

    committees = abcrules_gurobi.__gurobi_optphragmen(
        profile, committeesize, resolute=resolute, verbose=verbose)
    committees = sort_committees(committees)

    # optional output
    if verbose:
        print(str_committees_header(committees, winning=True))
        print(str_candsets(committees, names=profile.names))
    # end of optional output

    return committees


def compute_phragmen_enestroem(profile, committeesize, algorithm="standard",
                               resolute=True, verbose=0):
    """"Phragmen-Enestroem (aka Phragmen's first method, Enestroem's method)

    In every round the candidate with the highest combined budget of
    their supporters is put in the committee.
    Method described in:
    https://arxiv.org/pdf/1611.08826.pdf (Section 18.5, Page 59)
    """
    enough_approved_candidates(profile, committeesize)
    if not profile.has_unit_weights():
        raise ValueError(rules["phrag-enestr"].shortname +
                         " is only defined for unit weights (weight=1)")

    if algorithm != "standard":
        raise NotImplementedError(
            "Algorithm " + str(algorithm)
            + " not specified for compute_phragmen_enestroem")

    num_voters = len(profile)

    start_budget = {i: Fraction(profile[i].weight)
                    for i in range(num_voters)}
    price = Fraction(sum(start_budget.values()), committeesize)

    cands = range(profile.num_cand)

    committees = [(start_budget, set())]
    for _ in range(committeesize):
        # here the committees with i+1 candidates are
        # stored (together with budget)
        next_committees = []
        # loop in case multiple possible committees
        # with i filled candidates
        for committee in committees:
            budget, comm = committee
            curr_cands = set(cands) - comm
            support = {c: 0 for c in curr_cands}
            for nr, pref in enumerate(profile):
                voting_power = budget[nr]
                if voting_power <= 0:
                    continue
                for cand in pref:
                    if cand in curr_cands:
                        support[cand] += voting_power
            max_support = max(support.values())
            winners = [c for c, s in support.items()
                       if s == max_support]
            for cand in winners:
                b = dict(budget)  # copy of budget
                if max_support > price:  # supporters can afford it
                    # (voting_power - price) / voting_power
                    multiplier = Fraction(max_support - price,
                                          max_support)
                else:  # set supporters to 0
                    multiplier = 0
                for nr, pref in enumerate(profile):
                    if cand in pref:
                        b[nr] *= multiplier
                c = comm.union([cand])  # new committee with candidate
                next_committees.append((b, c))

        if resolute:
            committees = [next_committees[0]]
        else:
            committees = next_committees
    committees = [comm for b, comm in committees]
    committees = sort_committees(committees)
    if resolute:
        committees = [committees[0]]

    # optional output
    if verbose:
        print(header(rules["phrag-enestr"].longname))

        print(str_committees_header(committees, winning=True))
        print(str_candsets(committees, names=profile.names))
    # end of optional output

    return committees


__RULESINFO = [
    ("av", "AV", "Approval Voting (AV)", compute_av,
     ("standard",), (True, False)),
    ("sav", "SAV", "Satisfaction Approval Voting (SAV)", compute_sav,
     ("standard",), (True, False)),
    ("pav", "PAV", "Proportional Approval Voting (PAV)", compute_pav,
     ("gurobi", "branch-and-bound"), (True, False)),
    ("slav", "SLAV", "Sainte-Laguë Approval Voting (SLAV)", compute_slav,
     ("gurobi", "branch-and-bound"), (True, False)),
    ("cc", "CC", "Approval Chamberlin-Courant (CC)", compute_cc,
     ("gurobi", "branch-and-bound"), (True, False)),
    ("geom2", "2-Geometric", "2-Geometric Rule",
     functools.partial(compute_thiele_method, "geom2"),
     ("gurobi", "branch-and-bound",), (True, False)),
    ("seqpav", "seq-PAV", "Sequential Proportional Approval Voting (seq-PAV)",
     compute_seqpav, ("standard",), (True, False)),
    ("revseqpav", "revseq-PAV",
     "Reverse Sequential Proportional Approval Voting (revseq-PAV)",
     compute_revseqpav, ("standard",), (True, False)),
    ("seqslav", "seq-SLAV",
     "Sequential Sainte-Laguë Approval Voting (seq-SLAV)",
     compute_seqslav, ("standard",), (True, False)),
    ("seqcc", "seq-CC", "Sequential Approval Chamberlin-Courant (seq-CC)",
     compute_seqcc, ("standard",), (True, False)),
    ("seqphrag", "seq-Phragmén", "Phragmén's Sequential Rule (seq-Phragmén)",
     compute_seqphragmen, ("standard",), (True, False)),
    ("optphrag", "opt-Phragmén", "Phragmén's Optimization Rule (opt-Phragmén)",
     compute_optphragmen, ("gurobi",), (True, False)),
    ("monroe", "Monroe", "Monroe's Approval Rule (Monroe)",
     compute_monroe, ("gurobi", "brute-force"), (True, False)),
    ("greedy-monroe", "Greedy Monroe", "Greedy Monroe",
     compute_greedy_monroe, ("standard",), (True,)),
    ("mav", "MAV", "Minimax Approval Voting (MAV)",
     compute_mav, ("gurobi", "brute-force"), (True, False)),
    ("lexmav", "lex-MAV", "Lexicographic Minimax Approval Voting (lex-MAV)",
     compute_lexmav, ("brute-force",), (True, False)),
    ("rule-x", "Rule X", "Rule X",
     compute_rule_x, ("standard",), (True, False)),
    ("phrag-enestr", "Phragmén-Eneström", "Method of Phragmén-Eneström",
     compute_phragmen_enestroem, ("standard",), (True, False))]
rules = {}
for ruleinfo in __RULESINFO:
    rules[ruleinfo[0]] = ABCRule(*ruleinfo)
# TODO: add other thiele methods
