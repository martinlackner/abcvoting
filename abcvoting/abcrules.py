# -*- coding: utf-8 -*-
"""Approval-based committee (ABC) voting rules"""


import sys
import functools
from itertools import combinations

try:
    from gmpy2 import mpq as Fraction
except ImportError:
    print("Warning: module gmpy2 not found, " + "resorting to Python's fractions.Fraction")
    from fractions import Fraction
from abcvoting import abcrules_gurobi, abcrules_ortools, abcrules_cvxpy, abcrules_mip
from abcvoting.misc import sorted_committees
from abcvoting.misc import hamming
from abcvoting.misc import check_enough_approved_candidates
from abcvoting.misc import str_committees_header, header
from abcvoting.misc import str_set_of_candidates, str_sets_of_candidates
from abcvoting import scores


########################################################################


class UnknownRuleIDError(ValueError):
    """Exception raised if unknown rule id is used"""

    def __init__(self, rule_id):
        message = 'Rule ID "' + str(rule_id) + '" is not known.'
        super(ValueError, self).__init__(message)


def is_algorithm_supported(algorithm):
    if "gurobi" in algorithm and not abcrules_gurobi.available:
        return False

    if algorithm.startswith("cvxpy"):
        if not abcrules_cvxpy.cvxpy_available or not abcrules_cvxpy.numpy_available:
            return False

        import cvxpy

        if algorithm == "cvxpy_glpk_mi" and cvxpy.GLPK_MI not in cvxpy.installed_solvers():
            return False
        elif algorithm == "cvxpy_cbc" and cvxpy.CBC not in cvxpy.installed_solvers():
            return False
        elif algorithm == "cvxpy_scip" and cvxpy.SCIP not in cvxpy.installed_solvers():
            return False
        elif algorithm == "cvxpy_gurobi" and not abcrules_gurobi.available:
            return False

    return True


class ABCRule:
    """Class for ABC rules containing basic information and function call"""

    def __init__(
        self, rule_id, shortname, longname, fct, algorithms=("standard",), resolute=(True, False)
    ):
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
        for algorithm in self.algorithms:
            if is_algorithm_supported(algorithm):
                return algorithm


def _init_rules():
    _THIELE_ALGORITHMS = (
        # TODO sort by speed, requires testing
        "gurobi",
        "branch-and-bound",
        "ortools_cbc",
        "ortools_gurobi",
        "mip_cbc",
        "mip_gurobi",
        "brute-force",
    )
    _RULESINFO = [
        ("av", "AV", "Approval Voting (AV)", compute_av, ("standard",), (True, False)),
        (
            "sav",
            "SAV",
            "Satisfaction Approval Voting (SAV)",
            compute_sav,
            ("standard",),
            (True, False),
        ),
        (
            "pav",
            "PAV",
            "Proportional Approval Voting (PAV)",
            compute_pav,
            # TODO sort by speed, requires testing
            (
                "gurobi",
                "branch-and-bound",
                "cvxpy_glpk_mi",
                "cvxpy_cbc",
                "cvxpy_gurobi",
                "ortools_cbc",
                "ortools_gurobi",
                "mip_cbc",
                "mip_gurobi",
                "brute-force",
            ),
            (True, False),
        ),
        (
            "slav",
            "SLAV",
            "Sainte-Laguë Approval Voting (SLAV)",
            compute_slav,
            # TODO sort by speed, requires testing
            _THIELE_ALGORITHMS,
            (True, False),
        ),
        (
            "cc",
            "CC",
            "Approval Chamberlin-Courant (CC)",
            compute_cc,
            # TODO sort by speed, requires testing
            _THIELE_ALGORITHMS,
            (True, False),
        ),
        (
            "seqpav",
            "seq-PAV",
            "Sequential Proportional Approval Voting (seq-PAV)",
            compute_seqpav,
            ("standard",),
            (True, False),
        ),
        (
            "revseqpav",
            "revseq-PAV",
            "Reverse Sequential Proportional Approval Voting (revseq-PAV)",
            compute_revseqpav,
            ("standard",),
            (True, False),
        ),
        (
            "seqslav",
            "seq-SLAV",
            "Sequential Sainte-Laguë Approval Voting (seq-SLAV)",
            compute_seqslav,
            ("standard",),
            (True, False),
        ),
        (
            "seqcc",
            "seq-CC",
            "Sequential Approval Chamberlin-Courant (seq-CC)",
            compute_seqcc,
            ("standard",),
            (True, False),
        ),
        (
            "seqphragmen",
            "seq-Phragmén",
            "Phragmén's Sequential Rule (seq-Phragmén)",
            compute_seqphragmen,
            ("standard", "exact-fractions"),
            (True, False),
        ),
        (
            "optphragmen",
            "opt-Phragmén",
            "Phragmén's Optimization Rule (opt-Phragmén)",
            compute_optphragmen,
            ("gurobi",),
            (True, False),
        ),
        (
            "monroe",
            "Monroe",
            "Monroe's Approval Rule (Monroe)",
            compute_monroe,
            ("gurobi", "ortools_cp", "brute-force"),
            (True, False),
        ),
        (
            "greedy-monroe",
            "Greedy Monroe",
            "Greedy Monroe",
            compute_greedy_monroe,
            ("standard",),
            (True,),
        ),
        (
            "minimaxav",
            "minimaxav",
            "Minimax Approval Voting (MAV)",
            compute_minimaxav,
            # TODO sort by speed, requires testing
            (
                "gurobi",
                "ortools_cp",
                "ortools_cbc",
                "ortools_gurobi",
                "brute-force",
                "mip_gurobi",
                "mip_cbc",
            ),
            (True, False),
        ),
        (
            "lexminimaxav",
            "lex-MAV",
            "Lexicographic Minimax Approval Voting (lex-MAV)",
            compute_lexminimaxav,
            ("brute-force",),
            (True, False),
        ),
        (
            "rule-x",
            "Rule X",
            "Rule X",
            compute_rule_x,
            ("standard", "exact-fractions"),
            (True, False),
        ),
        (
            "rule-x-without-2nd-phase",
            "Rule X (without 2nd phase)",
            "Rule X without the second (Phragmén) phase",
            compute_rule_x_without_2nd_phase,
            ("standard", "exact-fractions"),
            (True, False),
        ),
        (
            "phragmen-enestroem",
            "Phragmén-Eneström",
            "Method of Phragmén-Eneström",
            compute_phragmen_enestroem,
            ("standard", "exact-fractions"),
            (True, False),
        ),
        (
            "consensus-rule",
            "Consensus Rule",
            "Consensus Rule",
            compute_consensus_rule,
            ("standard", "exact-fractions"),
            (True, False),
        ),
    ]
    for parameter in [1.5, 2, 5]:
        _RULESINFO.append(
            (
                f"geom{parameter}",
                f"{parameter}-Geometric",
                f"{parameter}-Geometric Rule",
                functools.partial(compute_thiele_method, f"geom{parameter}"),
                _THIELE_ALGORITHMS,
                (True, False),
            )
        )

    rulesdict = {}
    for ruleinfo in _RULESINFO:
        rulesdict[ruleinfo[0]] = ABCRule(*ruleinfo)

    return rulesdict


########################################################################


def compute(rule_id, profile, committeesize, **kwargs):
    try:
        rule = rules[rule_id]
    except KeyError:
        raise UnknownRuleIDError(rule_id)
    return rule.compute(profile, committeesize, **kwargs)


def compute_thiele_method(
    scorefct_str, profile, committeesize, algorithm="branch-and-bound", resolute=False, verbose=0
):
    """Thiele methods

    Compute winning committees according to a Thiele method specified
    by a score function (scorefct_str).
    Examples of Thiele methods are PAV, CC, and SLAV.
    An exception is Approval Voting (AV), which should be computed using
    compute_av(). (AV is polynomial-time computable (separable) and can thus be
    computed much faster.)
    """
    check_enough_approved_candidates(profile, committeesize)
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
        if algorithm == "brute-force":
            print("Using a brute-force algorithm\n")
    # end of optional output

    if algorithm == "fastest":
        algorithm = rules[scorefct_str].fastest_algo()

    if algorithm == "gurobi":
        committees = abcrules_gurobi._gurobi_thiele_methods(
            profile, committeesize, scorefct, resolute
        )
        committees = sorted_committees(committees)
    elif algorithm == "branch-and-bound":
        committees = _thiele_methods_branchandbound(profile, committeesize, scorefct_str, resolute)
    elif algorithm == "brute-force":
        committees = _thiele_methods_bruteforce(profile, committeesize, scorefct_str, resolute)
    elif algorithm.startswith("cvxpy_"):
        committees = abcrules_cvxpy.cvxpy_thiele_methods(
            profile=profile,
            committeesize=committeesize,
            scorefct_str=scorefct_str,
            resolute=resolute,
            solver_id=algorithm[6:],
        )
    elif algorithm.startswith("mip_"):
        committees = abcrules_mip._mip_thiele_methods(
            profile,
            committeesize,
            scorefct,
            resolute,
            solver_id=algorithm[4:],
        )
    elif algorithm.startswith("ortools_"):
        committees = abcrules_ortools._ortools_thiele_methods(
            profile,
            committeesize,
            scorefct=scorefct,
            resolute=resolute,
            solver_id=algorithm[8:],
        )
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_thiele_method"
        )

    # optional output
    if verbose >= 2:
        print(
            "Optimal "
            + scorefct_str.upper()
            + "-score: "
            + str(scores.thiele_score(scorefct_str, profile, committees[0]))
        )
        print()
    if verbose:
        print(str_committees_header(committees, winning=True))
        print(str_sets_of_candidates(committees, cand_names=profile.cand_names))
    # end of optional output

    return committees


def _thiele_methods_bruteforce(profile, committeesize, scorefct_str, resolute):
    """Brute-force algorithm for computing Thiele methods
    Only intended for comparison, much slower than _thiele_methods_branchandbound()"""
    scorefct = scores.get_scorefct(scorefct_str, committeesize)

    opt_committees = []
    opt_thiele_score = -1
    for committee in combinations(profile.candidates, committeesize):
        score = scores.thiele_score(scorefct_str, profile, committee)
        if score > opt_thiele_score:
            opt_committees = [committee]
            opt_thiele_score = score
        elif score == opt_thiele_score:
            opt_committees.append(committee)

    committees = sorted_committees(opt_committees)
    if resolute:
        return [committees[0]]
    return committees


def _thiele_methods_branchandbound(profile, committeesize, scorefct_str, resolute):
    """Branch-and-bound algorithm to compute winning committees
    for Thiele methods"""
    scorefct = scores.get_scorefct(scorefct_str, committeesize)

    best_committees = []
    init_com = compute_seq_thiele_method(profile, committeesize, scorefct_str, resolute=True)[0]
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
            marg_util_cand = scores.marginal_thiele_scores_add(scorefct, profile, part_com)
            upper_bound = sum(
                sorted(marg_util_cand[largest_cand + 1 :])[-missing:]
            ) + scores.thiele_score(scorefct_str, profile, part_com)
            if upper_bound >= best_score:
                for cand in range(largest_cand + 1, profile.num_cand - missing + 1):
                    part_coms.insert(0, part_com + [cand])

    committees = sorted_committees(best_committees)
    if resolute:
        committees = [committees[0]]

    return committees


# Sequential PAV
def compute_seqpav(profile, committeesize, algorithm="standard", resolute=True, verbose=0):
    """Sequential PAV (seq-PAV)"""
    return compute_seq_thiele_method(
        profile, committeesize, "pav", algorithm=algorithm, resolute=resolute, verbose=verbose
    )


def compute_seqslav(profile, committeesize, algorithm="standard", resolute=True, verbose=0):
    """Sequential Sainte-Lague Approval Voting (SLAV)"""
    return compute_seq_thiele_method(
        profile, committeesize, "slav", algorithm=algorithm, resolute=resolute, verbose=verbose
    )


# Reverse Sequential PAV
def compute_revseqpav(profile, committeesize, algorithm="standard", resolute=True, verbose=0):
    """Reverse sequential PAV (revseq-PAV)"""
    return compute_revseq_thiele_method(
        profile, committeesize, "pav", algorithm=algorithm, resolute=resolute, verbose=verbose
    )


def compute_seqcc(profile, committeesize, algorithm="standard", resolute=True, verbose=0):
    """Sequential Chamberlin-Courant (seq-CC)"""
    return compute_seq_thiele_method(
        profile, committeesize, "cc", algorithm=algorithm, resolute=resolute, verbose=verbose
    )


def compute_sav(profile, committeesize, algorithm="standard", resolute=False, verbose=0):
    """Satisfaction Approval Voting (SAV)"""
    if algorithm == "fastest":
        algorithm = rules["sav"].fastest_algo()
    if algorithm == "standard":
        return compute_separable_rule("sav", profile, committeesize, resolute, verbose)
    else:
        raise NotImplementedError("Algorithm " + str(algorithm) + " not specified for compute_sav")


# Approval Voting (AV)
def compute_av(profile, committeesize, algorithm="standard", resolute=False, verbose=0):
    """Approval Voting"""
    if algorithm == "fastest":
        algorithm = rules["av"].fastest_algo()
    if algorithm == "standard":
        return compute_separable_rule("av", profile, committeesize, resolute, verbose)
    else:
        raise NotImplementedError("Algorithm " + str(algorithm) + " not specified for compute_av")


def compute_separable_rule(rule_id, profile, committeesize, resolute, verbose):
    check_enough_approved_candidates(profile, committeesize)
    score = [0] * profile.num_cand
    for voter in profile:
        for cand in voter.approved:
            if rule_id == "sav":
                # Satisfaction Approval Voting
                score[cand] += voter.weight / len(voter.approved)
            elif rule_id == "av":
                # (Classic) Approval Voting
                score[cand] += voter.weight
            else:
                raise UnknownRuleIDError(rule_id)

    # smallest score to be in the committee
    cutoff = sorted(score)[-committeesize]

    certain_cands = [cand for cand in profile.candidates if score[cand] > cutoff]
    possible_cands = [cand for cand in profile.candidates if score[cand] == cutoff]
    missing = committeesize - len(certain_cands)
    if len(possible_cands) == missing:
        # candidates with score[cand] == cutoff
        # are also certain candidates because all these candidates
        # are required to fill the committee
        certain_cands = sorted(certain_cands + possible_cands)
        possible_cands = []
        missing = 0

    if resolute:
        committees = sorted_committees([(certain_cands + possible_cands[:missing])])
    else:
        committees = sorted_committees(
            [
                (certain_cands + list(selection))
                for selection in combinations(possible_cands, missing)
            ]
        )

    # optional output
    if verbose:
        print(header(rules[rule_id].longname))
        if resolute:
            print("Computing only one winning committee (resolute=True)\n")
    if verbose >= 2:
        print("Scores of candidates:")
        for cand in profile.candidates:
            print(profile.cand_names[cand] + ": " + str(score[cand]))

        print("\nCandidates are contained in winning committees")
        print("if their score is >= " + str(cutoff) + ".")

        if len(certain_cands) > 0:
            print("\nThe following candidates are contained in")
            print("every winning committee:")
            namedset = [profile.cand_names[cand] for cand in certain_cands]
            print(" " + ", ".join(map(str, namedset)))
            print()

        if len(possible_cands) > 0:
            print("The following candidates are contained in")
            print("some of the winning committees:")
            namedset = [profile.cand_names[cand] for cand in possible_cands]
            print(" " + ", ".join(map(str, namedset)))
            print(
                "("
                + str(missing)
                + " of those candidates is contained\n"
                + " in every winning committee.)\n"
            )
    if verbose:
        print(str_committees_header(committees, winning=True))
        print(str_sets_of_candidates(committees, cand_names=profile.cand_names))
    # end of optional output

    return committees


def _seq_thiele_resolute(profile, committeesize, scorefct_str, verbose):
    """Compute a *resolute* reverse sequential Thiele method

    Tiebreaking between candidates in favor of candidate with smaller
    number/index (candidates with larger numbers get deleted first).
    """
    committee = []
    scorefct = scores.get_scorefct(scorefct_str, committeesize)

    # optional output
    if verbose >= 2:
        output = "starting with the empty committee (score = "
        output += str(scores.thiele_score(scorefct_str, profile, committee)) + ")"
        print(output + "\n")
    # end of optional output

    # build a committee starting with the empty set
    for _ in range(committeesize):
        additional_score_cand = scores.marginal_thiele_scores_add(scorefct, profile, committee)
        next_cand = additional_score_cand.index(max(additional_score_cand))
        committee.append(next_cand)
        # optional output
        if verbose >= 2:
            output = "adding candidate number "
            output += str(len(committee)) + ": "
            output += profile.cand_names[next_cand] + "\n"
            output += " score increases by "
            output += str(max(additional_score_cand))
            output += " to a total of "
            output += str(scores.thiele_score(scorefct_str, profile, committee))
            tied_cands = [
                cand
                for cand in range(len(additional_score_cand))
                if (cand > next_cand and additional_score_cand[cand] == max(additional_score_cand))
            ]
            if tied_cands:
                output += " tie broken in favor of " + str(next_cand)
                output += " candidates " + str_set_of_candidates(tied_cands)
                output += " would increase the score by the same amount ("
                output += str(max(additional_score_cand)) + ")"
            print(output + "\n")
        # end of optional output
    return [sorted(committee)]


def _seq_thiele_irresolute(profile, committeesize, scorefct_str):
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
            additional_score_cand = scores.marginal_thiele_scores_add(scorefct, profile, committee)
            for cand in profile.candidates:
                if additional_score_cand[cand] >= max(additional_score_cand):
                    next_comm = tuple(sorted(committee + (cand,)))
                    comm_scores_next[next_comm] = score + additional_score_cand[cand]
        comm_scores = comm_scores_next
    return sorted_committees(list(comm_scores.keys()))


def compute_seq_thiele_method(
    profile, committeesize, scorefct_str, algorithm="standard", resolute=True, verbose=0
):
    """Sequential Thiele methods"""

    check_enough_approved_candidates(profile, committeesize)

    if algorithm == "fastest":
        algorithm = rules["seq" + scorefct_str].fastest_algo()
    if algorithm != "standard":
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_seq_thiele_method"
        )

    # optional output
    if verbose:
        print(header(rules["seq" + scorefct_str].longname))
        if resolute:
            print("Computing only one winning committee (resolute=True)\n")
    # end of optional output

    if resolute:
        committees = _seq_thiele_resolute(profile, committeesize, scorefct_str, verbose=verbose)
    else:
        committees = _seq_thiele_irresolute(profile, committeesize, scorefct_str)

    # optional output
    if verbose:
        print(str_committees_header(committees, winning=True))
        print(str_sets_of_candidates(committees, cand_names=profile.cand_names))
        if resolute or len(committees) == 1:
            print(scorefct_str.upper() + "-score of winning committee:", end="")
        else:
            print(scorefct_str.upper() + "-score of winning committees:")
        for committee in committees:
            print(" " + str(scores.thiele_score(scorefct_str, profile, committee)))
        print()
    # end of optional output

    return sorted_committees(committees)


def _revseq_thiele_irresolute(profile, committeesize, scorefct_str):
    """Compute an *irresolute* sequential Thiele method

    Consider all possible ways to break ties between candidates
    (aka parallel universe tiebreaking)
    """
    scorefct = scores.get_scorefct(scorefct_str, committeesize)

    allcandcomm = tuple(profile.candidates)
    comm_scores = {allcandcomm: scores.thiele_score(scorefct_str, profile, allcandcomm)}

    for _ in range(profile.num_cand - committeesize):
        comm_scores_next = {}
        for committee, score in comm_scores.items():
            marg_util_cand = scores.marginal_thiele_scores_remove(scorefct, profile, committee)
            score_reduction = min(marg_util_cand)
            # find smallest elements in marg_util_cand and return indices
            cands_to_remove = [
                cand for cand in profile.candidates if marg_util_cand[cand] == min(marg_util_cand)
            ]
            for cand in cands_to_remove:
                next_comm = tuple(set(committee) - {cand})
                comm_scores_next[next_comm] = score - score_reduction
            comm_scores = comm_scores_next
    return sorted_committees(list(comm_scores.keys()))


def _revseq_thiele_resolute(profile, committeesize, scorefct_str, verbose):
    """Compute a *resolute* reverse sequential Thiele method

    Tiebreaking between candidates in favor of candidate with smaller
    number/index (candidates with smaller numbers are added first).
    """
    scorefct = scores.get_scorefct(scorefct_str, committeesize)
    committee = set(profile.candidates)

    # optional output
    if verbose >= 2:
        output = "full committee (" + str(len(committee))
        output += " candidates) has a total score of "
        output += str(scores.thiele_score(scorefct_str, profile, committee))
        print(output + "\n")
    # end of optional output

    for _ in range(profile.num_cand - committeesize):
        marg_util_cand = scores.marginal_thiele_scores_remove(scorefct, profile, committee)
        score_reduction = min(marg_util_cand)
        # find smallest elements in marg_util_cand and return indices
        cands_to_remove = [
            cand for cand in profile.candidates if marg_util_cand[cand] == min(marg_util_cand)
        ]
        committee.remove(cands_to_remove[-1])

        # optional output
        if verbose >= 2:
            rem_cand = cands_to_remove[-1]
            output = "removing candidate number "
            output += str(profile.num_cand - len(committee)) + ": "
            output += profile.cand_names[rem_cand] + "\n"
            output += " score decreases by "
            output += str(score_reduction)
            output += " to a total of "
            output += str(scores.thiele_score(scorefct_str, profile, committee))
            if len(cands_to_remove) > 1:
                output += " (tie between candidates "
                output += str_set_of_candidates(cands_to_remove) + ")\n"
            print(output + "\n")
        # end of optional output

    return sorted_committees([committee])


def compute_revseq_thiele_method(
    profile, committeesize, scorefct_str, algorithm="standard", resolute=True, verbose=0
):
    """Reverse sequential Thiele methods"""
    check_enough_approved_candidates(profile, committeesize)

    if algorithm == "fastest":
        algorithm = rules["revseq" + scorefct_str].fastest_algo()
    if algorithm != "standard":
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_revseq_thiele_method"
        )

    # optional output
    if verbose:
        print(header(rules["revseq" + scorefct_str].longname))
        if resolute:
            print("Computing only one winning committee (resolute=True)\n")
    # end of optional output

    if resolute:
        committees = _revseq_thiele_resolute(profile, committeesize, scorefct_str, verbose=verbose)
    else:
        committees = _revseq_thiele_irresolute(profile, committeesize, scorefct_str)

    # optional output
    if verbose:
        print(str_committees_header(committees, winning=True))
        print(str_sets_of_candidates(committees, cand_names=profile.cand_names))
    if verbose >= 2:
        if resolute or len(committees) == 1:
            print("PAV-score of winning committee:", end="")
        else:
            print("PAV-score of winning committees:")
        for committee in committees:
            print(" " + str(scores.thiele_score(scorefct_str, profile, committee)))
        print()
    # end of optional output

    return committees


def _minimaxav_bruteforce(profile, committeesize, resolute):
    """Brute-force algorithm for computing Minimax AV (MAV)"""
    opt_committees = []
    opt_minimaxav_score = profile.num_cand + 1
    for committee in combinations(profile.candidates, committeesize):
        score = scores.mavscore(profile, committee)
        if score < opt_minimaxav_score:
            opt_committees = [committee]
            opt_minimaxav_score = score
        elif score == opt_minimaxav_score:
            opt_committees.append(committee)

    committees = sorted_committees(opt_committees)
    if resolute:
        return [committees[0]]
    return committees


# Minimax Approval Voting
def compute_minimaxav(profile, committeesize, algorithm="brute-force", resolute=False, verbose=0):
    """Minimax AV (MAV)"""
    check_enough_approved_candidates(profile, committeesize)

    # optional output
    if verbose:
        print(header(rules["minimaxav"].longname))
        if resolute:
            print("Computing only one winning committee (resolute=True)\n")
    if verbose >= 3:
        if algorithm == "gurobi":
            print("Using the Gurobi ILP solver\n")
        if algorithm == "brute-force":
            print("Using a brute-force algorithm\n")
    # end of optional output

    if algorithm == "fastest":
        algorithm = rules["minimaxav"].fastest_algo()

    if algorithm == "gurobi":
        committees = abcrules_gurobi._gurobi_minimaxav(profile, committeesize, resolute)
        committees = sorted_committees(committees)
    elif algorithm.startswith("ortools_"):
        solver_id = algorithm[8:]
        committees = abcrules_ortools._ortools_minimaxav(
            profile, committeesize, resolute, solver_id
        )
        committees = sorted_committees(committees)
    elif algorithm.startswith("mip_"):
        solver_id = algorithm[4:]
        committees = abcrules_mip._mip_minimaxav(profile, committeesize, resolute, solver_id)
        committees = sorted_committees(committees)
    elif algorithm == "brute-force":
        committees = _minimaxav_bruteforce(profile, committeesize, resolute)
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_minimaxav"
        )

    opt_minimaxav_score = scores.mavscore(profile, committees[0])

    # optional output
    if verbose:
        print("Minimum maximal distance: " + str(opt_minimaxav_score))

        print(str_committees_header(committees, winning=True))
        print(str_sets_of_candidates(committees, cand_names=profile.cand_names))

        print("Corresponding distances to voters:")
        for committee in committees:
            print([hamming(voter.approved, committee) for voter in profile])
        print()
    # end of optional output

    return committees


# Lexicographic Minimax Approval Voting
def compute_lexminimaxav(
    profile, committeesize, algorithm="brute-force", resolute=False, verbose=0
):
    """Lexicographic Minimax AV"""
    check_enough_approved_candidates(profile, committeesize)

    if not profile.has_unit_weights():
        raise ValueError(
            rules["lexminimaxav"].shortname + " is only defined for unit weights (weight=1)"
        )

    if algorithm == "fastest":
        algorithm = rules["lexminimaxav"].fastest_algo()

    if algorithm != "brute-force":
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_lexminimaxav"
        )

    opt_committees = []
    opt_distances = [profile.num_cand + 1] * len(profile)
    for committee in combinations(profile.candidates, committeesize):
        distances = sorted([hamming(voter.approved, committee) for voter in profile], reverse=True)
        for i in range(len(distances)):
            if opt_distances[i] < distances[i]:
                break
            if opt_distances[i] > distances[i]:
                opt_distances = distances
                opt_committees = [committee]
                break
        else:
            opt_committees.append(committee)

    committees = sorted_committees(opt_committees)
    if resolute:
        committees = [committees[0]]

    # optional output
    if verbose:
        print(header(rules["lexminimaxav"].longname))
        if resolute:
            print("Computing only one winning committee (resolute=True)\n")

        print("Minimum maximal distance: " + str(max(opt_distances)))

        print(str_committees_header(committees, winning=True))
        print(str_sets_of_candidates(committees, cand_names=profile.cand_names))

        print("Corresponding distances to voters:")
        for committee in committees:
            print([hamming(voter.approved, committee) for voter in profile])
        print()
    # end of optional output

    return committees


# Proportional Approval Voting
def compute_pav(profile, committeesize, algorithm="branch-and-bound", resolute=False, verbose=0):
    """Proportional Approval Voting (PAV)"""
    return compute_thiele_method(
        "pav", profile, committeesize, algorithm=algorithm, resolute=resolute, verbose=verbose
    )


# Sainte-Lague Approval Voting
def compute_slav(profile, committeesize, algorithm="branch-and-bound", resolute=False, verbose=0):
    """Sainte-Lague Approval Voting (SLAV)"""
    return compute_thiele_method(
        "slav", profile, committeesize, algorithm=algorithm, resolute=resolute, verbose=verbose
    )


# Chamberlin-Courant
def compute_cc(profile, committeesize, algorithm="branch-and-bound", resolute=False, verbose=0):
    """Approval Chamberlin-Courant (CC)"""
    return compute_thiele_method(
        "cc", profile, committeesize, algorithm=algorithm, resolute=resolute, verbose=verbose
    )


def compute_monroe(profile, committeesize, algorithm="brute-force", resolute=False, verbose=0):
    """Monroe's rule"""
    check_enough_approved_candidates(profile, committeesize)

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
        raise ValueError(
            rules["monroe"].shortname + " is only defined for unit weights (weight=1)"
        )

    if algorithm == "fastest":
        algorithm = rules["monroe"].fastest_algo()

    if algorithm == "gurobi":
        committees = abcrules_gurobi._gurobi_monroe(profile, committeesize, resolute)
        committees = sorted_committees(committees)
    elif algorithm.startswith("ortools_"):
        solver_id = algorithm[8:]
        committees = abcrules_ortools._ortools_monroe(profile, committeesize, resolute, solver_id)
        committees = sorted_committees(committees)
    elif algorithm == "brute-force":
        committees = _monroe_bruteforce(profile, committeesize, resolute)
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_monroe"
        )

    # optional output
    if verbose:
        print("Optimal Monroe score: " + str(scores.monroescore(profile, committees[0])) + "\n")

        print(str_committees_header(committees, winning=True))
        print(str_sets_of_candidates(committees, cand_names=profile.cand_names))
    # end of optional output

    return committees


def _monroe_bruteforce(profile, committeesize, resolute):
    """A brute-force algorithm for computing Monroe's rule"""
    opt_committees = []
    opt_monroescore = -1
    for committee in combinations(profile.candidates, committeesize):
        score = scores.monroescore(profile, committee)
        if score > opt_monroescore:
            opt_committees = [committee]
            opt_monroescore = score
        elif scores.monroescore(profile, committee) == opt_monroescore:
            opt_committees.append(committee)

    committees = sorted_committees(opt_committees)
    if resolute:
        committees = [committees[0]]

    return committees


def compute_greedy_monroe(profile, committeesize, algorithm="standard", resolute=True, verbose=0):
    """"Greedy Monroe"""
    check_enough_approved_candidates(profile, committeesize)
    if not profile.has_unit_weights():
        raise ValueError(
            rules["greedy-monroe"].shortname + " is only defined for unit weights (weight=1)"
        )

    if not resolute:
        raise NotImplementedError("compute_greedy_monroe does not support resolute=False.")

    if algorithm == "fastest":
        algorithm = rules["greedy-monroe"].fastest_algo()

    if algorithm != "standard":
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_greedy_monroe"
        )

    num_voters = len(profile)
    committee = []

    remaining_voters = list(range(num_voters))
    remaining_cands = set(profile.candidates)

    assignment = []
    for t in range(committeesize):
        maxapprovals = -1
        selected = None
        for cand in remaining_cands:
            approvals = len([i for i in remaining_voters if cand in profile[i].approved])
            if approvals > maxapprovals:
                maxapprovals = approvals
                selected = cand

        # determine how many voters are removed (at most)
        if t < num_voters - committeesize * (num_voters // committeesize):
            num_remove = num_voters // committeesize + 1
        else:
            num_remove = num_voters // committeesize

        # only voters that approve the chosen candidate
        # are removed
        to_remove = [i for i in remaining_voters if selected in profile[i].approved]
        if len(to_remove) > num_remove:
            to_remove = to_remove[:num_remove]
        assignment.append((selected, to_remove))
        remaining_voters = [i for i in remaining_voters if i not in to_remove]
        committee.append(selected)
        remaining_cands.remove(selected)

    committees = sorted_committees([committee])

    # optional output
    if verbose:
        print(header(rules["greedy-monroe"].longname))

    if verbose >= 2:
        score1 = scores.monroescore(profile, committees[0])

        score2 = len(profile) - len(remaining_voters)
        print("The Monroe assignment computed by Greedy Monroe")
        print("has a Monroe score of " + str(score2) + ".")

        if score1 > score2:
            print(
                "Monroe assignment found by Greedy Monroe is not "
                + "optimal for the winning committee,"
            )
            print(
                "i.e., by redistributing voters to candidates a higher "
                + "satisfaction is possible "
                + "(without changing the committee)."
            )
            print("Optimal Monroe score of the winning committee is " + str(score1) + ".")

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
            print(" candidate " + profile.cand_names[cand] + " assigned to: ", end="")
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
        print(str_sets_of_candidates(committees, cand_names=profile.cand_names))
    # end of optional output

    return sorted_committees(committees)


def _seqphragmen_resolute(
    profile, committeesize, division, verbose=0, start_load=None, partial_committee=None
):
    """Algorithm for computing resolute seq-Phragmen  (1 winning committee)"""
    approvers_weight = {}
    for cand in profile.candidates:
        approvers_weight[cand] = sum(voter.weight for voter in profile if cand in voter.approved)

    load = start_load
    if load is None:
        load = {v: 0 for v, _ in enumerate(profile)}

    committee = partial_committee
    if partial_committee is None:
        committee = []  # build committees starting with the empty set

    for _ in range(len(committee), committeesize):
        approvers_load = {}
        for cand in profile.candidates:
            approvers_load[cand] = sum(
                voter.weight * load[v] for v, voter in enumerate(profile) if cand in voter.approved
            )
        new_maxload = [
            division(approvers_load[cand] + 1, approvers_weight[cand])
            if approvers_weight[cand] > 0
            else committeesize + 1
            for cand in profile.candidates
        ]
        # exclude committees already in the committee
        large = max(new_maxload) + 1
        for cand in profile.candidates:
            if cand in committee:
                new_maxload[cand] = large
        # find smallest maxload
        opt = min(new_maxload)
        next_cand = new_maxload.index(opt)
        # compute new loads and add new candidate
        for v, voter in enumerate(profile):
            if next_cand in voter.approved:
                load[v] = new_maxload[next_cand]
            else:
                load[v] = load[v]
        committee = sorted(committee + [next_cand])

        # optional output
        if verbose >= 2:
            output = "adding candidate number "
            output += str(len(committee)) + ": "
            output += profile.cand_names[next_cand] + "\n"
            output += " maximum load increased to "
            output += str(opt)
            print(output)
            print(" load distribution:")
            output = "  ("
            for v, _ in enumerate(profile):
                output += str(load[v]) + ", "
            print(output[:-2] + ")")
            tied_cands = [
                cand
                for cand in profile.candidates
                if cand > next_cand and new_maxload[cand] == opt
            ]
            if tied_cands:
                output = " tie broken in favor of " + profile.cand_names[next_cand]
                output += ",\n candidates " + str_set_of_candidates(tied_cands)
                output += " would increase the load to the same amount ("
                output += str(new_maxload) + ")"
                print(output)
            print()
        # end of optional output

    comm_loads = {tuple(committee): load}
    return [committee], comm_loads


def _seqphragmen_irresolute(
    profile, committeesize, division, start_load=None, partial_committee=None
):
    """Algorithm for computing irresolute seq-Phragmen (>=1 winning committees)"""
    approvers_weight = {}
    for cand in profile.candidates:
        approvers_weight[cand] = sum(voter.weight for voter in profile if cand in voter.approved)

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
            for cand in profile.candidates:
                approvers_load[cand] = sum(
                    voter.weight * load[v]
                    for v, voter in enumerate(profile)
                    if cand in voter.approved
                )
            new_maxload = [
                division(approvers_load[cand] + 1, approvers_weight[cand])
                if approvers_weight[cand] > 0
                else committeesize + 1
                for cand in profile.candidates
            ]
            # exclude committees already in the committee
            for cand in profile.candidates:
                if cand in committee:
                    new_maxload[cand] = sys.maxsize
            # compute new loads
            # and add new committees
            for cand in profile.candidates:
                if new_maxload[cand] <= min(new_maxload):
                    new_load = {}
                    for v, voter in enumerate(profile):
                        if cand in voter.approved:
                            new_load[v] = new_maxload[cand]
                        else:
                            new_load[v] = load[v]
                    new_comm = tuple(sorted(committee + (cand,)))
                    comm_loads_next[new_comm] = new_load
        comm_loads = comm_loads_next

    committees = sorted_committees(list(comm_loads.keys()))
    return committees, comm_loads


def compute_seqphragmen(
    profile, committeesize, algorithm="standard", resolute=True, verbose=False
):
    """Phragmen's sequential rule (seq-Phragmen)"""
    check_enough_approved_candidates(profile, committeesize)

    if algorithm == "fastest":
        algorithm = rules["seqphragmen"].fastest_algo()

    if algorithm == "standard":
        division = lambda x, y: x / y  # standard float division
    elif algorithm == "exact-fractions":
        division = Fraction  # using exact fractions
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_seqphragmen"
        )

    # optional output
    if verbose:
        print(header(rules["seqphragmen"].longname))
        if resolute:
            print("Computing only one winning committee (resolute=True)\n")
    # end of optional output

    if resolute:
        committees, comm_loads = _seqphragmen_resolute(
            profile, committeesize, division, verbose=verbose
        )
    else:
        committees, comm_loads = _seqphragmen_irresolute(profile, committeesize, division)

    # optional output
    if verbose:
        print(str_committees_header(committees, winning=True))
        print(str_sets_of_candidates(committees, cand_names=profile.cand_names))
    if verbose >= 2:
        if resolute or len(committees) == 1:
            print("corresponding load distribution:")
        else:
            print("corresponding load distributions:")
        for committee in committees:
            output = "("
            for v, _ in enumerate(profile):
                output += str(comm_loads[tuple(committee)][v]) + ", "
            print(output[:-2] + ")")
    # end of optional output

    return sorted_committees(committees)


def _rule_x_get_min_q(profile, budget, cand, division):
    rich = set([v for v, voter in enumerate(profile) if cand in voter.approved])
    poor = set()

    while len(rich) > 0:
        poor_budget = sum(budget[v] for v in poor)
        q = division(1 - poor_budget, len(rich))
        new_poor = set([v for v in rich if budget[v] < q])
        if len(new_poor) == 0:
            return q
        rich -= new_poor
        poor.update(new_poor)

    return None  # not sufficient budget available


def compute_rule_x(
    profile,
    committeesize,
    algorithm="standard",
    resolute=True,
    verbose=0,
    skip_phragmen_phase=False,
):
    """Rule X

    See https://arxiv.org/pdf/1911.11747.pdf, page 7

    skip_phragmen_phase : bool, optional
        omit the second phase (that uses seq-Phragmen)
        may result in a committee that is too small
    """
    check_enough_approved_candidates(profile, committeesize)
    if not profile.has_unit_weights():
        raise ValueError(
            rules["rule-x"].shortname + " is only defined for unit weights (weight=1)"
        )

    if algorithm == "fastest":
        algorithm = rules["rule-x"].fastest_algo()

    if algorithm == "standard":
        division = lambda x, y: x / y  # standard float division
    elif algorithm == "exact-fractions":
        division = Fraction  # using exact fractions
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_rule_x"
        )

    # optional output
    if verbose:
        print(header(rules["rule-x"].longname))
        if resolute:
            print("Computing only one winning committee (resolute=True)\n")
    # end of optional output

    start_budget = {v: division(committeesize, len(profile)) for v, _ in enumerate(profile)}
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

            curr_cands = set(profile.candidates) - committee
            min_q = {}
            for cand in curr_cands:
                q = _rule_x_get_min_q(profile, budget, cand, division)
                if q is not None:
                    min_q[cand] = q

            if len(min_q) > 0:  # one or more candidates are affordable
                next_cands = [cand for cand in min_q.keys() if min_q[cand] == min(min_q.values())]
                for next_cand in next_cands:
                    new_budget = dict(budget)
                    for v, voter in enumerate(profile):
                        if next_cand in voter.approved:
                            new_budget[v] -= min(budget[v], min_q[next_cand])
                    new_comm = set(committee)
                    new_comm.add(next_cand)
                    next_commbudgets.append((new_comm, new_budget))

                    # optional output
                    if resolute and verbose >= 2:
                        output = "adding candidate number "
                        output += str(len(committee)) + ": "
                        output += profile.cand_names[next_cand] + "\n"
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
                            output += profile.cand_names[next_cand] + ","
                            output += "\n candidates "
                            output += str_set_of_candidates(next_cands[1:])
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
                        start_load[v] = division(committeesize, len(profile)) - budget[v]

                    # optional output
                    if resolute and verbose >= 2:
                        print("starting loads (= budget spent):")
                        output = "  ("
                        for v, _ in enumerate(profile):
                            output += str(start_load[v]) + ", "
                        print(output[:-2] + ")\n")
                    # end of optional output

                    if resolute:
                        committees, _ = _seqphragmen_resolute(
                            profile,
                            committeesize,
                            division,
                            verbose=verbose,
                            partial_committee=list(committee),
                            start_load=start_load,
                        )
                    else:
                        committees, _ = _seqphragmen_irresolute(
                            profile,
                            committeesize,
                            division,
                            partial_committee=list(committee),
                            start_load=start_load,
                        )
                    final_committees.update([tuple(committee) for committee in committees])
                    # after filling the remaining spots these committees
                    # have size committeesize

            commbugdets = next_commbudgets

    final_committees.update([tuple(committee) for committee, _ in commbugdets])

    committees = sorted_committees(final_committees)
    if resolute:
        committees = committees[:1]

    # optional output
    if verbose:
        print(str_committees_header(committees, winning=True))
        print(str_sets_of_candidates(committees, cand_names=profile.cand_names))
    # end of optional output

    return sorted_committees(committees)


def compute_rule_x_without_2nd_phase(
    profile, committeesize, algorithm="standard", resolute=True, verbose=0
):
    """Rule X with skip_phragmen_phase=True"""
    return compute_rule_x(
        profile,
        committeesize,
        algorithm,
        resolute=resolute,
        verbose=verbose,
        skip_phragmen_phase=True,
    )


def compute_optphragmen(profile, committeesize, algorithm="gurobi", resolute=False, verbose=0):
    """opt-Phragmen

    Warning: does not include the lexicographic optimization as specified
    in Markus Brill, Rupert Freeman, Svante Janson and Martin Lackner.
    Phragmen's Voting Methods and Justified Representation.
    http://martin.lackner.xyz/publications/phragmen.pdf

    Instead: minimizes the maximum load (without consideration of the
             second-, third-, ...-largest load
    """
    check_enough_approved_candidates(profile, committeesize)

    # optional output
    if verbose:
        print(header(rules["optphragmen"].longname))
        if resolute:
            print("Computing only one winning committee (resolute=True)\n")
    if verbose >= 3:
        if algorithm == "gurobi":
            print("Using the Gurobi ILP solver")
    # end of optional output

    if algorithm == "fastest":
        algorithm = rules["optphragmen"].fastest_algo()

    if algorithm != "gurobi":
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_optphragmen"
        )

    committees = abcrules_gurobi._gurobi_optphragmen(
        profile, committeesize, resolute=resolute, verbose=verbose
    )
    committees = sorted_committees(committees)

    # optional output
    if verbose:
        print(str_committees_header(committees, winning=True))
        print(str_sets_of_candidates(committees, cand_names=profile.cand_names))
    # end of optional output

    return committees


def compute_phragmen_enestroem(
    profile, committeesize, algorithm="standard", resolute=True, verbose=0
):
    """Phragmen-Enestroem (aka Phragmen's first method, Enestroem's method)

    In every round the candidate with the highest combined budget of
    their supporters is put in the committee.
    Method described in:
    https://arxiv.org/pdf/1611.08826.pdf (Section 18.5, Page 59)
    """
    check_enough_approved_candidates(profile, committeesize)
    if not profile.has_unit_weights():
        raise ValueError(
            rules["phragmen-enestroem"].shortname + " is only defined for unit weights (weight=1)"
        )

    if algorithm == "fastest":
        algorithm = rules["phragmen-enestroem"].fastest_algo()

    if algorithm == "standard":
        division = lambda x, y: x / y  # standard float division
    elif algorithm == "exact-fractions":
        division = Fraction  # using exact fractions
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_phragmen_enestroem"
        )

    num_voters = len(profile)

    initial_voter_budget = {i: profile[i].weight for i in range(num_voters)}

    # price for adding a candidate to the committee
    price = division(sum(initial_voter_budget.values()), committeesize)

    voter_budgets_for_partial_committee = [(initial_voter_budget, set())]
    for _ in range(committeesize):
        next_committees = []
        for committee in voter_budgets_for_partial_committee:
            budget, committee = committee
            curr_cands = set(profile.candidates) - committee
            support = {cand: 0 for cand in curr_cands}
            for nr, voter in enumerate(profile):
                voting_power = budget[nr]
                if voting_power <= 0:
                    continue
                for cand in voter.approved:
                    if cand in curr_cands:
                        support[cand] += voting_power
            max_support = max(support.values())
            winners = [c for c, s in support.items() if s == max_support]
            for cand in winners:
                new_budget = dict(budget)  # copy of budget
                if max_support > price:  # supporters can afford it
                    # (voting_power - price) / voting_power
                    multiplier = division(max_support - price, max_support)
                else:  # supporters can't afford it, set budget to 0
                    multiplier = 0
                for nr, voter in enumerate(profile):
                    if cand in voter.approved:
                        new_budget[nr] *= multiplier
                next_committees.append((new_budget, committee.union([cand])))

        if resolute:
            voter_budgets_for_partial_committee = [next_committees[0]]
        else:
            voter_budgets_for_partial_committee = next_committees

    # get rid of duplicates
    committees = set(
        [tuple(sorted(committee)) for _, committee in voter_budgets_for_partial_committee]
    )
    # sort committees
    committees = sorted_committees(set(committee) for committee in committees)
    if resolute:
        committees = [committees[0]]

    # optional output
    if verbose:
        print(header(rules["phragmen-enestroem"].longname))
        print(str_committees_header(committees, winning=True))
        print(str_sets_of_candidates(committees, cand_names=profile.cand_names))
    # end of optional output

    return committees


def compute_consensus_rule(profile, committeesize, algorithm="standard", resolute=True, verbose=0):
    """Consensus rule,
    based on Perpetual Consensus from
    Martin Lackner Perpetual Voting: Fairness in Long-Term Decision Making
    In Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI 2020)
    """
    check_enough_approved_candidates(profile, committeesize)

    if algorithm == "fastest":
        algorithm = rules["consensus-rule"].fastest_algo()

    if algorithm == "standard":
        division = lambda x, y: x / y  # standard float division
    elif algorithm == "exact-fractions":
        division = Fraction  # using exact fractions
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_consensus_rule"
        )

    num_voters = len(profile)

    initial_voter_budget = {i: 0 for i in range(num_voters)}

    voter_budgets_for_partial_committee = [(initial_voter_budget, set())]
    for _ in range(committeesize):
        next_committees = []
        for budget, committee in voter_budgets_for_partial_committee:
            for i in range(num_voters):
                budget[i] += profile[i].weight  # weight is 1 by default
            available_candidates = set(profile.candidates) - committee
            support = {cand: 0 for cand in available_candidates}
            supporters = {cand: [] for cand in available_candidates}
            for i, voter in enumerate(profile):
                if budget[i] <= 0:
                    continue
                for cand in voter.approved:
                    if cand in available_candidates:
                        support[cand] += budget[i]
                        supporters[cand].append(i)
            max_support = max(support.values())
            winners = [c for c, s in support.items() if s == max_support]
            for cand in winners:
                new_budget = dict(budget)  # copy of budget
                for i in supporters[cand]:
                    new_budget[i] -= division(num_voters, len(supporters[cand]))
                next_committees.append((new_budget, committee.union([cand])))

        if resolute:
            voter_budgets_for_partial_committee = [next_committees[0]]
        else:
            voter_budgets_for_partial_committee = next_committees

    # get rid of duplicates
    committees = set(
        [tuple(sorted(committee)) for _, committee in voter_budgets_for_partial_committee]
    )
    # sort committees
    committees = sorted_committees(set(committee) for committee in committees)
    if resolute:
        committees = [committees[0]]

    # optional output
    if verbose:
        print(header(rules["consensus-rule"].longname))
        print(str_committees_header(committees, winning=True))
        print(str_sets_of_candidates(committees, cand_names=profile.cand_names))
    # end of optional output

    return committees


rules = _init_rules()
