# -*- coding: utf-8 -*-
"""Approval-based committee (ABC) voting rules."""

import functools
from itertools import combinations
from collections import namedtuple
from abcvoting.output import output
from abcvoting import abcrules_gurobi, abcrules_ortools, abcrules_cvxpy, abcrules_mip
from abcvoting.misc import sorted_committees
from abcvoting.misc import hamming
from abcvoting.misc import check_enough_approved_candidates
from abcvoting.misc import str_committees_header, header
from abcvoting.misc import str_set_of_candidates, str_sets_of_candidates
from abcvoting import scores

try:
    from gmpy2 import mpq as Fraction
except ImportError:
    output.warning("Warning: module gmpy2 not found, resorting to Python's fractions.Fraction")
    from fractions import Fraction

########################################################################


MAIN_RULE_IDS = [
    "av",
    "sav",
    "pav",
    "slav",
    "cc",
    "geom2",
    "seqpav",
    "revseqpav",
    "seqslav",
    "seqcc",
    "seqphragmen",
    "minimaxphragmen",
    "monroe",
    "greedy-monroe",
    "minimaxav",
    "lexminimaxav",
    "rule-x",
    "rule-x-without-2nd-phase",
    "phragmen-enestroem",
    "consensus-rule",
]


RuleInfo = namedtuple(
    "RuleInfo", ["shortname", "longname", "function", "algorithms", "resolute_values"]
)


class UnknownRuleIDError(ValueError):
    """UnknownRuleIDError exception raised if an unknown rule id is used."""

    def __init__(self, rule_id):
        message = 'Rule ID "' + str(rule_id) + '" is not known.'
        super(ValueError, self).__init__(message)


def is_algorithm_available(algorithm):
    """Verify whether algorithm `algorithm` is supported on the current machine.

    In particular, the functions verifies that required modules and solvers are available.
    """
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


def fastest_algo(rule_id):
    """Return fastest *available* algorithm for ABC rule `rule_id`.

    This function rests on the assumption that the available algorithms returned by
    get_ruleinfo() are sorted by speed."""
    algorithms = get_algorithms(rule_id)
    for algorithm in algorithms:
        if is_algorithm_available(algorithm):
            return algorithm
    raise RuntimeError(
        f"No algorithm available for {rule_id} "
        f"that is supported on this machine. "
        f"({', '.join(algorithms)} not supported)",
    )


def get_ruleinfo(rule_id):
    _THIELE_ALGORITHMS = (
        # TODO sort by speed, requires testing
        "gurobi",
        "branch-and-bound",
        "mip_cbc",
        "mip_gurobi",
        "brute-force",
    )
    _RULESINFO = {
        "av": RuleInfo("AV", "Approval Voting (AV)", compute_av, ("standard",), (True, False)),
        "sav": RuleInfo(
            "SAV",
            "Satisfaction Approval Voting (SAV)",
            compute_sav,
            ("standard",),
            (True, False),
        ),
        "pav": RuleInfo(
            "PAV",
            "Proportional Approval Voting (PAV)",
            compute_pav,
            # TODO sort by speed, requires testing
            _THIELE_ALGORITHMS + ("cvxpy_glpk_mi", "cvxpy_cbc", "cvxpy_scip"),
            (True, False),
        ),
        "slav": RuleInfo(
            "SLAV",
            "Sainte-Laguë Approval Voting (SLAV)",
            compute_slav,
            _THIELE_ALGORITHMS,
            (True, False),
        ),
        "cc": RuleInfo(
            "CC",
            "Approval Chamberlin-Courant (CC)",
            compute_cc,
            # TODO sort by speed, requires testing
            _THIELE_ALGORITHMS + ("ortools_cp",),
            (True, False),
        ),
        "seqpav": RuleInfo(
            "seq-PAV",
            "Sequential Proportional Approval Voting (seq-PAV)",
            compute_seqpav,
            ("standard",),
            (True, False),
        ),
        "revseqpav": RuleInfo(
            "revseq-PAV",
            "Reverse Sequential Proportional Approval Voting (revseq-PAV)",
            compute_revseqpav,
            ("standard",),
            (True, False),
        ),
        "seqslav": RuleInfo(
            "seq-SLAV",
            "Sequential Sainte-Laguë Approval Voting (seq-SLAV)",
            compute_seqslav,
            ("standard",),
            (True, False),
        ),
        "seqcc": RuleInfo(
            "seq-CC",
            "Sequential Approval Chamberlin-Courant (seq-CC)",
            compute_seqcc,
            ("standard",),
            (True, False),
        ),
        "seqphragmen": RuleInfo(
            "seq-Phragmén",
            "Phragmén's Sequential Rule (seq-Phragmén)",
            compute_seqphragmen,
            ("standard", "exact-fractions"),
            (True, False),
        ),
        "minimaxphragmen": RuleInfo(
            "minimax-Phragmén",
            "Phragmén's Minimax Rule (minimax-Phragmén)",
            compute_minimaxphragmen,
            ("gurobi", "mip_gurobi", "mip_cbc"),
            (True, False),
        ),
        "monroe": RuleInfo(
            "Monroe",
            "Monroe's Approval Rule (Monroe)",
            compute_monroe,
            ("gurobi", "mip_gurobi", "mip_cbc", "ortools_cp", "brute-force"),
            (True, False),
        ),
        "greedy-monroe": RuleInfo(
            "Greedy Monroe",
            "Greedy Monroe",
            compute_greedy_monroe,
            ("standard",),
            (True,),
        ),
        "minimaxav": RuleInfo(
            "minimaxav",
            "Minimax Approval Voting (MAV)",
            compute_minimaxav,
            # TODO sort by speed, requires testing
            ("gurobi", "ortools_cp", "brute-force", "mip_gurobi", "mip_cbc"),
            (True, False),
        ),
        "lexminimaxav": RuleInfo(
            "lex-MAV",
            "Lexicographic Minimax Approval Voting (lex-MAV)",
            compute_lexminimaxav,
            ("brute-force",),
            (True, False),
        ),
        "rule-x": RuleInfo(
            "Rule X",
            "Rule X",
            compute_rule_x,
            ("standard", "exact-fractions"),
            (True, False),
        ),
        "rule-x-without-2nd-phase": RuleInfo(
            "Rule X (without 2nd phase)",
            "Rule X without the second (Phragmén) phase",
            compute_rule_x_without_2nd_phase,
            ("standard", "exact-fractions"),
            (True, False),
        ),
        "phragmen-enestroem": RuleInfo(
            "Phragmén-Eneström",
            "Method of Phragmén-Eneström",
            compute_phragmen_enestroem,
            ("standard", "exact-fractions"),
            (True, False),
        ),
        "consensus-rule": RuleInfo(
            "Consensus Rule",
            "Consensus Rule",
            compute_consensus_rule,
            ("standard", "exact-fractions"),
            (True, False),
        ),
    }

    if rule_id in _RULESINFO:
        return _RULESINFO[rule_id]

    if rule_id.startswith("geom"):
        parameter = rule_id[4:]
        return RuleInfo(
            f"{parameter}-Geometric",
            f"{parameter}-Geometric Rule",
            functools.partial(compute_thiele_method, scorefct_id=rule_id),
            _THIELE_ALGORITHMS,
            (True, False),
        )

    # handle sequential and reverse sequential Thiele methods
    # that are not explicitly included in the list above
    if rule_id.startswith("seq") or rule_id.startswith("revseq"):
        if rule_id.startswith("seq"):
            scorefct_id = rule_id[3:]  # score function id of Thiele method
        else:
            scorefct_id = rule_id[6:]  # score function id of Thiele method

        try:
            scores.get_scorefct(scorefct_id)
        except scores.UnknownScoreFunctionError:
            raise UnknownRuleIDError(rule_id)

        if rule_id == "av":
            raise UnknownRuleIDError(rule_id)  # seq-AV and revseq-AV are equivalent to AV

        # sequential Thiele methods
        if rule_id.startswith("seq"):
            return RuleInfo(
                f"seq-{get_shortname(scorefct_id)}",
                f"Sequential {get_longname(scorefct_id)}",
                functools.partial(compute_seq_thiele_method, scorefct_id=scorefct_id),
                ("standard",),
                (True, False),
            )
        # reverse sequential Thiele methods
        if rule_id.startswith("revseq"):
            return RuleInfo(
                f"revseq-{get_shortname(scorefct_id)}",
                f"Reverse Sequential {get_longname(scorefct_id)}",
                functools.partial(compute_revseq_thiele_method, scorefct_id=scorefct_id),
                ("standard",),
                (True, False),
            )

    raise UnknownRuleIDError(rule_id)


def get_shortname(rule_id):
    return get_ruleinfo(rule_id).shortname


def get_longname(rule_id):
    return get_ruleinfo(rule_id).longname


def get_algorithms(rule_id):
    return get_ruleinfo(rule_id).algorithms


def get_available_algorithms(rule_id):
    return [
        algorithm for algorithm in get_algorithms(rule_id) if is_algorithm_available(algorithm)
    ]


def get_resolute_values(rule_id):
    return get_ruleinfo(rule_id).resolute_values


def _algorithm_fullnames(algorithm):
    msg = "Algorithm: "
    fullnames = {
        "gurobi": "Gurobi ILP solver",
        "branch-and-bound": "branch-and-bound",
        "brute-force": "brute-force",
        "mip_cbc": "CBC ILP solver via Python MIP library",
        "mip_gurobi": "Gurobi ILP solver via Python MIP library",
        "cvxpy_gurobi": "Gurobi ILP solver via CVXPY library",
        "cvxpy_scip": "SCIP ILP solver via CVXPY library",
        "cvxpy_glpk_mi": "GLPK ILP solver via CVXPY library",
        "cvxpy_cbc": "CBC ILP solver via CVXPY library",
        "standard": "Standard algorithm",
        "exact-fractions": "Standard algorithm with exact fractions",
        "ortools_cp": "OR-Tools CP-SAT solver",
    }
    try:
        return msg + fullnames[algorithm]
    except KeyError:
        raise ValueError(f"Algorithm {algorithm} not known.")


########################################################################


def compute(rule_id, profile, committeesize, **kwargs):
    """Compute rule given by `rule_id`."""
    return get_ruleinfo(rule_id).function(profile, committeesize, **kwargs)


def compute_thiele_method(
    profile, committeesize, scorefct_id, algorithm="branch-and-bound", resolute=False
):
    """Thiele methods.

    Compute winning committees according to a Thiele method specified
    by a score function (scorefct_id).
    Examples of Thiele methods are PAV, CC, and SLAV.
    An exception is Approval Voting (AV), which should be computed using
    compute_av(). (AV is polynomial-time computable (separable) and can thus be
    computed much faster.)
    """
    check_enough_approved_candidates(profile, committeesize)
    scorefct = scores.get_scorefct(scorefct_id, committeesize)

    if algorithm == "fastest":
        algorithm = fastest_algo(scorefct_id)

    if algorithm == "gurobi":
        committees = abcrules_gurobi._gurobi_thiele_methods(
            profile, committeesize, scorefct, resolute
        )
        committees = sorted_committees(committees)
    elif algorithm == "branch-and-bound":
        committees, detailed_info = _thiele_methods_branchandbound(
            profile, committeesize, scorefct_id, resolute
        )
    elif algorithm == "brute-force":
        committees, detailed_info = _thiele_methods_bruteforce(
            profile, committeesize, scorefct_id, resolute
        )
    elif algorithm.startswith("cvxpy_"):
        committees = abcrules_cvxpy.cvxpy_thiele_methods(
            profile=profile,
            committeesize=committeesize,
            scorefct_id=scorefct_id,
            resolute=resolute,
            solver_id=algorithm[6:],
        )
    elif algorithm.startswith("mip_"):
        committees = abcrules_mip._mip_thiele_methods(
            profile,
            committeesize,
            scorefct=scorefct,
            resolute=resolute,
            solver_id=algorithm[4:],
        )
    elif algorithm == "ortools_cp" and scorefct_id == "cc":
        committees = abcrules_ortools._ortools_cc(profile, committeesize, resolute)
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_thiele_method"
        )

    # optional output
    output.info(header(get_longname(scorefct_id)))
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(_algorithm_fullnames(algorithm) + "\n")

    output.details(
        f"Optimal {scorefct_id.upper()}-score: "
        f"{scores.thiele_score(scorefct_id, profile, committees[0])}\n"
    )

    output.info(str_committees_header(committees, winning=True))
    output.info(str_sets_of_candidates(committees, cand_names=profile.cand_names))
    # end of optional output

    return committees


def _thiele_methods_bruteforce(profile, committeesize, scorefct_id, resolute):
    """Brute-force algorithm for Thiele methods (PAV, CC, etc.).

    Only intended for comparison, much slower than _thiele_methods_branchandbound()
    """
    opt_committees = []
    opt_thiele_score = -1
    for committee in combinations(profile.candidates, committeesize):
        score = scores.thiele_score(scorefct_id, profile, committee)
        if score > opt_thiele_score:
            opt_committees = [committee]
            opt_thiele_score = score
        elif score == opt_thiele_score:
            opt_committees.append(committee)

    committees = sorted_committees(opt_committees)
    detailed_info = {}
    if resolute:
        committees = [committees[0]]
    return committees, detailed_info


def _thiele_methods_branchandbound(profile, committeesize, scorefct_id, resolute):
    """Branch-and-bound algorithm for Thiele methods."""
    scorefct = scores.get_scorefct(scorefct_id, committeesize)

    best_committees = []
    init_com, _ = _seq_thiele_resolute(profile, committeesize, scorefct_id)
    init_com = init_com[0]
    best_score = scores.thiele_score(scorefct_id, profile, init_com)
    part_coms = [[]]
    while part_coms:
        part_com = part_coms.pop(0)
        # potential committee, check if at least as good
        # as previous best committee
        if len(part_com) == committeesize:
            score = scores.thiele_score(scorefct_id, profile, part_com)
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
            ) + scores.thiele_score(scorefct_id, profile, part_com)
            if upper_bound >= best_score:
                for cand in range(largest_cand + 1, profile.num_cand - missing + 1):
                    part_coms.insert(0, part_com + [cand])

    committees = sorted_committees(best_committees)
    if resolute:
        committees = [committees[0]]

    detailed_info = {}
    return committees, detailed_info


def compute_pav(profile, committeesize, algorithm="branch-and-bound", resolute=False):
    """Proportional Approval Voting (PAV)."""
    return compute_thiele_method(
        profile, committeesize, "pav", algorithm=algorithm, resolute=resolute
    )


def compute_slav(profile, committeesize, algorithm="branch-and-bound", resolute=False):
    """Sainte-Lague Approval Voting (SLAV)."""
    return compute_thiele_method(
        profile, committeesize, "slav", algorithm=algorithm, resolute=resolute
    )


def compute_cc(profile, committeesize, algorithm="branch-and-bound", resolute=False):
    """Approval Chamberlin-Courant (CC)."""
    return compute_thiele_method(
        profile, committeesize, "cc", algorithm=algorithm, resolute=resolute
    )


def compute_seq_thiele_method(
    profile, committeesize, scorefct_id, algorithm="standard", resolute=True
):
    """Sequential Thiele methods."""
    check_enough_approved_candidates(profile, committeesize)
    scores.get_scorefct(scorefct_id, committeesize)  # check that scorefct_id is valid

    if algorithm == "fastest":
        algorithm = fastest_algo("seq" + scorefct_id)
    if algorithm != "standard":
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_seq_thiele_method"
        )

    if resolute:
        committees, detailed_info = _seq_thiele_resolute(profile, committeesize, scorefct_id)
    else:
        committees, detailed_info = _seq_thiele_irresolute(profile, committeesize, scorefct_id)

    # optional output
    output.info(header(get_longname("seq" + scorefct_id)))
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(_algorithm_fullnames(algorithm) + "\n")
    output.details(
        f"starting with the empty committee (score = "
        f"{scores.thiele_score(scorefct_id, profile, [])})\n"
    )
    if resolute:
        committee = []
        for i, next_cand in enumerate(detailed_info["next_cand"]):
            tied_cands = detailed_info["tied_cands"][i]
            delta_score = detailed_info["delta_score"][i]
            committee.append(next_cand)
            output.details(f"adding candidate number {i+1}: {profile.cand_names[next_cand]}")
            output.details(
                f" score increases by {delta_score} to"
                f" a total of {scores.thiele_score(scorefct_id, profile, committee)}"
            )
            if tied_cands:
                output.details(f" tie broken in favor of {next_cand}")
                output.details(
                    f" candidates "
                    f"{str_set_of_candidates(tied_cands, cand_names=profile.cand_names)}"
                    f" would increase the score by the same amount {delta_score})"
                )
            output.details("")

    output.info(str_committees_header(committees, winning=True))
    output.info(str_sets_of_candidates(committees, cand_names=profile.cand_names))

    output.details(scorefct_id.upper() + "-score of winning committee(s):")
    for committee in committees:
        output.details(
            f" {str_set_of_candidates(committee, cand_names=profile.cand_names)}: "
            f"{scores.thiele_score(scorefct_id, profile, committee)}"
        )
    output.details("\n")
    # end of optional output

    return sorted_committees(committees)


def _seq_thiele_resolute(profile, committeesize, scorefct_id):
    """Compute one winning committee (=resolute) for sequential Thiele methods.

    Tiebreaking between candidates in favor of candidate with smaller
    number/index (candidates with larger numbers get deleted first).
    """
    committee = []
    scorefct = scores.get_scorefct(scorefct_id, committeesize)
    detailed_info = {"next_cand": [], "tied_cands": [], "delta_score": []}

    # build a committee starting with the empty set
    for _ in range(committeesize):
        additional_score_cand = scores.marginal_thiele_scores_add(scorefct, profile, committee)
        next_cand = additional_score_cand.index(max(additional_score_cand))
        committee.append(next_cand)
        tied_cands = [
            cand
            for cand in range(len(additional_score_cand))
            if (cand > next_cand and additional_score_cand[cand] == max(additional_score_cand))
        ]
        detailed_info["next_cand"].append(next_cand)
        detailed_info["tied_cands"].append(tied_cands)
        detailed_info["delta_score"].append(max(additional_score_cand))

    return sorted_committees([committee]), detailed_info


def _seq_thiele_irresolute(profile, committeesize, scorefct_id):
    """Compute all winning committee (=irresolute) for sequential Thiele methods.

    Consider all possible ways to break ties between candidates
    (aka parallel universe tiebreaking)
    """
    scorefct = scores.get_scorefct(scorefct_id, committeesize)

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
    detailed_info = {}
    return sorted_committees(list(comm_scores.keys())), detailed_info


# Sequential PAV
def compute_seqpav(profile, committeesize, algorithm="standard", resolute=True):
    """Sequential PAV (seq-PAV)."""
    return compute_seq_thiele_method(
        profile, committeesize, "pav", algorithm=algorithm, resolute=resolute
    )


def compute_seqslav(profile, committeesize, algorithm="standard", resolute=True):
    """Sequential Sainte-Lague Approval Voting (SLAV)."""
    return compute_seq_thiele_method(
        profile, committeesize, "slav", algorithm=algorithm, resolute=resolute
    )


def compute_seqcc(profile, committeesize, algorithm="standard", resolute=True):
    """Sequential Chamberlin-Courant (seq-CC)."""
    return compute_seq_thiele_method(
        profile, committeesize, "cc", algorithm=algorithm, resolute=resolute
    )


def compute_revseq_thiele_method(
    profile, committeesize, scorefct_id, algorithm="standard", resolute=True
):
    """Reverse sequential Thiele methods."""
    check_enough_approved_candidates(profile, committeesize)
    scores.get_scorefct(scorefct_id, committeesize)  # check that scorefct_id is valid

    if algorithm == "fastest":
        algorithm = fastest_algo("revseq" + scorefct_id)
    if algorithm != "standard":
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_revseq_thiele_method"
        )

    if resolute:
        committees, detailed_info = _revseq_thiele_resolute(profile, committeesize, scorefct_id)
    else:
        committees, detailed_info = _revseq_thiele_irresolute(profile, committeesize, scorefct_id)

    # optional output
    output.info(header(get_longname("revseq" + scorefct_id)))
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(_algorithm_fullnames(algorithm) + "\n")

    if resolute:
        committee = set(profile.candidates)
        output.details(
            f"full committee ({len(committee)} candidates) has a total score of "
            f"{scores.thiele_score(scorefct_id, profile, committee)}\n"
        )
        for i, next_cand in enumerate(detailed_info["next_cand"]):
            committee.remove(next_cand)
            tied_cands = detailed_info["tied_cands"][i]
            delta_score = detailed_info["delta_score"][i]
            output.details(
                f"removing candidate number {profile.num_cand - len(committee)}: "
                f"{profile.cand_names[next_cand]}"
            )
            output.details(
                f" score decreases by {delta_score} to a total of "
                f"{scores.thiele_score(scorefct_id, profile, committee)}"
            )
            if tied_cands:
                output.details(
                    f" (tie between candidates "
                    f"{str_set_of_candidates(tied_cands, cand_names=profile.cand_names)}"
                )
                output.details(f" tie broken to the disadvantage of {next_cand}")
                output.details(
                    f" candidates "
                    f"{str_set_of_candidates(tied_cands, cand_names=profile.cand_names)}"
                    f" would decrease the score by the same amount {delta_score})"
                )
            output.details("")

    output.info(str_committees_header(committees, winning=True))
    output.info(str_sets_of_candidates(committees, cand_names=profile.cand_names))

    msg = "PAV-score of winning committee:"
    if not resolute and len(committees) != 1:
        msg += "\n"
    for committee in committees:
        msg += " " + str(scores.thiele_score(scorefct_id, profile, committee))
    msg += "\n"
    output.details(msg)
    # end of optional output

    return committees


def _revseq_thiele_resolute(profile, committeesize, scorefct_id):
    """Compute one winning committee (=resolute) for reverse sequential Thiele methods.

    Tiebreaking between candidates in favor of candidate with smaller
    number/index (candidates with smaller numbers are added first).
    """
    scorefct = scores.get_scorefct(scorefct_id, committeesize)
    committee = set(profile.candidates)

    detailed_info = {"next_cand": [], "tied_cands": [], "delta_score": []}

    for _ in range(profile.num_cand - committeesize):
        marg_util_cand = scores.marginal_thiele_scores_remove(scorefct, profile, committee)
        # find smallest elements in marg_util_cand and return indices
        cands_to_remove = [
            cand for cand in profile.candidates if marg_util_cand[cand] == min(marg_util_cand)
        ]
        next_cand = cands_to_remove[-1]
        tied_cands = cands_to_remove[:-1]
        committee.remove(next_cand)

        detailed_info["next_cand"].append(next_cand)
        detailed_info["tied_cands"].append(tied_cands)
        detailed_info["delta_score"].append(min(marg_util_cand))

    return sorted_committees([committee]), detailed_info


def _revseq_thiele_irresolute(profile, committeesize, scorefct_id):
    """Compute all winning committee (=irresolute) for reverse sequential Thiele methods.

    Consider all possible ways to break ties between candidates
    (aka parallel universe tiebreaking)
    """
    scorefct = scores.get_scorefct(scorefct_id, committeesize)

    allcandcomm = tuple(profile.candidates)
    comm_scores = {allcandcomm: scores.thiele_score(scorefct_id, profile, allcandcomm)}

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

    detailed_info = {}
    return sorted_committees(list(comm_scores.keys())), detailed_info


# Reverse Sequential PAV
def compute_revseqpav(profile, committeesize, algorithm="standard", resolute=True):
    """Reverse sequential PAV (revseq-PAV)."""
    return compute_revseq_thiele_method(
        profile, committeesize, "pav", algorithm=algorithm, resolute=resolute
    )


def compute_separable_rule(profile, committeesize, rule_id, algorithm, resolute=True):
    """Separable rules (such as AV and SAV)."""
    check_enough_approved_candidates(profile, committeesize)

    if algorithm == "fastest":
        algorithm = fastest_algo("av")
    if algorithm == "standard":
        pass
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_separable_rule"
        )

    committees, detailed_info = _separable_rule_algorithm(
        profile, committeesize, rule_id, resolute
    )

    # optional output
    output.info(header(get_longname(rule_id)))
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(_algorithm_fullnames(algorithm) + "\n")

    score = detailed_info["score"]
    msg = "Scores of candidates:\n"
    for cand in profile.candidates:
        msg += (profile.cand_names[cand] + ": " + str(score[cand])) + "\n"

    cutoff = detailed_info["cutoff"]
    msg += "\nCandidates are contained in winning committees\n"
    msg += "if their score is >= " + str(cutoff) + "."
    output.details(msg)

    certain_cands = detailed_info["certain_cands"]
    if len(certain_cands) > 0:
        msg = "\nThe following candidates are contained in\n"
        msg += "every winning committee:\n"
        namedset = [profile.cand_names[cand] for cand in certain_cands]
        msg += (" " + ", ".join(map(str, namedset))) + "\n"
        output.details(msg)

    possible_cands = detailed_info["possible_cands"]
    missing = detailed_info["missing"]
    if len(possible_cands) > 0:
        msg = "The following candidates are contained in\n"
        msg += "some of the winning committees:\n"
        namedset = [profile.cand_names[cand] for cand in possible_cands]
        msg += (" " + ", ".join(map(str, namedset))) + "\n"
        msg += f"({missing} of those candidates are contained\n in every winning committee.)\n"
        output.details(msg)

    output.info(str_committees_header(committees, winning=True))
    output.info(str_sets_of_candidates(committees, cand_names=profile.cand_names))
    # end of optional output

    return committees


def _separable_rule_algorithm(profile, committeesize, rule_id, resolute):
    """Algorithm for separable rules (such as AV and SAV)."""
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
    detailed_info = {
        "certain_cands": certain_cands,
        "possible_cands": possible_cands,
        "missing": missing,
        "cutoff": cutoff,
        "score": score,
    }
    return committees, detailed_info


def compute_sav(profile, committeesize, algorithm="standard", resolute=False):
    """Satisfaction Approval Voting (SAV)."""
    return compute_separable_rule(profile, committeesize, "sav", algorithm, resolute)


def compute_av(profile, committeesize, algorithm="standard", resolute=False):
    """Approval Voting (AV)."""
    return compute_separable_rule(profile, committeesize, "av", algorithm, resolute)


def compute_minimaxav(profile, committeesize, algorithm="brute-force", resolute=False):
    """Minimax Approval Voting (MAV)."""
    check_enough_approved_candidates(profile, committeesize)

    if algorithm == "fastest":
        algorithm = fastest_algo("minimaxav")

    if algorithm == "gurobi":
        committees = abcrules_gurobi._gurobi_minimaxav(profile, committeesize, resolute)
        committees = sorted_committees(committees)
    elif algorithm == "ortools_cp":
        committees = abcrules_ortools._ortools_minimaxav(profile, committeesize, resolute)
        committees = sorted_committees(committees)
    elif algorithm.startswith("mip_"):
        solver_id = algorithm[4:]
        committees = abcrules_mip._mip_minimaxav(profile, committeesize, resolute, solver_id)
        committees = sorted_committees(committees)
    elif algorithm == "brute-force":
        committees, detailed_info = _minimaxav_bruteforce(profile, committeesize, resolute)
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_minimaxav"
        )

    # optional output
    output.info(header(get_longname("minimaxav")))
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(_algorithm_fullnames(algorithm) + "\n")

    opt_minimaxav_score = scores.mavscore(profile, committees[0])
    output.info(str_committees_header(committees, winning=True))
    output.info(str_sets_of_candidates(committees, cand_names=profile.cand_names))
    output.details("Minimum maximal distance: " + str(opt_minimaxav_score))
    msg = "Corresponding distances to voters:\n"
    for committee in committees:
        msg += str([hamming(voter.approved, committee) for voter in profile]) + "\n"
    output.details(msg)
    # end of optional output

    return committees


def _minimaxav_bruteforce(profile, committeesize, resolute):
    """Brute-force algorithm for Minimax AV (MAV)."""
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
    detailed_info = {}
    if resolute:
        committees = [committees[0]]
    return committees, detailed_info


def compute_lexminimaxav(profile, committeesize, algorithm="brute-force", resolute=False):
    """Lexicographic Minimax AV (lex-MAV)."""
    check_enough_approved_candidates(profile, committeesize)

    if not profile.has_unit_weights():
        raise ValueError(
            get_shortname("lexminimaxav") + " is only defined for unit weights (weight=1)"
        )

    if algorithm == "fastest":
        algorithm = fastest_algo("lexminimaxav")

    if algorithm == "brute-force":
        committees, detailed_info = _lexminimaxav_bruteforce(profile, committeesize, resolute)
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_lexminimaxav"
        )

    # optional output
    opt_distances = detailed_info["opt_distances"]
    output.info(header(get_longname("lexminimaxav")))
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(_algorithm_fullnames(algorithm) + "\n")
    output.info(str_committees_header(committees, winning=True))
    output.info(str_sets_of_candidates(committees, cand_names=profile.cand_names))
    output.details("Minimum maximal distance: " + str(max(opt_distances)))
    msg = "Corresponding distances to voters:\n"
    for committee in committees:
        msg += str([hamming(voter.approved, committee) for voter in profile])
    output.details(msg + "\n")
    # end of optional output

    return committees


def _lexminimaxav_bruteforce(profile, committeesize, resolute):
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
    detailed_info = {"opt_distances": opt_distances}
    if resolute:
        committees = [committees[0]]
    return committees, detailed_info


def compute_monroe(profile, committeesize, algorithm="brute-force", resolute=False):
    """Monroe's rule."""
    check_enough_approved_candidates(profile, committeesize)

    if not profile.has_unit_weights():
        raise ValueError(get_shortname("monroe") + " is only defined for unit weights (weight=1)")

    if algorithm == "fastest":
        algorithm = fastest_algo("monroe")

    if algorithm == "gurobi":
        committees = abcrules_gurobi._gurobi_monroe(profile, committeesize, resolute)
        committees = sorted_committees(committees)
    elif algorithm == "ortools_cp":
        committees = abcrules_ortools._ortools_monroe(profile, committeesize, resolute)
        committees = sorted_committees(committees)
    elif algorithm.startswith("mip_"):
        committees = abcrules_mip._mip_monroe(
            profile,
            committeesize,
            resolute=resolute,
            solver_id=algorithm[4:],
        )
    elif algorithm == "brute-force":
        committees, detailed_info = _monroe_bruteforce(profile, committeesize, resolute)
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_monroe"
        )

    # optional output
    output.info(header(get_longname("monroe")))
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(_algorithm_fullnames(algorithm) + "\n")
    output.info("Optimal Monroe score: " + str(scores.monroescore(profile, committees[0])) + "\n")

    output.info(str_committees_header(committees, winning=True))
    output.info(str_sets_of_candidates(committees, cand_names=profile.cand_names))
    # end of optional output

    return committees


def _monroe_bruteforce(profile, committeesize, resolute):
    """Brute-force algorithm for Monroe's rule."""
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

    detailed_info = {}
    return committees, detailed_info


def compute_greedy_monroe(profile, committeesize, algorithm="standard", resolute=True):
    """Greedy Monroe."""
    check_enough_approved_candidates(profile, committeesize)
    if not profile.has_unit_weights():
        raise ValueError(
            get_shortname("greedy-monroe") + " is only defined for unit weights (weight=1)"
        )

    if not resolute:
        raise NotImplementedError("compute_greedy_monroe does not support resolute=False.")

    if algorithm == "fastest":
        algorithm = fastest_algo("greedy-monroe")

    if algorithm != "standard":
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_greedy_monroe"
        )

    committees, detailed_info = _greedy_monroe_algorithm(profile, committeesize)

    # optional output
    output.info(header(get_longname("greedy-monroe")))
    output.details(_algorithm_fullnames(algorithm) + "\n")
    remaining_voters = detailed_info["remaining_voters"]
    assignment = detailed_info["assignment"]

    score1 = scores.monroescore(profile, committees[0])
    score2 = len(profile) - len(remaining_voters)
    output.details("The Monroe assignment computed by Greedy Monroe")
    output.details("has a Monroe score of " + str(score2) + ".")

    if score1 > score2:
        output.details(
            "Monroe assignment found by Greedy Monroe is not "
            + "optimal for the winning committee,"
        )
        output.details(
            "i.e., by redistributing voters to candidates a higher "
            + "satisfaction is possible "
            + "(without changing the committee)."
        )
        output.details("Optimal Monroe score of the winning committee is " + str(score1) + ".")

    # build actual Monroe assignment for winning committee
    num_voters = len(profile)
    for t, district in enumerate(assignment):
        cand, voters = district
        if t < num_voters - committeesize * (num_voters // committeesize):
            missing = num_voters // committeesize + 1 - len(voters)
        else:
            missing = num_voters // committeesize - len(voters)
        for _ in range(missing):
            v = remaining_voters.pop()
            voters.append(v)

    msg = "Assignment (unsatisfatied voters marked with *):\n\n"
    for cand, voters in assignment:
        msg += " candidate " + profile.cand_names[cand] + " assigned to: "
        assing_msg = ""
        for v in sorted(voters):
            assing_msg += str(v)
            if cand not in profile[v].approved:
                assing_msg += "*"
            assing_msg += ", "
        msg += assing_msg[:-2] + "\n"
    output.details(msg)

    output.info(str_committees_header(committees, winning=True))
    output.info(str_sets_of_candidates(committees, cand_names=profile.cand_names))
    # end of optional output

    return sorted_committees(committees)


def _greedy_monroe_algorithm(profile, committeesize):
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

    detailed_info = {"remaining_voters": remaining_voters, "assignment": assignment}
    return sorted_committees([committee]), detailed_info


def compute_seqphragmen(profile, committeesize, algorithm="standard", resolute=True):
    """Phragmen's sequential rule (seq-Phragmen)."""
    check_enough_approved_candidates(profile, committeesize)

    if algorithm == "fastest":
        algorithm = fastest_algo("seqphragmen")

    if algorithm == "standard":
        division = lambda x, y: x / y  # standard float division
    elif algorithm == "exact-fractions":
        division = Fraction  # using exact fractions
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_seqphragmen"
        )

    if resolute:
        committees, detailed_info = _seqphragmen_resolute(
            profile,
            committeesize,
            division,
        )
    else:
        committees, detailed_info = _seqphragmen_irresolute(profile, committeesize, division)

    # optional output
    output.info(header(get_longname("seqphragmen")))
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(_algorithm_fullnames(algorithm) + "\n")

    if resolute:
        committee = []
        for i, next_cand in enumerate(detailed_info["next_cand"]):
            tied_cands = detailed_info["tied_cands"][i]
            max_load = detailed_info["max_load"][i]
            load = detailed_info["load"][i]
            committee.append(next_cand)
            output.details(
                f"adding candidate number {i+1}: {profile.cand_names[next_cand]}\n"
                f" maximum load increased to {max_load}"
            )
            output.details(" load distribution:")
            msg = "  ("
            for v, _ in enumerate(profile):
                msg += str(load[v]) + ", "
            output.details(msg[:-2] + ")")
            if tied_cands:
                msg = " tie broken in favor of " + profile.cand_names[next_cand]
                msg += ",\n candidates " + str_set_of_candidates(
                    tied_cands, cand_names=profile.cand_names
                )
                msg += f" would increase the load to the same amount ({max_load})"
                output.details(msg)
            output.details("")

    output.info(str_committees_header(committees, winning=True))
    output.info(str_sets_of_candidates(committees, cand_names=profile.cand_names))

    if resolute or len(committees) == 1:
        output.details("corresponding load distribution:")
    else:
        output.details("corresponding load distributions:")
    for committee in committees:
        msg = "("
        for v, _ in enumerate(profile):
            msg += str(detailed_info["comm_loads"][tuple(committee)][v]) + ", "
        output.details(msg[:-2] + ")")
    # end of optional output

    return sorted_committees(committees)


def _seqphragmen_resolute(
    profile, committeesize, division, start_load=None, partial_committee=None
):
    """Algorithm for computing resolute seq-Phragmen (1 winning committee)."""
    approvers_weight = {}
    for cand in profile.candidates:
        approvers_weight[cand] = sum(voter.weight for voter in profile if cand in voter.approved)
    load = start_load
    if load is None:
        load = [0 for _ in range(len(profile))]
    committee = partial_committee
    if partial_committee is None:
        committee = []  # build committees starting with the empty set

    detailed_info = {
        "next_cand": [],
        "tied_cands": [],
        "load": [],
        "max_load": [],
    }

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
        tied_cands = [
            cand for cand in profile.candidates if cand > next_cand and new_maxload[cand] == opt
        ]
        # compute new loads and add new candidate
        for v, voter in enumerate(profile):
            if next_cand in voter.approved:
                load[v] = new_maxload[next_cand]

        committee = sorted(committee + [next_cand])
        detailed_info["next_cand"].append(next_cand)
        detailed_info["tied_cands"].append(tied_cands)
        detailed_info["load"].append(list(load))  # create copy of `load`
        detailed_info["max_load"].append(opt)

    comm_loads = {tuple(committee): load}
    detailed_info["comm_loads"] = comm_loads
    return [committee], detailed_info


def _seqphragmen_irresolute(
    profile, committeesize, division, start_load=None, partial_committee=None
):
    """Algorithm for computing irresolute seq-Phragmen (all winning committees)."""
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
                    new_maxload[cand] = committeesize + 1  # that's larger than any possible value
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
    detailed_info = {"comm_loads": comm_loads}
    return committees, detailed_info


def compute_rule_x(
    profile,
    committeesize,
    algorithm="standard",
    resolute=True,
    skip_phragmen_phase=False,
):
    """Rule X.

    See https://arxiv.org/pdf/1911.11747.pdf, page 7

    skip_phragmen_phase : bool, optional
        omit the second phase (that uses seq-Phragmen)
        may result in a committee that is too small
    """
    check_enough_approved_candidates(profile, committeesize)
    if not profile.has_unit_weights():
        raise ValueError(get_shortname("rule-x") + " is only defined for unit weights (weight=1)")

    if algorithm == "fastest":
        algorithm = fastest_algo("rule-x")

    if algorithm == "standard":
        division = lambda x, y: x / y  # standard float division
    elif algorithm == "exact-fractions":
        division = Fraction  # using exact fractions
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_rule_x"
        )

    committees, detailed_info = _rule_x_algorithm(
        profile, committeesize, resolute, division, skip_phragmen_phase
    )

    # optional output
    if skip_phragmen_phase:
        output.info(header(get_longname("rule-x-without-2nd-phase")))
    else:
        output.info(header(get_longname("rule-x")))
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(_algorithm_fullnames(algorithm) + "\n")

    if resolute:
        start_budget = detailed_info["start_budget"]
        output.details("Phase 1:\n")
        output.details("starting budget:")
        msg = "  ("
        for v, _ in enumerate(profile):
            msg += str(start_budget[v]) + ", "
        output.details(msg[:-2] + ")\n")

        committee = []
        for i, next_cand in enumerate(detailed_info["next_cand"]):
            committee.append(next_cand)
            budget = detailed_info["budget"][i]
            cost = detailed_info["cost"][i]
            tied_cands = detailed_info["tied_cands"][i]
            msg = f"adding candidate number {i+1}: {profile.cand_names[next_cand]}\n"
            msg += f" with maxmimum cost per voter q = {cost}"
            output.details(msg)
            output.details(" remaining budget:")
            msg = "  ("
            for v, _ in enumerate(profile):
                msg += str(budget[v]) + ", "
            output.details(msg[:-2] + ")")
            if tied_cands:
                msg = " tie broken in favor of "
                msg += profile.cand_names[next_cand] + ","
                msg += "\n candidates "
                msg += str_set_of_candidates(tied_cands, cand_names=profile.cand_names)
                msg += " are tied"
                output.details(msg)
            output.details("")

        if detailed_info["phragmen_start_load"]:  # the second phase (seq-Phragmen) was used
            output.details("Phase 2 (seq-Phragmén):\n")
            output.details("starting loads (= budget spent):")
            msg = "  ("
            for v, _ in enumerate(profile):
                msg += str(detailed_info["phragmen_start_load"][v]) + ", "
            output.details(msg[:-2] + ")\n")

            detailed_info_phragmen = detailed_info["phragmen_phase"]
            for i, next_cand in enumerate(detailed_info_phragmen["next_cand"]):
                tied_cands = detailed_info_phragmen["tied_cands"][i]
                max_load = detailed_info_phragmen["max_load"][i]
                load = detailed_info_phragmen["load"][i]
                committee.append(next_cand)
                output.details(
                    f"adding candidate number {len(committee)}: {profile.cand_names[next_cand]}\n"
                    f" maximum load increased to {max_load}"
                )
                output.details(" load distribution:")
                msg = "  ("
                for v, _ in enumerate(profile):
                    msg += str(load[v]) + ", "
                output.details(msg[:-2] + ")")
                if tied_cands:
                    msg = " tie broken in favor of " + profile.cand_names[next_cand]
                    msg += ",\n candidates " + str_set_of_candidates(
                        tied_cands, cand_names=profile.cand_names
                    )
                    msg += f" would increase the load to the same amount ({max_load})"
                    output.details(msg)
                output.details("")

    output.info(str_committees_header(committees, winning=True))
    output.info(str_sets_of_candidates(committees, cand_names=profile.cand_names))
    # end of optional output

    return sorted_committees(committees)


def _rule_x_algorithm(profile, committeesize, resolute, division, skip_phragmen_phase=False):
    """Algorithm for Rule X."""

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

    start_budget = {v: division(committeesize, len(profile)) for v, _ in enumerate(profile)}
    commbugdets = [(set(), start_budget)]
    final_committees = set()

    detailed_info = {
        "next_cand": [],
        "cost": [],
        "tied_cands": [],
        "budget": [],
        "start_budget": start_budget,
        "phragmen_start_load": None,
    }
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
                tied_cands = [cand for cand in min_q.keys() if min_q[cand] == min(min_q.values())]
                for next_cand in tied_cands:
                    new_budget = dict(budget)
                    for v, voter in enumerate(profile):
                        if next_cand in voter.approved:
                            new_budget[v] -= min(budget[v], min_q[next_cand])
                    new_comm = set(committee)
                    new_comm.add(next_cand)
                    next_commbudgets.append((new_comm, new_budget))

                    if resolute:
                        detailed_info["next_cand"].append(next_cand)
                        detailed_info["tied_cands"].append(tied_cands[1:])
                        detailed_info["cost"].append(min(min_q.values()))
                        detailed_info["budget"].append(new_budget)
                        break

            else:  # no affordable candidates remain
                if skip_phragmen_phase:
                    final_committees.add(tuple(committee))
                else:
                    # fill committee via seq-Phragmen

                    # translate budget to loads
                    start_load = [
                        division(committeesize, len(profile)) - budget[v]
                        for v in range(len(profile))
                    ]
                    detailed_info["phragmen_start_load"] = list(start_load)  # make a copy

                    if resolute:
                        committees, detailed_info_phragmen = _seqphragmen_resolute(
                            profile,
                            committeesize,
                            division,
                            partial_committee=list(committee),
                            start_load=start_load,
                        )
                    else:
                        committees, detailed_info_phragmen = _seqphragmen_irresolute(
                            profile,
                            committeesize,
                            division,
                            partial_committee=list(committee),
                            start_load=start_load,
                        )
                    final_committees.update([tuple(committee) for committee in committees])
                    detailed_info["phragmen_phase"] = detailed_info_phragmen
                    # after filling the remaining spots these committees
                    # have size committeesize

            commbugdets = next_commbudgets

    final_committees.update([tuple(committee) for committee, _ in commbugdets])

    committees = sorted_committees(final_committees)
    if resolute:
        committees = [committees[0]]
    return committees, detailed_info


def compute_rule_x_without_2nd_phase(profile, committeesize, algorithm="standard", resolute=True):
    """Rule X with skip_phragmen_phase=True.

    May return committees with fewer than `committeesize` candidates.
    """
    return compute_rule_x(
        profile,
        committeesize,
        algorithm,
        resolute=resolute,
        skip_phragmen_phase=True,
    )


def compute_minimaxphragmen(profile, committeesize, algorithm="gurobi", resolute=False):
    """Phragmen's minimax rule (minimax-Phragmen).

    Minimizes the maximum load.

    Warning: does not include the lexicographic optimization as specified
    in Markus Brill, Rupert Freeman, Svante Janson and Martin Lackner.
    Phragmen's Voting Methods and Justified Representation.
    https://arxiv.org/abs/2102.12305
    Instead: minimizes the maximum load (without consideration of the
             second-, third-, ...-largest load
    """
    check_enough_approved_candidates(profile, committeesize)

    if algorithm == "fastest":
        algorithm = fastest_algo("minimaxphragmen")

    if algorithm == "gurobi":
        committees = abcrules_gurobi._gurobi_minimaxphragmen(
            profile, committeesize, resolute=resolute
        )
    elif algorithm.startswith("mip_"):
        committees = abcrules_mip._mip_minimaxphragmen(
            profile,
            committeesize,
            resolute=resolute,
            solver_id=algorithm[4:],
        )
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_minimaxphragmen"
        )

    # optional output
    output.info(header(get_longname("minimaxphragmen")))
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(_algorithm_fullnames(algorithm) + "\n")
    output.info(str_committees_header(committees, winning=True))
    output.info(str_sets_of_candidates(committees, cand_names=profile.cand_names))
    # end of optional output

    return committees


def compute_phragmen_enestroem(profile, committeesize, algorithm="standard", resolute=True):
    """Phragmen-Enestroem (aka Phragmen's first method, Enestroem's method).

    In every round the candidate with the highest combined budget of
    their supporters is put in the committee.
    Method described in:
    https://arxiv.org/pdf/1611.08826.pdf (Section 18.5, Page 59)
    """
    check_enough_approved_candidates(profile, committeesize)
    if not profile.has_unit_weights():
        raise ValueError(
            get_shortname("phragmen-enestroem") + " is only defined for unit weights (weight=1)"
        )

    if algorithm == "fastest":
        algorithm = fastest_algo("phragmen-enestroem")

    if algorithm == "standard":
        division = lambda x, y: x / y  # standard float division
    elif algorithm == "exact-fractions":
        division = Fraction  # using exact fractions
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_phragmen_enestroem"
        )
    committees, detailed_info = _phragmen_enestroem_algorithm(
        profile, committeesize, resolute, division
    )

    # optional output
    output.info(header(get_longname("phragmen-enestroem")))
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(_algorithm_fullnames(algorithm) + "\n")
    output.info(str_committees_header(committees, winning=True))
    output.info(str_sets_of_candidates(committees, cand_names=profile.cand_names))
    # end of optional output

    return committees


def _phragmen_enestroem_algorithm(profile, committeesize, resolute, division):
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

    # get rid of duplicates and sort committees
    committees = sorted_committees(
        set([tuple(sorted(committee)) for _, committee in voter_budgets_for_partial_committee])
    )
    if resolute:
        committees = [committees[0]]
    detailed_info = {}
    return committees, detailed_info


def compute_consensus_rule(profile, committeesize, algorithm="standard", resolute=True):
    """Consensus rule.

    Based on Perpetual Consensus from
    Martin Lackner Perpetual Voting: Fairness in Long-Term Decision Making
    In Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI 2020)
    """
    check_enough_approved_candidates(profile, committeesize)

    if algorithm == "fastest":
        algorithm = fastest_algo("consensus-rule")

    if algorithm == "standard":
        division = lambda x, y: x / y  # standard float division
    elif algorithm == "exact-fractions":
        division = Fraction  # using exact fractions
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for compute_consensus_rule"
        )
    committees, detailed_info = _consensus_rule_algorithm(
        profile, committeesize, resolute, division
    )

    # optional output
    output.info(header(get_longname("consensus-rule")))
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(_algorithm_fullnames(algorithm) + "\n")
    output.info(str_committees_header(committees, winning=True))
    output.info(str_sets_of_candidates(committees, cand_names=profile.cand_names))
    # end of optional output

    return committees


def _consensus_rule_algorithm(profile, committeesize, resolute, division):
    """Algorithm for the consensus rule."""
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
    detailed_info = {}
    return committees, detailed_info
