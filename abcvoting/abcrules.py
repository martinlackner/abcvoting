# -*- coding: utf-8 -*-
"""Approval-based committee (ABC) voting rules."""

import functools
import itertools
from abcvoting.output import output, DETAILS
from abcvoting import abcrules_gurobi, abcrules_ortools, abcrules_mip, misc
from abcvoting.misc import str_committees_with_header, header, str_set_of_candidates
from abcvoting.misc import sorted_committees, CandidateSet
from abcvoting import scores
from fractions import Fraction
import math
import random

gmpy2_available = True
try:
    from gmpy2 import mpq
except ImportError:
    gmpy2_available = False

########################################################################


MAIN_RULE_IDS = [
    "av",
    "sav",
    "pav",
    "slav",
    "cc",
    "lexcc",
    "geom2",
    "seqpav",
    "revseqpav",
    "seqslav",
    "seqcc",
    "seqphragmen",
    "minimaxphragmen",
    "leximinphragmen",  # TODO: called leximax-Phragmen in https://arxiv.org/abs/2102.12305
    "monroe",
    "greedy-monroe",
    "minimaxav",
    "lexminimaxav",
    "rule-x",
    "phragmen-enestroem",
    "consensus-rule",
    "trivial",
    "rsd",
]
"""
List of rule identifiers (`rule_id`) for the main ABC rules included in abcvoting.

This selection is somewhat arbitrary. But all really important rules (that are implemented) 
are contained in this list.
"""

ALGORITHM_NAMES = {
    "gurobi": "Gurobi ILP solver",
    "branch-and-bound": "branch-and-bound",
    "brute-force": "brute-force",
    "mip-cbc": "CBC ILP solver via Python MIP library",
    "mip-gurobi": "Gurobi ILP solver via Python MIP library",
    # "cvxpy_gurobi": "Gurobi ILP solver via CVXPY library",
    # "cvxpy_scip": "SCIP ILP solver via CVXPY library",
    # "cvxpy_glpk_mi": "GLPK ILP solver via CVXPY library",
    # "cvxpy_cbc": "CBC ILP solver via CVXPY library",
    "standard": "Standard algorithm",
    "standard-fractions": "Standard algorithm (using standard Python fractions)",
    "gmpy2-fractions": "Standard algorithm (using gmpy2 fractions)",
    "float-fractions": "Standard algorithm (using floats instead of fractions)",
    "ortools-cp": "OR-Tools CP-SAT solver",
}
"""
A dictionary containing mapping all valid algorithm identifiers to full names (i.e., descriptions).
"""


FLOAT_ISCLOSE_REL_TOL = 1e-12
"""
The relative tolerance when comparing floats.

See also: `math.isclose() <https://docs.python.org/3/library/math.html#math.isclose>`_.
"""

FLOAT_ISCLOSE_ABS_TOL = 1e-12
"""
The absolute tolerance when comparing floats.

See also: `math.isclose() <https://docs.python.org/3/library/math.html#math.isclose>`_.
"""

MAX_NUM_OF_COMMITTEES_DEFAULT = None
"""
The  maximum number of committees that is returned by an ABC voting rule.

If `MAX_NUM_OF_COMMITTEES_DEFAULT` ist set to `None`, then there is no constraint
on the maximum number of committees.
Can be overridden with the parameter `max_num_of_committees` in any `compute` function. 
"""


class Rule:
    """
    A class that contains the main information about an ABC rule.

    Parameters
    ----------
        rule_id : str
            The rule identifier.

        shortname : str
            The name of the ABC rule (shortened).

        longname : str
            The full name of the ABC rule.

        compute_fct : func
            Function used to compute this rule.

        algorithms : tuple of str
            List of algorithms that compute this rule.

        resolute_values : tuple of bool
            Values that the `resolute` can take.
    """

    def __init__(
        self,
        rule_id,
        shortname,
        longname,
        compute_fct,
        algorithms,
        resolute_values,  # A list containing True, False, or both.
        # The value at position 0 is the default.
        # (e.g., False for optimization-based rules, True for sequential rules.)
    ):
        self.rule_id = rule_id
        self.shortname = shortname
        self.longname = longname
        self._compute_fct = compute_fct
        self.algorithms = algorithms
        self.resolute_values = resolute_values

        # find all *available* algorithms for this ABC rule
        self.available_algorithms = []
        for algorithm in algorithms:
            if algorithm in available_algorithms:
                self.available_algorithms.append(algorithm)

    def fastest_available_algorithm(self):
        """
        Return the fastest algorithm for this rule that is available on this system.

        An algorithm may not be available because its requirements are not satisfied. For example,
        some algorithms require Gurobi, others require gmpy2 - both of which are not requirements
        for abcvoting.

        Returns
        -------
            str
        """
        if self.available_algorithms:
            # This rests on the assumption that ``self.algorithms`` are sorted by speed.
            return self.available_algorithms[0]
        else:
            raise NoAvailableAlgorithm(self.rule_id, self.algorithms)

    def compute(self, profile, committeesize, **kwargs):
        """
        Compute rule using self._compute_fct.

        Parameters
        ----------
            profile : abcvoting.preferences.Profile
                A profile.

            committeesize : int
                The desired committee size.

            **kwargs : dict
                Optional arguments for computing the rule (e.g., `resolute`).

        Returns
        -------
            list of CandidateSet
                A list of winning committees.
        """
        return self._compute_fct(profile, committeesize, **kwargs)

    def verify_compute_parameters(
        self,
        profile,
        committeesize,
        algorithm,
        resolute,
        max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
    ):
        """
        Basic checks for parameter values when computing an ABC rule.

        Parameters
        ----------

            profile : abcvoting.preferences.Profile
                A profile.

            committeesize : int
                The desired committee size.

            algorithm : str, optional
                The algorithm to be used.

            resolute : bool
                Return only one winning committee.

                If `resolute=False`, all winning committees are computed
                (subject to `max_num_of_committees`).

            max_num_of_committees : int, optional
                At most `max_num_of_committees` winning committees are computed.

                If `max_num_of_committees=None`, the number of winning committees is not
                restricted.
                The default value of `max_num_of_committees` can be modified via the constant
                `MAX_NUM_OF_COMMITTEES_DEFAULT`.

            Returns
            -------
                bool
        """
        if committeesize < 1:
            raise ValueError(f"Parameter `committeesize` must be a positive integer.")

        if committeesize > profile.num_cand:
            raise ValueError(
                "Parameter `committeesize` must be smaller or equal to"
                " the total number of candidates."
            )

        if len(profile) == 0:
            raise ValueError("The given profile contains no voters (len(profile) == 0).")

        if algorithm not in self.algorithms:
            raise UnknownAlgorithm(self.rule_id, algorithm)

        if resolute not in self.resolute_values:
            raise NotImplementedError(
                f"ABC rule {self.rule_id} does not support resolute={resolute}."
            )

        if max_num_of_committees is not None and max_num_of_committees < 1:
            raise ValueError(
                "Parameter `max_num_of_committees` must be None or a positive integer."
            )


class UnknownRuleIDError(ValueError):
    """
    Error: unknown rule id.

    Parameters
    ----------
        rule_id : str
            The unknown rule identifier.
    """

    def __init__(self, rule_id):
        message = 'Rule ID "' + str(rule_id) + '" is not known.'
        super(ValueError, self).__init__(message)


class UnknownAlgorithm(ValueError):
    """
    Error: unknown algorithm for a given ABC rule.

    Parameters
    ----------
        rule_id : str
            The ABC rule for which the algorithm is not known.

        algorithm : str
            The unknown algorithm.
    """

    def __init__(self, rule_id, algorithm):
        message = f"Algorithm {algorithm} not specified for ABC rule {rule_id}."
        super(ValueError, self).__init__(message)


class NoAvailableAlgorithm(ValueError):
    """
    Exception: none of the implemented algorithms are available.

    This error occurs because no solvers are installed.

    Parameters
    ----------
        rule_id : str
            The ABC rule for which no algorithm are available.

        algorithms : tuple of str
            List of algorithms for this rule (none of which are available).
    """

    def __init__(self, rule_id, algorithms):
        message = (
            f"None of the implemented algorithms are available for ABC rule {rule_id}\n"
            f"(because the solvers for the following algorithms are not installed: "
            f"{algorithms}) "
        )
        super(ValueError, self).__init__(message)


def _available_algorithms():
    """Verify which algorithms are supported on the current machine.

    This is done by verifying that the required modules and solvers are available.
    """
    available_algorithms = []

    for algorithm in ALGORITHM_NAMES.keys():

        if "gurobi" in algorithm and not abcrules_gurobi.gurobipy_available:
            continue

        if algorithm == "gmpy2-fractions" and not gmpy2_available:
            continue

        available_algorithms.append(algorithm)

    return available_algorithms


available_algorithms = _available_algorithms()


def get_rule(rule_id):
    """
    Get instance of `Rule` for the ABC rule specified by `rule_id`.

    Parameters
    ----------
        rule_id : str
            The rule identifier.

    Returns
    -------
        Rule
            A corresponding `Rule` object.
    """
    _THIELE_ALGORITHMS = (
        # algorithms sorted by speed
        "gurobi",
        "mip-gurobi",
        "mip-cbc",
        "branch-and-bound",
        "brute-force",
    )
    _RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES = (False, True)
    _RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES = (True, False)
    if rule_id == "av":
        return Rule(
            rule_id=rule_id,
            shortname="AV",
            longname="Approval Voting (AV)",
            compute_fct=compute_av,
            algorithms=("standard",),
            resolute_values=_RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES,
        )
    if rule_id == "sav":
        return Rule(
            rule_id=rule_id,
            shortname="SAV",
            longname="Satisfaction Approval Voting (SAV)",
            compute_fct=compute_sav,
            algorithms=("standard",),
            resolute_values=_RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES,
        )
    if rule_id == "pav":
        return Rule(
            rule_id=rule_id,
            shortname="PAV",
            longname="Proportional Approval Voting (PAV)",
            compute_fct=compute_pav,
            algorithms=_THIELE_ALGORITHMS,
            resolute_values=_RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES,
        )
    if rule_id == "slav":
        return Rule(
            rule_id=rule_id,
            shortname="SLAV",
            longname="Sainte-Laguë Approval Voting (SLAV)",
            compute_fct=compute_slav,
            algorithms=_THIELE_ALGORITHMS,
            resolute_values=_RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES,
        )
    if rule_id == "cc":
        return Rule(
            rule_id=rule_id,
            shortname="CC",
            longname="Approval Chamberlin-Courant (CC)",
            compute_fct=compute_cc,
            algorithms=(
                # algorithms sorted by speed
                "gurobi",
                "mip-gurobi",
                "ortools-cp",
                "branch-and-bound",
                "brute-force",
                "mip-cbc",
            ),
            resolute_values=_RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES,
        )
    if rule_id == "lexcc":
        return Rule(
            rule_id=rule_id,
            shortname="lex-CC",
            longname="Lexicographic Chamberlin-Courant (lex-CC)",
            compute_fct=compute_lexcc,
            # algorithms sorted by speed
            algorithms=("gurobi", "mip-gurobi", "brute-force", "mip-cbc"),
            resolute_values=_RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES,
        )
    if rule_id == "seqpav":
        return Rule(
            rule_id=rule_id,
            shortname="seq-PAV",
            longname="Sequential Proportional Approval Voting (seq-PAV)",
            compute_fct=compute_seqpav,
            algorithms=("standard",),
            resolute_values=_RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES,
        )
    if rule_id == "revseqpav":
        return Rule(
            rule_id=rule_id,
            shortname="revseq-PAV",
            longname="Reverse Sequential Proportional Approval Voting (revseq-PAV)",
            compute_fct=compute_revseqpav,
            algorithms=("standard",),
            resolute_values=_RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES,
        )
    if rule_id == "seqslav":
        return Rule(
            rule_id=rule_id,
            shortname="seq-SLAV",
            longname="Sequential Sainte-Laguë Approval Voting (seq-SLAV)",
            compute_fct=compute_seqslav,
            algorithms=("standard",),
            resolute_values=_RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES,
        )
    if rule_id == "seqcc":
        return Rule(
            rule_id=rule_id,
            shortname="seq-CC",
            longname="Sequential Approval Chamberlin-Courant (seq-CC)",
            compute_fct=compute_seqcc,
            algorithms=("standard",),
            resolute_values=_RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES,
        )
    if rule_id == "seqphragmen":
        return Rule(
            rule_id=rule_id,
            shortname="seq-Phragmén",
            longname="Phragmén's Sequential Rule (seq-Phragmén)",
            compute_fct=compute_seqphragmen,
            algorithms=("float-fractions", "gmpy2-fractions", "standard-fractions"),
            resolute_values=_RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES,
        )
    if rule_id == "minimaxphragmen":
        return Rule(
            rule_id=rule_id,
            shortname="minimax-Phragmén",
            longname="Phragmén's Minimax Rule (minimax-Phragmén)",
            compute_fct=compute_minimaxphragmen,
            algorithms=("gurobi", "mip-gurobi", "mip-cbc"),
            resolute_values=_RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES,
        )
    if rule_id == "leximinphragmen":
        return Rule(
            rule_id=rule_id,
            shortname="leximin-Phragmén",
            longname="Phragmén's Leximin Rule (leximin-Phragmén)",
            compute_fct=compute_leximinphragmen,
            algorithms=("gurobi",),  # TODO: "mip-gurobi", "mip-cbc"),
            resolute_values=_RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES,
        )
    if rule_id == "monroe":
        return Rule(
            rule_id=rule_id,
            shortname="Monroe",
            longname="Monroe's Approval Rule (Monroe)",
            compute_fct=compute_monroe,
            algorithms=(
                # algorithms sorted by speed
                "gurobi",
                "mip-gurobi",
                "mip-cbc",
                "ortools-cp",
                "brute-force",
            ),
            resolute_values=_RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES,
        )
    if rule_id == "greedy-monroe":
        return Rule(
            rule_id=rule_id,
            shortname="Greedy Monroe",
            longname="Greedy Monroe",
            compute_fct=compute_greedy_monroe,
            algorithms=("standard",),
            resolute_values=(True,),
        )
    if rule_id == "minimaxav":
        return Rule(
            rule_id=rule_id,
            shortname="minimaxav",
            longname="Minimax Approval Voting (MAV)",
            compute_fct=compute_minimaxav,
            algorithms=("gurobi", "mip-gurobi", "ortools-cp", "mip-cbc", "brute-force"),
            # algorithms sorted by speed. however, for small profiles with a small committee size,
            # brute-force is often the fastest
            resolute_values=_RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES,
        )
    if rule_id == "lexminimaxav":
        return Rule(
            rule_id=rule_id,
            shortname="lex-MAV",
            longname="Lexicographic Minimax Approval Voting (lex-MAV)",
            compute_fct=compute_lexminimaxav,
            algorithms=("gurobi", "brute-force"),
            resolute_values=_RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES,
        )
    if rule_id == "rule-x":
        return Rule(
            rule_id=rule_id,
            shortname="Rule X",
            longname="Rule X (aka Method of Equal Shares)",
            compute_fct=compute_rule_x,
            algorithms=("float-fractions", "gmpy2-fractions", "standard-fractions"),
            resolute_values=_RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES,
        )
    if rule_id == "rule-x-without-phragmen-phase":
        return Rule(
            rule_id=rule_id,
            shortname="Rule X without Phragmén phase",
            longname="Rule X without the Phragmén phase (second phase)",
            compute_fct=functools.partial(compute_rule_x, skip_phragmen_phase=True),
            algorithms=("float-fractions", "gmpy2-fractions", "standard-fractions"),
            resolute_values=_RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES,
        )
    if rule_id == "phragmen-enestroem":
        return Rule(
            rule_id=rule_id,
            shortname="Phragmén-Eneström",
            longname="Method of Phragmén-Eneström",
            compute_fct=compute_phragmen_enestroem,
            algorithms=("float-fractions", "gmpy2-fractions", "standard-fractions"),
            resolute_values=_RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES,
        )
    if rule_id == "consensus-rule":
        return Rule(
            rule_id=rule_id,
            shortname="Consensus Rule",
            longname="Consensus Rule",
            compute_fct=compute_consensus_rule,
            algorithms=("float-fractions", "gmpy2-fractions", "standard-fractions"),
            resolute_values=_RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES,
        )
    if rule_id == "trivial":
        return Rule(
            rule_id=rule_id,
            shortname="Trivial Rule",
            longname="Trivial Rule",
            compute_fct=compute_trivial_rule,
            algorithms=("standard",),
            resolute_values=_RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES,
        )
    if rule_id == "rsd":
        return Rule(
            rule_id=rule_id,
            shortname="Random Serial Dictator",
            longname="Random Serial Dictator",
            compute_fct=compute_rsd,
            algorithms=("standard",),
            resolute_values=(True,),
        )
    if rule_id.startswith("geom"):
        parameter = rule_id[4:]
        return Rule(
            rule_id=rule_id,
            shortname=f"{parameter}-Geometric",
            longname=f"{parameter}-Geometric Rule",
            compute_fct=functools.partial(compute_thiele_method, rule_id),
            algorithms=_THIELE_ALGORITHMS,
            resolute_values=_RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES,
        )

    # handle sequential and reverse sequential Thiele methods
    # that are not explicitly included in the list above
    if rule_id.startswith("seq") or rule_id.startswith("revseq"):
        if rule_id.startswith("seq"):
            scorefct_id = rule_id[3:]  # score function id of Thiele method
        else:
            scorefct_id = rule_id[6:]  # score function id of Thiele method

        try:
            scores.get_marginal_scorefct(scorefct_id)
        except scores.UnknownScoreFunctionError:
            raise UnknownRuleIDError(rule_id)

        if rule_id == "av":
            raise UnknownRuleIDError(rule_id)  # seq-AV and revseq-AV are equivalent to AV

        # sequential Thiele methods
        if rule_id.startswith("seq"):
            return Rule(
                rule_id=rule_id,
                shortname=f"seq-{get_rule(scorefct_id).shortname}",
                longname=f"Sequential {get_rule(scorefct_id).longname}",
                compute_fct=functools.partial(compute_seq_thiele_method, scorefct_id),
                algorithms=("standard",),
                resolute_values=_RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES,
            )
        # reverse sequential Thiele methods
        if rule_id.startswith("revseq"):
            return Rule(
                rule_id=rule_id,
                shortname=f"revseq-{get_rule(scorefct_id).shortname}",
                longname=f"Reverse Sequential {get_rule(scorefct_id).longname}",
                compute_fct=functools.partial(compute_revseq_thiele_method, scorefct_id),
                algorithms=("standard",),
                resolute_values=_RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES,
            )

    raise UnknownRuleIDError(rule_id)


########################################################################


def compute(rule_id, profile, committeesize, result=None, **kwargs):
    """
    Compute winning committees with an ABC rule given by `rule_id`.

    Parameters
    ----------
        rule_id : str
            The rule identifier.

        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        result : list of CandidateSet, optional
            Expected winning committees.

            This is used in unit tests to verify correctness. Raises `ValueError` if
            `result` is different from actual winning committees.

        **kwargs : dict
            Optional arguments for computing the rule (e.g., `resolute`).

    Returns
    -------
        list of CandidateSet
            A list of the winning committees.

            If `resolute=True`, the list contains only one winning committee.
    """
    rule = get_rule(rule_id)
    committees = rule.compute(profile=profile, committeesize=committeesize, **kwargs)
    if result is not None:
        # verify that the parameter `result` is indeed the result of computing the ABC rule
        resolute = kwargs.get("resolute", rule.resolute_values[0])
        misc.verify_expected_committees_equals_actual_committees(
            actual_committees=committees,
            expected_committees=result,
            resolute=resolute,
            shortname=rule.shortname,
        )
    return committees


def compute_thiele_method(
    scorefct_id,
    profile,
    committeesize,
    algorithm="fastest",
    resolute=False,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Compute winning committees with Thiele methods.

    Compute winning committees according to a Thiele method specified
    by a score function (scorefct_id).
    Examples of Thiele methods are PAV, CC, and SLAV.
    An exception is Approval Voting (AV), which should be computed using
    compute_av(). (AV is polynomial-time computable (separable) and can thus be
    computed much faster.)

    For a mathematical description of Thiele methods, see e.g.
    "Multi-Winner Voting with Approval Preferences"
    by Martin Lackner and Piotr Skowron
    <https://arxiv.org/abs/2007.01795>

    Parameters
    ----------
        scorefct_id : str
            A string identifying the score function that defines the Thiele method.

        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
            At most `max_num_of_committees` winning committees are computed.

            If `max_num_of_committees=None`, the number of winning committees is not restricted.
            The default value of `max_num_of_committees` can be modified via the constant
            `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.

            If `resolute=True`, the list contains only one winning committee.
    """
    rule = get_rule(scorefct_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )

    if algorithm == "gurobi":
        committees = abcrules_gurobi._gurobi_thiele_methods(
            scorefct_id=scorefct_id,
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    elif algorithm == "branch-and-bound":
        committees, detailed_info = _thiele_methods_branchandbound(
            scorefct_id=scorefct_id,
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    elif algorithm == "brute-force":
        committees, detailed_info = _thiele_methods_bruteforce(
            scorefct_id=scorefct_id,
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    elif algorithm.startswith("mip-"):
        committees = abcrules_mip._mip_thiele_methods(
            scorefct_id=scorefct_id,
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
            solver_id=algorithm[4:],
        )
    elif algorithm == "ortools-cp" and scorefct_id == "cc":
        committees = abcrules_ortools._ortools_cc(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    else:
        raise UnknownAlgorithm(scorefct_id, algorithm)

    # optional output
    output.info(header(rule.longname))
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")

    output.details(
        f"Optimal {scorefct_id.upper()}-score: "
        f"{scores.thiele_score(scorefct_id, profile, committees[0])}\n"
    )
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )
    # end of optional output

    return committees


def _thiele_methods_bruteforce(
    scorefct_id,
    profile,
    committeesize,
    resolute,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Brute-force algorithm for Thiele methods (PAV, CC, etc.).

    Only intended for comparison, much slower than _thiele_methods_branchandbound()
    """
    opt_committees = []
    opt_thiele_score = -1
    for committee in itertools.combinations(profile.candidates, committeesize):
        score = scores.thiele_score(scorefct_id, profile, committee)
        if score > opt_thiele_score:
            opt_committees = [committee]
            opt_thiele_score = score
        elif score == opt_thiele_score:
            if not resolute:
                opt_committees.append(committee)

    committees = sorted_committees(opt_committees)
    if max_num_of_committees is not None:
        committees = committees[:max_num_of_committees]
    detailed_info = {}
    if resolute:
        committees = [committees[0]]
    return committees, detailed_info


def _thiele_methods_branchandbound(
    scorefct_id,
    profile,
    committeesize,
    resolute,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Branch-and-bound algorithm for Thiele methods.
    """
    marginal_scorefct = scores.get_marginal_scorefct(scorefct_id, committeesize)

    best_committees = []
    init_com, _ = _seq_thiele_resolute(scorefct_id, profile, committeesize)
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
            marg_util_cand = scores.marginal_thiele_scores_add(
                marginal_scorefct, profile, part_com
            )
            upper_bound = sum(
                sorted(marg_util_cand[largest_cand + 1 :])[-missing:]
            ) + scores.thiele_score(scorefct_id, profile, part_com)
            if upper_bound >= best_score:
                for cand in range(largest_cand + 1, profile.num_cand - missing + 1):
                    part_coms.insert(0, part_com + [cand])

    committees = sorted_committees(best_committees)
    if max_num_of_committees is not None:
        committees = committees[:max_num_of_committees]
    if resolute:
        committees = [committees[0]]

    detailed_info = {}
    return committees, detailed_info


def compute_pav(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=False,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Compute winning committees with Proportional Approval Voting (PAV).

    This ABC rule belongs to the class of Thiele methods.

    For a mathematical description of this rule, see e.g.
    "Multi-Winner Voting with Approval Preferences"
    by Martin Lackner and Piotr Skowron
    <https://arxiv.org/abs/2007.01795>

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for PAV:

            .. doctest::

                >>> get_rule("pav").algorithms
                ('gurobi', 'mip-gurobi', 'mip-cbc', 'branch-and-bound', 'brute-force')

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    return compute_thiele_method(
        scorefct_id="pav",
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )


def compute_slav(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=False,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Compute winning committees with Sainte-Lague Approval Voting (SLAV).

    This ABC rule belongs to the class of Thiele methods.

    For a mathematical description of this rule, see e.g.
    Martin Lackner and Piotr Skowron
    Utilitarian Welfare and Representation Guarantees of Approval-Based Multiwinner Rules
    In Artificial Intelligence, 288: 103366, 2020.
    <https://arxiv.org/abs/1801.01527>

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for SLAV:

            .. doctest::

                >>> get_rule("slav").algorithms
                ('gurobi', 'mip-gurobi', 'mip-cbc', 'branch-and-bound', 'brute-force')

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    return compute_thiele_method(
        scorefct_id="slav",
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )


def compute_cc(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=False,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Compute winning committees with Approval Chamberlin-Courant (CC).

    This ABC rule belongs to the class of Thiele methods.

    For a mathematical description of this rule, see e.g.
    "Multi-Winner Voting with Approval Preferences"
    by Martin Lackner and Piotr Skowron
    <https://arxiv.org/abs/2007.01795>

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for Approval Chamberlin-Courant (CC):

            .. doctest::

                >>> get_rule("cc").algorithms
                ('gurobi', 'mip-gurobi', 'ortools-cp', 'branch-and-bound', 'brute-force', 'mip-cbc')

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    return compute_thiele_method(
        scorefct_id="cc",
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )


def compute_lexcc(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=False,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Compute winning committees with a Lexicographic Chamberlin-Courant (lex-CC).

    This ABC rule is a lexicographic variant of Approval Chamberlin-Courant (CC). It maximizes the
    CC score, i.e., the number of voters with at least one approved
    candidate in the winning committee. If there is more than one such committee, it chooses the
    committee with most voters having at least two approved candidates in the committee. This
    tie-breaking continues with values of 3, 4, .., k if necessary.

    This rule can be seen as an analogue to the leximin social welfare ordering for utility
    functions.

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for Lexicographic Chamberlin-Courant (lex-CC):

            .. doctest::

                >>> get_rule("lexcc").algorithms
                ('gurobi', 'mip-gurobi', 'brute-force', 'mip-cbc')

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    rule_id = "lexcc"
    rule = get_rule(rule_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )

    if algorithm == "brute-force":
        committees, detailed_info = _lexcc_bruteforce(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    elif algorithm == "gurobi":
        committees, detailed_info = abcrules_gurobi._gurobi_lexcc(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    elif algorithm.startswith("mip-"):
        committees, detailed_info = abcrules_mip._mip_lexcc(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
            solver_id=algorithm[4:],
        )
    else:
        raise UnknownAlgorithm(rule_id, algorithm)

    # optional output
    output.info(header(rule.longname))
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")
    output.details(f"At-least-ell scores:")
    output.details(
        "\n".join(
            f" at-least-{ell+1}: {score}"
            for ell, score in enumerate(detailed_info["opt_score_vector"])
        )
        + "\n"
    )
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )
    # end of optional output

    return committees


def _lexcc_bruteforce(profile, committeesize, resolute, max_num_of_committees):
    opt_committees = []
    opt_score_vector = [0] * committeesize
    for committee in itertools.combinations(profile.candidates, committeesize):
        score_vector = [
            scores.thiele_score(f"atleast{ell}", profile, committee)
            for ell in range(1, committeesize + 1)
        ]
        for i in range(committeesize):
            if opt_score_vector[i] > score_vector[i]:
                break
            if opt_score_vector[i] < score_vector[i]:
                opt_score_vector = score_vector
                opt_committees = [committee]
                break
        else:
            opt_committees.append(committee)

    committees = sorted_committees(opt_committees)
    detailed_info = {"opt_score_vector": opt_score_vector}
    if resolute:
        committees = [committees[0]]
    if max_num_of_committees is not None:
        committees = committees[:max_num_of_committees]
    return committees, detailed_info


def compute_seq_thiele_method(
    scorefct_id,
    profile,
    committeesize,
    algorithm="fastest",
    resolute=True,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Sequential Thiele methods.

    For a mathematical description of these rules, see e.g.
    "Multi-Winner Voting with Approval Preferences"
    by Martin Lackner and Piotr Skowron
    <https://arxiv.org/abs/2007.01795>

    Parameters
    ----------
        scorefct_id : str
            A string identifying the score function that defines the Thiele method.

        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    scores.get_marginal_scorefct(scorefct_id, committeesize)  # check that scorefct_id is valid
    rule_id = "seq" + scorefct_id
    rule = get_rule(rule_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )

    if algorithm == "standard":
        if resolute:
            committees, detailed_info = _seq_thiele_resolute(scorefct_id, profile, committeesize)
        else:
            committees, detailed_info = _seq_thiele_irresolute(
                scorefct_id, profile, committeesize, max_num_of_committees
            )
    else:
        raise UnknownAlgorithm(rule_id, algorithm)

    # optional output
    output.info(header(rule.longname))
    if not resolute:
        output.info(
            "Computing all possible winning committees for any tiebreaking order\n"
            " (aka parallel universes tiebreaking) (resolute=False)\n"
        )
    if output.verbosity <= DETAILS:  # skip thiele_score() calculations if not necessary
        output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")
        if resolute:
            output.details(
                f"starting with the empty committee (score = "
                f"{scores.thiele_score(scorefct_id, profile, [])})\n"
            )
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
                if len(tied_cands) > 1:
                    output.details(f" tie broken in favor of {next_cand},\n")
                    output.details(
                        f" candidates "
                        f"{str_set_of_candidates(tied_cands, cand_names=profile.cand_names)} "
                        f"are tied (all would increase the score by the same amount {delta_score})"
                    )
                output.details("")
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )

    if output.verbosity <= DETAILS:  # skip thiele_score() calculations if not necessary
        output.details(scorefct_id.upper() + "-score of winning committee(s):")
        for committee in committees:
            output.details(
                f" {str_set_of_candidates(committee, cand_names=profile.cand_names)}: "
                f"{scores.thiele_score(scorefct_id, profile, committee)}"
            )
        output.details("\n")
    # end of optional output

    return sorted_committees(committees)


def _seq_thiele_resolute(scorefct_id, profile, committeesize):
    """Compute one winning committee (=resolute) for sequential Thiele methods.

    Tiebreaking between candidates in favor of candidate with smaller
    number/index (candidates with larger numbers get deleted first).
    """
    committee = []
    marginal_scorefct = scores.get_marginal_scorefct(scorefct_id, committeesize)
    detailed_info = {"next_cand": [], "tied_cands": [], "delta_score": []}

    # build a committee starting with the empty set
    for _ in range(committeesize):
        additional_score_cand = scores.marginal_thiele_scores_add(
            marginal_scorefct, profile, committee
        )
        tied_cands = [
            cand
            for cand in range(len(additional_score_cand))
            if additional_score_cand[cand] == max(additional_score_cand)
        ]
        next_cand = tied_cands[0]  # tiebreaking in favor of candidate with smallest index
        committee.append(next_cand)
        detailed_info["next_cand"].append(next_cand)
        detailed_info["tied_cands"].append(tied_cands)
        detailed_info["delta_score"].append(max(additional_score_cand))

    return sorted_committees([committee]), detailed_info


def _seq_thiele_irresolute(scorefct_id, profile, committeesize, max_num_of_committees):
    """Compute all winning committee (=irresolute) for sequential Thiele methods.

    Consider all possible ways to break ties between candidates
    (aka parallel universe tiebreaking)
    """
    marginal_scorefct = scores.get_marginal_scorefct(scorefct_id, committeesize)

    # build committees starting with the empty set
    partial_committees = [()]
    winning_committees = set()

    while partial_committees:
        new_partial_committees = []
        committee = partial_committees.pop()
        # marginal utility gained by adding candidate to the committee
        additional_score_cand = scores.marginal_thiele_scores_add(
            marginal_scorefct, profile, committee
        )
        for cand in profile.candidates:
            if additional_score_cand[cand] >= max(additional_score_cand):
                new_committee = committee + (cand,)

                if len(new_committee) == committeesize:
                    new_committee = tuple(sorted(new_committee))
                    winning_committees.add(new_committee)  # remove duplicate committees
                    if (
                        max_num_of_committees is not None
                        and len(winning_committees) == max_num_of_committees
                    ):
                        # sufficiently many winning committees found
                        detailed_info = {}
                        return sorted_committees(winning_committees), detailed_info
                else:
                    # partial committee
                    new_partial_committees.append(new_committee)
        # add new partial committees in reversed order, so that tiebreaking is correct
        partial_committees += reversed(new_partial_committees)

    detailed_info = {}
    return sorted_committees(winning_committees), detailed_info


# Sequential PAV
def compute_seqpav(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=True,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Sequential PAV (seq-PAV).

    For a mathematical description of this rule, see e.g.
    "Multi-Winner Voting with Approval Preferences"
    by Martin Lackner and Piotr Skowron
    <https://arxiv.org/abs/2007.01795>

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for Sequential PAV:

            .. doctest::

                >>> get_rule("seqpav").algorithms
                ('standard',)

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    return compute_seq_thiele_method(
        scorefct_id="pav",
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )


def compute_seqslav(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=True,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Sequential Sainte-Lague Approval Voting (SLAV).

    For a mathematical description of SLAV, see e.g.
    Martin Lackner and Piotr Skowron
    Utilitarian Welfare and Representation Guarantees of Approval-Based Multiwinner Rules
    In Artificial Intelligence, 288: 103366, 2020.
    <https://arxiv.org/abs/1801.01527>

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for Sequential SLAV:

            .. doctest::

                >>> get_rule("seqslav").algorithms
                ('standard',)

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    return compute_seq_thiele_method(
        scorefct_id="slav",
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )


def compute_seqcc(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=True,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Sequential Chamberlin-Courant (seq-CC).

    For a mathematical description of this rule, see e.g.
    "Multi-Winner Voting with Approval Preferences"
    by Martin Lackner and Piotr Skowron
    <https://arxiv.org/abs/2007.01795>

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for Sequential CC:

            .. doctest::

                >>> get_rule("seqcc").algorithms
                ('standard',)

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    return compute_seq_thiele_method(
        scorefct_id="cc",
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )


def compute_revseq_thiele_method(
    scorefct_id,
    profile,
    committeesize,
    algorithm="fastest",
    resolute=True,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Reverse sequential Thiele methods.

    For a mathematical description of these rules, see e.g.
    "Multi-Winner Voting with Approval Preferences"
    by Martin Lackner and Piotr Skowron
    <https://arxiv.org/abs/2007.01795>

    Parameters
    ----------
        scorefct_id : str
            A string identifying the score function that defines the Thiele method.

        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    scores.get_marginal_scorefct(scorefct_id, committeesize)  # check that scorefct_id is valid
    rule_id = "revseq" + scorefct_id
    rule = get_rule(rule_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )

    if algorithm == "standard":
        if resolute:
            committees, detailed_info = _revseq_thiele_resolute(
                scorefct_id=scorefct_id,
                profile=profile,
                committeesize=committeesize,
            )
        else:
            committees, detailed_info = _revseq_thiele_irresolute(
                scorefct_id=scorefct_id,
                profile=profile,
                committeesize=committeesize,
                max_num_of_committees=max_num_of_committees,
            )
    else:
        raise UnknownAlgorithm(rule_id, algorithm)

    # optional output
    output.info(header(rule.longname))
    if not resolute:
        output.info(
            "Computing all possible winning committees for any tiebreaking order\n"
            " (aka parallel universes tiebreaking) (resolute=False)\n"
        )
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")

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
            if len(tied_cands) > 1:
                output.details(f" tie broken to the disadvantage of {next_cand},\n")
                output.details(
                    f" candidates "
                    f"{str_set_of_candidates(tied_cands, cand_names=profile.cand_names)}"
                    f" are tied (all would decrease the score by the same amount {delta_score})"
                )
            output.details("")
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )

    msg = "PAV-score of winning committee:"
    if not resolute and len(committees) != 1:
        msg += "\n"
    for committee in committees:
        msg += " " + str(scores.thiele_score(scorefct_id, profile, committee))
    msg += "\n"
    output.details(msg)
    # end of optional output

    return committees


def _revseq_thiele_resolute(scorefct_id, profile, committeesize):
    """Compute one winning committee (=resolute) for reverse sequential Thiele methods.

    Tiebreaking between candidates in favor of candidate with smaller
    number/index (candidates with smaller numbers are added first).
    """
    marginal_scorefct = scores.get_marginal_scorefct(scorefct_id, committeesize)
    committee = set(profile.candidates)

    detailed_info = {"next_cand": [], "tied_cands": [], "delta_score": []}

    for _ in range(profile.num_cand - committeesize):
        marg_util_cand = scores.marginal_thiele_scores_remove(
            marginal_scorefct, profile, committee
        )
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


def _revseq_thiele_irresolute(scorefct_id, profile, committeesize, max_num_of_committees):
    """
    Compute all winning committee (=irresolute) for reverse sequential Thiele methods.

    Consider all possible ways to break ties between candidates
    (aka parallel universe tiebreaking)
    """
    marginal_scorefct = scores.get_marginal_scorefct(scorefct_id, committeesize)

    full_committee = tuple(profile.candidates)
    comm_scores = {full_committee: scores.thiele_score(scorefct_id, profile, full_committee)}

    for _ in range(profile.num_cand - committeesize):
        comm_scores_next = {}
        for committee, score in comm_scores.items():
            marg_util_cand = scores.marginal_thiele_scores_remove(
                marginal_scorefct, profile, committee
            )
            score_reduction = min(marg_util_cand)
            # find smallest elements in marg_util_cand and return indices
            cands_to_remove = [
                cand for cand in profile.candidates if marg_util_cand[cand] == min(marg_util_cand)
            ]
            for cand in cands_to_remove:
                next_committee = tuple(set(committee) - {cand})
                comm_scores_next[next_committee] = score - score_reduction
            comm_scores = comm_scores_next

    committees = sorted_committees(list(comm_scores.keys()))
    if max_num_of_committees is not None:
        committees = committees[:max_num_of_committees]
    detailed_info = {}
    return committees, detailed_info


# Reverse Sequential PAV
def compute_revseqpav(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=True,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Reverse Sequential PAV (revseq-PAV).

    For a mathematical description of this rule, see e.g.
    "Multi-Winner Voting with Approval Preferences"
    by Martin Lackner and Piotr Skowron
    <https://arxiv.org/abs/2007.01795>

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for Reverse Sequential PAV:

            .. doctest::

                >>> get_rule("revseqpav").algorithms
                ('standard',)

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    return compute_revseq_thiele_method(
        scorefct_id="pav",
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )


def compute_separable_rule(
    rule_id,
    profile,
    committeesize,
    algorithm,
    resolute=True,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Separable rules (such as AV and SAV).

    For a mathematical description of separable rules (for ranking-based rules), see
    E. Elkind, P. Faliszewski, P. Skowron, and A. Slinko.
    Properties of multiwinner voting rules.
    Social Choice and Welfare, 48(3):599–632, 2017.
    <https://link.springer.com/article/10.1007/s00355-017-1026-z>

    Parameters
    ----------
        rule_id : str
            The rule identifier for a separable rule.

        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for AV:

            .. doctest::

                >>> get_rule("av").algorithms
                ('standard',)

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    rule = get_rule(rule_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )

    if algorithm == "standard":
        committees, detailed_info = _separable_rule_algorithm(
            rule_id=rule_id,
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    else:
        raise UnknownAlgorithm(rule_id, algorithm)

    # optional output
    output.info(header(rule.longname))
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")

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
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )
    # end of optional output

    return committees


def _separable_rule_algorithm(rule_id, profile, committeesize, resolute, max_num_of_committees):
    """
    Algorithm for separable rules (such as AV and SAV).
    """
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
        if max_num_of_committees is None:
            committees = sorted_committees(
                [
                    (certain_cands + list(selection))
                    for selection in itertools.combinations(possible_cands, missing)
                ]
            )
        else:
            committees = []
            for selection in itertools.combinations(possible_cands, missing):
                committees.append(certain_cands + list(selection))
                if len(committees) >= max_num_of_committees:
                    break
            committees = sorted_committees(committees)
    detailed_info = {
        "certain_cands": certain_cands,
        "possible_cands": possible_cands,
        "missing": missing,
        "cutoff": cutoff,
        "score": score,
    }
    return committees, detailed_info


def compute_sav(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=False,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Compute winning committees with Satisfaction Approval Voting (SAV).

    For a mathematical description of this rule, see e.g.
    "Multi-Winner Voting with Approval Preferences"
    by Martin Lackner and Piotr Skowron
    <https://arxiv.org/abs/2007.01795>

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for SAV:

            .. doctest::

                >>> get_rule("sav").algorithms
                ('standard',)

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    return compute_separable_rule(
        rule_id="sav",
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )


def compute_av(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=False,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Approval Voting (AV).

    AV is both a Thiele method and a separable rule. Seperable rules can be computed much
    faster than Thiele methods (in general), thus `compute_separable_rule` is used
    to compute AV.

    For a mathematical description of this rule, see e.g.
    "Multi-Winner Voting with Approval Preferences"
    by Martin Lackner and Piotr Skowron
    <https://arxiv.org/abs/2007.01795>

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for AV:

            .. doctest::

                >>> get_rule("av").algorithms
                ('standard',)

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    return compute_separable_rule(
        rule_id="av",
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )


def compute_minimaxav(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=False,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Compute winning committees with Minimax Approval Voting (MAV).

    For a mathematical description of this rule, see e.g.
    "Multi-Winner Voting with Approval Preferences"
    by Martin Lackner and Piotr Skowron
    <https://arxiv.org/abs/2007.01795>

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for Minimax AV:

            .. doctest::

                >>> get_rule("minimaxav").algorithms
                ('gurobi', 'mip-gurobi', 'ortools-cp', 'mip-cbc', 'brute-force')

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    rule_id = "minimaxav"
    rule = get_rule(rule_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )

    if algorithm == "gurobi":
        committees = abcrules_gurobi._gurobi_minimaxav(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    elif algorithm == "ortools-cp":
        committees = abcrules_ortools._ortools_minimaxav(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    elif algorithm.startswith("mip-"):
        solver_id = algorithm[4:]
        committees = abcrules_mip._mip_minimaxav(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
            solver_id=solver_id,
        )
    elif algorithm == "brute-force":
        committees, detailed_info = _minimaxav_bruteforce(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    else:
        raise UnknownAlgorithm(rule_id, algorithm)

    # optional output
    output.info(header(rule.longname))
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")

    opt_minimaxav_score = scores.minimaxav_score(profile, committees[0])
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )
    output.details("Minimum maximal distance: " + str(opt_minimaxav_score))
    msg = "Corresponding distances to voters:\n"
    for committee in committees:
        msg += str([misc.hamming(voter.approved, committee) for voter in profile]) + "\n"
    output.details(msg)
    # end of optional output

    return committees


def _minimaxav_bruteforce(profile, committeesize, resolute, max_num_of_committees):
    """Brute-force algorithm for Minimax AV (MAV)."""
    opt_committees = []
    opt_minimaxav_score = profile.num_cand + 1
    for committee in itertools.combinations(profile.candidates, committeesize):
        score = scores.minimaxav_score(profile, committee)
        if score < opt_minimaxav_score:
            opt_committees = [committee]
            opt_minimaxav_score = score
        elif score == opt_minimaxav_score:
            opt_committees.append(committee)

    committees = sorted_committees(opt_committees)
    detailed_info = {}
    if resolute:
        committees = [committees[0]]
    if max_num_of_committees is not None:
        committees = committees[:max_num_of_committees]
    return committees, detailed_info


def compute_lexminimaxav(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=False,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Compute winning committees with Lexicographic Minimax AV (lex-MAV).

    For a mathematical description of this rule, see e.g.
    "Multi-Winner Voting with Approval Preferences"
    by Martin Lackner and Piotr Skowron
    <https://arxiv.org/abs/2007.01795>
    (Remark 2)

    If `lexicographic_tiebreaking` is True, compute all winning committees and choose the
    lexicographically smallest. This is a deterministic form of tiebreaking; if only resolute=True,
    it is not guaranteed how ties are broken.

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for Lexicographic Minimax AV:

            .. doctest::

                >>> get_rule("lexminimaxav").algorithms
                ('gurobi', 'brute-force')

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    rule_id = "lexminimaxav"
    rule = get_rule(rule_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )

    if not profile.has_unit_weights():
        raise ValueError(f"{rule.shortname} is only defined for unit weights (weight=1)")

    if algorithm == "brute-force":
        committees, detailed_info = _lexminimaxav_bruteforce(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    elif algorithm == "gurobi":
        committees, detailed_info = abcrules_gurobi._gurobi_lexminimaxav(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    else:
        raise UnknownAlgorithm(rule_id, algorithm)

    # optional output
    opt_distances = detailed_info["opt_distances"]
    output.info(header(rule.longname))
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )
    output.details("Minimum maximal distance: " + str(max(opt_distances)))
    msg = "Corresponding distances to voters:\n"
    for committee in committees:
        msg += str([misc.hamming(voter.approved, committee) for voter in profile])
    output.details(msg + "\n")
    # end of optional output

    return committees


def _lexminimaxav_bruteforce(profile, committeesize, resolute, max_num_of_committees):
    opt_committees = []
    opt_distances = [profile.num_cand + 1] * len(profile)
    for committee in itertools.combinations(profile.candidates, committeesize):
        distances = sorted(
            [misc.hamming(voter.approved, set(committee)) for voter in profile], reverse=True
        )
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
    if max_num_of_committees is not None:
        committees = committees[:max_num_of_committees]
    return committees, detailed_info


def compute_monroe(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=False,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Compute winning committees with Monroe's rule.

    For a mathematical description of this rule, see e.g.
    "Multi-Winner Voting with Approval Preferences"
    by Martin Lackner and Piotr Skowron
    <https://arxiv.org/abs/2007.01795>

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for Monroe:

            .. doctest::

                >>> get_rule("monroe").algorithms
                ('gurobi', 'mip-gurobi', 'mip-cbc', 'ortools-cp', 'brute-force')

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    rule_id = "monroe"
    rule = get_rule(rule_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )

    if not profile.has_unit_weights():
        raise ValueError(f"{rule.shortname} is only defined for unit weights (weight=1)")

    if algorithm == "gurobi":
        committees = abcrules_gurobi._gurobi_monroe(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    elif algorithm == "ortools-cp":
        committees = abcrules_ortools._ortools_monroe(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    elif algorithm.startswith("mip-"):
        committees = abcrules_mip._mip_monroe(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
            solver_id=algorithm[4:],
        )
    elif algorithm == "brute-force":
        committees, detailed_info = _monroe_bruteforce(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    else:
        raise UnknownAlgorithm(rule_id, algorithm)

    # optional output
    output.info(header(rule.longname))
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")
    output.details(
        "Optimal Monroe score: " + str(scores.monroescore(profile, committees[0])) + "\n"
    )
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )
    # end of optional output

    return committees


def _monroe_bruteforce(profile, committeesize, resolute, max_num_of_committees):
    """
    Brute-force algorithm for Monroe's rule.
    """
    opt_committees = []
    opt_monroescore = -1
    for committee in itertools.combinations(profile.candidates, committeesize):
        score = scores.monroescore(profile, committee)
        if score > opt_monroescore:
            opt_committees = [committee]
            opt_monroescore = score
        elif scores.monroescore(profile, committee) == opt_monroescore:
            opt_committees.append(committee)

    committees = sorted_committees(opt_committees)
    if max_num_of_committees is not None:
        committees = committees[:max_num_of_committees]
    if resolute:
        committees = [committees[0]]

    detailed_info = {}
    return committees, detailed_info


def compute_greedy_monroe(
    profile, committeesize, algorithm="fastest", resolute=True, max_num_of_committees=None
):
    """
    Compute winning committees with Greedy Monroe.

    For a mathematical description of this rule, see e.g.
    "Multi-Winner Voting with Approval Preferences"
    by Martin Lackner and Piotr Skowron
    <https://arxiv.org/abs/2007.01795>

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for Greedy Monroe:

            .. doctest::

                >>> get_rule("greedy-monroe").algorithms
                ('standard',)

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    rule_id = "greedy-monroe"
    rule = get_rule(rule_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(profile, committeesize, algorithm, resolute)

    if not profile.has_unit_weights():
        raise ValueError(f"{rule.shortname} is only defined for unit weights (weight=1)")

    if algorithm == "standard":
        committees, detailed_info = _greedy_monroe_algorithm(profile, committeesize)
    else:
        raise UnknownAlgorithm(rule_id, algorithm)

    # optional output
    output.info(header(rule.longname))
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")
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
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )
    # end of optional output

    return sorted_committees(committees)


def _greedy_monroe_algorithm(profile, committeesize):
    """
    Algorithm for Greedy Monroe.
    """
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


def compute_seqphragmen(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=True,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Compute winning committees with Phragmen's sequential rule (seq-Phragmen).

    For a mathematical description of this rule, see e.g.
    "Multi-Winner Voting with Approval Preferences"
    by Martin Lackner and Piotr Skowron
    <https://arxiv.org/abs/2007.01795>

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for Phragmen's sequential rule (seq-Phragmen):

            .. doctest::

                >>> get_rule("seqphragmen").algorithms
                ('float-fractions', 'gmpy2-fractions', 'standard-fractions')

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    rule_id = "seqphragmen"
    rule = get_rule(rule_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )

    if resolute:
        committees, detailed_info = _seqphragmen_resolute(
            profile=profile,
            committeesize=committeesize,
            algorithm=algorithm,
        )
    else:
        committees, detailed_info = _seqphragmen_irresolute(
            profile=profile,
            committeesize=committeesize,
            algorithm=algorithm,
            max_num_of_committees=max_num_of_committees,
        )

    # optional output
    output.info(header(rule.longname))
    if not resolute:
        output.info(
            "Computing all possible winning committees for any tiebreaking order\n"
            " (aka parallel universes tiebreaking) (resolute=False)\n"
        )
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")

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
                # f"\n (continuous model: time t_{i+1} = {max_load})"
            )
            output.details(" load distribution:")
            msg = "  ("
            for v, _ in enumerate(profile):
                msg += str(load[v]) + ", "
            output.details(msg[:-2] + ")")
            if len(tied_cands) > 1:
                msg = " tie broken in favor of " + profile.cand_names[next_cand]
                msg += ",\n candidates " + str_set_of_candidates(
                    tied_cands, cand_names=profile.cand_names
                )
                msg += f" are tied (for all those new maximum load = {max_load})."
                output.details(msg)
            output.details("")
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )

    if resolute or len(committees) == 1:
        output.details("corresponding load distribution:")
    else:
        output.details("corresponding load distributions:")
    for committee, load in detailed_info["committee_load_pairs"].items():
        msg = f"{str_set_of_candidates(committee, cand_names=profile.cand_names)}: ("
        for v, _ in enumerate(profile):
            msg += str(load[v]) + ", "
        output.details(msg[:-2] + ")\n")
    # end of optional output

    return sorted_committees(committees)


def _seqphragmen_resolute(
    profile, committeesize, algorithm, start_load=None, partial_committee=None
):
    """
    Algorithm for computing resolute seq-Phragmen (1 winning committee).
    """
    if algorithm == "float-fractions":
        division = lambda x, y: x / y  # standard float division
    elif algorithm == "standard-fractions":
        division = Fraction  # using Python built-in fractions
    elif algorithm == "gmpy2-fractions":
        if not gmpy2_available:
            raise ImportError(
                'Module gmpy2 not available, required for algorithm "gmpy2-fractions"'
            )
        division = mpq  # using gmpy2 fractions
    else:
        raise UnknownAlgorithm("seqphragmen", algorithm)

    approvers_weight = {}
    for cand in profile.candidates:
        approvers_weight[cand] = sum(voter.weight for voter in profile if cand in voter.approved)
    load = start_load
    if load is None:
        load = [0 for _ in range(len(profile))]
    max_start_load = max(load)
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
        for cand in profile.candidates:
            if cand in committee:
                new_maxload[cand] = committeesize + 2  # that's larger than any possible value
        # find smallest maxload
        opt = min(new_maxload)
        if algorithm == "float-fractions":
            tied_cands = [
                cand
                for cand in profile.candidates
                if math.isclose(
                    new_maxload[cand],
                    opt,
                    rel_tol=FLOAT_ISCLOSE_REL_TOL,
                    abs_tol=FLOAT_ISCLOSE_ABS_TOL,
                )
            ]
        else:
            tied_cands = [cand for cand in profile.candidates if new_maxload[cand] == opt]
        next_cand = tied_cands[0]
        # compute new loads and add new candidate
        for v, voter in enumerate(profile):
            if next_cand in voter.approved:
                load[v] = new_maxload[next_cand]

        committee = sorted(committee + [next_cand])
        detailed_info["next_cand"].append(next_cand)
        detailed_info["tied_cands"].append(tied_cands)
        detailed_info["load"].append(list(load))  # create copy of `load`
        detailed_info["max_load"].append(opt)

    detailed_info["committee_load_pairs"] = {tuple(committee): load}
    return [committee], detailed_info


def _seqphragmen_irresolute(
    profile,
    committeesize,
    algorithm,
    max_num_of_committees,
    start_load=None,
    partial_committee=None,
):
    """Algorithm for computing irresolute seq-Phragmen (all winning committees)."""
    if algorithm == "float-fractions":
        division = lambda x, y: x / y  # standard float division
    elif algorithm == "standard-fractions":
        division = Fraction  # using Python built-in fractions
    elif algorithm == "gmpy2-fractions":
        if not gmpy2_available:
            raise ImportError(
                'Module gmpy2 not available, required for algorithm "gmpy2-fractions"'
            )
        division = mpq  # using gmpy2 fractions
    else:
        raise UnknownAlgorithm("seqphragmen", algorithm)

    approvers_weight = {}
    for cand in profile.candidates:
        approvers_weight[cand] = sum(voter.weight for voter in profile if cand in voter.approved)

    load = start_load
    if load is None:
        load = {v: 0 for v, _ in enumerate(profile)}

    if partial_committee is None:
        partial_committee = ()  # build committees starting with the empty set
    else:
        partial_committee = tuple(partial_committee)
    committee_load_pairs = [(partial_committee, load)]
    committees = set()
    detailed_info = {"committee_load_pairs": {}}

    while committee_load_pairs:
        committee, load = committee_load_pairs.pop()
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
        for cand in profile.candidates:
            if cand in committee:
                new_maxload[cand] = committeesize + 2  # that's larger than any possible value
        # compute new loads
        new_committee_load_pairs = []
        for cand in profile.candidates:
            if algorithm == "float-fractions":
                select_cand = math.isclose(
                    new_maxload[cand],
                    min(new_maxload),
                    rel_tol=FLOAT_ISCLOSE_REL_TOL,
                    abs_tol=FLOAT_ISCLOSE_ABS_TOL,
                )
            else:
                select_cand = new_maxload[cand] <= min(new_maxload)

            if select_cand:
                new_load = [0] * len(profile)
                for v, voter in enumerate(profile):
                    if cand in voter.approved:
                        new_load[v] = new_maxload[cand]
                    else:
                        new_load[v] = load[v]
                new_committee = committee + (cand,)

                if len(new_committee) == committeesize:
                    new_committee = tuple(sorted(new_committee))
                    committees.add(new_committee)  # remove duplicate committees
                    detailed_info["committee_load_pairs"][new_committee] = new_load
                    if (
                        max_num_of_committees is not None
                        and len(committees) == max_num_of_committees
                    ):
                        # sufficiently many winning committees found
                        return sorted_committees(committees), detailed_info
                else:
                    # partial committee
                    new_committee_load_pairs.append((new_committee, new_load))
        # add new committee/load pairs in reversed order, so that tiebreaking is correct
        committee_load_pairs += reversed(new_committee_load_pairs)

    return sorted_committees(committees), detailed_info


def compute_rule_x(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=True,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
    skip_phragmen_phase=False,
):
    """
    Compute winning committees with Rule X (aka Method of Equal Shares).

    For a mathematical description of this rule, see e.g.
    "Multi-Winner Voting with Approval Preferences"
    by Martin Lackner and Piotr Skowron
    <https://arxiv.org/abs/2007.01795>
    See also <https://arxiv.org/pdf/1911.11747.pdf>, page 7

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for Rule X (aka Method of Equal Shares):

            .. doctest::

                >>> get_rule("rule-x").algorithms
                ('float-fractions', 'gmpy2-fractions', 'standard-fractions')

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

        skip_phragmen_phase : bool, default=False
             Omit the second phase (that uses seq-Phragmen).

             May result in a committee that is too small (length smaller than `committeesize`).

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    if skip_phragmen_phase:
        rule_id = "rule-x-without-phragmen-phase"
    else:
        rule_id = "rule-x"
    rule = get_rule(rule_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )

    if not profile.has_unit_weights():
        raise ValueError(f"{rule.shortname} is only defined for unit weights (weight=1)")

    committees, detailed_info = _rule_x_algorithm(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
        skip_phragmen_phase=skip_phragmen_phase,
    )

    # optional output
    output.info(header(rule.longname))
    if not resolute:
        output.info(
            "Computing all possible winning committees for any tiebreaking order\n"
            " (aka parallel universes tiebreaking) (resolute=False)\n"
        )
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")

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
            if len(tied_cands) > 1:
                msg = " tie broken in favor of "
                msg += profile.cand_names[next_cand] + ","
                msg += "\n candidates "
                msg += str_set_of_candidates(tied_cands, cand_names=profile.cand_names)
                msg += f" are tied (all would impose a maximum cost of {cost})."
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
                    # f"\n (continuous model: time t_{len(committee)} = {max_load})"
                )
                output.details(" load distribution:")
                msg = "  ("
                for v, _ in enumerate(profile):
                    msg += str(load[v]) + ", "
                output.details(msg[:-2] + ")")
                if len(tied_cands) > 1:
                    output.details(
                        f" tie broken in favor of {profile.cand_names[next_cand]},\n"
                        f" candidates "
                        f"{str_set_of_candidates(tied_cands, cand_names=profile.cand_names)}"
                        f" are tied"
                        f" (for any of those, the new maximum load would be {max_load})."
                    )
                output.details("")
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )
    # end of optional output

    return sorted_committees(committees)


def _rule_x_algorithm(
    profile, committeesize, algorithm, resolute, max_num_of_committees, skip_phragmen_phase=False
):
    """Algorithm for Rule X."""

    def _rule_x_get_min_q(profile, budget, cand, division):
        rich = set([v for v, voter in enumerate(profile) if cand in voter.approved])
        poor = set()
        while len(rich) > 0:
            poor_budget = sum(budget[v] for v in poor)
            q = division(1 - poor_budget, len(rich))
            if algorithm == "float-fractions":
                # due to float imprecision, values very close to `q` count as `q`
                new_poor = set(
                    v
                    for v in rich
                    if budget[v] < q
                    and not math.isclose(
                        budget[v],
                        q,
                        rel_tol=FLOAT_ISCLOSE_REL_TOL,
                        abs_tol=FLOAT_ISCLOSE_ABS_TOL,
                    )
                )
            else:
                new_poor = set([v for v in rich if budget[v] < q])
            if len(new_poor) == 0:
                return q
            rich -= new_poor
            poor.update(new_poor)
        return None  # not sufficient budget available

    def find_minimum_dict_entries(dictx):
        if algorithm == "float-fractions":
            min_entries = [
                cand
                for cand in dictx.keys()
                if math.isclose(
                    dictx[cand],
                    min(dictx.values()),
                    rel_tol=FLOAT_ISCLOSE_REL_TOL,
                    abs_tol=FLOAT_ISCLOSE_ABS_TOL,
                )
            ]
        else:
            min_entries = [cand for cand in dictx.keys() if dictx[cand] == min(dictx.values())]
        return min_entries

    def phragmen_phase(_committee, _budget):
        # translate budget to loads
        start_load = [-_budget[v] for v in range(len(profile))]
        detailed_info["phragmen_start_load"] = list(start_load)  # make a copy

        if resolute:
            committees, detailed_info_phragmen = _seqphragmen_resolute(
                profile=profile,
                committeesize=committeesize,
                algorithm=algorithm,
                partial_committee=list(_committee),
                start_load=start_load,
            )
        else:
            committees, detailed_info_phragmen = _seqphragmen_irresolute(
                profile=profile,
                committeesize=committeesize,
                algorithm=algorithm,
                max_num_of_committees=None,
                # TODO: would be nice to have max_num_of_committees=max_num_of_committees
                #       but there is the issue that some of these committees might be
                #       already contained in `winning_committees` - so we need more
                partial_committee=list(_committee),
                start_load=start_load,
            )
        winning_committees.update([tuple(sorted(committee)) for committee in committees])
        detailed_info["phragmen_phase"] = detailed_info_phragmen
        # after filling the remaining spots these committees
        # have size `committeesize`

    if algorithm == "float-fractions":
        division = lambda x, y: x / y  # standard float division
    elif algorithm == "standard-fractions":
        division = Fraction  # using Python built-in fractions
    elif algorithm == "gmpy2-fractions":
        if not gmpy2_available:
            raise ImportError(
                'Module gmpy2 not available, required for algorithm "gmpy2-fractions"'
            )
        division = mpq  # using gmpy2 fractions
    else:
        raise UnknownAlgorithm("rule-x", algorithm)

    if resolute:
        max_num_of_committees = 1  # same algorithm for resolute==True and resolute==False

    start_budget = {v: division(committeesize, len(profile)) for v, _ in enumerate(profile)}
    committee_bugdet_pairs = [(tuple(), start_budget)]
    winning_committees = set()
    detailed_info = {
        "next_cand": [],
        "cost": [],
        "tied_cands": [],
        "budget": [],
        "start_budget": start_budget,
        "phragmen_start_load": None,
    }

    while committee_bugdet_pairs:
        committee, budget = committee_bugdet_pairs.pop()

        available_candidates = [cand for cand in profile.candidates if cand not in committee]
        min_q = {}
        for cand in available_candidates:
            q = _rule_x_get_min_q(profile, budget, cand, division)
            if q is not None:
                min_q[cand] = q

        if len(min_q) > 0:  # one or more candidates are affordable
            # choose those candidates that require the smallest budget
            tied_cands = find_minimum_dict_entries(min_q)

            new_committee_budget_pairs = []
            for next_cand in sorted(tied_cands):
                new_budget = dict(budget)
                for v, voter in enumerate(profile):
                    if next_cand in voter.approved:
                        new_budget[v] -= min(budget[v], min_q[next_cand])

                new_committee = committee + (next_cand,)

                if resolute:
                    detailed_info["next_cand"].append(next_cand)
                    detailed_info["tied_cands"].append(tied_cands)
                    detailed_info["cost"].append(min(min_q.values()))
                    detailed_info["budget"].append(new_budget)

                if len(new_committee) == committeesize:
                    new_committee = tuple(sorted(new_committee))
                    winning_committees.add(new_committee)  # remove duplicate committees
                    if (
                        max_num_of_committees is not None
                        and len(winning_committees) == max_num_of_committees
                    ):
                        # sufficiently many winning committees found
                        return sorted_committees(winning_committees), detailed_info
                else:
                    # partial committee
                    new_committee_budget_pairs.append((new_committee, new_budget))

                if resolute:
                    break

            # add new committee/budget pairs in reversed order, so that tiebreaking is correct
            committee_bugdet_pairs += reversed(new_committee_budget_pairs)

        else:  # no affordable candidates remain
            if skip_phragmen_phase:
                winning_committees.add(tuple(sorted(committee)))
            else:
                # fill committee via seq-Phragmen
                phragmen_phase(committee, budget)

        if max_num_of_committees is not None and len(winning_committees) >= max_num_of_committees:
            winning_committees = sorted_committees(winning_committees)[:max_num_of_committees]
            break

    return sorted_committees(winning_committees), detailed_info


def compute_minimaxphragmen(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=False,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Compute winning committees with Phragmen's minimax rule (minimax-Phragmen).

    Minimizes the maximum load.

    For a mathematical description of this rule, see e.g.
    "Multi-Winner Voting with Approval Preferences"
    by Martin Lackner and Piotr Skowron
    <https://arxiv.org/abs/2007.01795>

    .. important::

        Warning: does not include the lexicographic optimization as specified
        in Markus Brill, Rupert Freeman, Svante Janson and Martin Lackner.
        Phragmen's Voting Methods and Justified Representation.
        <https://arxiv.org/abs/2102.12305>
        Instead: minimizes the maximum load (without consideration of the second-,
        third-, ...-largest load

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for Phragmen's minimax rule (minimax-Phragmen):

            .. doctest::

                >>> get_rule("minimaxphragmen").algorithms
                ('gurobi', 'mip-gurobi', 'mip-cbc')

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    rule_id = "minimaxphragmen"
    rule = get_rule(rule_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )

    if algorithm == "gurobi":
        committees = abcrules_gurobi._gurobi_minimaxphragmen(
            profile,
            committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    elif algorithm.startswith("mip-"):
        committees = abcrules_mip._mip_minimaxphragmen(
            profile,
            committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
            solver_id=algorithm[4:],
        )
    else:
        raise UnknownAlgorithm(rule_id, algorithm)

    # optional output
    output.info(header(rule.longname))
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )
    # end of optional output

    return committees


def compute_leximinphragmen(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=False,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
    lexicographic_tiebreaking=False,
):
    """
    Compute winning committees with Phragmen's leximin rule (leximin-Phragmen).

    Lexicographically minimizes loads.
    Details in
    Markus Brill, Rupert Freeman, Svante Janson and Martin Lackner.
    Phragmen's Voting Methods and Justified Representation.
    <https://arxiv.org/abs/2102.12305>

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for Phragmen's leximin rule (leximin-Phragmen):

            .. doctest::

                >>> get_rule("leximinphragmen").algorithms
                ('gurobi',)

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

        lexicographic_tiebreaking : bool
            Require lexicographic tiebreaking among tied committees.

            This requires the computation of *all* winning committees and is therefore very slow.

            .. important::

                `lexicographic_tiebreaking=True` is only valid in "combination with resolute=True.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    rule_id = "leximinphragmen"
    rule = get_rule(rule_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )

    if lexicographic_tiebreaking:
        if not resolute:
            raise ValueError(
                "lexicographic_tiebreaking=True is only valid in "
                "combination with resolute=True."
            )
        resolute = False  # compute all committees to break ties correctly

    if algorithm == "gurobi":
        committees = abcrules_gurobi._gurobi_leximinphragmen(
            profile,
            committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    # elif algorithm.startswith("mip-"):
    #     committees = abcrules_mip._mip_leximinphragmen(
    #         profile,
    #         committeesize,
    #         resolute=resolute,
    #         max_num_of_committees=max_num_of_committees,
    #         solver_id=algorithm[4:],
    #     )
    else:
        raise UnknownAlgorithm(rule_id, algorithm)

    if lexicographic_tiebreaking:
        committees = sorted_committees(committees)[:1]

    # optional output
    output.info(header(rule.longname))
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )
    # end of optional output

    return committees


def compute_phragmen_enestroem(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=True,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Compute winning committees with Phragmen-Enestroem.

    This ABC rule is also known as Phragmen's first method and Enestroem's method.

    In every round the candidate with the highest combined budget of
    their supporters is put in the committee.
    Method described in:
    Svante Janson
    Phragmén's and Thiele's election methods
    <https://arxiv.org/pdf/1611.08826.pdf> (Section 18.5, Page 59)

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for Phragmen-Enestroem:

            .. doctest::

                >>> get_rule("phragmen-enestroem").algorithms
                ('float-fractions', 'gmpy2-fractions', 'standard-fractions')

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    rule_id = "phragmen-enestroem"
    rule = get_rule(rule_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )

    if not profile.has_unit_weights():
        raise ValueError(f"{rule.shortname} is only defined for unit weights (weight=1)")

    committees, detailed_info = _phragmen_enestroem_algorithm(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )

    # optional output
    output.info(header(rule.longname))
    if not resolute:
        output.info(
            "Computing all possible winning committees for any tiebreaking order\n"
            " (aka parallel universes tiebreaking) (resolute=False)\n"
        )
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )
    # end of optional output

    return committees


def _phragmen_enestroem_algorithm(
    profile, committeesize, algorithm, resolute, max_num_of_committees
):
    """
    Algorithm computing Phragmen-Enestroem.
    """
    if algorithm == "float-fractions":
        division = lambda x, y: x / y  # standard float division
    elif algorithm == "standard-fractions":
        division = Fraction  # using Python built-in fractions
    elif algorithm == "gmpy2-fractions":
        if not gmpy2_available:
            raise ImportError(
                'Module gmpy2 not available, required for algorithm "gmpy2-fractions"'
            )
        division = mpq  # using gmpy2 fractions
    else:
        raise UnknownAlgorithm("phragmen-enestroem", algorithm)

    if resolute:
        max_num_of_committees = 1  # same algorithm for resolute==True and resolute==False

    initial_voter_budget = [voter.weight for voter in profile]
    # price for adding a candidate to the committee
    price = division(sum(initial_voter_budget), committeesize)
    committee_budget_pairs = [(tuple(), initial_voter_budget)]
    committees = set()

    while committee_budget_pairs:
        committee, budget = committee_budget_pairs.pop()
        available_candidates = [cand for cand in profile.candidates if cand not in committee]
        support = {cand: 0 for cand in available_candidates}
        for i, voter in enumerate(profile):
            voting_power = budget[i]
            if voting_power <= 0:
                continue
            for cand in voter.approved:
                if cand in available_candidates:
                    support[cand] += voting_power
        max_support = max(support.values())
        if algorithm == "float-fractions":
            tied_cands = [
                cand
                for cand, supp in support.items()
                if math.isclose(
                    supp,
                    max_support,
                    rel_tol=FLOAT_ISCLOSE_REL_TOL,
                    abs_tol=FLOAT_ISCLOSE_ABS_TOL,
                )
            ]
        else:
            tied_cands = sorted([cand for cand, supp in support.items() if supp == max_support])
        assert tied_cands, "_phragmen_enestroem_algorithm: no candidate with max support (??)"

        new_committee_budget_pairs = []
        for cand in tied_cands:
            new_budget = list(budget)  # copy of budget
            if max_support > price:  # supporters can afford it
                multiplier = division(max_support - price, max_support)
            else:  # supporters can't afford it, set budget to 0
                multiplier = 0
            for i, voter in enumerate(profile):
                if cand in voter.approved:
                    new_budget[i] *= multiplier
            new_committee = committee + (cand,)

            if len(new_committee) == committeesize:
                new_committee = tuple(sorted(new_committee))
                committees.add(new_committee)  # remove duplicate committees
                if max_num_of_committees is not None and len(committees) == max_num_of_committees:
                    # sufficiently many winning committees found
                    detailed_info = {}
                    return sorted_committees(committees), detailed_info
            else:
                # partial committee
                new_committee_budget_pairs.append((new_committee, new_budget))
        # add new committee/budget pairs in reversed order, so that tiebreaking is correct
        committee_budget_pairs += reversed(new_committee_budget_pairs)

    detailed_info = {}
    return sorted_committees(committees), detailed_info


def compute_consensus_rule(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=True,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Compute winning committees with the Consensus rule.

    Based on Perpetual Consensus from
    Martin Lackner Perpetual Voting: Fairness in Long-Term Decision Making
    In Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI 2020)

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for the Consensus rule:

            .. doctest::

                >>> get_rule("consensus-rule").algorithms
                ('float-fractions', 'gmpy2-fractions', 'standard-fractions')

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    rule_id = "consensus-rule"
    rule = get_rule(rule_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )

    committees, detailed_info = _consensus_rule_algorithm(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )

    # optional output
    output.info(header(rule.longname))
    if not resolute:
        output.info(
            "Computing all possible winning committees for any tiebreaking order\n"
            " (aka parallel universes tiebreaking) (resolute=False)\n"
        )
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )
    # end of optional output

    return committees


def _consensus_rule_algorithm(profile, committeesize, algorithm, resolute, max_num_of_committees):
    """
    Algorithm for computing the consensus rule.
    """
    if algorithm == "float-fractions":
        division = lambda x, y: x / y  # standard float division
    elif algorithm == "standard-fractions":
        division = Fraction  # using Python built-in fractions
    elif algorithm == "gmpy2-fractions":
        if not gmpy2_available:
            raise ImportError(
                'Module gmpy2 not available, required for algorithm "gmpy2-fractions"'
            )
        division = mpq  # using gmpy2 fractions
    else:
        raise UnknownAlgorithm("consensus-rule", algorithm)

    if resolute:
        max_num_of_committees = 1  # same algorithm for resolute==True and resolute==False

    initial_voter_budget = [0] * len(profile)
    committee_budget_pairs = [(tuple(), initial_voter_budget)]
    committees = set()

    while committee_budget_pairs:
        committee, budget = committee_budget_pairs.pop()

        for i, _ in enumerate(profile):
            budget[i] += profile[i].weight  # weight is 1 by default
        available_candidates = [cand for cand in profile.candidates if cand not in committee]
        support = {cand: 0 for cand in available_candidates}
        supporters = {cand: [] for cand in available_candidates}
        for i, voter in enumerate(profile):
            if (budget[i] <= 0) or (
                algorithm == "float-fractions"
                and math.isclose(
                    budget[i],
                    0,
                    rel_tol=FLOAT_ISCLOSE_REL_TOL,
                    abs_tol=FLOAT_ISCLOSE_ABS_TOL,
                )
            ):
                continue
            for cand in voter.approved:
                if cand in available_candidates:
                    support[cand] += budget[i]
                    supporters[cand].append(i)
        max_support = max(support.values())
        if algorithm == "float-fractions":
            tied_cands = [
                cand
                for cand, supp in support.items()
                if math.isclose(
                    supp,
                    max_support,
                    rel_tol=FLOAT_ISCLOSE_REL_TOL,
                    abs_tol=FLOAT_ISCLOSE_ABS_TOL,
                )
            ]
        else:
            tied_cands = sorted([cand for cand, supp in support.items() if supp == max_support])
        assert tied_cands, "_consensus_rule_algorithm: no candidate with max support (??)"

        new_committee_budget_pairs = []
        for cand in tied_cands:
            new_budget = list(budget)  # copy of budget
            for i in supporters[cand]:
                new_budget[i] -= division(len(profile), len(supporters[cand]))
            new_committee = committee + (cand,)

            if len(new_committee) == committeesize:
                new_committee = tuple(sorted(new_committee))
                committees.add(new_committee)  # remove duplicate committees
                if max_num_of_committees is not None and len(committees) == max_num_of_committees:
                    # sufficiently many winning committees found
                    detailed_info = {}
                    return sorted_committees(committees), detailed_info
            else:
                # partial committee
                new_committee_budget_pairs.append((new_committee, new_budget))
        # add new committee/budget pairs in reversed order, so that tiebreaking is correct
        committee_budget_pairs += reversed(new_committee_budget_pairs)

    detailed_info = {}
    return sorted_committees(committees), detailed_info


def compute_trivial_rule(
    profile,
    committeesize,
    algorithm="standard",
    resolute=False,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Compute winning committees with the trivial rule (all committees are winning).

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for the trivial rule:

            .. doctest::

                >>> get_rule("trivial").algorithms
                ('standard',)

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    rule_id = "trivial"
    rule = get_rule(rule_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )

    if algorithm == "standard":
        if resolute:
            committees = [range(committeesize)]
        else:
            all_committees = itertools.combinations(profile.candidates, committeesize)
            if max_num_of_committees is None:
                committees = list(all_committees)
            else:
                committees = itertools.islice(all_committees, max_num_of_committees)
        committees = [CandidateSet(comm) for comm in committees]
    else:
        raise UnknownAlgorithm(rule_id, algorithm)

    # optional output
    output.info(header(rule.longname))
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )
    # end of optional output

    return sorted_committees(committees)


def compute_rsd(
    profile, committeesize, algorithm="standard", resolute=True, max_num_of_committees=None
):
    """
    Compute winning committees with the Random Serial Dictator rule.

    This rule randomy selects a permutation of voters. The first voter in this permutation
    adds all approved candidates to the winning committee, then the second voter,
    then the third, etc. At some point, a voter has more approved candidates than
    can be added to the winning committee. In this case, as many as possible are added
    (candidates with smaller index first). In this way, the winning committee is constructed.

    .. important::

        This algorithm is not deterministic and relies on the Python module `random`.

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for Random Serial Dictator:

            .. doctest::

                >>> get_rule("rsd").algorithms
                ('standard',)

        resolute : bool, optional
            Return only one winning committee.

            If `resolute=False`, all winning committees are computed (subject to
            `max_num_of_committees`).

        max_num_of_committees : int, optional
             At most `max_num_of_committees` winning committees are computed.

             If `max_num_of_committees=None`, the number of winning committees is not restricted.
             The default value of `max_num_of_committees` can be modified via the constant
             `MAX_NUM_OF_COMMITTEES_DEFAULT`.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    rule_id = "rsd"
    rule = get_rule(rule_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(profile, committeesize, algorithm, resolute)

    if not profile.has_unit_weights():
        raise ValueError(f"{rule.shortname} is only implemented for unit weights (weight=1).")

    if algorithm == "standard":
        approval_sets = [sorted(voter.approved) for voter in profile]
        # random order of dictators
        random.shuffle(approval_sets)
        committee = set()
        for approved in approval_sets:
            if len(committee) + len(approved) <= committeesize:
                committee.update(approved)
            else:
                for cand in approved:
                    committee.add(cand)
                    if len(committee) == committeesize:
                        break
            if len(committee) == committeesize:
                break
        else:
            remaining_candidates = [cand for cand in profile.candidates if cand not in committee]
            num_missing_candidates = committeesize - len(committee)
            committee.update(random.sample(remaining_candidates, num_missing_candidates))
    else:
        raise UnknownAlgorithm(rule_id, algorithm)

    committees = [committee]

    # optional output
    output.info(header(rule.longname))
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )
    # end of optional output

    return sorted_committees(committees)
