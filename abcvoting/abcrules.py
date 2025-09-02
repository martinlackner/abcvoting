"""Approval-based committee (ABC) voting rules.

Module Attributes
-----------------
MAIN_RULE_IDS : list of str
    List of rule identifiers (`rule_id`) for the main ABC rules included in abcvoting.
    This selection is somewhat arbitrary. But all really important rules (that are implemented)
    are contained in this list.

ALGORITHM_NAMES : dict of str to str
    A dictionary mapping valid algorithm identifiers to their full names or descriptions.
    These identifiers are used to select different solver backends and algorithmic approaches
    within the ABC voting framework.

MAX_NUM_OF_COMMITTEES_DEFAULT : None or int
    The maximum number of committees that an ABC voting rule may return.
    If set to `None`, there is no limit on the number of returned committees.
    This value can be overridden via the `max_num_of_committees` parameter
    in `compute_*` functions.
"""

import functools
import itertools
import random
import math
from fractions import Fraction
from abcvoting.output import output, DETAILS
from abcvoting import abcrules_gurobi, abcrules_ortools, abcrules_mip, abcrules_pulp, misc, scores
from abcvoting.misc import str_committees_with_header, header, str_set_of_candidates
from abcvoting.misc import sorted_committees, CandidateSet

try:
    from gmpy2 import mpq
except ImportError:
    mpq = None


# List of rule identifiers (`rule_id`)
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
    "leximaxphragmen",
    "maximin-support",
    "monroe",
    "greedy-monroe",
    "minimaxav",
    "lexminimaxav",
    "equal-shares",
    "equal-shares-with-av-completion",
    "equal-shares-with-increment-completion",
    "phragmen-enestroem",
    "consensus-rule",
    "trivial",
    "rsd",
    "eph",
]

# A dictionary containing mapping all valid algorithm identifiers
# to full names (i.e., descriptions).
ALGORITHM_NAMES = {
    "gurobi": "Gurobi ILP solver",
    "pulp-highs": "ILP solver via Python PuLP library",
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

# The  maximum number of committees that is returned by an ABC voting rule.
MAX_NUM_OF_COMMITTEES_DEFAULT = None


class Rule:
    """
    A class that contains the main information about an ABC rule.

    Parameters
    ----------
        rule_id : str
            The rule identifier.
    """

    _THIELE_ALGORITHMS = (
        # algorithms sorted by speed
        "gurobi",
        "pulp-highs",
        "mip-gurobi",
        "mip-cbc",
        "branch-and-bound",
        "brute-force",
    )
    _RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES = (False, True)
    _RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES = (True, False)

    def __init__(
        self,
        rule_id,
    ):
        self.rule_id = rule_id
        if rule_id == "av":
            self.shortname = "AV"
            self.longname = "Approval Voting (AV)"
            self.compute_fct = compute_av
            self.algorithms = ("standard",)
            self.resolute_values = self._RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES
        elif rule_id == "sav":
            self.shortname = "SAV"
            self.longname = "Satisfaction Approval Voting (SAV)"
            self.compute_fct = compute_sav
            self.algorithms = ("standard",)
            self.resolute_values = self._RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES
        elif rule_id == "pav":
            self.shortname = "PAV"
            self.longname = "Proportional Approval Voting (PAV)"
            self.compute_fct = compute_pav
            self.algorithms = self._THIELE_ALGORITHMS
            self.resolute_values = self._RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES
        elif rule_id == "slav":
            self.shortname = "SLAV"
            self.longname = "Sainte-Laguë Approval Voting (SLAV)"
            self.compute_fct = compute_slav
            self.algorithms = self._THIELE_ALGORITHMS
            self.resolute_values = self._RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES
        elif rule_id == "cc":
            self.shortname = "CC"
            self.longname = "Approval Chamberlin-Courant (CC)"
            self.compute_fct = compute_cc
            self.algorithms = (
                # algorithms sorted by speed
                "gurobi",
                "pulp-highs",
                "mip-gurobi",
                "ortools-cp",
                "branch-and-bound",
                "brute-force",
                "mip-cbc",
            )
            self.resolute_values = self._RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES
        elif rule_id == "lexcc":
            self.shortname = "lex-CC"
            self.longname = "Lexicographic Chamberlin-Courant (lex-CC)"
            self.compute_fct = compute_lexcc
            # algorithms sorted by speed
            self.algorithms = ("gurobi", "pulp-highs", "brute-force")
            self.resolute_values = self._RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES
        elif rule_id == "seqpav":
            self.shortname = "seq-PAV"
            self.longname = "Sequential Proportional Approval Voting (seq-PAV)"
            self.compute_fct = compute_seqpav
            self.algorithms = ("standard",)
            self.resolute_values = self._RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES
        elif rule_id == "revseqpav":
            self.shortname = "revseq-PAV"
            self.longname = "Reverse Sequential Proportional Approval Voting (revseq-PAV)"
            self.compute_fct = compute_revseqpav
            self.algorithms = ("standard",)
            self.resolute_values = self._RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES
        elif rule_id == "seqslav":
            self.shortname = "seq-SLAV"
            self.longname = "Sequential Sainte-Laguë Approval Voting (seq-SLAV)"
            self.compute_fct = compute_seqslav
            self.algorithms = ("standard",)
            self.resolute_values = self._RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES
        elif rule_id == "seqcc":
            self.shortname = "seq-CC"
            self.longname = "Sequential Approval Chamberlin-Courant (seq-CC)"
            self.compute_fct = compute_seqcc
            self.algorithms = ("standard",)
            self.resolute_values = self._RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES
        elif rule_id == "seqphragmen":
            self.shortname = "seq-Phragmén"
            self.longname = "Phragmén's Sequential Rule (seq-Phragmén)"
            self.compute_fct = compute_seqphragmen
            self.algorithms = ("float-fractions", "gmpy2-fractions", "standard-fractions")
            self.resolute_values = self._RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES
        elif rule_id == "minimaxphragmen":
            self.shortname = "minimax-Phragmén"
            self.longname = "Phragmén's Minimax Rule (minimax-Phragmén)"
            self.compute_fct = compute_minimaxphragmen
            self.algorithms = ("gurobi", "pulp-highs", "mip-gurobi", "mip-cbc")
            self.resolute_values = self._RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES
        elif rule_id == "leximaxphragmen":
            self.shortname = "leximax-Phragmén"
            self.longname = "Phragmén's Leximax Rule (leximax-Phragmén)"
            self.compute_fct = compute_leximaxphragmen
            self.algorithms = ("gurobi",)  # "pulp-highs" is too slow
            self.resolute_values = self._RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES
        elif rule_id == "maximin-support":
            self.shortname = "Maximin-Support"
            self.longname = "Maximin Support Method (MMS)"
            self.compute_fct = compute_maximin_support
            self.algorithms = ("gurobi", "mip-gurobi", "mip-cbc")
            self.resolute_values = self._RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES
        elif rule_id == "monroe":
            self.shortname = "Monroe"
            self.longname = "Monroe's Approval Rule (Monroe)"
            self.compute_fct = compute_monroe
            self.algorithms = (
                # algorithms sorted by speed
                "gurobi",
                "pulp-highs",
                "mip-gurobi",
                "mip-cbc",
                "ortools-cp",
                "brute-force",
            )
            self.resolute_values = self._RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES
        elif rule_id == "greedy-monroe":
            self.shortname = "Greedy Monroe"
            self.longname = "Greedy Monroe"
            self.compute_fct = compute_greedy_monroe
            self.algorithms = ("standard",)
            self.resolute_values = (True,)
        elif rule_id == "minimaxav":
            self.shortname = "minimaxav"
            self.longname = "Minimax Approval Voting (MAV)"
            self.compute_fct = compute_minimaxav
            self.algorithms = (
                "gurobi",
                "pulp-highs",
                "mip-gurobi",
                "ortools-cp",
                "mip-cbc",
                "brute-force",
            )
            # algorithms sorted by speed. however, for small profiles with a small committee size,
            # brute-force is often the fastest
            self.resolute_values = self._RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES
        elif rule_id == "lexminimaxav":
            self.shortname = "lex-MAV"
            self.longname = "Lexicographic Minimax Approval Voting (lex-MAV)"
            self.compute_fct = compute_lexminimaxav
            self.algorithms = ("gurobi", "pulp-highs", "brute-force")
            self.resolute_values = self._RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES
        elif rule_id in ["rule-x", "equal-shares", "equal-shares-with-seqphragmen-completion"]:
            self.shortname = "Equal Shares"
            self.longname = "Method of Equal Shares (aka Rule X) with Phragmén phase"
            self.compute_fct = compute_equal_shares
            self.algorithms = ("float-fractions", "gmpy2-fractions", "standard-fractions")
            self.resolute_values = self._RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES
        elif rule_id in ["equal-shares-with-av-completion"]:
            self.shortname = "Equal Shares with AV completion"
            self.longname = "Method of Equal Shares (aka Rule X) with AV completion"
            self.compute_fct = functools.partial(compute_equal_shares, completion="av")
            self.algorithms = ("float-fractions", "gmpy2-fractions", "standard-fractions")
            self.resolute_values = self._RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES
        elif rule_id in ["equal-shares-with-increment-completion"]:
            self.shortname = "Equal Shares with increment completion"
            self.longname = "Method of Equal Shares (aka Rule X) with increment completion"
            self.compute_fct = functools.partial(compute_equal_shares, completion="increment")
            self.algorithms = ("float-fractions", "gmpy2-fractions", "standard-fractions")
            self.resolute_values = (True,)  # this rule is ill-defined for resolute=False
        elif rule_id in [
            "rule-x-without-phragmen-phase",
            "equal-shares-without-phragmen-phase",
            "equal-shares-without-completion",
        ]:
            self.shortname = "Equal Shares without completion"
            self.longname = "Method of Equal Shares (aka Rule X) without completion (second phase)"
            self.compute_fct = functools.partial(compute_equal_shares, completion=None)
            self.algorithms = ("float-fractions", "gmpy2-fractions", "standard-fractions")
            self.resolute_values = self._RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES
        elif rule_id == "phragmen-enestroem":
            self.shortname = "Phragmén-Eneström"
            self.longname = "Method of Phragmén-Eneström"
            self.compute_fct = compute_phragmen_enestroem
            self.algorithms = ("float-fractions", "gmpy2-fractions", "standard-fractions")
            self.resolute_values = self._RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES
        elif rule_id == "consensus-rule":
            self.shortname = "Consensus Rule"
            self.longname = "Consensus Rule"
            self.compute_fct = compute_consensus_rule
            self.algorithms = ("float-fractions", "gmpy2-fractions", "standard-fractions")
            self.resolute_values = self._RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES
        elif rule_id == "trivial":
            self.shortname = "Trivial Rule"
            self.longname = "Trivial Rule"
            self.compute_fct = compute_trivial_rule
            self.algorithms = ("standard",)
            self.resolute_values = self._RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES
        elif rule_id == "rsd":
            self.shortname = "Random Serial Dictator"
            self.longname = "Random Serial Dictator"
            self.compute_fct = compute_rsd
            self.algorithms = ("standard",)
            self.resolute_values = (True,)
        elif rule_id == "eph":
            self.shortname = "E Pluribus Hugo"
            self.longname = "E Pluribus Hugo (EPH)"
            self.compute_fct = compute_eph
            self.algorithms = ("float-fractions", "gmpy2-fractions", "standard-fractions")
            self.resolute_values = (False, True)
        elif rule_id.startswith("geom"):
            parameter = rule_id[4:]
            self.shortname = f"{parameter}-Geometric"
            self.longname = f"{parameter}-Geometric Rule"
            self.compute_fct = functools.partial(compute_thiele_method, rule_id)
            self.algorithms = self._THIELE_ALGORITHMS
            self.resolute_values = self._RESOLUTE_VALUES_FOR_OPTIMIZATION_BASED_RULES
        elif rule_id.startswith("seq") or rule_id.startswith("revseq"):
            # handle sequential and reverse sequential Thiele methods
            # that are not explicitly included in the list above
            if rule_id.startswith("seq"):
                scorefct_id = rule_id[3:]  # score function id of Thiele method
            else:
                scorefct_id = rule_id[6:]  # score function id of Thiele method

            try:
                scores.get_marginal_scorefct(scorefct_id)
            except scores.UnknownScoreFunctionError as error:
                raise UnknownRuleIDError(rule_id) from error

            if rule_id == "av":
                raise UnknownRuleIDError(rule_id)  # seq-AV and revseq-AV are equivalent to AV

            # sequential Thiele methods
            optrule = Rule(scorefct_id)
            if rule_id.startswith("seq"):
                self.shortname = f"seq-{optrule.shortname}"
                self.longname = f"Sequential {optrule.longname}"
                self.compute_fct = functools.partial(compute_seq_thiele_method, scorefct_id)
                self.algorithms = ("standard",)
                self.resolute_values = self._RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES
            # reverse sequential Thiele methods
            elif rule_id.startswith("revseq"):
                self.shortname = f"revseq-{optrule.shortname}"
                self.longname = f"Reverse Sequential {optrule.longname}"
                self.compute_fct = functools.partial(compute_revseq_thiele_method, scorefct_id)
                self.algorithms = ("standard",)
                self.resolute_values = self._RESOLUTE_VALUES_FOR_SEQUENTIAL_RULES
        else:
            raise UnknownRuleIDError(rule_id)

        # find all *available* algorithms for this ABC rule
        self.available_algorithms = []
        for algorithm in self.algorithms:
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
        return self.compute_fct(profile, committeesize, **kwargs)

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
            raise ValueError("Parameter `committeesize` must be a positive integer.")

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
                f'ABC rule with rule_id "{self.rule_id}" does not support resolute={resolute}.'
            )

        if (max_num_of_committees is not None and not isinstance(max_num_of_committees, int)) or (
            max_num_of_committees is not None and max_num_of_committees < 1
        ):
            raise ValueError(
                "Parameter `max_num_of_committees` must be None or a positive integer."
            )

        if max_num_of_committees is not None and resolute:
            raise ValueError(
                "Parameter `max_num_of_committees` cannot be used when `resolute` is set to True."
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
        message = f'Rule ID "{rule_id}" is not known.'
        super().__init__(message)


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
        super().__init__(message)


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
        super().__init__(message)


def _available_algorithms():
    """Verify which algorithms are supported on the current machine.

    This is done by verifying that the required modules and solvers are available.
    """
    available = []

    for algorithm in ALGORITHM_NAMES:
        if "gurobi" in algorithm and not abcrules_gurobi.gb:
            continue
        if algorithm.startswith("pulp-") and not abcrules_pulp.pulp:
            continue
        if algorithm == "gmpy2-fractions" and not mpq:
            continue
        if algorithm == "ortools-cp" and not abcrules_ortools.cp_model:
            continue
        if algorithm.startswith("mip-") and abcrules_mip.mip is None:
            continue
        available.append(algorithm)

    return available


available_algorithms = _available_algorithms()


def get_rule(rule_id):
    """
    Get instance of `Rule` for the ABC rule specified by `rule_id`.

    .. deprecated:: 2.3.0
       Function `get_rule(rule_id)` is deprecated, use `Rule(rule_id)` instead.

    Parameters
    ----------
        rule_id : str
            The rule identifier.

    Returns
    -------
        Rule
            A corresponding `Rule` object.
    """
    return Rule(rule_id)


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
    rule = Rule(rule_id)
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
    lexicographic_tiebreaking=False,
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
    "Multi-Winner Voting with Approval Preferences".
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>

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

        lexicographic_tiebreaking : bool
            Require lexicographic tiebreaking among tied committees.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.

            If `resolute=True`, the list contains only one winning committee.
    """
    rule = Rule(scorefct_id)
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
            lexicographic_tiebreaking=lexicographic_tiebreaking,
        )
    elif algorithm.startswith("pulp-"):
        committees = abcrules_pulp._pulp_thiele_methods(
            scorefct_id=scorefct_id,
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
            solver_id=algorithm[5:],
            lexicographic_tiebreaking=lexicographic_tiebreaking,
        )
    elif algorithm == "branch-and-bound":
        # lexicographic tiebreaking works automatically for brute-force
        committees, detailed_info = _thiele_methods_branchandbound(
            scorefct_id=scorefct_id,
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    elif algorithm == "brute-force":
        # lexicographic tiebreaking works automatically for brute-force
        committees, detailed_info = _thiele_methods_bruteforce(
            scorefct_id=scorefct_id,
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    elif algorithm.startswith("mip-"):
        if lexicographic_tiebreaking:
            raise NotImplementedError(
                f"Lexicographic tiebreaking is not implemented for {algorithm}."
            )
        committees = abcrules_mip._mip_thiele_methods(
            scorefct_id=scorefct_id,
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
            solver_id=algorithm[4:],
        )
    elif algorithm == "ortools-cp" and scorefct_id == "cc":
        if lexicographic_tiebreaking:
            raise NotImplementedError(
                f"Lexicographic tiebreaking is not implemented for {algorithm}."
            )
        committees = abcrules_ortools._ortools_cc(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    else:
        raise UnknownAlgorithm(scorefct_id, algorithm)

    # optional output
    output.info(header(rule.longname), wrap=False)
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
    lexicographic_tiebreaking=False,
):
    """
    Compute winning committees with Proportional Approval Voting (PAV).

    This ABC rule belongs to the class of Thiele methods.

    For a mathematical description of this rule, see e.g.
    "Multi-Winner Voting with Approval Preferences".
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>

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

                >>> Rule("pav").algorithms
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

        lexicographic_tiebreaking : bool
            Require lexicographic tiebreaking among tied committees.

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
        lexicographic_tiebreaking=lexicographic_tiebreaking,
    )


def compute_slav(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=False,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
    lexicographic_tiebreaking=False,
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

                >>> Rule("slav").algorithms
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

        lexicographic_tiebreaking : bool
            Require lexicographic tiebreaking among tied committees.

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
        lexicographic_tiebreaking=lexicographic_tiebreaking,
    )


def compute_cc(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=False,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
    lexicographic_tiebreaking=False,
):
    """
    Compute winning committees with Approval Chamberlin-Courant (CC).

    This ABC rule belongs to the class of Thiele methods.

    For a mathematical description of this rule, see e.g.
    "Multi-Winner Voting with Approval Preferences".
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>

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

                >>> Rule("cc").algorithms  # doctest: +NORMALIZE_WHITESPACE
                ('gurobi', 'mip-gurobi', 'ortools-cp', 'branch-and-bound', 'brute-force',
                 'mip-cbc')

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
        lexicographic_tiebreaking=lexicographic_tiebreaking,
    )


def compute_lexcc(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=False,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
    lexicographic_tiebreaking=False,
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

    .. important::

        Very slow due to lexicographic optimization.

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

                >>> Rule("lexcc").algorithms
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

        lexicographic_tiebreaking : bool
            Require lexicographic tiebreaking among tied committees.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    rule_id = "lexcc"
    rule = Rule(rule_id)
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
        # Lexicographic tie-breaking works automatically for brute-force
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
            lexicographic_tiebreaking=lexicographic_tiebreaking,
        )
    elif algorithm.startswith("pulp-"):
        committees, detailed_info = abcrules_pulp._pulp_lexcc(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
            solver_id=algorithm[5:],
            lexicographic_tiebreaking=lexicographic_tiebreaking,
        )
    elif algorithm.startswith("mip-"):
        # lexicographic tiebreaking works automatically for brute-force
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
    output.info(header(rule.longname), wrap=False)
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")
    output.details("At-least-ell scores:")
    for ell, score in enumerate(detailed_info["opt_score_vector"]):
        output.details(f"at-least-{ell + 1}: {score}", indent=" ")
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
    "Multi-Winner Voting with Approval Preferences".
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>

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
    scores.get_marginal_scorefct(scorefct_id, committeesize)  # check that `scorefct_id` is valid
    rule_id = "seq" + scorefct_id
    rule = Rule(rule_id)
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
    output.info(header(rule.longname), wrap=False)
    if not resolute:
        output.info("Computing all possible winning committees for any tiebreaking order")
        output.info(" (aka parallel universes tiebreaking) (resolute=False)\n")
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
                output.details(f"adding candidate number {i + 1}: {profile.cand_names[next_cand]}")
                output.details(
                    f"score increases by {delta_score} to"
                    f" a total of {scores.thiele_score(scorefct_id, profile, committee)}",
                    indent=" ",
                )
                if len(tied_cands) > 1:
                    output.details(f"tie broken in favor of {next_cand},\n", indent=" ")
                    output.details(
                        f"candidates "
                        f"{str_set_of_candidates(tied_cands, cand_names=profile.cand_names)} "
                        "are tied"
                    )
                    output.details(
                        f"(all would increase the score by the same amount {delta_score})",
                        indent=" ",
                    )
                output.details("")
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )

    if output.verbosity <= DETAILS:  # skip thiele_score() calculations if not necessary
        output.details(scorefct_id.upper() + "-score of winning committee(s):")
        for committee in committees:
            output.details(
                f"{str_set_of_candidates(committee, cand_names=profile.cand_names)}: "
                f"{scores.thiele_score(scorefct_id, profile, committee)}",
                indent=" ",
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
    "Multi-Winner Voting with Approval Preferences".
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>

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

                >>> Rule("seqpav").algorithms
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

                >>> Rule("seqslav").algorithms
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
    "Multi-Winner Voting with Approval Preferences".
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>

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

                >>> Rule("seqcc").algorithms
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
    "Multi-Winner Voting with Approval Preferences".
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>

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
    rule = Rule(rule_id)
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
    output.info(header(rule.longname), wrap=False)
    if not resolute:
        output.info("Computing all possible winning committees for any tiebreaking order")
        output.info(" (aka parallel universes tiebreaking) (resolute=False)\n")
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
                f"score decreases by {delta_score} to a total of "
                f"{scores.thiele_score(scorefct_id, profile, committee)}",
                indent=" ",
            )
            if len(tied_cands) > 1:
                output.details(f"tie broken to the disadvantage of {next_cand},", indent=" ")
                output.details(
                    f"candidates "
                    f"{str_set_of_candidates(tied_cands, cand_names=profile.cand_names)}"
                    " are tied",
                    indent=" ",
                )
                output.details(
                    f"(all would decrease the score by the same amount {delta_score})", indent=" "
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
        # find smallest elements in `marg_util_cand` and return indices
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
                marginal_scorefct, profile, set(committee)
            )
            score_reduction = min(marg_util_cand)
            # find smallest elements in `marg_util_cand` and return indices
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
    "Multi-Winner Voting with Approval Preferences".
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>

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

                >>> Rule("revseqpav").algorithms
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

                >>> Rule("av").algorithms
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
    rule = Rule(rule_id)
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
    output.info(header(rule.longname), wrap=False)
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


def _separable_rule_algorithm(
    rule_id, profile, committeesize, resolute, max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT
):
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
        committees = sorted_committees([certain_cands + possible_cands[:missing]])
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
    "Multi-Winner Voting with Approval Preferences".
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>

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

                >>> Rule("sav").algorithms
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
    "Multi-Winner Voting with Approval Preferences".
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>

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

                >>> Rule("av").algorithms
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
    lexicographic_tiebreaking=False,
):
    """
    Compute winning committees with Minimax Approval Voting (MAV).

    For a mathematical description of this rule, see e.g.
    "Multi-Winner Voting with Approval Preferences".
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>

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

                >>> Rule("minimaxav").algorithms
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

        lexicographic_tiebreaking : bool
            Require lexicographic tiebreaking among tied committees.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    rule_id = "minimaxav"
    rule = Rule(rule_id)
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
            lexicographic_tiebreaking=lexicographic_tiebreaking,
        )
    elif algorithm.startswith("pulp-"):
        committees = abcrules_pulp._pulp_minimaxav(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
            solver_id=algorithm[5:],
            lexicographic_tiebreaking=lexicographic_tiebreaking,
        )
    elif algorithm == "ortools-cp":
        if lexicographic_tiebreaking:
            raise NotImplementedError(f"Lexicographic tiebreaking not available with {algorithm}.")
        committees = abcrules_ortools._ortools_minimaxav(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    elif algorithm.startswith("mip-"):
        if lexicographic_tiebreaking:
            raise NotImplementedError(f"Lexicographic tiebreaking not available with {algorithm}.")
        solver_id = algorithm[4:]
        committees = abcrules_mip._mip_minimaxav(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
            solver_id=solver_id,
        )
    elif algorithm == "brute-force":
        # Lexicographic tiebreaking works automatically for brute-force
        committees, detailed_info = _minimaxav_bruteforce(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    else:
        raise UnknownAlgorithm(rule_id, algorithm)

    # optional output
    output.info(header(rule.longname), wrap=False)
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
    lexicographic_tiebreaking=False,
):
    """
    Compute winning committees with Lexicographic Minimax AV (lex-MAV).

    For a mathematical description of this rule, see e.g.
    "Multi-Winner Voting with Approval Preferences".
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>
    (Remark 2)

    If `lexicographic_tiebreaking` is True, compute all winning committees and choose the
    lexicographically smallest. This is a deterministic form of tiebreaking; if only resolute=True,
    it is not guaranteed how ties are broken.

    .. important::

        Very slow due to lexicographic optimization.

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

                >>> Rule("lexminimaxav").algorithms
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

        lexicographic_tiebreaking : bool
            Require lexicographic tiebreaking among tied committees.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    rule_id = "lexminimaxav"
    rule = Rule(rule_id)
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
        # Lexicographic tiebreaking works automatically for brute-force
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
            lexicographic_tiebreaking=lexicographic_tiebreaking,
        )
    elif algorithm.startswith("pulp-"):
        committees, detailed_info = abcrules_pulp._pulp_lexminimaxav(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
            solver_id=algorithm[5:],
            lexicographic_tiebreaking=lexicographic_tiebreaking,
        )
    else:
        raise UnknownAlgorithm(rule_id, algorithm)

    # optional output
    opt_distances = detailed_info["opt_distances"]
    output.info(header(rule.longname), wrap=False)
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
            (misc.hamming(voter.approved, set(committee)) for voter in profile), reverse=True
        )
        for i, dist in enumerate(distances):
            if opt_distances[i] < dist:
                break
            if opt_distances[i] > dist:
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
    lexicographic_tiebreaking=False,
):
    """
    Compute winning committees with Monroe's rule.

    For a mathematical description of this rule, see e.g.
    "Multi-Winner Voting with Approval Preferences".
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>

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

                >>> Rule("monroe").algorithms
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

        lexicographic_tiebreaking : bool
            Require lexicographic tiebreaking among tied committees.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    rule_id = "monroe"
    rule = Rule(rule_id)
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
            lexicographic_tiebreaking=lexicographic_tiebreaking,
        )
    elif algorithm.startswith("pulp-"):
        committees = abcrules_pulp._pulp_monroe(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
            solver_id=algorithm[5:],
            lexicographic_tiebreaking=lexicographic_tiebreaking,
        )
    elif algorithm == "ortools-cp":
        if lexicographic_tiebreaking:
            raise NotImplementedError(f"Lexicographic tiebreaking not available with {algorithm}.")
        committees = abcrules_ortools._ortools_monroe(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    elif algorithm.startswith("mip-"):
        if lexicographic_tiebreaking:
            raise NotImplementedError(f"Lexicographic tiebreaking not available with {algorithm}.")
        committees = abcrules_mip._mip_monroe(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
            solver_id=algorithm[4:],
        )
    elif algorithm == "brute-force":
        # Lexicographic tiebreaking works automatically for brute-force
        committees, detailed_info = _monroe_bruteforce(
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
        )
    else:
        raise UnknownAlgorithm(rule_id, algorithm)

    # optional output
    output.info(header(rule.longname), wrap=False)
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
    "Multi-Winner Voting with Approval Preferences".
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>

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

                >>> Rule("greedy-monroe").algorithms
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
    rule = Rule(rule_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(
        profile, committeesize, algorithm, resolute, max_num_of_committees
    )

    if not profile.has_unit_weights():
        raise ValueError(f"{rule.shortname} is only defined for unit weights (weight=1)")

    if algorithm == "standard":
        committees, detailed_info = _greedy_monroe_algorithm(profile, committeesize)
    else:
        raise UnknownAlgorithm(rule_id, algorithm)

    # optional output
    output.info(header(rule.longname), wrap=False)
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
    "Multi-Winner Voting with Approval Preferences".
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>

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

                >>> Rule("seqphragmen").algorithms
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
    rule = Rule(rule_id)
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
    output.info(header(rule.longname), wrap=False)
    if not resolute:
        output.info("Computing all possible winning committees for any tiebreaking order")
        output.info(" (aka parallel universes tiebreaking) (resolute=False)\n")
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")

    if resolute:
        committee = []
        for i, next_cand in enumerate(detailed_info["next_cand"]):
            tied_cands = detailed_info["tied_cands"][i]
            max_load = detailed_info["max_load"][i]
            load = detailed_info["load"][i]
            committee.append(next_cand)
            output.details(f"adding candidate number {i + 1}: {profile.cand_names[next_cand]}")
            output.details(
                f"maximum load increased to {max_load}",
                indent=" ",
                # f"\n (continuous model: time t_{i+1} = {max_load})"
            )
            output.details(" load distribution:")
            msg = "("
            for v, _ in enumerate(profile):
                msg += str(load[v]) + ", "
            output.details(msg[:-2] + ")", indent="  ")
            if len(tied_cands) > 1:
                output.details(
                    f"tie broken in favor of {profile.cand_names[next_cand]},", indent=" "
                )
                output.details(
                    "candidates "
                    f"{str_set_of_candidates(tied_cands, cand_names=profile.cand_names)}"
                    f" are tied",
                    indent=" ",
                )
                output.details(
                    f"(for all those new maximum load = {max_load}).",
                    indent=" ",
                )
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
        if not mpq:
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
            (
                division(approvers_load[cand] + 1, approvers_weight[cand])
                if approvers_weight[cand] > 0
                else committeesize + 1
            )
            for cand in profile.candidates
        ]
        # exclude committees already in the committee
        for cand in profile.candidates:
            if cand in committee:
                new_maxload[cand] = committeesize + 2  # that's larger than any possible value
        opt = min(new_maxload)
        if algorithm == "float-fractions":
            tied_cands = [
                cand for cand in profile.candidates if misc.isclose(new_maxload[cand], opt)
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
        if not mpq:
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
            (
                division(approvers_load[cand] + 1, approvers_weight[cand])
                if approvers_weight[cand] > 0
                else committeesize + 1
            )
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
                select_cand = misc.isclose(new_maxload[cand], min(new_maxload))
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
    Compute winning committees with the Method of Equal Shares (aka Rule X).

    .. deprecated:: 2.4.0
        Use :func:`compute_equal_shares` instead. (Rule X has been renamed by the authors
        to Method of Equal Shares and this appears to be the new standard name used in the
        literature by now.)

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for the Method of Equal Shares:

            .. doctest::

                >>> Rule("equal-shares").algorithms
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
        completion = None
    else:
        completion = "seqphragmen"
    return compute_equal_shares(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
        completion=completion,
    )


def compute_equal_shares(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=True,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
    completion="seqphragmen",
):
    """
    Compute winning committees with the Method of Equal Shares (aka Rule X).

    For a mathematical description of this rule, see e.g.
    "Multi-Winner Voting with Approval Preferences".
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>
    See also <https://arxiv.org/pdf/1911.11747.pdf>, page 7

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for the Method of Equal Shares:

            .. doctest::

                >>> Rule("equal-shares").algorithms
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

        completion : str, default="seqphragmen"
             As Equal Shares does not necessarily return the desired number of committee members,
             it requires an additional method to fill the remaining seats. The default is
             to use Sequential Phragmén.

             The following options are available:

             - `"seqphragmen"`: Use Sequential Phragmén to fill seats.
             - `"av"`: Use Approval Voting to fill seats (i.e., take those with most approvals).
             - `"increment"`: Increase the budget of voters by virtually incrementing
               the committee size. This step is repeated until a committee of size `committeesize`
               is found.
             - None: Do not fill the remaining seats. The resulting committees may contain
               fewer than `committeesize` members.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    if not completion:
        rule_id = "equal-shares-without-completion"
    elif completion == "seqphragmen":
        rule_id = "equal-shares"
    elif completion == "av":
        rule_id = "equal-shares-with-av-completion"
    elif completion == "increment":
        rule_id = "equal-shares-with-increment-completion"
    else:
        raise ValueError(f"completion argument {completion} unknown.")
    rule = Rule(rule_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )

    if completion == "increment":
        committees, detailed_info = _equal_shares_algorithm_with_increment_completion(
            profile=profile,
            committeesize=committeesize,
            algorithm=algorithm,
        )
    else:
        committees, detailed_info = _equal_shares_algorithm(
            profile=profile,
            committeesize=committeesize,
            algorithm=algorithm,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
            completion=completion,
        )

    # optional output
    output.info(header(rule.longname), wrap=False)
    if not resolute:
        output.info("Computing all possible winning committees for any tiebreaking order")
        output.info(" (aka parallel universes tiebreaking) (resolute=False)\n")
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")

    if "too_few_approved_candidates" in detailed_info.keys():
        output.details(
            "There are fewer candidates approved by at least one voter than"
            " the desired committee size. Thus Equal Shares returns all approved candidates"
            " and fills the committee with non-approved candidates."
        )
    elif completion == "increment":
        output.details("Incrementing starting budget of voters to fill the committee.")
        output.details(
            "Successful for a (virtual) committee size of "
            f"{detailed_info['increment_committeesize']}."
        )
    elif resolute:
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
            output.details(f"adding candidate number {i + 1}: {profile.cand_names[next_cand]}")
            output.details(f"with maxmimum cost per voter q = {cost}", indent=" ")
            output.details(" remaining budget:")
            msg = "("
            for v, _ in enumerate(profile):
                msg += str(budget[v]) + ", "
            output.details(msg[:-2] + ")", indent="  ")
            if len(tied_cands) > 1:
                output.details(
                    f"tie broken in favor of {profile.cand_names[next_cand]},", indent=" "
                )
                output.details(
                    "candidates "
                    f"{str_set_of_candidates(tied_cands, cand_names=profile.cand_names)}"
                    f" are tied",
                    indent=" ",
                )
                output.details(f"(all would impose a maximum cost of {cost}).", indent=" ")
            output.details("")

        if detailed_info["phragmen_start_load"]:  # the second phase (seq-Phragmen) was used
            phragmen_start_load = detailed_info["phragmen_start_load"]
            output.details("Phase 2 (seq-Phragmén):\n")
            output.details("starting loads (= budget spent):")
            msg = "("
            for v, _ in enumerate(profile):
                msg += str(phragmen_start_load[v]) + ", "
            output.details(msg[:-2] + ")\n", indent="  ")

            detailed_info_phragmen = detailed_info["phragmen_phase"]
            for i, next_cand in enumerate(detailed_info_phragmen["next_cand"]):
                tied_cands = detailed_info_phragmen["tied_cands"][i]
                max_load = detailed_info_phragmen["max_load"][i]
                load = detailed_info_phragmen["load"][i]
                committee.append(next_cand)
                output.details(
                    f"adding candidate number {len(committee)}: {profile.cand_names[next_cand]}"
                )
                output.details(
                    f"maximum load increased to {max_load}",
                    indent=" ",
                    # f"\n (continuous model: time t_{len(committee)} = {max_load})"
                )
                output.details(" load distribution:")
                msg = "("
                for v, _ in enumerate(profile):
                    msg += str(load[v]) + ", "
                output.details(msg[:-2] + ")", indent="  ")
                if len(tied_cands) > 1:
                    output.details(
                        f"tie broken in favor of {profile.cand_names[next_cand]},", indent=" "
                    )
                    output.details(
                        "candidates "
                        f"{str_set_of_candidates(tied_cands, cand_names=profile.cand_names)}"
                        " are tied",
                        indent=" ",
                    )
                    output.details(
                        f"(for any of those, the new maximum load would be {max_load}).",
                        indent=" ",
                    )
                output.details("")
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )
    # end of optional output

    return sorted_committees(committees)


def _equal_shares_algorithm(
    profile,
    committeesize,
    algorithm,
    resolute,
    max_num_of_committees=None,
    completion="seqphragmen",
    per_voter_budget=None,
):
    """Algorithm for the Method of Equal Shares."""

    def _equal_shares_get_min_q(profile, budget, cand, division):
        rich = {v for v, voter in enumerate(profile) if cand in voter.approved}
        poor = set()
        while len(rich) > 0:
            poor_budget = sum(budget[v] for v in poor)
            _q = division(1 - poor_budget, sum(profile[v].weight for v in rich))
            if algorithm == "float-fractions":
                # due to float imprecision, values very close to `q` count as `q`
                new_poor = {
                    v
                    for v in rich
                    if budget[v] < _q * profile[v].weight
                    and not misc.isclose(budget[v], _q * profile[v].weight)
                }
            else:
                new_poor = {v for v in rich if budget[v] < _q * profile[v].weight}
            if len(new_poor) == 0:
                return _q
            rich -= new_poor
            poor.update(new_poor)
        return None  # not sufficient budget available

    def find_minimum_dict_entries(dictx):
        if algorithm == "float-fractions":
            min_entries = [
                cand for cand in dictx.keys() if misc.isclose(dictx[cand], min(dictx.values()))
            ]
        else:
            min_entries = [cand for cand in dictx.keys() if dictx[cand] == min(dictx.values())]
        return min_entries

    def phragmen_phase(_committee, _budget):
        # translate budget to loads
        start_load = [-_budget[v] / profile[v].weight for v in range(len(profile))]
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
        # after filling the remaining spots these committees have size `committeesize`

    def av_phase(_committee):
        num_missing = committeesize - len(_committee)
        for _ in range(committeesize):
            new_winning_committees = []
            av_committees, detailed_info_av = _separable_rule_algorithm(
                rule_id="av",
                profile=profile,
                committeesize=num_missing,
                resolute=False,
                max_num_of_committees=None,
            )
            for comm in av_committees:
                new_comm = set(_committee).union(comm)
                if len(new_comm) == committeesize:
                    new_winning_committees.append(new_comm)
                if len(new_comm) > committeesize:
                    raise RuntimeError("Critical bug. This condition should not be satisfiable.")

            if new_winning_committees:
                winning_committees.update(
                    [tuple(sorted(committee)) for committee in new_winning_committees]
                )
                detailed_info["av_phase"] = detailed_info_av
                break
            else:
                num_missing += 1
        else:
            raise RuntimeError("Critical bug. This for-loop should terminate earlier.")

    if algorithm == "float-fractions":
        division = lambda x, y: x / y  # standard float division
    elif algorithm == "standard-fractions":
        division = Fraction  # using Python built-in fractions
    elif algorithm == "gmpy2-fractions":
        if not mpq:
            raise ImportError(
                'Module gmpy2 not available, required for algorithm "gmpy2-fractions"'
            )
        division = mpq  # using gmpy2 fractions
    else:
        raise UnknownAlgorithm("equal-shares", algorithm)

    if resolute:
        max_num_of_committees = 1  # same algorithm for resolute==True and resolute==False

    if per_voter_budget:
        start_budget = {vi: voter.weight * per_voter_budget for vi, voter in enumerate(profile)}
    else:
        start_budget = {
            vi: division(voter.weight * committeesize, profile.total_weight())
            for vi, voter in enumerate(profile)
        }
    committee_budget_pairs = [(tuple(), start_budget)]
    winning_committees = set()
    detailed_info = {
        "next_cand": [],
        "cost": [],
        "tied_cands": [],
        "budget": [],
        "start_budget": start_budget,
        "phragmen_start_load": None,
    }

    while committee_budget_pairs:
        committee, budget = committee_budget_pairs.pop()

        available_candidates = [cand for cand in profile.candidates if cand not in committee]
        min_q = {}
        for cand in available_candidates:
            q = _equal_shares_get_min_q(profile, budget, cand, division)
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
                        new_budget[v] -= min(budget[v], min_q[next_cand] * voter.weight)

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
            committee_budget_pairs += reversed(new_committee_budget_pairs)

        else:  # no affordable candidates remain
            if not completion or completion.lower == "none":
                winning_committees.add(tuple(sorted(committee)))
            elif completion == "seqphragmen":
                # fill committee via seq-Phragmen
                phragmen_phase(committee, budget)
            elif completion == "av":
                av_phase(committee)
            else:
                raise ValueError(f"completion argument {completion} unknown.")

        if max_num_of_committees is not None and len(winning_committees) >= max_num_of_committees:
            winning_committees = sorted_committees(winning_committees)[:max_num_of_committees]
            break

    return sorted_committees(winning_committees), detailed_info


def _equal_shares_algorithm_with_increment_completion(
    profile,
    committeesize,
    algorithm,
):
    if len(profile.approved_candidates()) < committeesize:
        # There are fewer candidates approved by at least one voter than `committeesize`.
        # Should return the approved candidates and fill the committee with non-approved
        # candidates. This is done, e.g., by AV.
        committees, _ = _separable_rule_algorithm(
            rule_id="av",
            profile=profile,
            committeesize=committeesize,
            resolute=True,  # increment completion is ill-defined for resolute=False
        )
        detailed_info = {"too_few_approved_candidates": True}
        return committees, detailed_info

    for increment_committeesize in range(
        committeesize, math.ceil(committeesize * profile.total_weight() + 1)
    ):
        committees, detailed_info = _equal_shares_algorithm(
            profile,
            committeesize,
            algorithm,
            resolute=True,
            completion=None,
            per_voter_budget=Fraction(increment_committeesize, profile.total_weight()),
        )
        detailed_info["increment_committeesize"] = increment_committeesize
        committees = [comm for comm in committees if len(comm) == committeesize]
        if any(len(comm) > committeesize for comm in committees):
            raise RuntimeError("Critical bug. This condition should not be satisfiable.")
        if committees:
            return committees, detailed_info
    else:
        raise RuntimeError(
            "Critical bug. This LOC should not be reachable; for-loop should terminate earlier."
        )


def compute_minimaxphragmen(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=False,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
    lexicographic_tiebreaking=False,
):
    """
    Compute winning committees with Phragmen's minimax rule (minimax-Phragmen).

    Minimizes the maximum load.

    For a mathematical description of this rule, see e.g.
    "Multi-Winner Voting with Approval Preferences".
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>

    Does not include the lexicographic optimization as specified
    in Markus Brill, Rupert Freeman, Svante Janson and Martin Lackner.
    Phragmen's Voting Methods and Justified Representation.
    <https://arxiv.org/abs/2102.12305>
    Instead: minimizes the maximum load (without consideration of the second-,
    third-, ...-largest load.
    The lexicographic method is this one: :func:`compute_leximaxphragmen`.

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

                >>> Rule("minimaxphragmen").algorithms
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

        lexicographic_tiebreaking : bool
            Require lexicographic tiebreaking among tied committees.

    Returns
    -------
        list of CandidateSet
            A list of winning committees.
    """
    rule_id = "minimaxphragmen"
    rule = Rule(rule_id)
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
            lexicographic_tiebreaking=lexicographic_tiebreaking,
        )
    elif algorithm.startswith("pulp-"):
        committees = abcrules_pulp._pulp_minimaxphragmen(
            profile,
            committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
            solver_id=algorithm[5:],
            lexicographic_tiebreaking=lexicographic_tiebreaking,
        )
    elif algorithm.startswith("mip-"):
        if lexicographic_tiebreaking:
            raise NotImplementedError(f"Lexicographic tiebreaking not available with {algorithm}.")
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
    output.info(header(rule.longname), wrap=False)
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )
    # end of optional output

    return committees


def compute_leximaxphragmen(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=False,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
    lexicographic_tiebreaking=False,
):
    """
    Compute winning committees with Phragmen's leximax rule (leximax-Phragmen).

    Lexicographically minimize the maximum loads.
    Details in
    Markus Brill, Rupert Freeman, Svante Janson and Martin Lackner.
    Phragmen's Voting Methods and Justified Representation.
    <https://arxiv.org/abs/2102.12305>

    .. important::

        Very slow due to lexicographic optimization.

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for Phragmen's leximax rule (leximax-Phragmen):

            .. doctest::

                >>> Rule("leximaxphragmen").algorithms
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
    rule_id = "leximaxphragmen"
    rule = Rule(rule_id)
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
        committees = abcrules_gurobi._gurobi_leximaxphragmen(
            profile,
            committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
            lexicographic_tiebreaking=lexicographic_tiebreaking,
        )
    # elif algorithm.startswith("mip-"):
    #     committees = abcrules_mip._mip_leximaxphragmen(
    #         profile,
    #         committeesize,
    #         resolute=resolute,
    #         max_num_of_committees=max_num_of_committees,
    #         solver_id=algorithm[4:],
    #     )
    elif algorithm.startswith("pulp-"):
        if lexicographic_tiebreaking:
            raise NotImplementedError(f"Lexicographic tiebreaking not available with {algorithm}.")
        committees = abcrules_pulp._pulp_leximaxphragmen(
            profile,
            committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
            solver_id=algorithm[5:],
            lexicographic_tiebreaking=lexicographic_tiebreaking,
        )
    else:
        raise UnknownAlgorithm(rule_id, algorithm)

    # optional output
    output.info(header(rule.longname), wrap=False)
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )
    # end of optional output

    return committees


def compute_maximin_support(
    profile,
    committeesize,
    algorithm="fastest",
    resolute=True,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Compute winning committees with the maximin support method (MMS).

    Details in
    Luis Sánchez-Fernández, Norberto Fernández, Jesús A. Fisteus, Markus Brill
    The maximin support method: an extension of the D'Hondt method
    to approval-based multiwinner elections
    <https://arxiv.org/abs/1609.05370>

    Parameters
    ----------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int
            The desired committee size.

        algorithm : str, optional
            The algorithm to be used.

            The following algorithms are available for the maximin support method (MMS):

            .. doctest::

                >>> Rule("maximin-support").algorithms
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
    rule_id = "maximin-support"
    rule = Rule(rule_id)
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
        scorefct = abcrules_gurobi._gurobi_maximin_support_scorefct
    elif algorithm.startswith("mip-"):
        solver_id = algorithm[4:]
        scorefct = functools.partial(
            abcrules_mip._mip_maximin_support_scorefct, solver_id=solver_id
        )
    else:
        raise UnknownAlgorithm(rule_id, algorithm)

    if resolute:
        committees, detailed_info = _maximin_support_resolute(scorefct, profile, committeesize)
    else:
        committees, detailed_info = _maximin_support_irresolute(
            scorefct, profile, committeesize, max_num_of_committees
        )

    # optional output
    output.info(header(rule.longname), wrap=False)
    if not resolute:
        output.info("Computing all possible winning committees for any tiebreaking order")
        output.info(" (aka parallel universes tiebreaking) (resolute=False)\n")
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")
    if resolute:
        output.details("starting with the empty committee\n")
        committee = []
        for i, next_cand in enumerate(detailed_info["next_cand"]):
            tied_cands = detailed_info["tied_cands"][i]
            support_value = detailed_info["support_value"][i]
            committee.append(next_cand)
            output.details(f"adding candidate number {i + 1}: {profile.cand_names[next_cand]}")
            output.details(
                f"giving a committee with maximin support value {support_value}",
                indent=" ",
            )
            if len(tied_cands) > 1:
                output.details(f"tie broken in favor of {next_cand},", indent=" ")
                output.details(
                    f"candidates "
                    f"{str_set_of_candidates(tied_cands, cand_names=profile.cand_names)} "
                    "are tied",
                    indent=" ",
                )
                output.details(
                    f"(all would give the same maximin support value {support_value})",
                    indent=" ",
                )
            output.details("")
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )
    # end of optional output

    return sorted_committees(committees)


def _maximin_support_resolute(scorefct, profile, committeesize):
    """Compute one winning committee (=resolute) for the maximin support method (MMS).

    Tiebreaking between candidates in favor of candidate with smaller
    number/index (candidates with larger numbers get deleted first).
    """
    committee = []
    remaining_cands = set(profile.candidates)
    detailed_info = {"next_cand": [], "tied_cands": [], "support_value": []}

    # build a committee starting with the empty set
    for _ in range(committeesize):
        additional_score_cand = scorefct(profile, committee)
        highest_score = max(additional_score_cand[cand] for cand in remaining_cands)
        tied_cands = [
            cand
            for cand in remaining_cands
            if additional_score_cand[cand] >= highest_score - 1e-7  # ILP float accuracy
        ]
        next_cand = tied_cands[0]  # tiebreaking in favor of candidate with smallest index
        committee.append(next_cand)
        remaining_cands.remove(next_cand)
        detailed_info["next_cand"].append(next_cand)
        detailed_info["tied_cands"].append(tied_cands)
        detailed_info["support_value"].append(max(additional_score_cand))

    return sorted_committees([committee]), detailed_info


def _maximin_support_irresolute(scorefct, profile, committeesize, max_num_of_committees):
    """Compute all winning committee (=irresolute) for the maximin support method (MMS).

    Consider all possible ways to break ties between candidates
    (aka parallel universe tiebreaking)
    """
    # build committees starting with the empty set
    partial_committees = [()]
    winning_committees = set()

    while partial_committees:
        new_partial_committees = []
        committee = partial_committees.pop()
        additional_score_cand = scorefct(profile, committee)
        remaining_cands = set(profile.candidates) - set(committee)
        highest_score = max(additional_score_cand[cand] for cand in remaining_cands)
        for cand in remaining_cands:
            if additional_score_cand[cand] >= highest_score - 1e-7:  # ILP float accuracy
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

                >>> Rule("phragmen-enestroem").algorithms
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
    rule = Rule(rule_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )

    committees, detailed_info = _phragmen_enestroem_algorithm(
        profile=profile,
        committeesize=committeesize,
        algorithm=algorithm,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )

    # optional output
    output.info(header(rule.longname), wrap=False)
    if not resolute:
        output.info("Computing all possible winning committees for any tiebreaking order")
        output.info(" (aka parallel universes tiebreaking) (resolute=False)\n")
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
        if not mpq:
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
                cand for cand, supp in support.items() if misc.isclose(supp, max_support)
            ]
        else:
            tied_cands = sorted(cand for cand, supp in support.items() if supp == max_support)
        if not tied_cands:
            raise RuntimeError("_phragmen_enestroem_algorithm: no candidate with max support (??)")

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

                >>> Rule("consensus-rule").algorithms
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
    rule = Rule(rule_id)
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
    output.info(header(rule.longname), wrap=False)
    if not resolute:
        output.info("Computing all possible winning committees for any tiebreaking order")
        output.info(" (aka parallel universes tiebreaking) (resolute=False)\n")
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
        if not mpq:
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
            if (budget[i] <= 0) or (algorithm == "float-fractions" and misc.isclose(budget[i], 0)):
                continue
            for cand in voter.approved:
                if cand in available_candidates:
                    support[cand] += budget[i]
                    supporters[cand].append(i)
        max_support = max(support.values())
        if algorithm == "float-fractions":
            tied_cands = [
                cand for cand, supp in support.items() if misc.isclose(supp, max_support)
            ]
        else:
            tied_cands = sorted(cand for cand, supp in support.items() if supp == max_support)
        if not tied_cands:
            raise RuntimeError("_consensus_rule_algorithm: no candidate with max support (??)")

        new_committee_budget_pairs = []
        for cand in tied_cands:
            new_budget = list(budget)  # copy of budget
            for i in supporters[cand]:
                new_budget[i] -= profile[i].weight * division(
                    profile.total_weight(), sum(profile[vi].weight for vi in supporters[cand])
                )
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

                >>> Rule("trivial").algorithms
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
    rule = Rule(rule_id)
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
    output.info(header(rule.longname), wrap=False)
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

                >>> Rule("rsd").algorithms
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
    rule = Rule(rule_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(
        profile, committeesize, algorithm, resolute, max_num_of_committees
    )

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
    output.info(header(rule.longname), wrap=False)
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )
    # end of optional output

    return sorted_committees(committees)


def compute_eph(
    profile,
    committeesize,
    algorithm="float-fractions",
    resolute=False,
    max_num_of_committees=MAX_NUM_OF_COMMITTEES_DEFAULT,
):
    """
    Compute winning committees with the "E Pluribus Hugo" (EPH) voting rule.

    This rule is used by the Hugo Awards as a shortlisting voting rule. It is described in the
    following paper under the name "Single Divisible Vote with Least-Popular Elimination
    (SDV-LPE)":
    "A proportional voting system for awards nominations resistant to voting blocs."
    Jameson Quinn, and Bruce Schneier.
    <https://www.schneier.com/wp-content/uploads/2016/05/Proportional_Voting_System.pdf>

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

                >>> Rule("eph").algorithms
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
    rule_id = "eph"
    rule = Rule(rule_id)
    if algorithm == "fastest":
        algorithm = rule.fastest_available_algorithm()
    rule.verify_compute_parameters(
        profile, committeesize, algorithm, resolute, max_num_of_committees
    )

    committees, detailed_info = _eph_algorithm(
        rule_id=rule_id,
        profile=profile,
        algorithm=algorithm,
        committeesize=committeesize,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
    )

    # optional output
    output.info(header(rule.longname), wrap=False)
    if resolute:
        output.info("Computing only one winning committee (resolute=True)\n")
    output.details(f"Algorithm: {ALGORITHM_NAMES[algorithm]}\n")
    output.info(
        str_committees_with_header(committees, cand_names=profile.cand_names, winning=True)
    )
    # end of optional output

    return sorted_committees(committees)


def _eph_algorithm(rule_id, profile, algorithm, committeesize, resolute, max_num_of_committees):
    """Algorithm for computing the "E Pluribus Hugo" (EPH) voting rule."""

    if algorithm == "float-fractions":
        division = lambda x, y: x / y  # standard float division
    elif algorithm == "standard-fractions":
        division = Fraction  # using Python built-in fractions
    elif algorithm == "gmpy2-fractions":
        if not mpq:
            raise ImportError(
                'Module gmpy2 not available, required for algorithm "gmpy2-fractions"'
            )
        division = mpq  # using gmpy2 fractions
    else:
        raise UnknownAlgorithm(rule_id, algorithm)

    if resolute:
        max_num_of_committees = 1  # same algorithm for resolute==True and resolute==False

    remaining_candidates = set(profile.candidates)
    while True:
        sdv_score = {cand: 0 for cand in remaining_candidates}
        av_score = {cand: 0 for cand in remaining_candidates}
        for voter in profile:
            remaining_approved = [cand for cand in remaining_candidates if cand in voter.approved]
            for cand in remaining_approved:
                sdv_score[cand] += division(voter.weight, len(remaining_approved))
                av_score[cand] += voter.weight

        cutoff_sdv = sorted(sdv_score.values())[1]  # 2nd smallest value
        elimination_cands = [
            cand
            for cand in remaining_candidates
            if (sdv_score[cand] <= cutoff_sdv)
            or (algorithm == "float-fractions" and misc.isclose(sdv_score[cand], cutoff_sdv))
        ]
        cutoff_av = min(av_score[cand] for cand in elimination_cands)
        elimination_cands = [cand for cand in elimination_cands if av_score[cand] <= cutoff_av]
        if len(remaining_candidates) - len(elimination_cands) <= committeesize:
            num_cands_to_be_eliminated = len(remaining_candidates) - committeesize
            committees = sorted_committees(
                [
                    (remaining_candidates - set(selection))
                    for selection in itertools.combinations(
                        elimination_cands, num_cands_to_be_eliminated
                    )
                ]
            )
            detailed_info = {}
            return committees[:max_num_of_committees], detailed_info
        remaining_candidates -= set(elimination_cands)
