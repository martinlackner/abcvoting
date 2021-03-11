"""
Unit tests for: abcrules.py, abcrules_gurobi.py, abcrules_mip.py and abcrules_cvxpy.py
"""

import re

import pytest

from abcvoting.abcrules_cvxpy import cvxpy_thiele_methods
from abcvoting.abcrules_gurobi import _gurobi_thiele_methods
from abcvoting.output import VERBOSITY_TO_NAME, WARNING, INFO, DETAILS, DEBUG, output
from abcvoting.preferences import Profile, Voter
from abcvoting import abcrules, misc, scores

MARKS = {
    "gurobi": pytest.mark.gurobi,
    "cvxpy_scip": [pytest.mark.cvxpy, pytest.mark.scip],
    "cvxpy_glpk_mi": [pytest.mark.cvxpy, pytest.mark.glpk_mi],
    "cvxpy_cbc": [pytest.mark.cvxpy, pytest.mark.cbc],
    "cvxpy_gurobi": [pytest.mark.cvxpy, pytest.mark.gurobi],
    "ortools_cp": [pytest.mark.ortools],
    "ortools_cbc": [pytest.mark.ortools, pytest.mark.cbc],
    "ortools_gurobi": [pytest.mark.ortools, pytest.mark.gurobi],
    "ortools_scip": [pytest.mark.ortools, pytest.mark.scip],
    "mip_cbc": [pytest.mark.mip],  # does not require cbc because cbc is included in mip
    "mip_gurobi": [pytest.mark.mip, pytest.mark.gurobi],
    "brute-force": [],
    "branch-and-bound": [],
    "standard": [],
    "exact-fractions": [],
    "fastest": [],
}


class CollectRules:
    """
    Collect all ABC rules that are available for unittesting.
    Exclude Gurobi-based rules if Gurobi is not available
    """

    def __init__(self):
        self.rule_algorithm_resolute = []
        self.rule_algorithm_onlyresolute = []
        self.rule_algorithm_onlyirresolute = []
        for rule_id in abcrules.MAIN_RULE_IDS:
            if abcrules.get_ruleinfo(rule_id).algorithms:
                for algorithm in list(abcrules.get_algorithms(rule_id)) + ["fastest"]:
                    for resolute in abcrules.get_resolute_values(rule_id):
                        if algorithm in MARKS:
                            if algorithm == "fastest":
                                try:
                                    actual_algorithm = abcrules.fastest_algo(rule_id)
                                except RuntimeError:
                                    continue
                            else:
                                actual_algorithm = algorithm
                            instance = pytest.param(
                                rule_id, algorithm, resolute, marks=MARKS[actual_algorithm]
                            )
                            instance_no_resolute_param = pytest.param(
                                rule_id, algorithm, marks=MARKS[actual_algorithm]
                            )
                        else:
                            raise ValueError(
                                f"Algorithm {algorithm} (for {rule_id}) "
                                f"not known in unit tests "
                                f"(pytest marks are missing)."
                            )

                        self.rule_algorithm_resolute.append(instance)
                        if resolute:
                            self.rule_algorithm_onlyresolute.append(instance_no_resolute_param)
                        else:
                            self.rule_algorithm_onlyirresolute.append(instance_no_resolute_param)


class CollectInstances:
    def __init__(self):
        self.instances = []

        # first profile
        profile = Profile(6)
        committeesize = 4
        approval_sets = [{0, 4, 5}, {0}, {1, 4, 5}, {1}, {2, 4, 5}, {2}, {3, 4, 5}, {3}]
        profile.add_voters(approval_sets)
        tests = {
            "seqpav": [
                {0, 1, 4, 5},
                {0, 2, 4, 5},
                {0, 3, 4, 5},
                {1, 2, 4, 5},
                {1, 3, 4, 5},
                {2, 3, 4, 5},
            ],
            "av": [
                {0, 1, 4, 5},
                {0, 2, 4, 5},
                {0, 3, 4, 5},
                {1, 2, 4, 5},
                {1, 3, 4, 5},
                {2, 3, 4, 5},
            ],
            "sav": [
                {0, 1, 2, 3},
                {0, 1, 2, 4},
                {0, 1, 2, 5},
                {0, 1, 3, 4},
                {0, 1, 3, 5},
                {0, 1, 4, 5},
                {0, 2, 3, 4},
                {0, 2, 3, 5},
                {0, 2, 4, 5},
                {0, 3, 4, 5},
                {1, 2, 3, 4},
                {1, 2, 3, 5},
                {1, 2, 4, 5},
                {1, 3, 4, 5},
                {2, 3, 4, 5},
            ],
            "pav": [
                {0, 1, 4, 5},
                {0, 2, 4, 5},
                {0, 3, 4, 5},
                {1, 2, 4, 5},
                {1, 3, 4, 5},
                {2, 3, 4, 5},
            ],
            "geom2": [
                {0, 1, 2, 4},
                {0, 1, 2, 5},
                {0, 1, 3, 4},
                {0, 1, 3, 5},
                {0, 1, 4, 5},
                {0, 2, 3, 4},
                {0, 2, 3, 5},
                {0, 2, 4, 5},
                {0, 3, 4, 5},
                {1, 2, 3, 4},
                {1, 2, 3, 5},
                {1, 2, 4, 5},
                {1, 3, 4, 5},
                {2, 3, 4, 5},
            ],
            "revseqpav": [
                {0, 1, 2, 4},
                {0, 1, 2, 5},
                {0, 1, 3, 4},
                {0, 1, 3, 5},
                {0, 1, 4, 5},
                {0, 2, 3, 4},
                {0, 2, 3, 5},
                {0, 2, 4, 5},
                {0, 3, 4, 5},
                {1, 2, 3, 4},
                {1, 2, 3, 5},
                {1, 2, 4, 5},
                {1, 3, 4, 5},
                {2, 3, 4, 5},
            ],
            "minimaxav": [
                {0, 1, 2, 3},
                {0, 1, 2, 4},
                {0, 1, 2, 5},
                {0, 1, 3, 4},
                {0, 1, 3, 5},
                {0, 1, 4, 5},
                {0, 2, 3, 4},
                {0, 2, 3, 5},
                {0, 2, 4, 5},
                {0, 3, 4, 5},
                {1, 2, 3, 4},
                {1, 2, 3, 5},
                {1, 2, 4, 5},
                {1, 3, 4, 5},
                {2, 3, 4, 5},
            ],
            "lexminimaxav": [
                {0, 1, 4, 5},
                {0, 2, 4, 5},
                {0, 3, 4, 5},
                {1, 2, 4, 5},
                {1, 3, 4, 5},
                {2, 3, 4, 5},
            ],
            "seqphragmen": [
                {0, 1, 4, 5},
                {0, 2, 4, 5},
                {0, 3, 4, 5},
                {1, 2, 4, 5},
                {1, 3, 4, 5},
                {2, 3, 4, 5},
            ],
            "minimaxphragmen": [{0, 1, 2, 3}],
            "cc": [{0, 1, 2, 3}],
            "seqcc": [
                {0, 1, 2, 4},
                {0, 1, 2, 5},
                {0, 1, 3, 4},
                {0, 1, 3, 5},
                {0, 2, 3, 4},
                {0, 2, 3, 5},
                {1, 2, 3, 4},
                {1, 2, 3, 5},
            ],
            "revseqcc": [{0, 1, 2, 3}],
            "monroe": [{0, 1, 2, 3}],
            "greedy-monroe": [{0, 2, 3, 4}],
            "slav": [
                {0, 1, 2, 3},
                {0, 1, 2, 4},
                {0, 1, 2, 5},
                {0, 1, 3, 4},
                {0, 1, 3, 5},
                {0, 2, 3, 4},
                {0, 2, 3, 5},
                {1, 2, 3, 4},
                {1, 2, 3, 5},
            ],
            "seqslav": [
                {0, 1, 2, 4},
                {0, 1, 2, 5},
                {0, 1, 3, 4},
                {0, 1, 3, 5},
                {0, 1, 4, 5},
                {0, 2, 3, 4},
                {0, 2, 3, 5},
                {0, 2, 4, 5},
                {0, 3, 4, 5},
                {1, 2, 3, 4},
                {1, 2, 3, 5},
                {1, 2, 4, 5},
                {1, 3, 4, 5},
                {2, 3, 4, 5},
            ],
            "rule-x": [
                {0, 1, 4, 5},
                {0, 2, 4, 5},
                {0, 3, 4, 5},
                {1, 2, 4, 5},
                {1, 3, 4, 5},
                {2, 3, 4, 5},
            ],
            "rule-x-without-2nd-phase": [{4, 5}],
            "phragmen-enestroem": [
                {0, 1, 4, 5},
                {0, 2, 4, 5},
                {0, 3, 4, 5},
                {1, 2, 4, 5},
                {1, 3, 4, 5},
                {2, 3, 4, 5},
            ],
            "consensus-rule": [
                {0, 1, 2, 4},
                {0, 1, 2, 5},
                {0, 1, 3, 4},
                {0, 1, 3, 5},
                {0, 1, 4, 5},
                {0, 2, 3, 4},
                {0, 2, 3, 5},
                {0, 2, 4, 5},
                {0, 3, 4, 5},
                {1, 2, 3, 4},
                {1, 2, 3, 5},
                {1, 2, 4, 5},
                {1, 3, 4, 5},
                {2, 3, 4, 5},
            ],
        }
        self.instances.append((profile, tests, committeesize))

        # first profile now with reversed list of voters
        approval_sets.reverse()
        profile = Profile(6)
        profile.add_voters(approval_sets)
        # Greedy Monroe yields a different result
        # for a different voter ordering
        tests = dict(tests)
        tests["greedy-monroe"] = [{0, 1, 2, 4}]
        committeesize = 4
        self.instances.append((profile, tests, committeesize))

        # second profile
        profile = Profile(5)
        committeesize = 3
        approval_sets = [
            {0, 1, 2},
            {0, 1, 2},
            {0, 1, 2},
            {0, 1, 2},
            {0, 1, 2},
            {0, 1},
            {3, 4},
            {3, 4},
            {3},
        ]
        profile.add_voters(approval_sets)

        tests = {
            "seqpav": [{0, 1, 3}],
            "av": [{0, 1, 2}],
            "sav": [{0, 1, 3}],
            "pav": [{0, 1, 3}],
            "geom2": [{0, 1, 3}],
            "revseqpav": [{0, 1, 3}],
            "minimaxav": [{0, 1, 3}, {0, 2, 3}, {1, 2, 3}],
            "lexminimaxav": [{0, 1, 3}],
            "seqphragmen": [{0, 1, 3}],
            "minimaxphragmen": [{0, 1, 3}, {0, 2, 3}, {1, 2, 3}],
            "cc": [{0, 1, 3}, {0, 2, 3}, {0, 3, 4}, {1, 2, 3}, {1, 3, 4}],
            "seqcc": [{0, 1, 3}, {0, 2, 3}, {0, 3, 4}, {1, 2, 3}, {1, 3, 4}],
            "revseqcc": [{0, 1, 3}, {0, 2, 3}, {0, 3, 4}, {1, 2, 3}, {1, 3, 4}],
            "monroe": [{0, 1, 3}, {0, 2, 3}, {1, 2, 3}],
            "greedy-monroe": [{0, 1, 3}],
            "seqslav": [{0, 1, 3}],
            "slav": [{0, 1, 3}],
            "rule-x": [{0, 1, 3}],
            "rule-x-without-2nd-phase": [{0, 1, 3}],
            "phragmen-enestroem": [{0, 1, 3}],
            "consensus-rule": [{0, 1, 3}],
        }
        self.instances.append((profile, tests, committeesize))

        # and a third profile
        profile = Profile(6)
        committeesize = 4
        approval_sets = [
            {0, 3, 4, 5},
            {1, 2},
            {0, 2, 5},
            {2},
            {0, 1, 2, 3, 4},
            {0, 3, 4},
            {0, 2, 4},
            {0, 1},
        ]
        profile.add_voters(approval_sets)

        tests = {
            "seqpav": [{0, 1, 2, 4}],
            "av": [{0, 1, 2, 4}, {0, 2, 3, 4}],
            "sav": [{0, 1, 2, 4}],
            "pav": [{0, 1, 2, 4}],
            "geom2": [{0, 1, 2, 4}],
            "revseqpav": [{0, 1, 2, 4}],
            "minimaxav": [{0, 1, 2, 3}, {0, 1, 2, 4}, {0, 2, 3, 4}, {0, 2, 3, 5}, {0, 2, 4, 5}],
            "lexminimaxav": [{0, 1, 2, 4}],
            "seqphragmen": [{0, 1, 2, 4}],
            "minimaxphragmen": [
                {0, 1, 2, 3},
                {0, 1, 2, 4},
                {0, 1, 2, 5},
                {0, 2, 3, 4},
                {0, 2, 3, 5},
                {0, 2, 4, 5},
                {1, 2, 3, 4},
                {1, 2, 3, 5},
                {1, 2, 4, 5},
            ],
            "cc": [
                {0, 1, 2, 3},
                {0, 1, 2, 4},
                {0, 1, 2, 5},
                {0, 2, 3, 4},
                {0, 2, 3, 5},
                {0, 2, 4, 5},
                {1, 2, 3, 4},
                {1, 2, 3, 5},
                {1, 2, 4, 5},
            ],
            "seqcc": [
                {0, 1, 2, 3},
                {0, 1, 2, 4},
                {0, 1, 2, 5},
                {0, 2, 3, 4},
                {0, 2, 3, 5},
                {0, 2, 4, 5},
            ],
            "revseqcc": [
                {0, 1, 2, 3},
                {0, 1, 2, 4},
                {0, 1, 2, 5},
                {0, 2, 3, 4},
                {0, 2, 3, 5},
                {0, 2, 4, 5},
                {1, 2, 3, 4},
                {1, 2, 3, 5},
                {1, 2, 4, 5},
            ],
            "monroe": [
                {0, 1, 2, 3},
                {0, 1, 2, 4},
                {0, 1, 2, 5},
                {0, 2, 3, 4},
                {0, 2, 3, 5},
                {0, 2, 4, 5},
                {1, 2, 3, 4},
                {1, 2, 3, 5},
                {1, 2, 4, 5},
            ],
            "greedy-monroe": [{0, 1, 2, 4}],
            "seqslav": [{0, 1, 2, 4}],
            "slav": [{0, 1, 2, 4}],
            "rule-x": [{0, 1, 2, 4}],
            "rule-x-without-2nd-phase": [{0, 2}],
            "phragmen-enestroem": [{0, 1, 2, 4}],
            "consensus-rule": [{0, 1, 2, 4}],
        }
        self.instances.append((profile, tests, committeesize))

        # and a fourth profile
        profile = Profile(4)
        committeesize = 2
        approval_sets = [{0, 1, 3}, {0, 1}, {0, 1}, {0, 3}, {2, 3}]
        profile.add_voters(approval_sets)

        tests = {
            "seqpav": [{0, 3}],
            "av": [{0, 1}, {0, 3}],
            "sav": [{0, 1}, {0, 3}],
            "pav": [{0, 3}],
            "geom2": [{0, 3}],
            "revseqpav": [{0, 3}],
            "minimaxav": [{0, 3}, {1, 3}],
            "lexminimaxav": [{0, 3}],
            "seqphragmen": [{0, 3}],
            "minimaxphragmen": [{0, 3}, {1, 3}],
            "cc": [{0, 2}, {0, 3}, {1, 3}],
            "seqcc": [{0, 2}, {0, 3}],
            "revseqcc": [{0, 2}, {0, 3}, {1, 3}],
            "monroe": [{0, 3}, {1, 3}],
            "greedy-monroe": [{0, 3}],
            "seqslav": [{0, 3}],
            "slav": [{0, 3}],
            "rule-x": [{0, 3}],
            "rule-x-without-2nd-phase": [{0}],
            "phragmen-enestroem": [{0, 3}],
            "consensus-rule": [{0, 3}],
        }
        self.instances.append((profile, tests, committeesize))

        # add a fifth profile
        # this tests a corner case of minimax
        profile = Profile(10)
        committeesize = 2
        approval_sets = [range(5), range(5, 10)]
        profile.add_voters(approval_sets)
        one_each = [{i, j} for i in range(5) for j in range(5, 10)]
        all_possibilities = [{i, j} for i in range(10) for j in range(10) if i != j]
        tests = {
            "seqpav": one_each,
            "av": all_possibilities,
            "sav": all_possibilities,
            "pav": one_each,
            "geom2": one_each,
            "revseqpav": one_each,
            "minimaxav": one_each,
            "lexminimaxav": one_each,
            "seqphragmen": one_each,
            "minimaxphragmen": one_each,
            "cc": one_each,
            "seqcc": one_each,
            "revseqcc": one_each,
            "monroe": one_each,
            "greedy-monroe": one_each,
            "seqslav": one_each,
            "slav": one_each,
            "rule-x": one_each,
            "rule-x-without-2nd-phase": one_each,
            "phragmen-enestroem": one_each,
            "consensus-rule": one_each,
        }
        self.instances.append((profile, tests, committeesize))


testinsts = CollectInstances()
testrules = CollectRules()


def remove_solver_output(out):
    """Remove extra, unwanted solver output (e.g. from Gurobi)."""
    filter_patterns = (
        (
            "\n--------------------------------------------\n"
            "Warning: your license will expire in .*\n"
            "--------------------------------------------\n\n"
        ),
        "Using license file.*\n",
        "Academic license - for non-commercial use only.*\n",
    )

    for filter_pattern in filter_patterns:
        out = re.sub(filter_pattern, "", out)

    return out


@pytest.mark.parametrize("rule_id", abcrules.MAIN_RULE_IDS)
def test_resolute_parameter(rule_id):
    for algorithm in abcrules.get_algorithms(rule_id):
        resolute_values = abcrules.get_resolute_values(rule_id)
        assert len(resolute_values) in [1, 2]
        # resolute=True should be default
        if len(resolute_values) == 2:
            assert resolute_values[0]
        # raise NotImplementedError if value for resolute is not implemented
        for resolute in [False, True]:
            if resolute not in resolute_values:
                profile = Profile(5)
                committeesize = 1
                approval_sets = [{0, 1, 2}, {1}, {1, 2}, {0}]
                profile.add_voters(approval_sets)

                with pytest.raises(NotImplementedError):
                    abcrules.compute(
                        rule_id, profile, committeesize, algorithm=algorithm, resolute=resolute
                    )


@pytest.mark.parametrize("rule_id, algorithm, resolute", testrules.rule_algorithm_resolute)
def test_abcrules__toofewcandidates(rule_id, algorithm, resolute):
    profile = Profile(5)
    committeesize = 4
    approval_sets = [{0, 1, 2}, {1}, {1, 2}, {0}]
    profile.add_voters(approval_sets)

    with pytest.raises(ValueError):
        abcrules.compute(
            rule_id,
            profile,
            committeesize,
            algorithm=algorithm,
            resolute=resolute,
        )
    with pytest.raises(ValueError):
        abcrules.get_ruleinfo(rule_id).function(
            profile,
            committeesize,
            algorithm=algorithm,
            resolute=resolute,
        )


def test_abcrules_wrong_rule_id():
    profile = Profile(3)
    with pytest.raises(abcrules.UnknownRuleIDError):
        abcrules.compute("a_rule_that_does_not_exist", profile, 3)


@pytest.mark.parametrize("rule_id, algorithm, resolute", testrules.rule_algorithm_resolute)
def test_abcrules_weightsconsidered(rule_id, algorithm, resolute):
    profile = Profile(3)
    profile.add_voter(Voter([0]))
    profile.add_voter(Voter([0]))
    profile.add_voter(Voter([1], 5))
    profile.add_voter(Voter([0]))
    committeesize = 1

    if "monroe" in rule_id or rule_id in [
        "lexminimaxav",
        "rule-x",
        "rule-x-without-2nd-phase",
        "phragmen-enestroem",
    ]:
        with pytest.raises(ValueError):
            abcrules.compute(rule_id, profile, committeesize, algorithm=algorithm)
        return

    result = abcrules.compute(
        rule_id, profile, committeesize, algorithm=algorithm, resolute=resolute
    )

    if rule_id == "minimaxav":
        # Minimax AV ignores weights by definition
        if resolute:
            assert result == [{0}] or result == [{1}] or result == [{2}]
        else:
            assert result == [{0}, {1}, {2}]
    else:
        assert len(result) == 1
        assert result[0] == {1}


@pytest.mark.parametrize("rule_id, algorithm, resolute", testrules.rule_algorithm_resolute)
def test_abcrules_correct_simple(rule_id, algorithm, resolute):
    def simple_checks(_committees):
        if rule_id == "rule-x-without-2nd-phase":
            assert _committees == [set()]
            return
        if resolute:
            assert len(_committees) == 1
        else:
            assert len(_committees) == 6

    profile = Profile(4)
    profile.add_voters([{0}, {1}, {2}, {3}])
    committeesize = 2

    committees = abcrules.compute(
        rule_id, profile, committeesize, algorithm=algorithm, resolute=resolute
    )
    simple_checks(committees)

    # call abcrules function differently, results should be the same
    committees = abcrules.get_ruleinfo(rule_id).function(
        profile,
        committeesize,
        algorithm=algorithm,
        resolute=resolute,
    )
    simple_checks(committees)


@pytest.mark.parametrize("rule_id, algorithm, resolute", testrules.rule_algorithm_resolute)
def test_abcrules_return_lists_of_sets(rule_id, algorithm, resolute):
    profile = Profile(4)
    profile.add_voters([{0}, [1], [2], {3}])
    committeesize = 2

    committees = abcrules.compute(
        rule_id, profile, committeesize, algorithm=algorithm, resolute=resolute
    )
    assert len(committees) >= 1
    for committee in committees:
        assert isinstance(committee, set)


@pytest.mark.parametrize("rule_id, algorithm, resolute", testrules.rule_algorithm_resolute)
def test_abcrules_handling_empty_ballots(rule_id, algorithm, resolute):
    profile = Profile(4)
    profile.add_voters([{0}, {1}, {2}])
    committeesize = 3

    committees = abcrules.compute(
        rule_id, profile, committeesize, algorithm=algorithm, resolute=resolute
    )

    assert committees == [{0, 1, 2}]

    profile.add_voters([[]])

    committees = abcrules.compute(
        rule_id, profile, committeesize, algorithm=algorithm, resolute=resolute
    )

    if rule_id == "rule-x-without-2nd-phase":
        assert committees == [set()]
    else:
        assert committees == [{0, 1, 2}]


@pytest.mark.parametrize(
    "algorithm",
    [
        pytest.param(algorithm, marks=MARKS[algorithm])
        for algorithm in abcrules.get_algorithms("monroe")
    ],
)
def test_monroe_indivisible(algorithm):
    profile = Profile(4)
    profile.add_voters([[0], [0], [0], [1, 2], [1, 2], [1], [3]])
    committeesize = 3

    assert abcrules.compute_monroe(
        profile, committeesize, algorithm=algorithm, resolute=False
    ) == [{0, 1, 2}, {0, 1, 3}, {0, 2, 3}]


@pytest.mark.parametrize(
    "algorithm",
    [
        pytest.param(algorithm, marks=MARKS[algorithm])
        for algorithm in abcrules.get_algorithms("minimaxphragmen")
    ],
)
def test_minimaxphragmen_does_not_use_lexicographic_optimization(algorithm):
    # this test shows that lexicographic optimization is not (yet)
    # implemented for opt-Phragmen (as it is described in
    # http://martin.lackner.xyz/publications/phragmen.pdf)

    profile = Profile(7)
    profile.add_voters([[6], [6], [1, 3], [1, 3], [1, 4], [2, 4], [2, 5], [2, 5]])
    committeesize = 3

    # without lexicographic optimization, this profile has 12 winning committees
    # (with lexicographic optimization only {1, 2, 6} is winning)
    committees = abcrules.compute(
        "minimaxphragmen", profile, committeesize, algorithm=algorithm, resolute=False
    )
    print(committees)
    assert len(committees) == 12


@pytest.mark.parametrize(
    "rule_id, algorithm, resolute",
    testrules.rule_algorithm_resolute,
)
@pytest.mark.parametrize("profile, exp_results, committeesize", testinsts.instances)
def test_abcrules_correct(rule_id, algorithm, resolute, profile, exp_results, committeesize):
    if rule_id.startswith("geom") and rule_id != "geom2":
        return  # correctness tests only for geom2
    if rule_id.startswith("seq") and rule_id not in ("seqpav", "seqslav", "seqcc"):
        return  # correctness tests only for selected sequential rules
    if rule_id.startswith("revseq") and rule_id != "revseqpav":
        return  # correctness tests only for selected reverse sequential rules (only for revseqpav)
    print(profile)
    committees = abcrules.compute(
        rule_id, profile, committeesize, algorithm=algorithm, resolute=resolute
    )
    print(f"output: {committees}")
    print(f"expected: {exp_results[rule_id]}")
    if resolute:
        assert len(committees) == 1
        assert committees[0] in exp_results[rule_id]
    else:
        # test unordered equality, this requires sets of sets, only possible with frozensets
        committees_ = {frozenset(committee) for committee in committees}
        exp_results_ = {frozenset(committee) for committee in exp_results[rule_id]}
        assert committees_ == exp_results_


def test_seqphragmen_irresolute():
    profile = Profile(3)
    profile.add_voters([[0, 1], [0, 1], [0], [1, 2], [2]])
    committeesize = 2
    committees = abcrules.compute("seqphragmen", profile, committeesize, resolute=False)
    assert committees == [{0, 1}, {0, 2}]

    committees = abcrules.compute("seqphragmen", profile, committeesize, resolute=True)
    assert committees == [{0, 2}]


def test_seqpav_irresolute():
    profile = Profile(3)
    profile.add_voters([[0, 1]] * 3 + [[0], [1, 2], [2], [2]])
    committeesize = 2

    committees = abcrules.compute("seqpav", profile, committeesize, resolute=False)
    assert committees == [{0, 1}, {0, 2}, {1, 2}]

    committees = abcrules.compute("seqpav", profile, committeesize, resolute=True)
    assert committees == [{0, 2}]


def test_gurobi_cant_compute_av():
    profile = Profile(4)
    profile.add_voters([[0, 1], [1, 2]])
    committeesize = 2

    scorefct = scores.get_scorefct("av", committeesize)

    with pytest.raises(ValueError):
        _gurobi_thiele_methods(profile, committeesize, scorefct, resolute=False)


@pytest.mark.cvxpy
def test_cvxpy_cant_compute_av():
    profile = Profile(4)
    profile.add_voters([[0, 1], [1, 2]])
    committeesize = 2

    with pytest.raises(ValueError):
        cvxpy_thiele_methods(profile, committeesize, "av", resolute=False, solver_id="glpk_mi")


def test_consensus_fails_lower_quota():
    profile = Profile(31)
    profile.add_voters(
        [[0]]
        + [[1, 2]] * 3
        + [[3, 4, 5]] * 5
        + [[6, 7, 8, 9, 10, 11, 12, 13, 14, 15]] * 18
        + [[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30]] * 27
    )
    committeesize = 30

    committees = abcrules.compute("consensus-rule", profile, committeesize, resolute=True)
    for committee in committees:
        assert not all(
            cand in committee
            for cand in [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        )
    # .. and thus the Consensus rule fails lower quota (and PJR and EJR):
    # the quota of the 27 voters is 15, but not all of their 15 approved candidates
    # are contained in a winning committee.


@pytest.mark.parametrize(
    "rule_id, algorithm",
    [
        pytest.param(rule_id, algorithm, marks=MARKS[algorithm])
        for rule_id in ["phragmen-enestroem", "seqphragmen", "pav", "seqpav", "revseqpav"]
        for algorithm in abcrules.get_algorithms(rule_id)
    ],
)
def test_jansonexamples(rule_id, algorithm):
    # example from Janson's survey (https://arxiv.org/pdf/1611.08826.pdf),
    # Example 3.7, 18.1
    profile = Profile(6)
    a = 0
    b = 1
    c = 2
    p = 3
    q = 4
    r = 5
    profile.add_voters(
        [[a, b, c]] * 1034 + [[p, q, r]] * 519 + [[a, b, q]] * 90 + [[a, p, q]] * 90
    )
    committeesize = 3

    committees = abcrules.compute(
        rule_id, profile, committeesize, algorithm=algorithm, resolute=False
    )
    assert committees == [{a, b, q}]


@pytest.mark.parametrize("rule_id", abcrules.MAIN_RULE_IDS)
@pytest.mark.parametrize("resolute", [True, False])
def test_unspecified_algorithms(rule_id, resolute):
    if resolute not in abcrules.get_resolute_values(rule_id):
        return
    profile = Profile(3)
    profile.add_voters([[0, 1], [1, 2]])
    committeesize = 2
    with pytest.raises(NotImplementedError):
        abcrules.compute(
            rule_id,
            profile,
            committeesize,
            algorithm="made-up-algorithm",
            resolute=resolute,
        )


@pytest.mark.parametrize("rule_id", abcrules.MAIN_RULE_IDS)
def test_fastest_algorithms(rule_id):
    profile = Profile(4)
    profile.add_voters([[0, 1], [1, 2], [0, 2, 3]])
    committeesize = 2
    try:
        algorithm = abcrules.fastest_algo(rule_id)
    except RuntimeError:
        pytest.skip("no supported algorithms for " + abcrules.get_shortname(rule_id))
    for resolute in abcrules.get_resolute_values(rule_id):
        abcrules.compute(rule_id, profile, committeesize, algorithm=algorithm, resolute=resolute)
    # second possibility
    abcrules.compute(rule_id, profile, committeesize, algorithm="fastest")


@pytest.mark.cvxpy
def test_cvxpy_wrong_score_fct():
    profile = Profile(4)
    profile.add_voters([[0, 1], [2, 3]])
    committeesize = 1
    with pytest.raises(NotImplementedError):
        cvxpy_thiele_methods(
            profile=profile,
            committeesize=committeesize,
            scorefct_id="non_existing",
            resolute=False,
            solver_id="glpk_mi",
        )


@pytest.mark.parametrize("sizemultiplier", [1, 2, 3, 4, 5])
def test_revseqpav_fails_EJR(sizemultiplier):
    # from "A Note on Justified RepresentationUnder the Reverse Sequential PAV rule"
    # by Haris Aziz
    # Proposition 2 for k=5

    num_cand = 12
    # candidates
    c, x1, x2, x3, x4, x5, x6, y1, y2, y3, y4, y5 = reversed(list(range(num_cand)))
    # reversed because c should be removed first in case of ties
    profile = Profile(num_cand)
    profile.add_voters([{c, x1, x3, x5}] * 4 * sizemultiplier)
    profile.add_voters([{c, x2, x4, x6}] * 4 * sizemultiplier)
    profile.add_voters([{x1}, {x2}, {x3}, {x4}, {x5}, {x6}] * sizemultiplier)
    profile.add_voters([[y1, y2, y3, y4, y5]] * 26 * sizemultiplier)
    assert len(profile) == 40 * sizemultiplier
    assert abcrules.compute_revseqpav(profile, 5) == [{y1, y2, y3, y4, y5}]


def test_seqphragmen_fails_EJR():
    # from "Phragmén's Voting Methods and Justified Representation"
    # by Markus Brill, Rupert Freeman, Svante Janson and Martin Lackner

    num_cand = 14
    # candidates
    a, b, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = list(range(num_cand))
    # reversed because c should be removed first in case of ties
    profile = Profile(num_cand)
    profile.add_voters([{a, b, c1}] * 2)
    profile.add_voters([{a, b, c2}] * 2)
    profile.add_voters([{c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12}] * 6)
    profile.add_voters([{c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12}] * 5)
    profile.add_voters([{c3, c4, c5, c6, c7, c8, c9, c10, c11, c12}] * 9)
    assert len(profile) == 24
    assert abcrules.compute_seqphragmen(profile, 12) == [
        {c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12}
    ]


@pytest.mark.parametrize("rule_id, algorithm, resolute", testrules.rule_algorithm_resolute)
@pytest.mark.parametrize("verbosity", VERBOSITY_TO_NAME.keys())
def test_output(capfd, rule_id, algorithm, resolute, verbosity):
    if algorithm == "fastest":
        return
        # not necessary, output for "fastest" is the same as
        # whatever algorithm is selected as fastest
        # (and "fastest" depends on the available solvers)

    if algorithm == "cvxpy_glpk_mi":
        # TODO unfortunately GLPK_MI prints "Long-step dual simplex will be used" to stderr and it
        #  would be very complicated to capture this on all platforms reliably, changing
        #  sys.stderr doesn't help.
        #  This seems to be fixed in GLPK 5.0 but not in GLPK 4.65. For some weird reason this
        #  test succeeds and does not need to be skipped when using conda-forge, although the
        #  version from conda-forge is given as glpk 4.65 he80fd80_1002.
        #  This could help to introduce a workaround: https://github.com/xolox/python-capturer
        #  Sage math is fighting the same problem: https://trac.sagemath.org/ticket/24824
        pytest.skip("GLPK_MI prints something to stderr, not easy to capture")

    # if "gurobi" in algorithm:
    #     # TODO Gurobi prints:
    #     #  "Academic license - for non-commercial use only"
    #     #  (if an acamedic license is used) as well as
    #     #  "Warning: parameter changes on this environment will not affect existing models."
    #     #  Thus this test fails.
    #     #
    #     # This message appears only sometimes, so even if this test succeeds for gurobi with
    #     # verbosity>=WARNING, it's probably just due to the test execution order, i.e. the
    #     # undesired message was simply printed earlier, or caused by use of different Gurobi
    #     # versions.
    #     pytest.skip("Gurobi always prints something when used with an academic license")

    output.set_verbosity(verbosity=verbosity)

    try:
        profile = Profile(2)
        profile.add_voters([[0]])
        committeesize = 1

        committees = abcrules.compute(
            rule_id, profile, committeesize, algorithm=algorithm, resolute=resolute
        )
        out = str(capfd.readouterr().out)

        # remove unwanted solver output
        out = remove_solver_output(out)

        if verbosity >= WARNING:
            assert out == ""
        else:
            assert len(out) > 0
            start_output = misc.header(abcrules.get_longname(rule_id)) + "\n"
            if resolute and False in abcrules.get_resolute_values(rule_id):
                # only if irresolute is available but resolute is chosen
                start_output += "Computing only one winning committee (resolute=True)\n\n"
            if verbosity <= DETAILS:
                start_output += abcrules._algorithm_fullnames(algorithm) + "\n"
            if verbosity <= DEBUG:
                assert start_output in out
            else:
                assert out.startswith(start_output)
            end_output = (
                f"{misc.str_committees_header(committees, winning=True)}\n"
                f"{misc.str_sets_of_candidates(committees, cand_names=profile.cand_names)}\n"
            )
            if verbosity == INFO:
                assert out.endswith(end_output)
            else:
                assert end_output in out

    finally:
        output.set_verbosity(verbosity=WARNING)
