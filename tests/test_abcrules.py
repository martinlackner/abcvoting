"""
Unit tests for abcrules.py and abcrules_ilp.py
"""

import pytest
from abcvoting.preferences import Profile, DichotomousPreferences
from abcvoting import abcrules


class CollectRules():
    """
    Collect all ABC rules that are available for unittesting.
    Exclude ILP-based rules if Gurobi is not available
    """
    def __init__(self):
        self.rules = abcrules.MWRULES
        self.ilp = [True, False]
        try:
            import gurobipy  # pylint: disable=unused-import
        except ImportError:
            self.rules = [rule for rule in self.rules
                          if rule[-4:] != "-ilp"]
            self.ilp = [False]
            print("Warning: Gurobi not found, "
                  + "ILP-based unittests are ignored.")


class CollectInstances():
    def __init__(self):
        self.instances = []

        # first profile
        profile = Profile(6)
        preflist = [[0, 4, 5], [0], [1, 4, 5], [1],
                    [2, 4, 5], [2], [3, 4, 5], [3]]
        profile.add_preferences(preflist)
        tests = {
            "seqpav": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5],
                       [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "av": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5],
                   [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "sav": [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5], [0, 1, 3, 4],
                    [0, 1, 3, 5], [0, 1, 4, 5], [0, 2, 3, 4], [0, 2, 3, 5],
                    [0, 2, 4, 5], [0, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 5],
                    [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "pav-ilp": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5],
                        [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "pav-noilp": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5],
                          [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "revseqpav": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5],
                          [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "minimaxav-noilp": [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5],
                                [0, 1, 3, 4], [0, 1, 3, 5], [0, 1, 4, 5],
                                [0, 2, 3, 4], [0, 2, 3, 5], [0, 2, 4, 5],
                                [0, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 5],
                                [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "minimaxav-ilp": [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5],
                              [0, 1, 3, 4], [0, 1, 3, 5], [0, 1, 4, 5],
                              [0, 2, 3, 4], [0, 2, 3, 5], [0, 2, 4, 5],
                              [0, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 5],
                              [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "lexminimaxav-noilp": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5],
                                   [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "phrag": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5],
                      [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "optphrag-ilp": [[0, 1, 2, 3]],
            "cc-ilp": [[0, 1, 2, 3]],
            "cc-noilp": [[0, 1, 2, 3]],
            "seqcc": [[0, 1, 2, 4], [0, 1, 2, 5], [0, 1, 3, 4], [0, 1, 3, 5],
                      [0, 2, 3, 4], [0, 2, 3, 5], [1, 2, 3, 4], [1, 2, 3, 5]],
            "revseqcc": [[0, 1, 2, 3]],
            "monroe-ilp": [[0, 1, 2, 3]],
            "monroe-noilp": [[0, 1, 2, 3]],
            "greedy-monroe": [[0, 2, 3, 4]],
            "slav-ilp": [[0, 1, 2, 3],
                         [0, 1, 2, 4], [0, 1, 2, 5],
                         [0, 1, 3, 4], [0, 1, 3, 5],
                         [0, 2, 3, 4], [0, 2, 3, 5],
                         [1, 2, 3, 4], [1, 2, 3, 5]],
            "slav-noilp": [[0, 1, 2, 3],
                           [0, 1, 2, 4], [0, 1, 2, 5],
                           [0, 1, 3, 4], [0, 1, 3, 5],
                           [0, 2, 3, 4], [0, 2, 3, 5],
                           [1, 2, 3, 4], [1, 2, 3, 5]],
            "seqslav": [[0, 1, 2, 4], [0, 1, 2, 5],
                        [0, 1, 3, 4], [0, 1, 3, 5],
                        [0, 2, 3, 4], [0, 2, 3, 5],
                        [1, 2, 3, 4], [1, 2, 3, 5]],
            "rule-x": [[0, 1, 4, 5], [0, 2, 4, 5],
                       [0, 3, 4, 5], [1, 2, 4, 5],
                       [1, 3, 4, 5], [2, 3, 4, 5]],
            "phragmen-enestroem": [[0, 1, 4, 5], [0, 2, 4, 5],
                                   [0, 3, 4, 5], [1, 2, 4, 5],
                                   [1, 3, 4, 5], [2, 3, 4, 5]],
        }
        committeesize = 4
        self.instances.append((profile, tests, committeesize))

        # first profile now with reversed preflist
        preflist.reverse()
        for p in preflist:
            p.reverse()
        profile = Profile(6)
        profile.add_preferences(preflist)
        # Greedy Monroe yields a different result
        # for a different voter ordering
        tests = dict(tests)
        tests["greedy-monroe"] = [[0, 1, 2, 4]]
        committeesize = 4
        self.instances.append((profile, tests, committeesize))

        # second profile
        profile = Profile(5)
        committeesize = 3
        preflist = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2],
                    [0, 1, 2], [0, 1], [3, 4], [3, 4], [3]]
        profile.add_preferences(preflist)

        tests = {
            "seqpav": [[0, 1, 3]],
            "av": [[0, 1, 2]],
            "sav": [[0, 1, 3]],
            "pav-ilp": [[0, 1, 3]],
            "pav-noilp": [[0, 1, 3]],
            "revseqpav": [[0, 1, 3]],
            "minimaxav-noilp": [[0, 1, 3], [0, 2, 3], [1, 2, 3]],
            "minimaxav-ilp": [[0, 1, 3], [0, 2, 3], [1, 2, 3]],
            "lexminimaxav-noilp": [[0, 1, 3]],
            "phrag": [[0, 1, 3]],
            "optphrag-ilp": [[0, 1, 3], [0, 2, 3], [1, 2, 3]],
            "cc-ilp": [[0, 1, 3], [0, 2, 3], [0, 3, 4],
                       [1, 2, 3], [1, 3, 4]],
            "cc-noilp": [[0, 1, 3], [0, 2, 3], [0, 3, 4],
                         [1, 2, 3], [1, 3, 4]],
            "seqcc": [[0, 1, 3], [0, 2, 3], [0, 3, 4],
                      [1, 2, 3], [1, 3, 4]],
            "revseqcc": [[0, 1, 3], [0, 2, 3], [0, 3, 4],
                         [1, 2, 3], [1, 3, 4]],
            "monroe-ilp": [[0, 1, 3], [0, 2, 3], [1, 2, 3]],
            "monroe-noilp": [[0, 1, 3], [0, 2, 3], [1, 2, 3]],
            "greedy-monroe": [[0, 1, 3]],
            "seqslav": [[0, 1, 3]],
            "slav-ilp": [[0, 1, 3]],
            "slav-noilp": [[0, 1, 3]],
            "rule-x": [[0, 1, 3]],
            "phragmen-enestroem": [[0, 1, 3]],
        }
        committeesize = 3
        self.instances.append((profile, tests, committeesize))

        # and a third profile
        profile = Profile(6)
        committeesize = 4
        preflist = [[0, 3, 4, 5], [1, 2], [0, 2, 5], [2],
                    [0, 1, 2, 3, 4], [0, 3, 4], [0, 2, 4], [0, 1]]
        profile.add_preferences(preflist)

        tests = {
            "seqpav": [[0, 1, 2, 4]],
            "av": [[0, 1, 2, 4], [0, 2, 3, 4]],
            "sav": [[0, 1, 2, 4]],
            "pav-ilp": [[0, 1, 2, 4]],
            "pav-noilp": [[0, 1, 2, 4]],
            "revseqpav": [[0, 1, 2, 4]],
            "minimaxav-noilp": [[0, 1, 2, 3], [0, 1, 2, 4],
                                [0, 2, 3, 4], [0, 2, 3, 5],
                                [0, 2, 4, 5]],
            "minimaxav-ilp": [[0, 1, 2, 3], [0, 1, 2, 4],
                              [0, 2, 3, 4], [0, 2, 3, 5],
                              [0, 2, 4, 5]],
            "lexminimaxav-noilp": [[0, 1, 2, 4]],
            "phrag": [[0, 1, 2, 4]],
            "optphrag-ilp": [[0, 1, 2, 3], [0, 1, 2, 4],
                             [0, 1, 2, 5], [0, 2, 3, 4],
                             [0, 2, 3, 5], [0, 2, 4, 5],
                             [1, 2, 3, 4], [1, 2, 3, 5],
                             [1, 2, 4, 5]],
            "cc-ilp": [[0, 1, 2, 3], [0, 1, 2, 4],
                       [0, 1, 2, 5], [0, 2, 3, 4],
                       [0, 2, 3, 5], [0, 2, 4, 5],
                       [1, 2, 3, 4], [1, 2, 3, 5],
                       [1, 2, 4, 5]],
            "cc-noilp": [[0, 1, 2, 3], [0, 1, 2, 4],
                         [0, 1, 2, 5], [0, 2, 3, 4],
                         [0, 2, 3, 5], [0, 2, 4, 5],
                         [1, 2, 3, 4], [1, 2, 3, 5],
                         [1, 2, 4, 5]],
            "seqcc": [[0, 1, 2, 3], [0, 1, 2, 4],
                      [0, 1, 2, 5], [0, 2, 3, 4],
                      [0, 2, 3, 5], [0, 2, 4, 5]],
            "revseqcc": [[0, 1, 2, 3], [0, 1, 2, 4],
                         [0, 1, 2, 5], [0, 2, 3, 4],
                         [0, 2, 3, 5], [0, 2, 4, 5],
                         [1, 2, 3, 4], [1, 2, 3, 5],
                         [1, 2, 4, 5]],
            "monroe-ilp": [[0, 1, 2, 3], [0, 1, 2, 4],
                           [0, 1, 2, 5], [0, 2, 3, 4],
                           [0, 2, 3, 5], [0, 2, 4, 5],
                           [1, 2, 3, 4], [1, 2, 3, 5],
                           [1, 2, 4, 5]],
            "monroe-noilp": [[0, 1, 2, 3], [0, 1, 2, 4],
                             [0, 1, 2, 5], [0, 2, 3, 4],
                             [0, 2, 3, 5], [0, 2, 4, 5],
                             [1, 2, 3, 4], [1, 2, 3, 5],
                             [1, 2, 4, 5]],
            "greedy-monroe": [[0, 1, 2, 4]],
            "seqslav": [[0, 1, 2, 4]],
            "slav-ilp": [[0, 1, 2, 4]],
            "slav-noilp": [[0, 1, 2, 4]],
            "rule-x": [[0, 1, 2, 4]],
            "phragmen-enestroem": [[0, 1, 2, 4]],
        }
        committeesize = 4
        self.instances.append((profile, tests, committeesize))

        # and a fourth profile
        profile = Profile(4)
        committeesize = 2
        preflist = [[0, 1, 3], [0, 1], [0, 1], [0, 3], [2, 3]]
        profile.add_preferences(preflist)

        tests = {
            "seqpav": [[0, 3]],
            "av": [[0, 1], [0, 3]],
            "sav": [[0, 1], [0, 3]],
            "pav-ilp": [[0, 3]],
            "pav-noilp": [[0, 3]],
            "revseqpav": [[0, 3]],
            "minimaxav-noilp": [[0, 3], [1, 3]],
            "minimaxav-ilp": [[0, 3], [1, 3]],
            "lexminimaxav-noilp": [[0, 3]],
            "phrag": [[0, 3]],
            "optphrag-ilp": [[0, 3], [1, 3]],
            "cc-ilp": [[0, 2], [0, 3], [1, 3]],
            "cc-noilp": [[0, 2], [0, 3], [1, 3]],
            "seqcc": [[0, 2], [0, 3]],
            "revseqcc": [[0, 2], [0, 3], [1, 3]],
            "monroe-ilp": [[0, 3], [1, 3]],
            "monroe-noilp": [[0, 3], [1, 3]],
            "greedy-monroe": [[0, 3]],
            "seqslav": [[0, 3]],
            "slav-ilp": [[0, 3]],
            "slav-noilp": [[0, 3]],
            "rule-x": [[0, 3]],
            "phragmen-enestroem": [[0, 3]],
        }
        committeesize = 2
        self.instances.append((profile, tests, committeesize))


testinsts = CollectInstances()
testrules = CollectRules()


@pytest.mark.parametrize(
    "resolute", [True, False]
)
@pytest.mark.parametrize(
    "rule", testrules.rules
)
def test_abcrules__toofewcandidates(rule, resolute):

    profile = Profile(5)
    committeesize = 4
    preflist = [[0, 1, 2], [1], [1, 2], [0]]
    profile.add_preferences(preflist)

    with pytest.raises(ValueError):
        abcrules.compute_rule(rule, profile, committeesize, resolute)


@pytest.mark.parametrize(
    "rule", testrules.rules
)
def test_abcrules_weightsconsidered(rule):
    profile = Profile(3)
    profile.add_preferences(DichotomousPreferences([0]))
    profile.add_preferences(DichotomousPreferences([0]))
    profile.add_preferences(DichotomousPreferences([1], 5))
    profile.add_preferences(DichotomousPreferences([0]))
    committeesize = 1

    if (("monroe" in rule
         or rule.startswith("lexminimaxav")
         or rule in ["rule-x", "phragmen-enestroem"])):
        with pytest.raises(ValueError):
            abcrules.compute_rule(rule, profile, committeesize)
    elif rule.startswith("minimaxav"):
        # Minimax AV ignores weights by definition
        result = abcrules.compute_rule(rule, profile, committeesize)
        assert len(result) == 3
        assert result == [[0], [1], [2]]
    else:
        result = abcrules.compute_rule(rule, profile, committeesize)
        assert len(result) == 1
        assert result[0] == [1]


@pytest.mark.parametrize(
    "rule", testrules.rules
)
def test_abcrules_correct_simple(rule):
    profile = Profile(4)
    profile.add_preferences([[0], [1], [2], [3]])
    committeesize = 2

    if rule == "greedy-monroe":   # always returns one committee
        return

    assert len(abcrules.compute_rule(
        rule, profile, committeesize)) == 6
    assert len(abcrules.compute_rule(
        rule, profile, committeesize, resolute=True)) == 1


@pytest.mark.parametrize(
    "ilp", testrules.ilp
)
def test_monroe_indivisible(ilp):
    profile = Profile(4)
    profile.add_preferences([[0], [0], [0], [1, 2], [1, 2], [1], [3]])
    committeesize = 3

    assert (abcrules.compute_monroe(profile, committeesize,
                                    ilp=ilp, resolute=False)
            == [[0, 1, 2], [0, 1, 3], [0, 2, 3]])


def test_optphrag_notiebreaking():
    # this test shows that tiebreaking is not (yet)
    # implemented for opt-Phragmen

    # requires Gurobi
    pytest.importorskip("gurobipy")
    profile = Profile(6)
    profile.add_preferences([[0], [0], [1, 3], [1, 3], [1, 4],
                             [2, 4], [2, 5], [2, 5]])
    committeesize = 3

    assert len(abcrules.compute_rule(
        "optphrag-ilp", profile, committeesize, resolute=False)) == 12


@pytest.mark.parametrize(
    "rule", testrules.rules
)
@pytest.mark.parametrize(
    "instance", testinsts.instances
)
def test_abcrules_correct(rule, instance):
    profile, exp_results, committeesize = instance
    output = abcrules.compute_rule(rule, profile,
                                   committeesize,
                                   resolute=False)
    assert output == exp_results[rule]


@pytest.mark.parametrize(
    "rule", testrules.rules
)
@pytest.mark.parametrize(
    "instance", testinsts.instances
)
def test_abcrules_correct_resolute(rule, instance):
    profile, exp_results, committeesize = instance
    output = abcrules.compute_rule(rule, profile,
                                   committeesize,
                                   resolute=True)
    assert len(output) == 1
    assert output[0] in exp_results[rule]
