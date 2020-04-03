"""
Unit tests for abcrules.py and abcrules_gurobi.py
"""

import pytest
from abcvoting.preferences import Profile, DichotomousPreferences
from abcvoting import abcrules


class CollectRules():
    """
    Collect all ABC rules that are available for unittesting.
    Exclude Gurobi-based rules if Gurobi is not available
    """
    def __init__(self):
        try:
            import gurobipy
            gurobipy.Model()
            self.gurobi_supported = True
        except ImportError:
            self.gurobi_supported = False
            print("Warning: Gurobi not found, "
                  + "Gurobi-based unittests are ignored.")

        self.instances = []
        self.res_instances = []
        self.irres_instances = []
        for rule in abcrules.rules.values():
            for alg in rule.algorithms:
                if alg == "gurobi" and not self.gurobi_supported:
                    continue
                for resolute in rule.resolute:
                    instance = (rule.rule_id, alg, resolute)
                    self.instances.append(instance)
                    if resolute:
                        self.res_instances.append(instance[:2])
                    else:
                        self.irres_instances.append(instance[:2])


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
            "pav": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5],
                    [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "geom2": [[0, 1, 2, 4], [0, 1, 2, 5], [0, 1, 3, 4],
                      [0, 1, 3, 5], [0, 1, 4, 5], [0, 2, 3, 4], [0, 2, 3, 5],
                      [0, 2, 4, 5], [0, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 5],
                      [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "revseqpav": [[0, 1, 2, 4], [0, 1, 2, 5], [0, 1, 3, 4],
                          [0, 1, 3, 5], [0, 1, 4, 5], [0, 2, 3, 4],
                          [0, 2, 3, 5], [0, 2, 4, 5], [0, 3, 4, 5],
                          [1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 4, 5],
                          [1, 3, 4, 5], [2, 3, 4, 5]],
            "mav": [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5],
                    [0, 1, 3, 4], [0, 1, 3, 5], [0, 1, 4, 5],
                    [0, 2, 3, 4], [0, 2, 3, 5], [0, 2, 4, 5],
                    [0, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 5],
                    [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "lexmav": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5],
                       [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "seqphrag": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5],
                         [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "optphrag": [[0, 1, 2, 3]],
            "cc": [[0, 1, 2, 3]],
            "seqcc": [[0, 1, 2, 4], [0, 1, 2, 5], [0, 1, 3, 4], [0, 1, 3, 5],
                      [0, 2, 3, 4], [0, 2, 3, 5], [1, 2, 3, 4], [1, 2, 3, 5]],
            "revseqcc": [[0, 1, 2, 3]],
            "monroe": [[0, 1, 2, 3]],
            "greedy-monroe": [[0, 2, 3, 4]],
            "slav": [[0, 1, 2, 3],
                     [0, 1, 2, 4], [0, 1, 2, 5],
                     [0, 1, 3, 4], [0, 1, 3, 5],
                     [0, 2, 3, 4], [0, 2, 3, 5],
                     [1, 2, 3, 4], [1, 2, 3, 5]],
            "seqslav": [[0, 1, 2, 4], [0, 1, 2, 5], [0, 1, 3, 4],
                        [0, 1, 3, 5], [0, 1, 4, 5], [0, 2, 3, 4], [0, 2, 3, 5],
                        [0, 2, 4, 5], [0, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 5],
                        [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "rule-x": [[0, 1, 4, 5], [0, 2, 4, 5],
                       [0, 3, 4, 5], [1, 2, 4, 5],
                       [1, 3, 4, 5], [2, 3, 4, 5]],
            "phrag-enestr": [[0, 1, 4, 5], [0, 2, 4, 5],
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
            "pav": [[0, 1, 3]],
            "geom2": [[0, 1, 3]],
            "revseqpav": [[0, 1, 3]],
            "mav": [[0, 1, 3], [0, 2, 3], [1, 2, 3]],
            "lexmav": [[0, 1, 3]],
            "seqphrag": [[0, 1, 3]],
            "optphrag": [[0, 1, 3], [0, 2, 3], [1, 2, 3]],
            "cc": [[0, 1, 3], [0, 2, 3], [0, 3, 4],
                   [1, 2, 3], [1, 3, 4]],
            "seqcc": [[0, 1, 3], [0, 2, 3], [0, 3, 4],
                      [1, 2, 3], [1, 3, 4]],
            "revseqcc": [[0, 1, 3], [0, 2, 3], [0, 3, 4],
                         [1, 2, 3], [1, 3, 4]],
            "monroe": [[0, 1, 3], [0, 2, 3], [1, 2, 3]],
            "greedy-monroe": [[0, 1, 3]],
            "seqslav": [[0, 1, 3]],
            "slav": [[0, 1, 3]],
            "rule-x": [[0, 1, 3]],
            "phrag-enestr": [[0, 1, 3]],
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
            "pav": [[0, 1, 2, 4]],
            "geom2": [[0, 1, 2, 4]],
            "revseqpav": [[0, 1, 2, 4]],
            "mav": [[0, 1, 2, 3], [0, 1, 2, 4],
                    [0, 2, 3, 4], [0, 2, 3, 5],
                    [0, 2, 4, 5]],
            "lexmav": [[0, 1, 2, 4]],
            "seqphrag": [[0, 1, 2, 4]],
            "optphrag": [[0, 1, 2, 3], [0, 1, 2, 4],
                         [0, 1, 2, 5], [0, 2, 3, 4],
                         [0, 2, 3, 5], [0, 2, 4, 5],
                         [1, 2, 3, 4], [1, 2, 3, 5],
                         [1, 2, 4, 5]],
            "cc": [[0, 1, 2, 3], [0, 1, 2, 4],
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
            "monroe": [[0, 1, 2, 3], [0, 1, 2, 4],
                       [0, 1, 2, 5], [0, 2, 3, 4],
                       [0, 2, 3, 5], [0, 2, 4, 5],
                       [1, 2, 3, 4], [1, 2, 3, 5],
                       [1, 2, 4, 5]],
            "greedy-monroe": [[0, 1, 2, 4]],
            "seqslav": [[0, 1, 2, 4]],
            "slav": [[0, 1, 2, 4]],
            "rule-x": [[0, 1, 2, 4]],
            "phrag-enestr": [[0, 1, 2, 4]],
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
            "pav": [[0, 3]],
            "geom2": [[0, 3]],
            "revseqpav": [[0, 3]],
            "mav": [[0, 3], [1, 3]],
            "lexmav": [[0, 3]],
            "seqphrag": [[0, 3]],
            "optphrag": [[0, 3], [1, 3]],
            "cc": [[0, 2], [0, 3], [1, 3]],
            "seqcc": [[0, 2], [0, 3]],
            "revseqcc": [[0, 2], [0, 3], [1, 3]],
            "monroe": [[0, 3], [1, 3]],
            "greedy-monroe": [[0, 3]],
            "seqslav": [[0, 3]],
            "slav": [[0, 3]],
            "rule-x": [[0, 3]],
            "phrag-enestr": [[0, 3]],
        }
        committeesize = 2
        self.instances.append((profile, tests, committeesize))


testinsts = CollectInstances()
testrules = CollectRules()


@pytest.mark.parametrize(
    "rule_instance", testrules.instances
)
@pytest.mark.parametrize(
    "verbose", [0, 1, 2]
)
def test_abcrules__toofewcandidates(rule_instance, verbose):
    rule_id, algorithm, resolute = rule_instance
    profile = Profile(5)
    committeesize = 4
    preflist = [[0, 1, 2], [1], [1, 2], [0]]
    profile.add_preferences(preflist)

    with pytest.raises(ValueError):
        abcrules.compute(
            rule_id, profile, committeesize, algorithm=algorithm,
            resolute=resolute, verbose=verbose)


@pytest.mark.parametrize(
    "rule_instance", testrules.instances
)
@pytest.mark.parametrize(
    "verbose", [0, 1, 2, 3]
)
def test_abcrules_weightsconsidered(rule_instance, verbose):
    rule_id, algorithm, resolute = rule_instance

    profile = Profile(3)
    profile.add_preferences(DichotomousPreferences([0]))
    profile.add_preferences(DichotomousPreferences([0]))
    profile.add_preferences(DichotomousPreferences([1], 5))
    profile.add_preferences(DichotomousPreferences([0]))
    committeesize = 1

    if (("monroe" in rule_id
         or rule_id in ["lexmav", "rule-x", "phrag-enestr"])):
        with pytest.raises(ValueError):
            abcrules.compute(rule_id, profile, committeesize,
                             algorithm=algorithm, verbose=verbose)
    elif rule_id == "mav":
        # Minimax AV ignores weights by definition
        result = abcrules.compute(
            rule_id, profile, committeesize,
            algorithm=algorithm, resolute=resolute, verbose=verbose)
        if resolute:
            assert result == [[0]]
        else:
            assert result == [[0], [1], [2]]
    else:
        result = abcrules.compute(
            rule_id, profile, committeesize, algorithm=algorithm,
            resolute=resolute, verbose=verbose)
        assert len(result) == 1
        assert result[0] == [1]


@pytest.mark.parametrize(
    "rule_instance", testrules.instances
)
@pytest.mark.parametrize(
    "verbose", [0, 1, 2, 3]
)
def test_abcrules_correct_simple(rule_instance, verbose):
    rule_id, algorithm, resolute = rule_instance
    profile = Profile(4)
    profile.add_preferences([[0], [1], [2], [3]])
    committeesize = 2

    committees = abcrules.compute(
        rule_id, profile, committeesize, algorithm=algorithm,
        resolute=resolute, verbose=verbose)

    if resolute:
        assert len(committees) == 1
    else:
        assert len(committees) == 6


@pytest.mark.parametrize(
    "algorithm", abcrules.rules["monroe"].algorithms
)
def test_monroe_indivisible(algorithm):
    if algorithm == "gurobi":
        pytest.importorskip("gurobipy")
    profile = Profile(4)
    profile.add_preferences([[0], [0], [0], [1, 2], [1, 2], [1], [3]])
    committeesize = 3

    assert (abcrules.compute_monroe(profile, committeesize,
                                    algorithm=algorithm, resolute=False)
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

    assert len(abcrules.rules["optphrag"].compute(
        profile, committeesize, algorithm="gurobi", resolute=False)) == 12


@pytest.mark.parametrize(
    "rule_instance", testrules.instances
)
@pytest.mark.parametrize(
    "verbose", [0, 1, 2, 3]
)
@pytest.mark.parametrize(
    "instance", testinsts.instances
)
def test_abcrules_correct(rule_instance, verbose, instance):
    rule_id, algorithm, resolute = rule_instance
    profile, exp_results, committeesize = instance
    print(profile)
    committees = abcrules.compute(
        rule_id, profile, committeesize, algorithm=algorithm,
        verbose=verbose, resolute=resolute)
    if resolute:
        assert len(committees) == 1
        assert committees[0] in exp_results[rule_id]
    else:
        assert committees == exp_results[rule_id]


def test_seqphragmen_irresolute():
    profile = Profile(3)
    profile.add_preferences([[0, 1], [0, 1], [0], [1, 2], [2]])
    committeesize = 2
    committees = abcrules.rules["seqphrag"].compute(
        profile, committeesize, resolute=False)
    assert committees == [[0, 1], [0, 2]]

    committees = abcrules.rules["seqphrag"].compute(
        profile, committeesize, resolute=True)
    assert committees == [[0, 2]]


def test_seqpav_irresolute():
    profile = Profile(3)
    profile.add_preferences([[0, 1]] * 3 + [[0], [1, 2], [2], [2]])
    committeesize = 2

    committees = abcrules.rules["seqpav"].compute(
        profile, committeesize, resolute=False)
    assert committees == [[0, 1], [0, 2], [1, 2]]

    committees = abcrules.rules["seqpav"].compute(
        profile, committeesize, resolute=True)
    assert committees == [[0, 2]]


testrules_jansonex = []
for rule_id, algorithm in testrules.irres_instances:
    if rule_id not in ["phragm-enestr", "seqphrag", "pav",
                       "seqpav", "revseqpav"]:
        continue
    testrules_jansonex.append((rule_id, algorithm))


@pytest.mark.parametrize(
    "rule_id, algorithm", testrules_jansonex
)
def test_jansonexamples(rule_id, algorithm):
    # example from Janson's survey (https://arxiv.org/pdf/1611.08826.pdf),
    # Example 3.7, 18.1
    profile = Profile(6)
    A = 0
    B = 1
    C = 2
    P = 3
    Q = 4
    R = 5
    profile.add_preferences([[A, B, C]] * 1034 + [[P, Q, R]] * 519 +
                            [[A, B, Q]] * 90 + [[A, P, Q]] * 90)
    committeesize = 3

    committees = abcrules.compute(
        rule_id, profile, committeesize, algorithm=algorithm, resolute=False)
    assert committees == [[A, B, Q]]


@pytest.mark.parametrize(
    "rule_instance", testrules.res_instances
)
@pytest.mark.parametrize(
    "verbose", [0, 1, 2, 3]
)
def test_tiebreaking_order(rule_instance, verbose):
    rule_id, algorithm = rule_instance
    profile = Profile(4)
    profile.add_preferences([[1]] * 2 + [[0]] * 2 + [[2]] * 2)
    committeesize = 1

    committees = abcrules.compute(
        rule_id, profile, committeesize, algorithm=algorithm,
        resolute=True, verbose=verbose)
    assert committees == [[0]]


@pytest.mark.parametrize(
    "rule", abcrules.rules.values()
)
@pytest.mark.parametrize(
    "verbose", [0, 1, 2, 3]
)
@pytest.mark.parametrize(
    "resolute", [True, False]
)
def test_unspecified_algorithms(rule, verbose, resolute):
    if resolute not in rule.resolute:
        return
    profile = Profile(3)
    profile.add_preferences([[0, 1], [1, 2]])
    committeesize = 2
    with pytest.raises(NotImplementedError):
        rule.compute(
            profile, committeesize, algorithm="made-up-algorithm",
            resolute=resolute, verbose=verbose)


testrules_fastest = []
for rule in abcrules.rules.values():
    for resolute in rule.resolute:
        testrules_fastest.append((rule, resolute))


@pytest.mark.parametrize(
    "rule, resolute", testrules_fastest
)
def test_fastest_algorithms(rule, resolute):
    profile = Profile(4)
    profile.add_preferences([[0, 1], [1, 2], [0, 2, 3]])
    committeesize = 2
    algo = rule.fastest_algo()
    if algo is None:
        pytest.skip("no supported algorithms for " + rule.shortname)
    rule.compute(
        profile, committeesize, algorithm=algo, resolute=resolute)


@pytest.mark.parametrize(
    "rule_instance", testrules.instances
)
@pytest.mark.parametrize(
    "verbose", [0, 1, 2, 3]
)
def test_output(capfd, rule_instance, verbose):
    rule_id, algorithm, resolute = rule_instance

    profile = Profile(2)
    profile.add_preferences([[0]])
    committeesize = 1
    abcrules.compute(
        rule_id, profile, committeesize,
        algorithm=algorithm, resolute=resolute,
        verbose=verbose)
    out = capfd.readouterr().out
    if verbose == 0:
        assert out == ""
    else:
        assert len(out) > 0
