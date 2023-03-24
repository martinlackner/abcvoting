"""
Unit tests for abcvoting/properties.py.
"""

import pytest
import os

import abcvoting.misc
from abcvoting.output import DETAILS, output
from abcvoting.preferences import Profile
from abcvoting import abcrules, properties, fileio

# set verbosity to DETAILS to increase unittest coverage
output.set_verbosity(verbosity=DETAILS)


# instances to check output of check_* property functions
def _create_handcrafted_instances():
    handcrafted_instances = []

    # very simple instance with committeesize 1
    profile = Profile(5)
    profile.add_voters([{1, 2, 3}, {0, 1}, {0, 1, 2}, {1, 2}])
    # cand 1 is approved by everyone
    for property_name in properties.PROPERTY_NAMES:
        handcrafted_instances.append((property_name, profile, {1}, True))
    # cand 4 is approved by no one
    for property_name in properties.PROPERTY_NAMES:
        handcrafted_instances.append((property_name, profile, {4}, False))

    # add an instance from
    # Lackner and Skowron, 2021, "Approval-Based Committee Voting", Example 20
    profile = Profile(4)
    profile.add_voters(
        [[0, 3]] + [[0, 1]] + [[1, 2]] + [[2, 3]] + [[0]] * 2 + [[1]] * 2 + [[2]] * 2 + [[3]] * 2
    )
    committee = {0, 1, 2}
    expected_result = True
    handcrafted_instances.append(("ejr", profile, committee, expected_result))

    # add an instance from
    # Aziz et al, 2016, "Justified Representation in Approval-Based Committee Voting", Example 4
    profile = Profile(6)
    profile.add_voters(
        [[0]] + [[1]] + [[2]] + [[3]] + [[0, 4, 5]] + [[1, 4, 5]] + [[2, 4, 5]] + [[3, 4, 5]]
    )
    committee = {0, 1, 2, 3}
    expected_result = False
    handcrafted_instances.append(("ejr", profile, committee, expected_result))

    # add an instance from
    # Aziz et al, 2016, "Justified Representation in Approval-Based Committee Voting", Example 5
    profile = Profile(6)
    profile.add_voters([[0]] * 2 + [[0, 1, 2]] * 2 + [[1, 2, 3]] * 2 + [[3, 4]] + [[3, 5]])
    committee = {0, 3, 4, 5}
    expected_result = False
    handcrafted_instances.append(("ejr", profile, committee, expected_result))

    # add an instance from
    # Aziz et al, 2016, "Justified Representation in Approval-Based Committee Voting", Example 6
    profile = Profile(12)
    profile.add_voters([[0, 10]] * 3 + [[0, 11]] * 3 + [[1, 2, 3, 4, 5, 6, 7, 8, 9]] * 14)
    committee = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    expected_result = True
    handcrafted_instances.append(("ejr", profile, committee, expected_result))

    # add an instance from
    # Aziz et al, 2016, "Justified Representation in Approval-Based Committee Voting", Example 8
    profile = Profile(5)
    profile.add_voters([[0, 1]] * 2 + [[2]] + [[3, 4]])
    committee = {0, 2, 3, 4}
    expected_result = False
    handcrafted_instances.append(("ejr", profile, committee, expected_result))

    # add an instance from
    # Brill et al, 2021, "Phragmen's Voting Methods and Justified Representation", Example 6
    profile = Profile(14)
    profile.add_voters(
        [[0, 1, 2]] * 2
        + [[0, 1, 3]] * 2
        + [[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]] * 6
        + [[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]] * 5
        + [[4, 5, 6, 7, 8, 9, 10, 11, 12, 13]] * 9
    )
    committee = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
    expected_result = False
    handcrafted_instances.append(("ejr", profile, committee, expected_result))

    # EJR+
    # Brill and Peters, 2023, "Robust and Verifiable Proportionality Axioms for Multiwinner Voting", Example 3 left
    profile = Profile(7)
    profile.add_voters(
        [[0, 1, 2]]
        + [[0, 1, 2, 3]] * 2
        + [[0, 1, 2, 3, 4]]
        + [[2, 3, 4]]
        + [[2, 3, 4, 5]]
        + [[2, 3, 4, 5]]
        + [[3, 4, 5, 6]]
    )
    committee = {0, 2, 4, 6}
    expected_result = True
    handcrafted_instances.append(("ejr", profile, committee, expected_result))
    expected_result = False
    handcrafted_instances.append(("ejr+", profile, committee, expected_result))

    # Brill and Peters, 2023, "Robust and Verifiable Proportionality Axioms for Multiwinner Voting", Example 3 right
    profile = Profile(7)
    profile.add_voters([[0]] + [[0, 1]] + [[0, 1, 2, 3]] + [[2, 3, 4]] * 3 + [[4, 5]] + [[5, 6]])
    committee = {0, 1, 2, 6}
    expected_result = True
    handcrafted_instances.append(("ejr", profile, committee, expected_result))
    expected_result = False
    handcrafted_instances.append(("ejr+", profile, committee, expected_result))

    # Brill and Peters, 2023, "Robust and Verifiable Proportionality Axioms for Multiwinner Voting", Remark 2 (core does not imply EJR+)
    profile = Profile(3)
    profile.add_voters([[0, 1]] + [[0, 2]])
    committee = {1, 2}
    expected_result = True
    handcrafted_instances.append(("core", profile, committee, expected_result))
    expected_result = False
    handcrafted_instances.append(("ejr+", profile, committee, expected_result))

    # add an instance from
    # Sanchez-Fernandez et al, 2017, "Proportional Justified Representation", Example 1
    profile = Profile(8)
    profile.add_voters([[0]] + [[1]] + [[2]] + [[3]] + [[4, 5, 6, 7]] * 6)
    committee = {0, 1, 2, 3, 4, 5, 6}
    expected_result = False
    handcrafted_instances.append(("pjr", profile, committee, expected_result))

    # for the second instance, the profile is the same
    committee = {1, 2, 3, 4, 5, 6, 7}
    expected_result = True
    handcrafted_instances.append(("pjr", profile, committee, expected_result))

    # add an instance from
    # Brill et al, 2021, "Phragmen's Voting Methods and Justified Representation", Example 5
    profile = Profile(6)
    profile.add_voters(
        [[0]] + [[1]] + [[2]] + [[3]] + [[0, 4, 5]] + [[1, 4, 5]] + [[2, 4, 5]] + [[3, 4, 5]]
    )
    committee = {0, 1, 2, 3}
    expected_result = True
    handcrafted_instances.append(("pjr", profile, committee, expected_result))

    # add an instance from
    # Brill et al, 2021, "Phragmen's Voting Methods and Justified Representation", Example 7
    profile = Profile(7)
    profile.add_voters([[0, 1, 2, 3]] * 67 + [[4]] * 12 + [[5]] * 11 + [[6]] * 10)
    committee = {0, 1, 2, 4, 5, 6}
    expected_result = False
    handcrafted_instances.append(("pjr", profile, committee, expected_result))

    # add an instance from
    # Lackner and Skowron, 2021, "Approval-Based Committee Voting", Example 23
    profile = Profile(15)
    profile.add_voters(
        [[0, 1, 2, 3]]
        + [[0, 1, 2, 4]]
        + [[0, 1, 2, 5]]
        + [[6, 7, 8]]
        + [[9, 10, 11]]
        + [[12, 13, 14]]
    )
    committee = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13}
    expected_result = True
    handcrafted_instances.append(("priceability", profile, committee, expected_result))
    committee = {0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14}
    expected_result = False
    handcrafted_instances.append(("priceability", profile, committee, expected_result))

    profile = Profile(3)
    profile.add_voters([[0, 1]] + [[0, 1, 2]] + [[2]])
    committee = {0, 1}
    expected_result = True
    handcrafted_instances.append(("priceability", profile, committee, expected_result))

    # add an instance from
    # Lackner and Skowron, 2021, "Approval-Based Committee Voting", Example 23
    profile = Profile(15)
    profile.add_voters(
        [[0, 1, 2, 3]]
        + [[0, 1, 2, 4]]
        + [[0, 1, 2, 5]]
        + [[6, 7, 8]]
        + [[9, 10, 11]]
        + [[12, 13, 14]]
    )
    committee = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13}
    expected_result = True
    handcrafted_instances.append(("stable-priceability", profile, committee, expected_result))
    committee = {0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14}
    expected_result = False
    handcrafted_instances.append(("stable-priceability", profile, committee, expected_result))

    profile = Profile(3)
    profile.add_voters([[0, 1]] + [[0, 1, 2]] + [[2]])
    committee = {0, 1, 2}
    expected_result = True
    handcrafted_instances.append(("stable-priceability", profile, committee, expected_result))
    committee = {0, 1}
    expected_result = False
    handcrafted_instances.append(("stable-priceability", profile, committee, expected_result))

    # add an instance from
    # Lackner and Skowron, 2021, "Approval-Based Committee Voting", Example 23
    profile = Profile(15)
    profile.add_voters(
        [[0, 1, 2, 3]]
        + [[0, 1, 2, 4]]
        + [[0, 1, 2, 5]]
        + [[6, 7, 8]]
        + [[9, 10, 11]]
        + [[12, 13, 14]]
    )
    committee = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13}
    expected_result = True
    handcrafted_instances.append(("core", profile, committee, expected_result))
    committee = {0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14}
    expected_result = False
    handcrafted_instances.append(("core", profile, committee, expected_result))

    profile = Profile(20)
    profile.add_voters([[i, i + 10] for i in range(10)])
    committee = set(range(20))
    expected_result = True
    handcrafted_instances.append(("core", profile, committee, expected_result))
    committee = set(range(9)).union({10})
    expected_result = False
    handcrafted_instances.append(("core", profile, committee, expected_result))

    # negative JR instance
    profile = Profile(12)
    profile.add_voters([[0, 1, 2]] * 2 + [[3]])
    committee = {0, 1, 2}
    expected_result = False
    handcrafted_instances.append(("jr", profile, committee, expected_result))

    # EJR+ implies EJR
    for property_name, profile, committee, expected_result in handcrafted_instances:
        if property_name == "ejr+" and expected_result:
            handcrafted_instances.append(("ejr", profile, committee, expected_result))

    # EJR implies PJR
    for property_name, profile, committee, expected_result in handcrafted_instances:
        if property_name == "ejr" and expected_result:
            handcrafted_instances.append(("pjr", profile, committee, expected_result))

    # PJR implies JR
    for property_name, profile, committee, expected_result in handcrafted_instances:
        if property_name == "pjr" and expected_result:
            handcrafted_instances.append(("jr", profile, committee, expected_result))

    return handcrafted_instances


def _list_abc_yaml_instances():
    currdir = os.path.dirname(os.path.abspath(__file__))
    return [
        currdir + "/test_instances/" + filename
        for filename in os.listdir(currdir + "/test_instances/")
        if filename.endswith(".abc.yaml")
    ]


check_property_instances = _create_handcrafted_instances()
abc_yaml_filenames = _list_abc_yaml_instances()


@pytest.mark.parametrize(
    "algorithm",
    ["brute-force", "fastest", pytest.param("gurobi", marks=pytest.mark.gurobipy), "nonsense"],
)
@pytest.mark.parametrize(
    "property_name, profile, committee, expected_result", check_property_instances
)
def test_property_functions_with_handcrafted_instances(
    property_name, algorithm, profile, committee, expected_result
):
    if algorithm == "nonsense":
        if property_name in ["jr", "ejr+"]:
            return  # no `algorithm` parameter
        with pytest.raises(NotImplementedError):
            properties.check(property_name, profile, committee, algorithm=algorithm)
    else:
        if algorithm == "brute-force" and property_name in ["priceability", "stable-priceability"]:
            return  # not supported
        else:
            assert (
                properties.check(property_name, profile, committee, algorithm=algorithm)
                == expected_result
            )


# to test the output of the brute-force vs gurobi counterparts
@pytest.mark.gurobipy
@pytest.mark.slow
@pytest.mark.parametrize(
    "abc_yaml_instance",
    [
        abc_yaml_filename
        for abc_yaml_filename in abc_yaml_filenames
        if abc_yaml_filename.__contains__("instanceS")
        or abc_yaml_filename.__contains__("instanceP")
        or abc_yaml_filename.__contains__("instanceM")
    ],
)
def test_matching_output_different_approaches(abc_yaml_instance):
    # read the instance from the file
    profile, _, compute_instances, _ = fileio.read_abcvoting_yaml_file(abc_yaml_instance)

    # get one sample committee as input for the functions
    input_committee = compute_instances[0]["result"][0]

    assert properties.check_pareto_optimality(
        profile, input_committee, algorithm="brute-force"
    ) == properties.check_pareto_optimality(profile, input_committee, algorithm="gurobi")
    assert properties.check_EJR(
        profile, input_committee, algorithm="brute-force"
    ) == properties.check_EJR(profile, input_committee, algorithm="gurobi")
    assert properties.check_PJR(
        profile, input_committee, algorithm="brute-force"
    ) == properties.check_PJR(profile, input_committee, algorithm="gurobi")
    assert properties.check_core(
        profile, input_committee, algorithm="brute-force"
    ) == properties.check_core(profile, input_committee, algorithm="gurobi")


@pytest.mark.gurobipy
@pytest.mark.slow
@pytest.mark.parametrize(
    "rule_id, property_name",
    [
        ("av", "pareto"),
        ("sav", "pareto"),
        ("pav", "pareto"),
        ("pav", "jr"),
        ("pav", "pjr"),
        ("pav", "ejr"),
        ("pav", "ejr+"),
        ("slav", "pareto"),
        ("cc", "jr"),
        ("geom2", "pareto"),
        ("seqphragmen", "jr"),
        ("seqphragmen", "pjr"),
        ("seqphragmen", "priceability"),
        ("leximaxphragmen", "jr"),
        ("leximaxphragmen", "pjr"),
        ("monroe", "jr"),
        ("greedy-monroe", "jr"),
        ("equal-shares", "jr"),
        ("equal-shares", "pjr"),
        ("equal-shares", "ejr"),
        ("equal-shares", "ejr+"),
        ("equal-shares", "priceability"),
        ("phragmen-enestroem", "jr"),
        ("phragmen-enestroem", "pjr"),
    ],
)
@pytest.mark.parametrize("abc_yaml_instance", abc_yaml_filenames)
def test_properties_with_rules(rule_id, property_name, abc_yaml_instance):
    # read the instance from the file
    profile, _, compute_instances, _ = fileio.read_abcvoting_yaml_file(abc_yaml_instance)
    print(profile)

    # get output computed by rule `rule_id` for this instance
    for computed_output in compute_instances:
        if computed_output["rule_id"] == rule_id:
            committees = computed_output["result"]
            break
    print(f"committees: {committees}")

    if committees is None:
        return

    # winning committees should satisfy `property_name`
    for committee in committees:
        assert properties.check(property_name, profile, committee)


# Test from literature: Lackner and Skowron 2020
# With given input profile, committee returned by Monroe Rule
# is not Pareto optimal
@pytest.mark.parametrize(
    "algorithm", ["brute-force", pytest.param("gurobi", marks=pytest.mark.gurobipy)]
)
def test_pareto_optimality_methods(algorithm):
    # profile with 4 candidates: a, b, c, d
    profile = Profile(4)

    # add voters in the profile
    profile.add_voters([[0]] * 2 + [[0, 2]] + [[0, 3]] + [[1, 2]] * 10 + [[1, 3]] * 10)

    # compute output committee from Monroe's Rule
    monroe_output = abcrules.compute_monroe(profile, 2)

    # Monroe's Rule should output winning committee {2, 3} for this input
    # It is not Pareto optimal because it is dominated by committee {0, 1}
    # Check using the methods
    is_pareto_optimal = properties.check_pareto_optimality(
        profile, monroe_output[0], algorithm=algorithm
    )

    assert monroe_output == [{2, 3}]
    assert abcvoting.misc.dominate(profile, {0, 1}, {2, 3})
    assert not is_pareto_optimal

    assert properties.check_pareto_optimality(profile, {0, 1}, algorithm=algorithm)
