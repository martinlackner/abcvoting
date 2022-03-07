"""
Unit tests for: properties.py
"""

import pytest
import os

import abcvoting.misc
from abcvoting.output import DETAILS, output
from abcvoting.preferences import Profile, Voter
from abcvoting import abcrules, properties, fileio

# set verbosity to DETAILS to increase unittest coverage
output.set_verbosity(verbosity=DETAILS)

# Test from literature: Lackner and Skowron 2020
# With given input profile, committee returned by Monroe Rule
# is not Pareto optimal
@pytest.mark.parametrize(
    "algorithm", ["brute-force", pytest.param("gurobi", marks=pytest.mark.gurobi)]
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


# instances to check output of EJR methods
EJR_instances = []

# add an instance from
# Lackner and Skowron, 2021, "Approval-Based Committee Voting", Example 20
profile = Profile(4)
profile.add_voters(
    [[0, 3]] + [[0, 1]] + [[1, 2]] + [[2, 3]] + [[0]] * 2 + [[1]] * 2 + [[2]] * 2 + [[3]] * 2
)
committee = {0, 1, 2}
expected_result = True
EJR_instances.append((profile, committee, expected_result))

# add an instance from
# Aziz et al, 2016, "Justified Representation in Approval-Based Committee Voting", Example 4
profile = Profile(6)
profile.add_voters(
    [[0]] + [[1]] + [[2]] + [[3]] + [[0, 4, 5]] + [[1, 4, 5]] + [[2, 4, 5]] + [[3, 4, 5]]
)
committee = {0, 1, 2, 3}
expected_result = False
EJR_instances.append((profile, committee, expected_result))

# add an instance from
# Aziz et al, 2016, "Justified Representation in Approval-Based Committee Voting", Example 5
profile = Profile(6)
profile.add_voters([[0]] * 2 + [[0, 1, 2]] * 2 + [[1, 2, 3]] * 2 + [[3, 4]] + [[3, 5]])
committee = {0, 3, 4, 5}
expected_result = False
EJR_instances.append((profile, committee, expected_result))

# add an instance from
# Aziz et al, 2016, "Justified Representation in Approval-Based Committee Voting", Example 6
profile = Profile(12)
profile.add_voters([[0, 10]] * 3 + [[0, 11]] * 3 + [[1, 2, 3, 4, 5, 6, 7, 8, 9]] * 14)
committee = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
expected_result = True
EJR_instances.append((profile, committee, expected_result))

# add an instance from
# Aziz et al, 2016, "Justified Representation in Approval-Based Committee Voting", Example 8
profile = Profile(5)
profile.add_voters([[0, 1]] * 2 + [[2]] + [[3, 4]])
committee = {0, 2, 3, 4}
expected_result = False
EJR_instances.append((profile, committee, expected_result))

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
EJR_instances.append((profile, committee, expected_result))


@pytest.mark.parametrize(
    "algorithm", ["brute-force", pytest.param("gurobi", marks=pytest.mark.gurobi)]
)
@pytest.mark.parametrize("profile, committee, expected_result", EJR_instances)
def test_EJR_methods(algorithm, profile, committee, expected_result):
    # check whether the committee satisfies EJR
    satisfies_EJR = properties.check_EJR(profile, committee, algorithm=algorithm)

    assert satisfies_EJR == expected_result


# instances to check output of PJR methods
PJR_instances = []

# add an instance from
# Sanchez-Fernandez et al, 2017, "Proportional Justified Representation", Example 1
profile = Profile(8)
profile.add_voters([[0]] + [[1]] + [[2]] + [[3]] + [[4, 5, 6, 7]] * 6)
committee = {0, 1, 2, 3, 4, 5, 6}
expected_result = False
PJR_instances.append((profile, committee, expected_result))

# for the second instance, the profile is the same
committee = {1, 2, 3, 4, 5, 6, 7}
expected_result = True
PJR_instances.append((profile, committee, expected_result))

# add an instance from
# Brill et al, 2021, "Phragmen's Voting Methods and Justified Representation", Example 5
profile = Profile(6)
profile.add_voters(
    [[0]] + [[1]] + [[2]] + [[3]] + [[0, 4, 5]] + [[1, 4, 5]] + [[2, 4, 5]] + [[3, 4, 5]]
)
committee = {0, 1, 2, 3}
expected_result = True
PJR_instances.append((profile, committee, expected_result))

# add an instance from
# Brill et al, 2021, "Phragmen's Voting Methods and Justified Representation", Example 7
profile = Profile(7)
profile.add_voters([[0, 1, 2, 3]] * 67 + [[4]] * 12 + [[5]] * 11 + [[6]] * 10)
committee = {0, 1, 2, 4, 5, 6}
expected_result = False
PJR_instances.append((profile, committee, expected_result))

# From Sanchez-Fernandez et al, 2017:
# "EJR implies PJR"
# Also adding the positive instances from test_EJR_methods()
for profile, committee, expected_result in EJR_instances:
    if expected_result:
        PJR_instances.append((profile, committee, expected_result))


@pytest.mark.parametrize(
    "algorithm", ["brute-force", pytest.param("gurobi", marks=pytest.mark.gurobi)]
)
@pytest.mark.parametrize("profile, committee, expected_result", PJR_instances)
def test_PJR_methods(algorithm, profile, committee, expected_result):
    # check whether the committee satisfies PJR
    satisfies_PJR = properties.check_PJR(profile, committee, algorithm=algorithm)

    assert satisfies_PJR == expected_result


# instances to check output of JR method
JR_instances = []

# From Sanchez-Fernandez et al, 2017:
# "PJR implies JR"
# adding the positive instances from test_PJR_methods()
for profile, committee, expected_result in PJR_instances:
    if expected_result:
        JR_instances.append((profile, committee, expected_result))


# negative JR instance
profile = Profile(12)
profile.add_voters([[0, 1, 2]] * 2 + [[3]])
committee = {0, 1, 2}
expected_result = False
JR_instances.append((profile, committee, expected_result))


@pytest.mark.parametrize("profile, committee, expected_result", JR_instances)
def test_JR_method(profile, committee, expected_result):
    # check whether the committee satisfies JR
    satisfies_JR = properties.check_JR(profile, committee)

    assert satisfies_JR == expected_result


def _list_abc_yaml_instances():
    currdir = os.path.dirname(os.path.abspath(__file__))
    filenames = [
        currdir + "/test_instances/" + filename
        for filename in os.listdir(currdir + "/test_instances/")
        if filename.endswith(".abc.yaml")
    ]

    return filenames


abc_yaml_filenames = _list_abc_yaml_instances()


# to test the output of the brute-force vs gurobi counterparts
@pytest.mark.gurobi
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


@pytest.mark.gurobi
@pytest.mark.slow
@pytest.mark.parametrize("abc_yaml_instance", abc_yaml_filenames)
def test_output_EJR_PAV(abc_yaml_instance):
    # read the instance from the file
    profile, _, compute_instances, _ = fileio.read_abcvoting_yaml_file(abc_yaml_instance)

    # get output computed by rule PAV for this instance
    for computed_output in compute_instances:
        if computed_output["rule_id"] == "pav":
            input_committee = computed_output["result"][0]
            break

    # winning committee for this profile computed by PAV should satisfy EJR
    assert properties.check_EJR(profile, input_committee, algorithm="gurobi")


@pytest.mark.gurobi
@pytest.mark.slow
@pytest.mark.parametrize("abc_yaml_instance", abc_yaml_filenames)
def test_output_PJR_seqPhragmen(abc_yaml_instance):
    # read the instance from the file
    profile, _, compute_instances, _ = fileio.read_abcvoting_yaml_file(abc_yaml_instance)

    # get output computed by rule seqPhragmen for this instance
    for computed_output in compute_instances:
        if computed_output["rule_id"] == "seqphragmen":
            input_committee = computed_output["result"][0]
            break

    # winning committee for this profile computed by seqPhragmen should satisfy PJR
    assert properties.check_PJR(profile, input_committee, algorithm="gurobi")


@pytest.mark.slow
@pytest.mark.parametrize("abc_yaml_instance", abc_yaml_filenames)
def test_output_JR_with_rules(abc_yaml_instance):
    # read the instance from the file
    profile, _, compute_instances, _ = fileio.read_abcvoting_yaml_file(abc_yaml_instance)

    for rule_id in ["monroe", "seqcc", "greedy-monroe", "cc"]:
        # get output computed by rule `rule_id` for this instance
        for computed_output in compute_instances:
            if computed_output["rule_id"] == rule_id:
                input_committee = computed_output["result"][0]
                break
        # winning committee should satisfy JR
        assert properties.check_JR(profile, input_committee)
