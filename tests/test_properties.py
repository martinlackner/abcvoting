"""
Unit tests for: properties.py
"""

import pytest
import os
import re
import random
from abcvoting.output import VERBOSITY_TO_NAME, WARNING, INFO, DETAILS, DEBUG, output
from abcvoting.preferences import Profile, Voter
from abcvoting import abcrules, properties, misc, scores, fileio

# Test from literature: Lackner and Skowron 2020
# With given input profile, committee returned by Monroe Rule
# is not Pareto optimal
@pytest.mark.parametrize("algorithm", ["brute-force", "gurobi"])
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

    assert not is_pareto_optimal
    assert monroe_output == [{2, 3}]


# instances to check output of EJR methods
EJR_instances = []

# create and add the first instance from literature:
# Lackner and Skowron, 2021
profile = Profile(4)
profile.add_voters(
    [[0, 3]] + [[0, 1]] + [[1, 2]] + [[2, 3]] + [[0]] * 2 + [[1]] * 2 + [[2]] * 2 + [[3]] * 2
)
committee = {0, 1, 2}
expected_result = True
EJR_instances.append((profile, committee, expected_result))

# create and add the second instance
# Aziz et al, 2016
profile = Profile(6)
profile.add_voters([[0]] * 2 + [[0, 1, 2]] * 2 + [[1, 2, 3]] * 2 + [[3, 4]] + [[3, 5]])
committee = {0, 3, 4, 5}
expected_result = False
EJR_instances.append((profile, committee, expected_result))

# create and add the third instance
# Brill et al, 2021
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


@pytest.mark.parametrize("algorithm", ["brute-force", "gurobi"])
@pytest.mark.parametrize("profile, committee, expected_result", EJR_instances)
def test_EJR_methods(algorithm, profile, committee, expected_result):
    # check whether the committee satisfies EJR
    satisfies_EJR = properties.check_EJR(profile, committee, algorithm=algorithm)

    assert satisfies_EJR == expected_result
