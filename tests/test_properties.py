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
