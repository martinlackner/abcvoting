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

# Test from literature: Lackner and Skowron 2021
# With given input profile, committee returned by Monroe Rule
# is not Pareto optimal
def test_pareto_optimality_methods():
    # profile with 4 candidates: a, b, c, d
    profile = Profile(4)

    # add voters in the profile
    profile.add_voter(Voter([0]))
    profile.add_voter(Voter([0]))

    profile.add_voter(Voter([0, 2]))

    profile.add_voter(Voter([0, 3]))

    for i in range(10):
        profile.add_voter(Voter([1, 2]))

    for i in range(10):
        profile.add_voter(Voter([1, 3]))

    # compute output committee from Monroe's Rule
    monroe_output = abcrules.compute_monroe(profile, 2)

    # Monroe's Rule should output winning committee {2, 3} for this input
    # It is not Pareto optimal because it is dominated by committee {0, 1}
    # Check using the methods
    is_pareto_optimal = properties.check_pareto_optimality(profile, monroe_output[0])

    assert is_pareto_optimal == False