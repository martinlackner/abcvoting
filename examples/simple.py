"""
Very simple example (compute PAV)
"""

import sys

sys.path.insert(0, "..")
from abcvoting.preferences import Profile
from abcvoting import abcrules


num_cand = 5
profile = Profile(num_cand)
profile.add_voters([[0, 1, 2], [0, 1], [0, 1], [1, 2], [3, 4], [3, 4]])
committeesize = 3
print("Computing a committee of size", committeesize)
print("with the Proportional Approval Voting (PAV) rule")
print("given a", profile)
committees = abcrules.compute_pav(profile, committeesize)
print("\nOutput: " + str(committees))
