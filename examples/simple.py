"""
Very simple example (compute PAV)
"""

from __future__ import print_function
import sys

sys.path.insert(0, "..")
from abcvoting.preferences import Profile
from abcvoting import abcrules


num_cand = 5
profile = Profile(num_cand)
profile.add_voters([{0, 1, 2}, {0, 1}, {0, 1}, {1, 2}, {3, 4}, {3, 4}])
committeesize = 3
print(
    f"Computing winning committees of size {committeesize}\n"
    f"with the Proportional Approval Voting (PAV) rule\n"
    f"given the following {profile}"
)
committees = abcrules.compute_pav(profile, committeesize, verbose=1)
