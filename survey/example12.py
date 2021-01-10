"""Example 12 (MAV)
from the survey: "Approval-Based Multi-Winner Voting:
Axioms, Algorithms, and Applications"
by Martin Lackner and Piotr Skowron
"""

from __future__ import print_function
import sys

sys.path.insert(0, "..")
from abcvoting import abcrules
from abcvoting.preferences import Profile
from abcvoting import misc


print(misc.header("Example 12", "*"))

# Approval profile
num_cand = 3
a, b, c = range(3)  # a = 0, b = 1, c = 2
approval_sets = [{a}] * 99 + [{b, c}]
cand_names = "abc"

profile = Profile(num_cand, cand_names=cand_names)
profile.add_voters(approval_sets)

print(misc.header("Input:"))
print(profile.str_compact())

committees = abcrules.compute_mav(profile, 1, verbose=2)


# verify correctness
assert committees == [{b}, {c}]
