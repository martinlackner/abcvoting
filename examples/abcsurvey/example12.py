"""Example 12 (MAV)
from the survey: "Approval-Based Multi-Winner Voting:
Axioms, Algorithms, and Applications"
by Martin Lackner and Piotr Skowron
"""

from abcvoting import abcrules
from abcvoting.preferences import Profile
from abcvoting import misc
from abcvoting.output import output
from abcvoting.output import DETAILS

output.set_verbosity(DETAILS)

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

committees = abcrules.compute_minimaxav(profile, 1)


# verify correctness
assert committees == [{b}, {c}]
