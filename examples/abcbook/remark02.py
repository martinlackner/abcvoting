"""
Remark 2.

From "Multi-Winner Voting with Approval Preferences"
by Martin Lackner and Piotr Skowron
http://dx.doi.org/10.1007/978-3-031-09016-5
"""

from abcvoting import abcrules
from abcvoting.preferences import Profile
from abcvoting import misc
from abcvoting.output import output
from abcvoting.output import DETAILS

output.set_verbosity(DETAILS)

print("Remark 2:\n*********\n")

# Approval profile
num_cand = 3
a, b, c = range(3)  # a = 0, b = 1, c = 2
approval_sets = [[a]] * 99 + [[a, b, c]]
cand_names = "abc"

profile = Profile(num_cand, cand_names=cand_names)
profile.add_voters(approval_sets)

print(misc.header("Input:"))
print(profile.str_compact())

committees_minimaxav = abcrules.compute_minimaxav(profile, 1)

committees_lexminimaxav = abcrules.compute_lexminimaxav(profile, 1)


# verify correctness
assert committees_minimaxav == [{a}, {b}, {c}]
assert committees_lexminimaxav == [{a}]
