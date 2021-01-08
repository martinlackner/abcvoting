"""Remark 2
from the survey: "Approval-Based Multi-Winner Voting:
Axioms, Algorithms, and Applications"
by Martin Lackner and Piotr Skowron
"""

import sys

sys.path.insert(0, "..")
from abcvoting import abcrules
from abcvoting.preferences import Profile
from abcvoting import misc


print("Remark 2:\n*********\n")

# Approval profile
num_cand = 3
a, b, c = range(3)  # a = 0, b = 1, c = 2
apprsets = [[a]] * 99 + [[a, b, c]]
cand_names = "abc"

profile = Profile(num_cand, cand_names=cand_names)
profile.add_voters(apprsets)

print(misc.header("Input:"))
print(profile.str_compact())

committees_mav = abcrules.compute_mav(profile, 1, verbose=2)

committees_lexmav = abcrules.compute_lexmav(profile, 1, verbose=2)


# verify correctness
assert committees_mav == [[a], [b], [c]]
assert committees_lexmav == [[a]]
