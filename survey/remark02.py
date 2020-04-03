"""Remark 2
from the survey: "Approval-Based Multi-Winner Voting:
Axioms, Algorithms, and Applications"
by Martin Lackner and Piotr Skowron
"""

from __future__ import print_function
import sys
sys.path.insert(0, '..')
from abcvoting import abcrules
from abcvoting.preferences import Profile
from abcvoting import misc


print("Remark 2:\n*********\n")

# Approval profile
num_cand = 3
a, b, c = list(range(3))  # a = 0, b = 1, c = 2
apprsets = [[a]] * 99 + [[a, b, c]]
names = "abc"

profile = Profile(num_cand, names=names)
profile.add_preferences(apprsets)

print(misc.header("Input:"))
print(profile.str_compact())

committees_mav = abcrules.compute_mav(profile, 1, verbose=2)

committees_lexmav = abcrules.compute_lexmav(profile, 1, verbose=2)


# verify correctness
assert committees_mav == [[a], [b], [c]]
assert committees_lexmav == [[a]]
