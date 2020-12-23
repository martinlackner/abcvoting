"""Remark 3
from the survey: "Approval-Based Multi-Winner Voting:
Axioms, Algorithms, and Applications"
by Martin Lackner and Piotr Skowron
"""

from __future__ import print_function
import sys
sys.path.insert(0, '../..')
from abcvoting import abcrules
from abcvoting.preferences import Profile
from abcvoting import misc


print("Remark 3:\n*********\n")

# Approval profile
num_cand = 4
a = 0
b = 1
c = 2
d = 3
apprsets = [[a, b], [a, b, d], [a, b, c], [a, c, d], [a, c, d], [b], [c], [d]]
names = "abcd"

profile = Profile(num_cand, names=names)
profile.add_voters(apprsets)

print(misc.header("Input:"))
print(profile.str_compact())

abcrules.compute_revseqpav(profile, 1, resolute=False, verbose=1)

abcrules.compute_av(profile, 1, verbose=1)
