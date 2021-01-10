"""Remark 3
from the survey: "Approval-Based Multi-Winner Voting:
Axioms, Algorithms, and Applications"
by Martin Lackner and Piotr Skowron
"""


import sys

sys.path.insert(0, "../..")
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
approval_sets = [[a, b], [a, b, d], [a, b, c], [a, c, d], [a, c, d], [b], [c], [d]]
cand_names = "abcd"

profile = Profile(num_cand, cand_names=cand_names)
profile.add_voters(approval_sets)

print(misc.header("Input:"))
print(profile.str_compact())

abcrules.compute_revseqpav(profile, 1, resolute=False, verbose=1)

abcrules.compute_av(profile, 1, verbose=1)
