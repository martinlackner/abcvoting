"""Example 5 (PAV, seq-PAV, revseq-PAV)
from the survey: "Approval-Based Multi-Winner Voting:
Axioms, Algorithms, and Applications"
by Martin Lackner and Piotr Skowron
"""

import sys

sys.path.insert(0, "..")
from abcvoting import abcrules
from abcvoting.preferences import Profile, ApprovalSet
from abcvoting import misc


print(misc.header("Example 5", "*"))

# Approval profile
num_cand = 4
a, b, c, d = range(4)  # a = 0, b = 1, c = 2, ...
cand_names = "abcd"

apprsets = [[a, b]] * 3 + [[a, d]] * 6 + [[b]] * 4 + [[c]] * 5 + [[c, d]] * 5
profile = Profile(num_cand, cand_names=cand_names)
profile.add_voters(apprsets)

print(misc.header("Input:"))
print(profile.str_compact())

committees_pav = abcrules.compute_pav(profile, 2, verbose=2)

committees_seqpav = abcrules.compute_seqpav(profile, 2, verbose=2)

committees_revseqpav = abcrules.compute_revseqpav(profile, 2, verbose=2)

# verify correctness
assert committees_pav == [[a, c]]
assert committees_seqpav == [[c, d]]
assert committees_revseqpav == [[c, d]]


print("\n")
print(misc.header("Example from Janson's survey (Example 13.3) / Thiele:", "*"))

# Approval profile
num_cand = 4
a, b, c, d = range(4)  # a = 0, b = 1, c = 2, ...
cand_names = "abcd"

profile = Profile(num_cand, cand_names=cand_names)
profile.add_voter(ApprovalSet([a, c, d], 960))
profile.add_voter(ApprovalSet([b, c, d], 3000))
profile.add_voter(ApprovalSet([b, c], 520))
profile.add_voter(ApprovalSet([a, b], 1620))
profile.add_voter(ApprovalSet([a, d], 1081))
profile.add_voter(ApprovalSet([a, c], 1240))
profile.add_voter(ApprovalSet([b, d], 360))
profile.add_voter(ApprovalSet([d], 360))
profile.add_voter(ApprovalSet([c], 120))
profile.add_voter(ApprovalSet([b], 60))

print(misc.header("Input:"))
print(profile.str_compact())

committees_pav = abcrules.compute_pav(profile, 2, verbose=2)

committees_seqpav = abcrules.compute_seqpav(profile, 2, verbose=2)

committees_revseqpav = abcrules.compute_revseqpav(profile, 2, verbose=2)


# verify correctness
assert committees_pav == [[a, b]]
assert committees_seqpav == [[a, c]]
assert committees_revseqpav == [[b, d]]
