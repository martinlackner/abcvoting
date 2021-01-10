"""Example 11 (Rule X, seq-Phragmen)
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


print(misc.header("Example 11", "*"))

# Approval profile
num_cand = 5
a, b, c, d, e = range(5)
# approval_sets = [[a, b, c]] * 5 + [[a, b, c, d, e]] * 5 + [[a, b, d, e]] * 2 + [[d, e]] * 3
# approval_sets = [[a, b]] + [[a, b, c]] * 3 + [[a, b, c, d, e]] * 5 + [[d, e]] * 2 + [[d]]
approval_sets = (
    [[a, b, c]] * 6 + [[a, b, c, d, e]] * 4 + [[a, b, d, e]] * 2 + [[d, e]] * 3
)
# approval_sets = [[a, b, c]] * 5 + [[a, b, c, e]] + [[a, b, c, d, e]] * 4 + [[a, b, d, e]] * 2 + [[d, e]] * 2 + [[d]]
cand_names = "abcde"

profile = Profile(num_cand, cand_names=cand_names)
profile.add_voters(approval_sets)

print(misc.header("Input:"))
print(profile.str_compact())

committees_rule_x = abcrules.compute_rule_x(profile, 4, resolute=False, verbose=2)

committees_seqphragmen = abcrules.compute_seqphragmen(
    profile, 4, resolute=False, verbose=1
)

# verify correctness
assert committees_rule_x == [{a, b, c, d}, {a, b, c, e}]
assert committees_seqphragmen == [{a, b, d, e}]
