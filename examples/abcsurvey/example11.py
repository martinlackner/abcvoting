"""Example 11 (Rule X, seq-Phragmen)
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

print(misc.header("Example 11", "*"))

# Approval profile
num_cand = 4
a, b, c, d = range(4)
approval_sets = [{c, d}, {c, d}, {c, d}, {a, b}, {a, b}, {a, c}, {a, c}, {b, d}]
cand_names = "abcd"

profile = Profile(num_cand, cand_names=cand_names)
profile.add_voters(approval_sets)

print(misc.header("Input:"))
print(profile.str_compact())

committees_rule_x = abcrules.compute_rule_x(
    profile, 3, resolute=False, algorithm="standard-fractions"
)

# detailed output is only available if resolute=True:
abcrules.compute_rule_x(profile, 3, resolute=True, algorithm="standard-fractions")

committees_seqphragmen = abcrules.compute_seqphragmen(
    profile, 3, resolute=False, algorithm="standard-fractions"
)

# detailed output is only available if resolute=True:
abcrules.compute_seqphragmen(profile, 3, resolute=True, algorithm="standard-fractions")

# verify correctness
assert committees_rule_x == [{a, c, d}]
assert committees_seqphragmen == [{b, c, d}]
