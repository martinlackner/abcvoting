"""
Example 2.12 (Method of Equal Shares, seq-Phragmen).

From "Multi-Winner Voting with Approval Preferences"
by Martin Lackner and Piotr Skowron
https://arxiv.org/abs/2007.01795
"""

from abcvoting import abcrules
from abcvoting.preferences import Profile
from abcvoting import misc
from abcvoting.output import output
from abcvoting.output import DETAILS

output.set_verbosity(DETAILS)

print(misc.header("Example 12", "*"))

# Approval profile
num_cand = 4
a, b, c, d = range(4)
approval_sets = [{c, d}, {c, d}, {c, d}, {a, b}, {a, b}, {a, c}, {a, c}, {b, d}]
cand_names = "abcd"

profile = Profile(num_cand, cand_names=cand_names)
profile.add_voters(approval_sets)

print(misc.header("Input:"))
print(profile.str_compact())

committees_equal_shares = abcrules.compute_equal_shares(
    profile, 3, resolute=True, algorithm="standard-fractions"
)

committees_seqphragmen = abcrules.compute_seqphragmen(
    profile, 3, resolute=True, algorithm="standard-fractions"
)

# verify correctness
assert committees_equal_shares == [{a, c, d}]
assert committees_seqphragmen == [{b, c, d}]
