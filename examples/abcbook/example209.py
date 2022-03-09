# -*- coding: utf-8 -*-
"""
Example 2.9 (seq-Phragmen).

From "Multi-Winner Voting with Approval Preferences"
by Martin Lackner and Piotr Skowron
https://arxiv.org/abs/2007.01795
"""

from abcvoting import abcrules
from abcvoting import misc
from abcvoting.preferences import Profile
from abcvoting.output import output, DETAILS

output.set_verbosity(DETAILS)

# the running example profile (Example 1)
num_cand = 8
a, b, c, d, e, f, g = range(7)  # a = 0, b = 1, c = 2, ...
approval_sets = [
    {a, b},
    {a, b},
    {a, b},
    {a, c},
    {a, c},
    {a, c},
    {a, d},
    {a, d},
    {b, c, f},
    {e},
    {f},
    {g},
]
profile = Profile(num_cand, cand_names="abcdefgh")
profile.add_voters(approval_sets)
committeesize = 4
#

print(misc.header("Example 9", "*"))

print(misc.header("Input (election instance from Example 1):"))
print(profile.str_compact())

committees = abcrules.compute_seqphragmen(profile, 4, algorithm="standard-fractions")


# verify correctness
a, b, c, d, e, f, g = range(7)  # a = 0, b = 1, c = 2, ...
assert committees == [{a, b, c, d}]
