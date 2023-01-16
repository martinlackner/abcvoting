"""
Example 2.8 (Greedy Monroe).

From "Multi-Winner Voting with Approval Preferences"
by Martin Lackner and Piotr Skowron
http://dx.doi.org/10.1007/978-3-031-09016-5
"""

from abcvoting import abcrules
from abcvoting import misc
from abcvoting.scores import monroescore
from abcvoting.preferences import Profile
from abcvoting.output import output, DETAILS

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

output.set_verbosity(DETAILS)

print(misc.header("Example 8", "*"))

print(misc.header("Input (election instance from Example 1):"))
print(profile.str_compact())

committees = abcrules.compute_greedy_monroe(profile, 4)


# verify correctness
a, b, c, d, e, f, g = range(7)  # a = 0, b = 1, c = 2, ...
assert committees == [{a, c, d, f}]
assert monroescore(profile, committees[0]) == 10
