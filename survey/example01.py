"""Example 1 (approval profile for running example)
from the survey: "Approval-Based Multi-Winner Voting:
Axioms, Algorithms, and Applications"
by Martin Lackner and Piotr Skowron
"""

from __future__ import print_function
import sys
sys.path.insert(0, '..')
from abcvoting.preferences import Profile
from abcvoting import misc


num_cand = 8
a, b, c, d, e, f, g = range(7)  # a = 0, b = 1, c = 2, ...
apprsets = [[a, b], [a, b], [a, b], [a, c], [a, c], [a, c],
            [a, d], [a, d], [b, c, f], [e], [f], [g]]
cand_names = "abcdefgh"
committeesize = 4

profile = Profile(num_cand, cand_names=cand_names)
profile.add_voters(apprsets)


if __name__ == "__main__":
    print(misc.header("Example 1", "*"))
    print(profile.str_compact())
    print("desired committee size k = " + str(committeesize))
