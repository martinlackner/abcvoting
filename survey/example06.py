"""Example 6 (Monroe)
from the survey: "Approval-Based Multi-Winner Voting:
Axioms, Algorithms, and Applications"
by Martin Lackner and Piotr Skowron
"""

import sys

sys.path.insert(0, "..")
from abcvoting import abcrules
from survey import example01 as ex1
from abcvoting.scores import monroescore
from abcvoting import misc


print(misc.header("Example 6", "*"))

print(misc.header("Input (election instance from Example 1):"))
print(ex1.profile.str_compact())

committees = abcrules.compute_monroe(ex1.profile, 4, verbose=2)


# verify correctness
a, b, c, d, e, f, g = range(7)  # a = 0, b = 1, c = 2, ...
assert len(committees) == 6
# Monroe-score of all committees is the same
score = monroescore(ex1.profile, committees[0])
for comm in committees:
    assert score == monroescore(ex1.profile, comm)
