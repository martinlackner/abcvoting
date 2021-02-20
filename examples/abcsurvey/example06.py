"""Example 6 (Monroe)
from the survey: "Approval-Based Multi-Winner Voting:
Axioms, Algorithms, and Applications"
by Martin Lackner and Piotr Skowron
"""

from abcvoting import abcrules
from examples.abcsurvey import example01 as ex1
from abcvoting.scores import monroescore
from abcvoting import misc
from abcvoting.output import output
from abcvoting.output import DETAILS

output.set_verbosity(DETAILS)

print(misc.header("Example 6", "*"))

print(misc.header("Input (election instance from Example 1):"))
print(ex1.profile.str_compact())

committees = abcrules.compute_monroe(ex1.profile, 4)


# verify correctness
a, b, c, d, e, f, g = range(7)  # a = 0, b = 1, c = 2, ...
assert len(committees) == 6
# Monroe-score of all committees is the same
score = monroescore(ex1.profile, committees[0])
for committee in committees:
    assert score == monroescore(ex1.profile, committee)
