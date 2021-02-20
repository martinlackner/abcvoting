"""Example 3 (CC)
from the survey: "Approval-Based Multi-Winner Voting:
Axioms, Algorithms, and Applications"
by Martin Lackner and Piotr Skowron
"""

from abcvoting import abcrules
from examples.abcsurvey import example01 as ex1
from abcvoting import misc
from abcvoting.output import output
from abcvoting.output import DETAILS

output.set_verbosity(DETAILS)

print(misc.header("Example 3", "*"))

print(misc.header("Input (election instance from Example 1):"))
print(ex1.profile.str_compact())

committees = abcrules.compute_cc(ex1.profile, 4)


# verify correctness
a, b, c, d, e, f, g = range(7)  # a = 0, b = 1, c = 2, ...
assert committees == [{a, e, f, g}]
