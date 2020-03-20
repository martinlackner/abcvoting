"""
Very simple example (compute PAV)
"""

from __future__ import print_function
import sys
sys.path.insert(0, '..')
from abcvoting.preferences import Profile
from abcvoting import abcrules
from abcvoting.committees import print_committees


# See whether the Gurobi ILP solver is available
ilp = True
try:
    import gurobipy  # pylint: disable=unused-import
    print("ILP solver Gurobi available.\n")
except ImportError:
    ilp = False
    print("ILP solver Gurobi not available (import gurobipy failed).\n")

print("****************************************")

# Compute PAV with or without Gurobi

num_cand = 5
profile = Profile(num_cand)
profile.add_preferences([[0, 1, 2], [0, 1], [0, 1], [1, 2], [3, 4], [3, 4]])
committeesize = 3
print("Computing a committee of size", committeesize, end=' ')
print("with the Proportional Approval Voting (PAV) rule")
print("given a", profile)
print("Output:")
committees = abcrules.compute_pav(profile, committeesize, ilp=ilp)
print_committees(committees)
