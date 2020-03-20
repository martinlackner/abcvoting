"""
Compute all implemented ABC rules for a profile
"""

from __future__ import print_function
import sys
sys.path.insert(0, '..')
from abcvoting.preferences import Profile
from abcvoting import abcrules
from abcvoting.committees import print_committees


def allrules(profile, committeesize, ilp=True, include_resolute=False):
    """Prints the winning committees for all implemented rules"""
    for rule in list(abcrules.MWRULES.keys()):
        if not ilp and "-ilp" in rule:
            continue
        print(abcrules.MWRULES[rule] + ":")
        committees = abcrules.compute_rule(rule, profile, committeesize)
        print_committees(committees)

        if include_resolute:
            print(abcrules.MWRULES[rule] + " (with tie-breaking):")
            committees = abcrules.compute_rule(
                rule, profile, committeesize, resolute=True)
            print_committees(committees)


# See whether the Gurobi ILP solver is available
ilp = True
try:
    import gurobipy  # pylint: disable=unused-import
    print("ILP solver Gurobi available.\n")
except ImportError:
    ilp = False
    print("ILP solver Gurobi not available (import gurobipy failed).\n")

print("****************************************")

# Compute all implemented multiwinner rules

num_cand = 6
profile = Profile(num_cand)
profile.add_preferences([[0, 4, 5], [0], [1, 4, 5], [1],
                         [2, 4, 5], [2], [3, 4, 5], [3]])
committeesize = 4

allrules(profile, committeesize, ilp=ilp)
