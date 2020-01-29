from __future__ import print_function

import sys
import random
sys.path.insert(0, '..')

import genprofiles
from preferences import Profile
import rules_approval
import committees

# See whether the Gurobi ILP solver is available
ilp = True
try:
    import gurobipy  # pylint: disable=unused-import
except ImportError:
    ilp = False
    print("ILP solver Gurobi not available (import gurobipy failed).")
    print()

print("****************************************")


random.seed(31415)

committeesize = 3
c_count = 10


def compute(ranking_list, cand_count, gen_profile_name):
    """As long as the parameters are not set to guarantee that
    one voter votes for at least committeesize candidates
    it can always happen that less candidates than committeesize are
    approved.
    For this it is recommended to use a loop and check before
    computing the rule.
    This method returns False in case of too few approved candidates."""
    profile = Profile(c_count)
    profile.add_preferences(rankings)
    try:
        committees.enough_approved_candidates(profile, committeesize)
    except Exception:
        return False
    print("Computing a committee of size", committeesize, end=' ')
    print("with the Proportional Approval Voting (PAV) rule")
    print("given a randomly generated profile through the method",
          gen_profile_name)
    print(profile)
    print("Output:")
    output = rules_approval.compute_pav(profile, committeesize, ilp=ilp)
    committees.print_committees(output)

    print("****************************************")
    return True


while True:
    rankings = genprofiles.random_urn_profile(c_count, 3,
                                              2, 0.4)
    if compute(rankings, c_count, "random_urn"):
        break
while True:
    rankings = genprofiles.\
        random_urn_party_list_profile(c_count, 3, 2, 0.4, uniform=False)
    if compute(rankings, c_count, "random_urn_party_list"):
        break


rankings = genprofiles.random_IC_profile(c_count, 3, 4)
if not compute(rankings, c_count, "random_IC"):
    print("This should not be possible as setsize=4 is larger than",
          "committeesize", committeesize)


rankings = \
    genprofiles.random_IC_party_list_profile(c_count, 3,
                                             2, uniform=True)
if not compute(rankings, c_count, "random_IC_party_list"):
    print("This should not be possible as two parties and 10 candidates"
          + " with uniform split equals 5 candidates per party",
          "which is greater than the committeesize", committeesize)


rankings = genprofiles.\
    random_mallows_profile(c_count, 4, 4, 0.7)
if not compute(rankings, c_count, "random_mallows"):
    print("This should not be possible as setsize=4 is larger than",
          "committeesize", committeesize)

while True:
    rankings= genprofiles.\
        random_2d_points_profile(c_count, 4, "twogroups",
                                 "uniform_square", 0.5, 1.9)
    if compute(rankings, c_count, "random_2d_points"):
        break

while True:
    rankings = genprofiles. \
        random_2d_points_party_list_profile(c_count, 4, 2, "twogroups",
                                            "normal", 0.5,
                                            uniform=False)
    if compute(rankings, c_count, "random_2d_points_party_list"):
        break

while True:
    rankings = genprofiles.random_IC_party_list_profile(
        10, 2, 3, uniform=False)
    if compute(rankings, c_count, "random_IC_party_list"):
        break
