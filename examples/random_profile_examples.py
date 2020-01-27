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
profiles = {}
profiles["random_urn"] = genprofiles.random_urn_profile(c_count, 3,
                                                        2, 0.4)


profiles["random_urn_party_list"] = genprofiles.\
    random_urn_party_list_profile(c_count, 3, 2, 0.4, uniform=True)
profiles["random_IC"] = genprofiles.random_IC_profile(c_count, 3, 4)
profiles["random_IC_party_list"] = \
    genprofiles.random_IC_party_list_profile(c_count, 3, 2,
                                             uniform=True)
profiles["random_mallows"] = genprofiles.\
    random_mallows_profile(c_count, 4, 4, 0.7)
profiles["random_2d_points"] = genprofiles.\
    random_2d_points_profile(c_count, 4, "twogroups", "uniform_square",
                             0.5, 1.9)
profiles["random_2d_points_party_list"] = genprofiles. \
    random_2d_points_party_list_profile(c_count, 4, 2, "twogroups",
                                        "normal", 0.5, uniform=True)


print("****************************************")

for gen_profile_name, rankings in profiles.items():
    profile = Profile(c_count)
    profile.add_preferences(rankings)
    print("Computing a committee of size", committeesize, end=' ')
    print("with the Proportional Approval Voting (PAV) rule")
    print("given a randomly generated profile through the method",
          gen_profile_name)
    print(profile)
    print("Output:")
    output = rules_approval.compute_pav(profile, committeesize, ilp=ilp)
    committees.print_committees(output)

    print("****************************************")


# The methods with the uniform parameter and not set to True
# are not guaranteed to return a profile with votes for
# more than one candidate
while True:
    profile = Profile(c_count)
    rankings = genprofiles.random_IC_party_list_profile(
        10, 2, 3, uniform=False)
    profile.add_preferences(rankings)
    try:
        committees.enough_approved_candidates(profile, committeesize)
        print("Computing a committe of size", committeesize, end=' ')
        print("with the Proportional Approval Voting (PAV) rule")
        print("given a randomly generated profile through the method",
              "random_IC_party_list")
        print(profile)
        print("Output:")
        output = rules_approval.compute_pav(profile, committeesize,
                                            ilp=ilp)
        committees.print_committees(output)

        print("****************************************")
        break
    except Exception:
        pass
