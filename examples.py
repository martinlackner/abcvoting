# Simple examples

# Author: Martin Lackner


from preferences import Profile
import rules_approval
import committees


ilp = True
try:
    import gurobipy
except ImportError:
    ilp = False
    print "ILP solver Gurobi not available (import gurobipy failed)."
    print

print "****************************************"

num_cand = 5
profile = Profile(num_cand)
profile.add_preferences([[0, 1, 2], [0, 1], [0, 1], [1, 2], [3, 4], [3, 4]])
committeesize = 3
print "Computing a committe of size", committeesize, "with Proportional Approval Voting given a", profile
print "Output:"
output = rules_approval.compute_pav(profile, committeesize, ilp=ilp)
committees.print_committees(output)

print "****************************************"

num_cand = 6
profile = Profile(num_cand)
profile.add_preferences([[0, 4, 5], [0], [1, 4, 5], [1], [2, 4, 5], [2], [3, 4, 5], [3]])
committeesize = 4

rules_approval.allrules(profile, committeesize, ilp=ilp)
