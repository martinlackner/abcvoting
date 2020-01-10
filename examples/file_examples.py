
import sys
sys.path.insert(0, '..')

import file_reader
from preferences import Profile
import rules_approval
import committees

# See whether the Gurobi ILP solver is available

ilp = True
try:
    import gurobipy  # pylint: disable=unused-import
except ImportError:
    ilp = False
    print "ILP solver Gurobi not available (import gurobipy failed)."
    print

print "****************************************"


profiles = file_reader.load_sois_from_dir("./examples/toi_examples/",
                                          max_approval_percent=0.7)
committeesize = 2

for candidate_map, rankings, cand_count in profiles:
    profile = Profile(cand_count)
    profile.add_preferences(rankings)
    print "Computing a committe of size", committeesize,
    print "with the Proportional Approval Voting (PAV) rule"
    print "given a", profile
    print "Output:"
    output = rules_approval.compute_pav(profile, committeesize, ilp=ilp)
    committees.print_committees(output)

    print "****************************************"