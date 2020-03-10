from __future__ import print_function

import sys
import os
sys.path.insert(0, '..')
from abcvoting import fileio
from abcvoting.preferences import Profile
from abcvoting import abcrules
from abcvoting import committees

# See whether the Gurobi ILP solver is available

ilp = True
try:
    import gurobipy  # pylint: disable=unused-import
except ImportError:
    ilp = False
    print("ILP solver Gurobi not available (import gurobipy failed).")
    print()

print("****************************************")

committeesize = 2

# directory of preflib.py
currdir = os.path.dirname(os.path.abspath(__file__))


def compute(appr_sets, cand_count):
    profile = Profile(cand_count)
    profile.add_preferences(appr_sets)
    print("Computing a committee of size", committeesize, end=' ')
    print("with the Proportional Approval Voting (PAV) rule")
    print("given a", profile)
    print("Output:")
    output = abcrules.compute_pav(profile, committeesize, ilp=ilp)
    committees.print_committees(output)

    print("****************************************")


profiles = fileio.load_election_files_from_dir(currdir + "/toi_examples/",
                                               max_approval_percent=0.7)

for candidate_map, appr_sets, cand_count in profiles:
    compute(appr_sets, cand_count)

# Example to read a single file
candidate_map, appr_sets, cand_count = \
    fileio.read_election_file(currdir + "/toi_examples/ex_2010.toi",
                              max_approval_percent=0.5)

compute(appr_sets, cand_count)


# Example with setsize
_, appr_sets, cand_count = \
    fileio.read_election_file(currdir + "/toi_examples/ex_2010.toi",
                              setsize=1)

compute(appr_sets, cand_count)
