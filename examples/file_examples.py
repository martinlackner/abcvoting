from __future__ import print_function

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
    print("ILP solver Gurobi not available (import gurobipy failed).")
    print()

print("****************************************")

committeesize = 2


def compute(rankings, cand_count):
    profile = Profile(cand_count)
    profile.add_preferences(rankings)
    print("Computing a committee of size", committeesize, end=' ')
    print("with the Proportional Approval Voting (PAV) rule")
    print("given a", profile)
    print("Output:")
    output = rules_approval.compute_pav(profile, committeesize, ilp=ilp)
    committees.print_committees(output)

    print("****************************************")


profiles = file_reader.load_election_files_from_dir("./examples/toi_examples/",
                                                    max_approval_percent=0.7)

for candidate_map, rankings, cand_count in profiles:
    compute(rankings, cand_count)


# Example to read a single file
candidate_map, rankings, cand_count = \
    file_reader.read_election_file("./examples/toi_examples/ex_2010.toi",
                                   max_approval_percent=0.5)

compute(rankings, cand_count)


# Example with setsize
_, rankings, cand_count = \
    file_reader.read_election_file("./examples/toi_examples/ex_2010.toi",
                                   setsize=1)

compute(rankings, cand_count)
