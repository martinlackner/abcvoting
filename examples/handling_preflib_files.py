from __future__ import print_function

import sys
import os
sys.path.insert(0, '..')
from abcvoting import fileio
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

profiles = fileio.load_preflib_files_from_dir(currdir + "/toi_examples/",
                                              appr_percent=0.7)

for profile in profiles:
    print("Computing a committee of size", committeesize, end=' ')
    print("with the Proportional Approval Voting (PAV) rule")
    print("given a", profile)
    print("Output:")
    output = abcrules.compute_pav(profile, committeesize, ilp=ilp)
    committees.print_committees(output)
    print("****************************************")


# Example to read a single file (parameter setsize)
profile = fileio.read_preflib_file(
    currdir + "/toi_examples/ex_2010.toi", setsize=1)

print("Computing a committee of size", committeesize, end=' ')
print("with Phragmen's sequential rule")
print("given a", profile)
print("Output:")
output = abcrules.compute_seqphragmen(profile, committeesize)
committees.print_committees(output)
print("****************************************")
