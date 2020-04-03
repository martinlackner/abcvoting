from __future__ import print_function

import sys
import os
sys.path.insert(0, '..')
from abcvoting import fileio
from abcvoting import abcrules
from abcvoting.misc import str_candsets


committeesize = 2

# directory of preflib.py
currdir = os.path.dirname(os.path.abspath(__file__))

profiles = fileio.load_preflib_files_from_dir(currdir + "/toi-files/",
                                              appr_percent=0.7)

for profile in profiles:
    print("Computing a committee of size", committeesize, end=' ')
    print("with the Proportional Approval Voting (PAV) rule")
    print("given a", profile)
    print("Output:")
    committees = abcrules.compute_pav(profile, committeesize)
    print(str_candsets(committees))
    print("****************************************")


# Example to read a single file (parameter setsize)
profile = fileio.read_preflib_file(
    currdir + "/toi-files/ex_2010.toi", setsize=1)

print("Computing a committee of size", committeesize, end=' ')
print("with Phragmen's sequential rule")
print("given a", profile)
print("Output:")
committees = abcrules.compute_seqphragmen(profile, committeesize)
print(str_candsets(committees))
print("****************************************")
