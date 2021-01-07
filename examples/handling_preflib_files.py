from __future__ import print_function

import sys
import os

sys.path.insert(0, "..")
from abcvoting import fileio
from abcvoting import abcrules
from abcvoting.misc import str_candsets
from abcvoting.preferences import Profile

currdir = os.path.dirname(os.path.abspath(__file__))


# Write a profile to toi file
profile = Profile(5, "ABCDE")
profile.add_voters([[0, 1], [1, 3, 4], [2], [3], [3]])
fileio.write_profile_to_preflib_toi_file(
    profile, currdir + "/toi-files/new_example.toi"
)


# Read a directory of preflib files (using parameter appr_percent)
profiles = fileio.load_preflib_files_from_dir(currdir + "/toi-files/", appr_percent=0.7)
# Compute PAV for each profile
committeesize = 2
for profile in profiles:
    print("Computing a committee of size", committeesize, end=" ")
    print("with the Proportional Approval Voting (PAV) rule")
    print("given a", profile)
    print("Output:")
    committees = abcrules.compute_pav(profile, committeesize)
    print(str_candsets(committees))
    print("****************************************")


# Read a preflib file (using parameter setsize)
profile = fileio.read_preflib_file(currdir + "/toi-files/example.toi", setsize=1)
# Compute Phragmen's sequential rule for this profile
print("Computing a committee of size", committeesize, end=" ")
print("with Phragmen's sequential rule")
print("given a", profile)
print("Output:")
committees = abcrules.compute_seqphragmen(profile, committeesize)
print(str_candsets(committees))
print("****************************************")
