"""
Compute all implemented ABC rules for a profile
"""

from __future__ import print_function
import sys

sys.path.insert(0, "..")
from abcvoting.preferences import Profile
from abcvoting import abcrules
from abcvoting.misc import str_sets_of_candidates


# Compute all implemented ABC rules with the default algorithms
# and resolute=True

num_cand = 6
profile = Profile(num_cand)
profile.add_voters([{0, 4, 5}, {0}, {1, 4, 5}, {1}, {2, 4, 5}, {2}, {3, 4, 5}, {3}])
committeesize = 4

"""Prints the winning committees for all implemented rules"""
for rule in abcrules.rules.values():
    print(rule.longname + ":")
    committees = rule.compute(profile, committeesize, resolute=True)
    print(str_sets_of_candidates(committees))
