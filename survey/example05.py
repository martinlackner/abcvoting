"""Example 5 (PAV, seq-PAV, revseq-PAV)
from the survey: "Approval-Based Multi-Winner Voting:
Axioms, Algorithms, and Applications"
by Martin Lackner and Piotr Skowron
"""

from __future__ import print_function
import sys
sys.path.insert(0, '..')
from abcvoting import abcrules
from abcvoting.preferences import Profile, DichotomousPreferences
from abcvoting import misc


print(misc.header("Example 5", "*"))

# Approval profile
num_cand = 4
a, b, c, d = range(4)  # a = 0, b = 1, c = 2, ...
names = "abcd"

apprsets = [[a, b]] * 3 + [[a, d]] * 6 + [[b]] * 4 + [[c]] * 5 + [[c, d]] * 5
profile = Profile(num_cand, names=names)
profile.add_preferences(apprsets)

print(misc.header("Input:"))
print(profile.str_compact())

committees_pav = abcrules.compute_pav(profile, 2, verbose=2)

committees_seqpav = abcrules.compute_seqpav(profile, 2, verbose=2)

committees_revseqpav = abcrules.compute_revseqpav(profile, 2, verbose=2)

# verify correctness
assert committees_pav == [[a, c]]
assert committees_seqpav == [[c, d]]
assert committees_revseqpav == [[c, d]]


print("\n")
print(misc.header(
    "Example from Janson's survey (Example 13.3) / Thiele:", "*"))

# Approval profile
num_cand = 4
a, b, c, d = range(4)  # a = 0, b = 1, c = 2, ...
names = "abcd"

profile = Profile(num_cand, names=names)
profile.add_preference(DichotomousPreferences([a, c, d], 960))
profile.add_preference(DichotomousPreferences([b, c, d], 3000))
profile.add_preference(DichotomousPreferences([b, c], 520))
profile.add_preference(DichotomousPreferences([a, b], 1620))
profile.add_preference(DichotomousPreferences([a, d], 1081))
profile.add_preference(DichotomousPreferences([a, c], 1240))
profile.add_preference(DichotomousPreferences([b, d], 360))
profile.add_preference(DichotomousPreferences([d], 360))
profile.add_preference(DichotomousPreferences([c], 120))
profile.add_preference(DichotomousPreferences([b], 60))

print(misc.header("Input:"))
print(profile.str_compact())

committees_pav = abcrules.compute_pav(profile, 2, verbose=2)

committees_seqpav = abcrules.compute_seqpav(profile, 2, verbose=2)

committees_revseqpav = abcrules.compute_revseqpav(profile, 2, verbose=2)


# verify correctness
assert committees_pav == [[a, b]]
assert committees_seqpav == [[a, c]]
assert committees_revseqpav == [[b, d]]
