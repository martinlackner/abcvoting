"""Proposition A.2
from the survey: "Approval-Based Multi-Winner Voting:
Axioms, Algorithms, and Applications"
by Martin Lackner and Piotr Skowron
"""

from __future__ import print_function
import sys
sys.path.insert(0, '..')
from abcvoting import abcrules
from abcvoting.preferences import Profile
from abcvoting import misc


print(misc.header("Proposition A.2", "*"))

num_cand = 3
a, b, c = (0, 1, 2)
apprsets = [[a]] * 2 + [[a, c]] * 3 + [[b, c]] * 3 + [[b]] * 2
names = "abcde"
profile = Profile(num_cand, names=names)
profile.add_preferences(apprsets)

print(misc.header("1st profile:"))
print(profile.str_compact())


print("winning committees for k=1 and k=2:")
for rule_id in ["pav", "cc", "monroe", "optphrag", "mav"]:
    comm1 = abcrules.compute(rule_id, profile, 1, resolute=True)[0]
    comm2 = abcrules.compute(rule_id, profile, 2, resolute=True)[0]
    print(" " + abcrules.rules[rule_id].shortname + ": "
          + misc.str_candset(comm1, names)
          + " vs " + misc.str_candset(comm2, names))


num_cand = 4
a, b, c, d = 0, 1, 2, 3
apprsets = ([[a]] * 6 + [[a, c]] * 4 + [[a, b, c]] * 2 + [[a]] * 2
            + [[a, d]] * 1 + [[b, d]] * 3)
names = "abcde"
profile = Profile(num_cand, names=names)
profile.add_preferences(apprsets)

print()
print(misc.header("2nd profile:"))
print(profile.str_compact())


print("winning committees for k=2 and k=3:")
for rule_id in ["greedy-monroe"]:
    comm1 = abcrules.compute(rule_id, profile, 2, resolute=True)[0]
    comm2 = abcrules.compute(rule_id, profile, 3, resolute=True)[0]
    print(" " + abcrules.rules[rule_id].shortname + ": "
          + misc.str_candset(comm1, names)
          + " vs " + misc.str_candset(comm2, names))


num_cand = 4
a, b, c, d = 0, 1, 2, 3
apprsets = [[a, c]] * 4 + [[a, d]] * 2 + [[b, d]] * 3 + [[b]]
names = "abcde"
profile = Profile(num_cand, names=names)
profile.add_preferences(apprsets)

print()
print(misc.header("3rd profile:"))
print(profile.str_compact())


print("winning committees for k=2 and k=3:")
comm1 = abcrules.compute("rule-x", profile, 2, resolute=True)[0]
comm2 = abcrules.compute("rule-x", profile, 3, resolute=True)[0]
print(" " + abcrules.rules[rule_id].shortname + ": "
      + misc.str_candset(comm1, names)
      + " vs " + misc.str_candset(comm2, names))

print("\n\nDetailed calculations:")
abcrules.compute("rule-x", profile, 2, resolute=True, verbose=2)
abcrules.compute("rule-x", profile, 3, resolute=True, verbose=2)

num_cand = 5
a, b, c, d, e = 0, 1, 2, 3, 4
apprsets = [[a, c]] * 1 + [[a, b]] * 1 + [[d, e]]
names = "abcde"
profile = Profile(num_cand, names=names)
profile.add_preferences(apprsets)

print()
print(misc.header("3rd profile:"))
print(profile.str_compact())


print("winning committees for k=2 and k=3:")
comm1 = abcrules.compute("mav", profile, 1, resolute=False, verbose=2)
print(comm1)
