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

###

num_cand = 3
a, b, c = (0, 1, 2)
apprsets = [[a]] * 2 + [[a, c]] * 3 + [[b, c]] * 3 + [[b]] * 2
names = "abcde"
profile = Profile(num_cand, names=names)
profile.add_voters(apprsets)

print(misc.header("1st profile:"))
print(profile.str_compact())


print("winning committees for k=1 and k=2:")
for rule_id in ["pav", "cc", "monroe", "optphrag", "mav"]:
    comm1 = abcrules.compute(rule_id, profile, 1, resolute=True)[0]
    comm2 = abcrules.compute(rule_id, profile, 2, resolute=True)[0]
    print(" " + abcrules.rules[rule_id].shortname + ": "
          + misc.str_candset(comm1, names)
          + " vs " + misc.str_candset(comm2, names))
    assert not all(cand in comm1 for cand in comm2)

###

num_cand = 4
a, b, c, d = 0, 1, 2, 3
apprsets = ([[a]] * 6 + [[a, c]] * 4 + [[a, b, c]] * 2 + [[a]] * 2
            + [[a, d]] * 1 + [[b, d]] * 3)
names = "abcde"
profile = Profile(num_cand, names=names)
profile.add_voters(apprsets)

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
    assert not all(cand in comm1 for cand in comm2)

###

num_cand = 6
a, b, c, d, e, f = range(num_cand)
apprsets = [[a, d, e], [a, c], [b, e], [c, d, f]]
names = "abcdef"
profile = Profile(num_cand, names=names)
profile.add_voters(apprsets)

print()
print(misc.header("3rd profile:"))
print(profile.str_compact())

print("winning committees for k=3 and k=4:")
comm1 = abcrules.compute("rule-x", profile, 3, resolute=True)[0]
comm2 = abcrules.compute("rule-x", profile, 4, resolute=True)[0]
print(" " + abcrules.rules["rule-x"].shortname + ": "
      + misc.str_candset(comm1, names)
      + " vs " + misc.str_candset(comm2, names))
assert not all(cand in comm1 for cand in comm2)

print("\n\nDetailed calculations:")
abcrules.compute("rule-x", profile, 3, resolute=True, verbose=2)
abcrules.compute("rule-x", profile, 4, resolute=True, verbose=2)
