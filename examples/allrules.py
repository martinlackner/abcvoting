"""
Compute all implemented ABC rules for a profile
"""

from abcvoting.preferences import Profile
from abcvoting import abcrules
from abcvoting.misc import str_sets_of_candidates


# Compute all implemented ABC rules with the default algorithms
# and resolute=True

num_cand = 6
profile = Profile(num_cand)
profile.add_voters([{0, 4, 5}, {0}, {1, 4, 5}, {1}, {2, 4, 5}, {2}, {3, 4, 5}, {3}])
committeesize = 4

"""Prints the winning committees for the main ABC rules"""
for rule_id in abcrules.MAIN_RULE_IDS:
    print(abcrules.get_rule(rule_id).longname + ":")
    committees = abcrules.compute(
        rule_id,
        profile,
        committeesize,
        algorithm="fastest",  # use the fastest algorithm that is available (usually Gurobi)
        resolute=True,  # only compute one winning committee
    )
    print(str_sets_of_candidates(committees))
