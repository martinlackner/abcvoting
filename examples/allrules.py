"""
Compute all major implemented ABC rules for a profile.
"""

from abcvoting.preferences import Profile
from abcvoting import abcrules
from abcvoting.output import output, INFO


output.set_verbosity(INFO)


num_cand = 6
profile = Profile(num_cand)
profile.add_voters([{0, 4, 5}, {0}, {1, 4, 5}, {1}, {2, 4, 5}, {2}, {3, 4, 5}, {3}])
committeesize = 4
print(f"Input: {profile}\n")

# Compute all major implemented ABC rules
for rule_id in abcrules.MAIN_RULE_IDS:
    try:
        abcrules.compute(rule_id, profile, committeesize)
    except abcrules.NoAvailableAlgorithm:
        print(
            f"Skipping the ABC rule {abcrules.Rule(rule_id).shortname}, "
            "since it requires a solver that is not installed on this machine."
        )
