"""
Very simple example (compute PAV)
"""

from abcvoting.preferences import Profile
from abcvoting import abcrules
from abcvoting.output import output, INFO

output.set_verbosity(INFO)

profile = Profile(num_cand=5)
profile.add_voters([{0, 1, 2}, {0, 1}, {0, 1}, {1, 2}, {3, 4}, {3, 4}])
committeesize = 3
print(
    f"Computing winning committees of size {committeesize}\n"
    f"with the Proportional Approval Voting (PAV) rule\n"
    f"given the following {profile}"
)
committees = abcrules.compute_pav(profile, committeesize)
