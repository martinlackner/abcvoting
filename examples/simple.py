"""
Very simple example (compute PAV)
"""

from abcvoting.output import INFO
from abcvoting.output import output
from abcvoting.preferences import Profile
from abcvoting import abcrules


output.set_verbosity(INFO)

num_cand = 5
profile = Profile(num_cand)
profile.add_voters([{0, 1, 2}, {0, 1}, {0, 1}, {1, 2}, {3, 4}, {3, 4}])
committeesize = 3
print(
    f"Computing winning committees of size {committeesize}\n"
    f"with the Proportional Approval Voting (PAV) rule\n"
    f"given the following {profile}"
)
committees = abcrules.compute_pav(profile, committeesize)
