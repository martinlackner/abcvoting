import random
from abcvoting import genprofiles
from abcvoting.misc import check_enough_approved_candidates


random.seed(31415)

committeesize = 3
num_cand = 10


def output(prof, gen_profile_name):
    print("Randomly generated profile via " + gen_profile_name + ":")
    print(str(prof))
    print("****************************************")


"""
For some methods, it might happen that fewer than
committeesize many candidates are approved (in total by all voters).
We thus recommended to verify this before computing the rule.
"""

while True:
    profile = genprofiles.random_urn_profile(num_cand, 5, 2, 0.4)
    try:
        check_enough_approved_candidates(profile, committeesize)
        break
    except ValueError:
        pass
output(profile, "random_urn")

while True:
    profile = genprofiles.random_urn_party_list_profile(num_cand, 3, 2, 0.4, uniform=False)
    try:
        check_enough_approved_candidates(profile, committeesize)
        break
    except ValueError:
        pass
output(profile, "random_urn_party_list")

profile = genprofiles.random_IC_profile(num_cand, 5, 4)
output(profile, "random_IC")

profile = genprofiles.random_IC_party_list_profile(num_cand, 5, 2, uniform=True)
output(profile, "random_IC_party_list")

profile = genprofiles.random_mallows_profile(num_cand, 4, 4, 0.7)
output(profile, "random_mallows")

while True:
    profile = genprofiles.random_2d_points_profile(
        num_cand, 4, "twogroups", "uniform_square", 0.5, 1.9
    )
    try:
        check_enough_approved_candidates(profile, committeesize)
        break
    except ValueError:
        pass
output(profile, "random_2d_points")
