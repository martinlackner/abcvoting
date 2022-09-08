"""
Compare axiomatic properties of two committees.
"""

from abcvoting.preferences import Profile
from abcvoting import abcrules, properties
from abcvoting.output import output, INFO


output.set_verbosity(INFO)


num_cand = 16
profile = Profile(num_cand)
profile.add_voters(
    [{1, 2, 3, 4}, {1, 2, 3, 5}, {1, 2, 3, 6}, {7, 8, 9}, {10, 11, 12}, {13, 14, 15}]
)
committeesize = 12
print(f"Input: {profile}\n")

committee_pav = abcrules.compute_pav(profile, committeesize, resolute=True)[0]
committee_phrag = abcrules.compute_seqphragmen(profile, committeesize, resolute=True)[0]

print("Results for PAV:")
prop_pav = properties.full_analysis(profile, committee_pav)
print()

print("Results for Sequential Phragmén:")
prop_phrag = properties.full_analysis(profile, committee_phrag)
print()

print("Sequential Phragmén satisfies priceability, PAV does not.")
assert prop_phrag["priceability"]
assert not prop_pav["priceability"]

print("Sequential Phragmén satisfies the core property, PAV does not.")
assert prop_phrag["core"]
assert not prop_pav["core"]
