"""Example how to generate random approval profiles."""

from abcvoting import generate
import numpy as np


generate.rng = np.random.default_rng(24121838)  # seed for random number generator (optional)

# specify dimensions of generated profiles
num_cand = 50
num_voters = 100

# specify probability distributions used to generate profiles
prob_distributions = []
# Disjoint Resampling
for num_groups in [2, 3, 4, 5]:
    for phi in np.linspace(0, 1 / num_groups, 5, endpoint=False):
        for p in np.linspace(0.1, 0.9, 9):
            prob_distributions.append(
                {"id": "Disjoint Resampling", "num_groups": num_groups, "p": p, "phi": phi}
            )
# Noise
for phi in np.linspace(0, 1, 10, endpoint=False):
    for p in np.linspace(0.1, 0.9, 9):
        prob_distributions.append({"id": "Noise", "p": p, "phi": phi})
# Truncated Urn
for replace in np.linspace(0, 1, 10, endpoint=False):
    for p in np.linspace(0.1, 0.9, 9):
        setsize = int(p * num_cand)
        prob_distributions.append({"id": "Truncated Urn", "replace": replace, "setsize": setsize})
# Euclidean VCR
for radius in np.linspace(0.005, 0.25, 100):
    for point_distribution in [
        generate.PointProbabilityDistribution("1d_interval"),
        generate.PointProbabilityDistribution("2d_square"),
    ]:
        voter_radius = radius / 2
        candidate_radius = radius / 2
        prob_distributions.append(
            {
                "id": "Euclidean VCR",
                "voter_prob_distribution": point_distribution,
                "candidate_prob_distribution": point_distribution,
                "voter_radius": voter_radius,
                "candidate_radius": candidate_radius,
            }
        )


print(
    f"Generating {len(prob_distributions)} profiles "
    f"with {num_voters} voters and {num_cand} candidates.\n"
    "This may take a while...\n"
)

# generate profiles
profiles = []
for prob_distribution in prob_distributions:
    profile = generate.random_profile(num_voters, num_cand, prob_distribution)
    profiles.append((prob_distribution, profile))

# print statistics
print(f"Done. Generated {len(profiles)} profiles.")
prob_dist_ids = sorted({prob_dist["id"] for prob_dist, profile in profiles})
for prob_dist_id in prob_dist_ids:
    count = len([profile for prob_dist, profile in profiles if prob_dist["id"] == prob_dist_id])
    print(f" {count} profiles via {prob_dist_id}")
