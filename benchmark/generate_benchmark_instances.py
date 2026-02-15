"""
Generate benchmark instances for ABC voting rules.

Creates instances using existing probability models from the abcvoting library,
with voter and candidate counts uniformly sampled from [3, 100].
"""

import os
import numpy as np
from abcvoting import generate, fileio

# Seed for reproducibility
SEED = 42

# Directory for generated instances
INSTANCES_DIR = os.path.join(os.path.dirname(__file__), "instances")

# Probability distributions to use for instance generation
PROB_DISTRIBUTIONS = [
    {"id": "IC", "p": 0.3},
    {"id": "IC", "p": 0.5},
    {"id": "IC fixed-size", "setsize": 2},
    {"id": "IC fixed-size", "setsize": 3},
    {"id": "IC fixed-size", "setsize": 4},
    {"id": "Truncated Mallows", "setsize": 3, "dispersion": 0.2},
    {"id": "Truncated Mallows", "setsize": 3, "dispersion": 0.5},
    {"id": "Truncated Mallows", "setsize": 4, "dispersion": 0.5},
    {"id": "Truncated Mallows", "setsize": 3, "dispersion": 0.8},
    {"id": "Urn fixed-size", "setsize": 2, "replace": 0.5},
    {"id": "Urn fixed-size", "setsize": 3, "replace": 0.5},
]


def generate_instances(num_instances=1000, output_dir=None, seed=None):
    """
    Generate benchmark instances using various probability distributions.

    Instances are sorted by num_cand, then committeesize, then num_voters.

    Parameters
    ----------
        num_instances : int
            Number of instances to generate.
        output_dir : str, optional
            Directory to save instances. Defaults to INSTANCES_DIR.
        seed : int, optional
            Random seed for reproducibility. Defaults to SEED.

    Returns
    -------
        list of str
            List of generated file paths.
    """
    if output_dir is None:
        output_dir = INSTANCES_DIR
    if seed is None:
        seed = SEED

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set random seed for reproducibility
    rng = np.random.default_rng(seed)

    # Generate all instance parameters upfront (before sorting) for determinism
    instance_params = []
    for _ in range(num_instances):
        num_voters = int(rng.integers(5, 101))
        num_cands = int(rng.integers(5, 101))
        prob_dist_idx = int(rng.integers(0, len(PROB_DISTRIBUTIONS)))
        committeesize = int(rng.integers(max(3, num_cands // 10), max(4, num_cands // 3)))
        # Generate a seed for this instance's profile generation
        instance_seed = int(rng.integers(0, 2**31))
        instance_params.append(
            (num_voters, num_cands, prob_dist_idx, committeesize, instance_seed)
        )

    # Sort by num_cand, then committeesize, then num_voters
    instance_params.sort(key=lambda x: (x[1], x[3], x[0]))

    generated_files = []

    for i, (num_voters, num_cands, prob_dist_idx, committeesize, instance_seed) in enumerate(
        instance_params
    ):
        prob_dist = dict(PROB_DISTRIBUTIONS[prob_dist_idx])  # copy to avoid modifying original

        # Use instance-specific seed for profile generation
        generate.rng = np.random.default_rng(instance_seed)

        # Generate the profile
        profile = generate.random_profile(num_voters, num_cands, prob_dist)

        # Create description
        dist_params = ", ".join(f"{k}={v}" for k, v in prob_dist.items() if k != "id")
        description = (
            f"Benchmark instance {i:04d}\n"
            f"Distribution: {prob_dist['id']}"
            + (f" ({dist_params})" if dist_params else "")
            + f"\nVoters: {num_voters}, Candidates: {num_cands}, Committee size: {committeesize}"
        )

        # Write to file
        filename = os.path.join(output_dir, f"instance{i:04d}.abc.yaml")
        fileio.write_abcvoting_instance_to_yaml_file(
            filename, profile, committeesize=committeesize, description=description
        )

        generated_files.append(filename)
        print(f"Generated {filename}")

    print(f"\nGenerated {len(generated_files)} instances in {output_dir}")
    return generated_files


def main():
    """Generate benchmark instances."""
    generate_instances()


if __name__ == "__main__":
    main()
