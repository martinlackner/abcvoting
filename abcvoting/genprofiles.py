"""
Random generation of approval profiles.
"""


import random
from math import fabs, sqrt
from abcvoting.preferences import Profile


def random_profile(num_voters, num_cand, prob_distribution, **kwargs):
    if prob_distribution == "IC":
        return random_IC_profile(num_cand, num_voters, **kwargs)
    elif prob_distribution == "Mallows":
        return random_mallows_profile(num_cand, num_voters, **kwargs)
    elif prob_distribution == "Urn":
        return random_urn_profile(num_cand, num_voters, **kwargs)
    else:
        raise ValueError(f"Probability model {prob_distribution} unknown.")


def random_urn_profile(num_cand, num_voters, setsize, replace):
    """
    Generate Polya Urn profile with fixed size approval sets.
    """
    currsize = 1.0
    approval_sets = []
    replacedsets = {}

    for _ in range(num_voters):
        r = random.random() * currsize
        if r < 1.0:
            # base case: sample uniformly at random
            randset = random.sample(range(num_cand), setsize)
            approval_sets.append(randset)
            key = tuple(set(randset))
            if key in replacedsets:
                replacedsets[key] += 1
            else:
                replacedsets[key] = 1
            currsize += replace
        else:
            # sample from one of the replaced ballots
            r = random.randint(0, sum(replacedsets.values()))
            for approval_set in replacedsets:
                count = replacedsets[approval_set]
                if r <= count:
                    approval_sets.append(list(approval_set))
                    break
                else:
                    r -= count
    profile = Profile(num_cand)
    profile.add_voters(approval_sets)
    return profile


def random_IC_profile(num_cand, num_voters, setsize):
    """
    Generates profile with random assignment of candidates to the fix size of setsize.
    """
    approval_sets = []
    for _ in range(num_voters):
        randset = random.sample(range(num_cand), setsize)
        approval_sets.append(randset)
    profile = Profile(num_cand)
    profile.add_voters(approval_sets)
    return profile


def random_2d_points_profile(
    num_cand, num_voters, candpointmode, voterpointmode, sigma, approval_threshold
):
    """
    Generate profiles from randomly generated 2d points.
    """
    voters = list(range(num_voters))
    cands = list(range(num_cand))

    voter_points = __generate_2d_points(voters, voterpointmode, sigma)
    cand_points = __generate_2d_points(cands, candpointmode, sigma)

    approval_sets = __get_profile_from_points(
        voters, cands, voter_points, cand_points, approval_threshold
    )
    profile = Profile(num_cand)
    profile.add_voters(approval_sets)
    return profile


def random_mallows_profile(num_cand, num_voters, setsize, dispersion):
    """
    Generates a Mallows Profile.

    After the definition for
    repeated insertion  mode (RIM) in
    https://icml.cc/2011/papers/135_icmlpaper.pdf
    """
    if not (0 < dispersion <= 1):
        raise Exception("Invalid dispersion, needs to be in (0, 1].")
    reference_ranking = list(range(num_cand))
    random.shuffle(reference_ranking)
    insert_dist = __compute_mallows_insert_distributions(num_cand, dispersion)
    approval_sets = []
    for _ in range(num_voters):
        vote = []
        for i, distribution in enumerate(insert_dist):
            pos = __select_pos(distribution)
            vote.insert(pos, reference_ranking[i])

        approval_sets.append(vote[:setsize])
    profile = Profile(num_cand)
    profile.add_voters(approval_sets)
    return profile


def __compute_mallows_insert_distributions(num_cand, dispersion):
    """
    Compute the insertion probability vectors for the dispersion and a given number of candidates.
    """
    distributions = []
    denominator = 0
    for i in range(num_cand):
        # compute the denominator = dispersion^0 + dispersion^1
        # + ... dispersion^(i-1)
        denominator += pow(dispersion, i)
        dist = []
        for j in range(i + 1):  # 0..i
            dist.append(pow(dispersion, i - j) / denominator)
        distributions.append(dist)
    return distributions


def __select_pos(distribution):
    """
    Returns a randomly selected value with the help of the distribution.
    """
    if round(sum(distribution), 10) != 1.0:
        raise Exception("Invalid Distribution", distribution, "sum:", sum(distribution))
    r = round(random.random(), 10)  # or random.uniform(0, 1)
    pos = -1
    s = 0
    for prob in distribution:
        pos += 1
        s += prob
        if s >= r:
            return pos

    return pos  # in case of rounding errors


def __generate_2d_points(agents, mode, sigma):
    """
    Generate a list of 2d coordinates subject to various distributions.
    """
    points = {}

    # normal distribution, 1/3 of agents centered on (-0.5,-0.5),
    #                      2/3 of agents on (0.5,0.5)
    if mode == "twogroups":
        for i in range(int(len(agents) // 3)):
            points[agents[i]] = (random.gauss(-0.5, sigma), random.gauss(-0.5, sigma))
        for i in range(int(len(agents) // 3), len(agents)):
            points[agents[i]] = (random.gauss(0.5, sigma), random.gauss(0.5, sigma))
    # normal distribution
    elif mode == "normal":
        for i in range(len(agents)):
            points[agents[i]] = (random.gauss(0.0, sigma), random.gauss(0.0, sigma))
    elif mode == "uniform_square":
        for a in agents:
            points[a] = (random.uniform(-1, 1), random.uniform(-1, 1))
    else:
        raise ValueError("mode", mode, "not known")
    return points


def __euclidean(p1, p2):
    return sqrt(fabs(p1[0] - p2[0]) ** 2 + fabs(p1[1] - p2[1]) ** 2)


def __get_profile_from_points(voters, cands, voter_points, cand_points, approval_threshold):
    """
    Generates a list of approval sets from 2d points according to approval_threshold.
    """
    profile = {}
    for v in voters:
        distances = {cand: __euclidean(voter_points[v], cand_points[cand]) for cand in cands}
        mindist = min(distances.values())
        profile[v] = [cand for cand in cands if distances[cand] <= mindist * approval_threshold]
    return list(profile.values())
