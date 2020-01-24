

import random
from time import time

random.seed(time())


# generate Polya Urn profile with fixed size approval sets
# Author: Martin Lackner
def random_urn_profile(num_cand, num_voters, setsize, replace):
    currsize = 1.
    apprsets = []
    replacedsets = {}

    for _ in range(num_voters):
        r = random.random() * currsize
        if r < 1.:
            # base case: sample uniformly at random
            randset = random.sample(range(num_cand), setsize)
            apprsets.append(randset)
            key = tuple(set(randset))
            if key in replacedsets:
                replacedsets[key] += 1
            else:
                replacedsets[key] = 1
            currsize += replace
        else:
            # sample from one of the replaced ballots
            r = random.randint(0, sum(replacedsets.values()))
            for apprset in replacedsets:
                count = replacedsets[apprset]
                if r < count:
                    apprsets.append(list(apprset))
                    break
                else:
                    r -= count

    return apprsets


def random_IC_profile(num_cand, num_voters, setsize):
    apprsets = []
    for _ in range(num_voters):
        randset = random.sample(range(num_cand), setsize)
        apprsets.append(randset)

    return apprsets


# After the definition for repeated  insertion  mode (RIM) in
# https://icml.cc/2011/papers/135_icmlpaper.pdf
def random_mallows_profile(num_cand, num_voters, setsize, dispersion):
    if not (0 <= dispersion <= 1):
        raise Exception("Invalid dispersion, needs to be in (0, 1].")
    reference_ranking = list(range(num_cand))
    random.shuffle(reference_ranking)
    insert_dist = compute_mallows_insert_distributions(
        num_cand, dispersion)
    rankings = []
    for v in range(num_voters):
        vote = []
        for i, distribution in enumerate(insert_dist):
            pos = select_pos(distribution)
            vote.insert(pos, reference_ranking[i])

        rankings.append(sorted(vote[:setsize]))

    return rankings


# For the dispersion and a given number of candidates, compute the
# insertion probability vectors.
def compute_mallows_insert_distributions(num_cand, dispersion):
    distributions = []
    denominator = 0
    for i in range(num_cand):
        # compute the denominator = dispersion^0 + dispersion^1
        # + ... dispersion^(i-1)
        denominator += pow(dispersion, i)
        dist = []
        for j in range(i+1):  # 0..i
            dist.append(pow(dispersion, i - j) / denominator)
        distributions.append(dist)

    return distributions


# Return a randomly selected value with the help of the distribution
def select_pos(distribution):
    if round(sum(distribution), 10) != 1.0:
        raise Exception("Invalid Distribution", distribution,
                        "sum:", sum(distribution))
    r = round(random.random(), 10)  # or random.uniform(0, 1)
    pos = -1
    s = 0
    for prob in distribution:
        pos += 1
        s += prob
        if s >= r:
            return pos

    return pos    # in case of rounding errors

