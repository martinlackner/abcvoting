

import random
from time import time

random.seed(time())


# Author: Martin Lackner
def random_urn_profile(num_cand, num_voters, setsize, replace):
    """Generate Polya Urn profile with fixed size approval sets."""
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
                if r <= count:
                    apprsets.append(list(apprset))
                    break
                else:
                    r -= count

    return apprsets


def random_urn_party_list_profile(num_cand, num_voters, num_parties,
                                  replace, uniform=False):
    """Generate Polya Urn profile from a number of parties.
    If uniform each party gets the same amount of candidates."""
    currsize = 1.
    apprsets = []
    replacedsets = {}
    parties = list(range(num_parties))
    party_cands = distribute_candidates_to_parties(
        num_cand, parties, uniform=uniform)
    for _ in range(num_voters):
        r = random.random() * currsize
        if r < 1.:
            # base case: sample uniformly at random
            party = random.choice(parties)
            randpartyset = list(party_cands[party])
            apprsets.append(randpartyset)
            if party in replacedsets:
                replacedsets[party] += 1
            else:
                replacedsets[party] = 1
            currsize += replace
        else:
            # sample from one of the parties
            r = random.randint(0, sum(replacedsets.values()))
            for party in replacedsets:
                count = replacedsets[party]
                if r <= count:
                    apprsets.append(list(party_cands[party]))
                    break
                else:
                    r -= count

    return apprsets


def random_IC_profile(num_cand, num_voters, setsize):
    """Generates profile with random assignment of candidates to
    the fix size of setsize."""
    apprsets = []
    for _ in range(num_voters):
        randset = random.sample(range(num_cand), setsize)
        apprsets.append(randset)

    return apprsets


def random_IC_party_list_profile(num_cand, num_voters, num_parties,
                                 uniform=False):
    """Generates profile with random assignment of parties.
    A party is a list of candidates.
    If uniform the number of candidates per party is the same,
    else at least 1."""
    parties = list(range(num_parties))
    party_cands = distribute_candidates_to_parties(
        num_cand, parties, uniform=uniform)
    apprsets = []
    for _ in range(num_voters):
        apprsets.append(party_cands[random.choice(parties)])
    return apprsets


def random_mallows_profile(num_cand, num_voters, setsize, dispersion):
    """Genearates a Mallows Profile after the definition for
    repeated insertion  mode (RIM) in
    https://icml.cc/2011/papers/135_icmlpaper.pdf"""
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


def compute_mallows_insert_distributions(num_cand, dispersion):
    """Computes the insertion probability vectors for
    the dispersion and a given number of candidates"""
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


def select_pos(distribution):
    """Returns a randomly selected value with the help of the
    distribution"""
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


def distribute_candidates_to_parties(num_cand, parties, uniform):
    """Distributes the candidates to the parties.
    Either uniformly distributed or randomly distributed with
    at least one candidate per party."""
    if num_cand < len(parties):
        raise Exception("Not enough candidates to split them between"
                        + "the parties.")
    if uniform:
        if num_cand % len(parties) != 0:
            raise Exception("To uniformly distribute candidates "
                            + "between parties the number of candidates"
                            + "needs to be divisible by the number of"
                            + "parties.")
        party_cands = {}
        party_size = int(num_cand / len(parties))
        for i, party in enumerate(parties):
            party_cands[party] = list(range(party_size*i,
                                            party_size*(i+1)))
        return party_cands
    else:  # not uniform
        num_parties = len(parties)
        party_cands = {}
        num_random_cands = num_cand - num_parties
        for i, party in enumerate(parties):
            party_cands[party] = [num_random_cands+i]
        for cand in range(num_random_cands):
            party = random.choice(parties)
            party_cands[party].append(cand)
        return party_cands
