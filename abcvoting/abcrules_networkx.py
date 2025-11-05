"""
Max-Flow Implementation for Maximin Support Method

Based on: "Efficient, traceable, and numerical error-free"
https://arxiv.org/abs/2309.15104 (October 17, 2023)
By Luis Sánchez-Fernández
From Universidad Carlos III de Madrid, Spain

Programmer: Amit Gini (S.C Student at Ariel University, Israel)
Date: 10-2025

To run these tests, execute:
    python -m doctest abcrules_networkx.py -v
"""

import networkx as nx
from abcvoting.preferences import Profile
from abcvoting import abcrules


def _nx_maximin_support_scorefct(profile, base_committee) -> list:
    """
    Compute maximin support scores using max-flow algorithm.

    This function is called iteratively during committee construction.
    For each candidate not yet in the committee, it computes what the
    maximin support value would be if that candidate were added.

    Parameters
    ----------
    profile : Profile
        Contains voter preferences and weights
        - profile.num_cand: number of candidates
        - profile.candidates: list of candidate indices [0, 1, 2, ...]
        - iterate as: for voter in profile:
            - voter.approved: set of approved candidates {0, 2, 5, ...}
            - voter.weight: weight of this voter (usually 1)

    base_committee : list or set
        Current partial committee (can be empty [])
        Example: [0, 3] means candidates 0 and 3 are already in committee

    Returns
    -------
    list
        scores where scores[cand] = maximin support of cand, list length is profile.num_cand

    Algorithm Overview
    ------------------
    For each candidate C not in base_committee:
        1. Form test_committee = base_committee ∪ {C}
        2. Build max-flow network:
           - Source → Voters (capacity = voter weight)
           - Voters → Approved candidates in test_committee (capacity = ∞ or voter weight)
           - Candidates → Sink (capacity = variable, to be maximized)
        3. Find maximum min-cut (maximin support value)
        4. Store in scores[C]
    """

    # Initialize scores, set to 0 for candidates in base_committee
    scores = [0] * profile.num_cand

    # Get candidates not yet in base_committee (chosen candidates)
    remaining_candidates = [cand for cand in profile.candidates if cand not in base_committee]

    # For each remaining candidate, compute score
    for added_cand in remaining_candidates:
        # Form test committee with added candidate
        committee = set(base_committee) | {added_cand}

        # Compute maximin support for this committee using max-flow
        support_value = _compute_maximin_support_via_maxflow(profile, committee)

        scores[added_cand] = support_value

    return scores
