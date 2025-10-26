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
        scores where scores[cand] = maximin support if cand added to committee
        Length: profile.num_cand

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

    Example Call Sequence
    ---------------------
    First call:  scorefct(profile, [])      → returns scores for adding any candidate to empty committee
    Second call: scorefct(profile, [2])     → returns scores for adding any candidate to {2}
    Third call:  scorefct(profile, [2, 5])  → returns scores for adding any candidate to {2, 5}
    ...until committee reaches desired size

    Examples
    --------
    Example 1: Simple 3 voters, 2 candidates
    >>> profile = Profile(num_cand=2)
    >>> profile.add_voters([
    ...     {0},      # Voter 1 approves A
    ...     {1},      # Voter 2 approves B
    ...     {0, 1}    # Voter 3 approves A and B
    ... ])
    >>> abcrules.compute("maximin-support", profile, committeesize=2, algorithm="nx-max-flow")
    [{0, 1}]

    Example 2: 5 voters, 4 candidates
    >>> profile = Profile(num_cand=4)
    >>> profile.add_voters([
    ...     {0, 1},   # Voter 1 approves A, B
    ...     {0, 2},   # Voter 2 approves A, C
    ...     {1, 2},   # Voter 3 approves B, C
    ...     {2},      # Voter 4 approves C
    ...     {1, 3}    # Voter 5 approves B, D
    ... ])
    >>> abcrules.compute("maximin-support", profile, committeesize=2, algorithm="nx-max-flow")
    [{1, 2}]

    Example 3: 7 voters, 5 candidates, committee size 3
    >>> profile = Profile(num_cand=5)
    >>> profile.add_voters([
    ...     {0, 1},      # Voter 1 approves A, B
    ...     {0, 2},      # Voter 2 approves A, C
    ...     {1, 2, 3},   # Voter 3 approves B, C, D
    ...     {1, 3},      # Voter 4 approves B, D
    ...     {2, 4},      # Voter 5 approves C, E
    ...     {3, 4},      # Voter 6 approves D, E
    ...     {2, 3}       # Voter 7 approves C, D
    ... ])
    >>> abcrules.compute("maximin-support", profile, committeesize=3, algorithm="nx-max-flow")
    [{1, 2, 3}]

    Example 4: 6 voters with duplicates, 4 candidates
    >>> profile = Profile(num_cand=4)
    >>> profile.add_voters([
    ...     {0, 1},   # Voter 1 approves A, B
    ...     {0, 1},   # Voter 2 approves A, B
    ...     {1, 2},   # Voter 3 approves B, C
    ...     {1, 2},   # Voter 4 approves B, C
    ...     {2, 3},   # Voter 5 approves C, D
    ...     {2, 3}    # Voter 6 approves C, D
    ... ])
    >>> abcrules.compute("maximin-support", profile, committeesize=2, algorithm="nx-max-flow")
    [{1, 2}]

    Example 5: 3 voters, 3 candidates with sparse approvals
    >>> profile = Profile(num_cand=3)
    >>> profile.add_voters([
    ...     {0},      # Voter 1 approves A
    ...     {0, 1},   # Voter 2 approves A, B
    ...     {2}       # Voter 3 approves C
    ... ])
    >>> abcrules.compute("maximin-support", profile, committeesize=2, algorithm="nx-max-flow")
    [{0, 1}]

    Example 6: 16 voters, 7 candidates with block structure
    >>> profile = Profile(num_cand=7)
    >>> profile.add_voters([
    ...     # Block 1 (prefers A/B/C)
    ...     {0, 1},      # Voter 1 approves A, B
    ...     {0, 2},      # Voter 2 approves A, C
    ...     {1, 2},      # Voter 3 approves B, C
    ...     {0, 1, 2},   # Voter 4 approves A, B, C
    ...     # Block 2 (prefers C/D/E)
    ...     {2, 3},      # Voter 5 approves C, D
    ...     {2, 4},      # Voter 6 approves C, E
    ...     {3, 4},      # Voter 7 approves D, E
    ...     {2, 3, 4},   # Voter 8 approves C, D, E
    ...     # Block 3 (prefers E/F/G)
    ...     {4, 5},      # Voter 9 approves E, F
    ...     {4, 6},      # Voter 10 approves E, G
    ...     {5, 6},      # Voter 11 approves F, G
    ...     {4, 5, 6},   # Voter 12 approves E, F, G
    ...     # Mixed voters
    ...     {1, 3},      # Voter 13 approves B, D
    ...     {2, 5},      # Voter 14 approves C, F
    ...     {3, 6},      # Voter 15 approves D, G
    ...     {0, 4, 6}    # Voter 16 approves A, E, G
    ... ])
    >>> abcrules.compute("maximin-support", profile, committeesize=4, algorithm="nx-max-flow")
    [{2, 3, 4, 6}]
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


def _compute_maximin_support_via_maxflow(profile, committee) -> float:
    """
    Compute the maximin support value for a given committee using max-flow.

    The maximin support is the maximum value of t such that each candidate
    in the committee can receive support of at least t from voters.

    Parameters
    ----------
    profile : Profile
        Voter preferences
    committee : set
        Set of candidate indices in the committee

    Returns
    -------
    float
        The maximin support value

    Algorithm
    ---------
    This is equivalent to finding the maximum value t such that there exists
    a feasible flow where:
    - Each voter i distributes their weight among approved candidates in committee
    - Each candidate j receives at least t units of flow
    - Flow conservation holds at all nodes

    This can be solved using binary search on t combined with max-flow
    feasibility checks, OR by using a minimum cut approach.

    Network Construction (for feasibility check with fixed t):
    - Source (s) → Voter nodes (v_i): capacity = voter[i].weight
    - Voter nodes (v_i) → Candidate nodes (c_j): capacity = ∞ (if j approved by i and j in committee)
    - Candidate nodes (c_j) → Sink (t): capacity = t (the value we're testing)

    Maximin support = maximum t such that max flow = sum of all voter weights
    """

    # TODO: Implement one of these approaches:

    # APPROACH 1: Binary search on support value
    # ------------------------------------------
    # Use binary search to find maximum t such that flow is feasible
    # For each t:
    #   - Build network with candidate→sink capacities = t
    #   - Check if max_flow equals total voter weight
    #   - If yes, t is feasible; try larger t
    #   - If no, t too large; try smaller t

    # APPROACH 2: Direct min-cut approach
    # ------------------------------------
    # Build network and find the minimum cut that separates source from sink
    # The value of this cut gives the maximin support directly

    return 0.0  # Empty implementation


# TODO: Remove the following test code when integrating into the main package
if __name__ == "__main__":
    import doctest

    print("Running doctests for maximin-support (nx-max-flow implementation)...")
    print(
        "Note: These tests will FAIL until _compute_maximin_support_via_maxflow is implemented.\n"
    )
    results = doctest.testmod(verbose=True)
    print(f"\n{results.attempted} tests attempted, {results.failed} failed")
    if results.failed == 0:
        print("All tests passed!")
    else:
        print(f"Expected failures since implementation returns 0.0")
