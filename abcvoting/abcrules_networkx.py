"""
Max-Flow Implementation for Maximin Support Method

Based on: "Efficient, traceable, and numerical error-free"
https://arxiv.org/abs/2309.15104 (October 17, 2023)
By Luis Sánchez-Fernández
From Universidad Carlos III de Madrid, Spain

Programmer: Amit Gini (S.C Student at Ariel University, Israel)
Date: 10-2025

To run these tests, execute:
    python -m doctest abcvoting/abcrules_networkx.py -v
"""

import networkx as nx


def _nx_maximin_support_scorefct(profile, base_committee):
    """
        Compute maximin support scores using max-flow algorithm.

        This function is called iteratively during committee construction.
        For each candidate not yet in the committee, using max-flow
        problems iteratively computes what the maximin support score
        be if that candidate were added, and return the support score list.

        Parameters
        ----------
        profile : Profile
            Contains voter preferences and weights
            - profile.num_cand: number of candidates
            - profile.candidates: list of candidate indices [0, 1, 2, ...]
            - iterate as: for voter in profile:
                - voter.approved: set of approved candidates {0, 2, 5, ...}
                - voter.weight: weight of this voter (usually 1)

        base_committee : list
            Current partial committee (can be empty [])
            Example: [0, 3] means candidates 0 and 3 are already in committee

        Returns
        -------
        list
            scores where scores[candidate_index] = maximin support of candidate.
            represented by candidate_index

        Algorithm Overview
        ------------------
        Notions:
        - N = [index∶[candidates]]
            - Key: Voters index, Value: list of candidates that key voter approved.
        - NS = {i∈N┤|N∩S≠∅}
            - Set of voters who approve at least one candidate
                in S.
        - S = base_committee ∪ {C}
            - Tested committee with candidate C added.
        - |NS_total_weight| = sum(voter.weight for voter in NS)
            - Total size of NS, with weighted voters.
        - Flow Value = maximum flow from source to sink returned from max-flow
          algorithm in networkx.
        - Target Flow = |NS_total_weight| * |S_size|
            - total weight of voters in NS * size of committee, needed to
              achieve maximin-support score for candidate C.

        1. For each candidate C not in base_committee:
            1.1. Form test_committee = base_committee ∪ {C}
            1.2. Build new profile of test_committee:
                - NS = {i∈N┤|N∩S≠∅} : voters who approve at least one
                  candidate in S.
            1.3. Build max-flow network with integer arithmetic:
               - Source → Voters: Edge capacity = voter.weight × |S|
                 (might have different weights for voters)
               - Voters → Candidates: Edge capacity = voter.weight × |S|
               - Candidates → Sink: Edge capacity = |NS_total_weight|
                 (Sum of weights of voters in NS)
            1.4. Compute maximum flow using max-flow algorithm in networkx,
                and return the Flow Value
            1.5. If Target Flow == Flow Value
                - Add maximin-support score of candidate C to scores list
                  (which is |NS_total_weight| / |S_size|)
            1.6. Otherwise:
                - Iteratively remove unsupported candidates if target flow is
                  not achieved (if none remain score of C is 0)
        7. Return the maximin-support score list

        Examples
        --------
        >>> from abcvoting.preferences import Profile, Voter

        ----------------------------------------------------------------
        Example 1: 3 candidates, 3 voters, k=2
        Voters: A:{1,2}, B:{2,3}, C:{1,2}
        Expected results based on hand calculations:

        # Setup profile
        >>> profile = Profile(4)
        >>> profile.add_voter(Voter([1, 2]))  # A: approves candidates 1,2
        >>> profile.add_voter(Voter([2, 3]))  # B: approves candidates 2,3
        >>> profile.add_voter(Voter([1, 2]))  # C: approves candidates 1,2

        # Iteration 1: base_committee = [], test each single candidate
        >>> scores = _nx_maximin_support_scorefct(profile, [])
        >>> scores[0]   # committee: [0]
        0.0
        >>> scores[1]   # committee: [1]
        2.0
        >>> scores[2]   # committee: [2]
        3.0
        >>> scores[3]   # committee: [3]
        1.0

        # Iteration 2: base_committee = [2], test adding each remaining candidate
        >>> scores = _nx_maximin_support_scorefct(profile, [2])
        >>> scores[0]   # committee: [2, 0]
        0.0
        >>> scores[1]   # committee: [2, 1]
        1.5
        >>> scores[2]   # committee: [2, 2]
        0
        >>> scores[3]   # committee: [2, 3]
        1.0

    ----------------------------------------------------------------
        Example 2: 5 candidates, 4 voters, k=3
        Voters: A:{1,3}, B:{2,0}, C:{0,3}, D:{4,3,1}

        >>> # Setup profile
        >>> profile = Profile(5)
        >>> profile.add_voter(Voter([1, 3]))        # A: 2 approvals
        >>> profile.add_voter(Voter([2, 0]))        # B: 2 approvals
        >>> profile.add_voter(Voter([0, 3]))        # C: 2 approvals
        >>> profile.add_voter(Voter([4, 3, 1]))     # D: 3 approval

        # Iteration 1: base_committee = []
        >>> scores = _nx_maximin_support_scorefct(profile, [])
        >>> scores[0]   # committee: [0]
        2.0
        >>> scores[1]   # committee: [1]
        2.0
        >>> scores[2]   # committee: [2]
        1.0
        >>> scores[3]   # committee: [3]
        3.0
        >>> scores[4]   # committee: [4]
        1.0

        # Iteration 2: base_committee = [3]
        >>> scores = _nx_maximin_support_scorefct(profile, [3])
        >>> scores[0]   # committee: [3, 0]
        2.0
        >>> scores[1]   # committee: [3, 1]
        1.5
        >>> scores[2]   # committee: [3, 2]
        1.0
        >>> scores[3]   # Already in committee
        0
        >>> scores[4]   # committee: [3, 4]
        1.0

        # Iteration 3: base_committee = [3, 0]
        >>> scores = _nx_maximin_support_scorefct(profile, [3, 0])
        >>> scores[0]               # Already in committee
        0
        >>> round(scores[1], 3)     # committee: [3,0,1]
        1.333
        >>> scores[2]               # committee: [3,0,2]
        1.0
        >>> scores[3]               # Already in committee
        0
        >>> scores[4]               # committee: [3,0,4]
        1.0

    ----------------------------------------------------------------
        Example 3: Weighted voters
        Testing that voter weights are properly considered

        # Setup profile with weights
        >>> profile = Profile(3)
        >>> profile.add_voter(Voter([1, 2], weight=1))  # Weight 2
        >>> profile.add_voter(Voter([2, 0], weight=2))  # Weight 1
        >>> profile.add_voter(Voter([1, 2], weight=1))  # Weight 1

        # Iteration 1: base_committee = []
        >>> scores = _nx_maximin_support_scorefct(profile, [])
        >>> scores[0]  # committee: [0]
        2.0
        >>> scores[1]  # committee: [1]
        2.0
        >>> scores[2]  # committee: [2]
        4.0

        # Iteration 2: base_committee = [2]
        >>> scores = _nx_maximin_support_scorefct(profile, [2])
        >>> scores[0]   # committee: [2, 0]
        2.0
        >>> scores[1]   # committee: [2, 1]
        2.0
        >>> scores[2]  # Already in committee
        0

        Notes
        -----
        - Candidates already in base_committee have score = 0
        - The algorithm uses integer arithmetic internally (capacities × |S|)
        - When target flow is achieved, all candidates receive equal support
        - Voter weights are properly considered in all capacity calculations
        - The iterative refinement removes unsupported candidates until target flow is achieved.
    """
    # Initialize scores, set to 0 for candidates in base_committee
    scores = [0] * profile.num_cand

    # Get candidates not yet in base_committee
    remaining_candidates = [cand for cand in profile.candidates if cand not in base_committee]

    # For each remaining candidate, compute maximin support score
    for added_cand in remaining_candidates:
        # Form test committee of the existing committee and
        # the remaining candidates (base_committee ∪ {C})
        S = set(base_committee) | {added_cand}

        # maximin support score for added candidate C
        support_value = 0.0

        if S:
            while True:
                # NS = {voter∈N|voter.approved ∩ S ≠ ∅}
                NS = []
                NS_total_weight = 0
                for voter in profile:
                    if voter.approved & S:
                        NS.append(voter)
                        NS_total_weight += voter.weight

                if not NS or NS_total_weight == 0:
                    support_value = 0.0
                    break

                S_size = len(S)
                target_flow = NS_total_weight * S_size

                # Build flow network
                G = nx.DiGraph()
                source = "source"
                sink = "sink"

                # Source → Voters
                for i, voter in enumerate(NS):
                    voter_node = f"voter_{i}"
                    G.add_edge(source, voter_node, capacity=voter.weight * S_size)

                    # Voters → Candidates
                    # (candidates in the test committee AND approved by at least one voter in NS)
                    for cand in S:
                        if cand in voter.approved:
                            cand_node = f"cand_{cand}"
                            G.add_edge(voter_node, cand_node, capacity=voter.weight * S_size)

                # Candidates → Sink
                for cand in S:
                    cand_node = f"cand_{cand}"
                    G.add_edge(cand_node, sink, capacity=NS_total_weight)

                # Compute max flow
                flow_value, flow_dict = nx.maximum_flow(G, source, sink)

                # Check if target achieved (with tolerance for floating point)
                if flow_value >= target_flow:
                    support_value = NS_total_weight / S_size
                    break

                # Target not achieved: remove unsupported candidates
                unsupported_candidates = set()
                for i, voter in enumerate(NS):
                    voter_node = f"voter_{i}"
                    # Expected flow: voter edge capacity (voter.weight * S_size)
                    expected_flow_for_voter = voter.weight * S_size

                    # Actual flow: sum of flow from voter_i to all candidates in S.
                    actual_flow_for_voter = sum(
                        flow_dict.get(voter_node, {}).get(f"cand_{cand}", 0) for cand in S
                    )

                    # If actual < expected, voter is constrained, can't send the expected flow.
                    # Mark their approved candidates as unsupported for removal.
                    # e.x. S=[2,3], voter_i.approved=[1,2] => voter_i constrained by [2]
                    #   => [2] is unsupported candidate and needs to be removed
                    # why not [3]?
                    # 3∉[1,2], means 3 not approved by voter_i therefore is not contained.
                    if actual_flow_for_voter < expected_flow_for_voter:
                        for cand in S:
                            if cand in voter.approved:
                                unsupported_candidates.add(cand)

                # Remove unsupported candidates from S
                S -= unsupported_candidates

                # If S is empty after removing unsupported candidates,
                # set support value to 0 and break
                if not S:
                    support_value = 0.0
                    break

        # Set support value for added candidate
        scores[added_cand] = support_value

    return scores
