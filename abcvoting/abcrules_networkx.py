"""
Max-Flow Implementation for Maximin Support Method

Based on: "Efficient, traceable, and numerical error-free"
https://arxiv.org/abs/2309.15104 (October 17, 2023)
By Luis Sánchez-Fernández
From Carlos III University, Madrid, Spain

Programmer: Amit Gini (B.Sc Computer Science Student at Ariel University, Israel)
Date: 10-2025

To run these tests, execute:
    python -m doctest abcvoting/abcrules_networkx.py -v
"""

import networkx as nx
import logging

logger = logging.getLogger(__name__)


def _compute_support_value(profile, S):
    """
    Compute the maximin support value for a given committee using max-flow algorithm.

    This function computes what the maximin support score would be for a test
    committee S by iteratively solving max-flow problems and removing unsupported
    candidates until the target flow is achieved.

    Parameters
    ----------
    profile : Profile
        Contains voter preferences and weights
        - iterate as: for voter in profile:
            - voter.approved: set of approved candidates {0, 2, 5, ...}
            - voter.weight: weight of this voter (usually 1)

    S : set
        Test committee (set of candidate indices)
        Example: {0, 3} means candidates 0 and 3 are in the test committee

    Returns
    -------
    float
        The maximin support value for the committee S.
        Returns 0.0 if no support can be achieved or if S is empty.

    Algorithm Overview
    ------------------
    Notions:
    - S: Test committee (set of candidate indices)
    - k_S_size = |S|: Size of the test committee
    - NS = {voter∈N | voter.approved ∩ S ≠ ∅}
        Set of voters who approve at least one candidate in S
    - NS_total_weight = sum(voter.weight for voter in NS)
        Total weight of voters in NS (with weighted voters)
    - Flow Value: Maximum flow from source to sink returned from max-flow
        algorithm in networkx
    - Target Flow = NS_total_weight × k_S_size
        Total weight of voters in NS × size of committee, needed to
        achieve maximin-support score

    Steps:
    1. Build NS: Find all voters who approve at least one candidate in S
    2. Build max-flow network with integer arithmetic:
       - Source → Voters: Edge capacity = voter.weight × k_S_size
         (might have different weights for voters)
       - Voters → Candidates: Edge capacity = voter.weight × k_S_size
       - Candidates → Sink: Edge capacity = NS_total_weight
         (Sum of weights of voters in NS)
    3. Compute maximum flow using max-flow algorithm in networkx
    4. If Flow Value >= Target Flow:
       - Return maximin-support score = NS_total_weight / k_S_size
    5. Otherwise:
       - Identify unsupported candidates (candidates that cannot receive
         sufficient flow from constrained voters)
       - Remove unsupported candidates from S
       - If S becomes empty, return 0.0
       - Otherwise, repeat from step 1 with the reduced S

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

    # Iteration 1: test each single candidate committee
    >>> _compute_support_value(profile, {0})   # committee: {0}
    0.0
    >>> _compute_support_value(profile, {1})   # committee: {1}
    2.0
    >>> _compute_support_value(profile, {2})   # committee: {2}
    3.0
    >>> _compute_support_value(profile, {3})   # committee: {3}
    1.0

    # Iteration 2: test committees with candidate 2 and each remaining candidate
    >>> _compute_support_value(profile, {2, 0})   # committee: {2, 0}
    0.0
    >>> _compute_support_value(profile, {2, 1})   # committee: {2, 1}
    1.5
    >>> _compute_support_value(profile, {2, 3})   # committee: {2, 3}
    1.0

    Notes
    -----
    - The algorithm uses integer arithmetic internally (capacities × k_S_size)
    - When target flow is achieved, all candidates receive equal support
    - Voter weights are properly considered in all capacity calculations
    - The iterative refinement removes unsupported candidates until target
      flow is achieved or no candidates remain
    """
    logger.info("[_compute_support_value] Computing support for S=%s", sorted(list(S)))
    support_value = 0.0

    if S:
        while True:
            # NS = {voter∈N|voter.approved ∩ S ≠ ∅}
            NS = []
            NS_total_weight = 0
            for i_voter in profile:
                if i_voter.approved & S:
                    NS.append(i_voter)
                    NS_total_weight += i_voter.weight

            if not NS or NS_total_weight == 0:
                logger.warning(
                    "[iteration] No supportive voters found "
                    "(NS is empty or has zero weight), support_value=0.0"
                )
                support_value = 0.0
                break

            k_S_size = len(S)  # size of the test committee
            target_flow = NS_total_weight * k_S_size

            logger.info(
                "[iteration] NS voters count=%d, NS_total_weight=%g, S_size=%d",
                len(NS),
                NS_total_weight,
                k_S_size,
            )
            logger.debug(
                "[iteration] NS voters: %s",
                [{"approved": sorted(list(v.approved)), "weight": v.weight} for v in NS],
            )

            # Build flow network
            G = nx.DiGraph()
            source = "source"
            sink = "sink"

            # Source → Voters
            for i, i_voter in enumerate(NS):
                i_voter_node = f"voter_{i}"
                G.add_edge(source, i_voter_node, capacity=i_voter.weight * k_S_size)

                # Voters → Candidates
                # (candidates in the test committee AND approved by at least one voter in NS)
                for c_cand in S:
                    if c_cand in i_voter.approved:
                        c_cand_node = f"cand_{c_cand}"
                        G.add_edge(i_voter_node, c_cand_node, capacity=i_voter.weight * k_S_size)

            # Candidates → Sink
            for c_cand in S:
                c_cand_node = f"cand_{c_cand}"
                G.add_edge(c_cand_node, sink, capacity=NS_total_weight)

            # Compute max flow
            flow_value, flow_dict = nx.maximum_flow(G, source, sink)

            logger.debug("[iteration] target_flow=%g, flow_value=%g", target_flow, flow_value)

            # Check if target achieved (with tolerance for floating point)
            # The opposite check than the condition checked in the article
            if flow_value >= (NS_total_weight * k_S_size):  # target flow
                support_value = NS_total_weight / k_S_size
                logger.info(
                    "[iteration] Target reached, support_value=%g (NS_total_weight=%g, S_size=%d)",
                    support_value,
                    NS_total_weight,
                    k_S_size,
                )
                break

            # Target not achieved: remove unsupported candidates
            unsupported_candidates = set()
            for i, i_voter in enumerate(NS):
                i_voter_node = f"voter_{i}"
                # Expected flow: voter edge capacity (voter.weight * S_size)
                expected_flow_for_voter = i_voter.weight * k_S_size

                # Actual flow: sum of flow from voter_i to all candidates in S.
                actual_flow_for_voter = sum(
                    flow_dict.get(i_voter_node, {}).get(f"cand_{cand}", 0) for cand in S
                )

                # If actual < expected, voter is constrained, can't send the expected flow.
                # Mark their approved candidates as unsupported for removal.
                # e.x. S=[2,3], voter_i.approved=[1,2] => voter_i constrained by [2]
                #   => [2] is unsupported candidate and needs to be removed
                # why not [3]?
                # 3∉[1,2], means 3 not approved by voter_i therefore is not contained.
                if actual_flow_for_voter < expected_flow_for_voter:
                    approved_in_S = sorted([c for c in S if c in i_voter.approved])
                    logger.debug(
                        "[iteration] voter_%d constrained: expected_flow=%g, "
                        "actual_flow=%g -> approved_in_S=%s",
                        i,
                        expected_flow_for_voter,
                        actual_flow_for_voter,
                        approved_in_S,
                    )
                    for c_cand in S:
                        if c_cand in i_voter.approved:
                            unsupported_candidates.add(c_cand)

            # Remove unsupported candidates from S
            logger.info(
                "[iteration] Removed %d unsupported candidates: %s",
                len(unsupported_candidates),
                sorted(list(unsupported_candidates)),
            )
            S -= unsupported_candidates

            # If S is empty after removing unsupported candidates,
            # set support value to 0 and break
            if not S:
                logger.warning(
                    "[iteration] S became empty after removing unsupported "
                    "candidates, returning 0.0"
                )
                support_value = 0.0
                break

    logger.info("[_compute_support_value] Final support_value=%g", support_value)
    return support_value


def _nx_maximin_support_scorefct(profile, base_committee):
    """
        Compute maximin support scores using max-flow algorithm.

        This function is called iteratively during committee construction.
        For each candidate not yet in the committee, it computes what the
        maximin support score would be if that candidate were added to the
        base committee, and returns the support score list.

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
            List of maximin support scores where scores[candidate_index] is the
            maximin support score for the candidate represented by candidate_index.
            Candidates already in base_committee have score 0.

        Algorithm Overview
        ------------------
        Notions:
        - N: Set of all voters in the profile
        - base_committee: Current partial committee (list of candidate indices)
        - C: Candidate being tested for addition to the committee
        - S = base_committee ∪ {C}: Test committee with candidate C added

        Steps:
        1. For each candidate C not in base_committee:
            1.1. Form test committee S = base_committee ∪ {C}
            1.2. Compute maximin support score for candidate C using
                 _compute_support_value(profile, S)
            1.3. Store the support score in scores[C]
        2. Return the maximin-support score list

        Examples
        --------
        >>> from abcvoting.preferences import Profile, Voter

    ----------------------------------------------------------------
        Example 1: 5 candidates, 4 voters, k=3
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
        Example 2: Weighted voters
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
        - This function iterates over all candidates not in base_committee and
          computes their maximin support scores using the max-flow algorithm
        - The actual max-flow computation is delegated to _compute_support_value
    """
    # Initialize scores, set to 0 for candidates in base_committee
    scores = [0] * profile.num_cand

    # Get candidates not yet in base_committee
    remaining_candidates = [cand for cand in profile.candidates if cand not in base_committee]

    logger.info(
        "[_nx_maximin_support_scorefct] base_committee=%s, remaining=%s",
        base_committee,
        remaining_candidates,
    )

    # For each remaining candidate, compute maximin support score
    for added_cand in remaining_candidates:
        logger.info("[_nx_maximin_support_scorefct] Evaluating candidate %d", added_cand)

        # Form test committee of the existing committee and
        # the remaining candidates (base_committee ∪ {C})
        S = set(base_committee) | {added_cand}

        # maximin support score for added candidate C
        support_value = _compute_support_value(profile, S)

        # Set support value for added candidate
        scores[added_cand] = support_value

    logger.info(
        "[_nx_maximin_support_scorefct] Computed scores for %d candidates",
        len(remaining_candidates),
    )
    logger.debug("[_nx_maximin_support_scorefct] Final scores=%s", scores)

    return scores
