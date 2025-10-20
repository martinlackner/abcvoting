"""
Pytest tests for maximin-support with max-flow algorithm implementation.

To run these tests, execute:
    pytest test_maximin_support.py
"""

import pytest
from abcvoting.preferences import Profile
from abcvoting import abcrules


def test_maximin_support():
    """
    Tests for the new max-flow algorithm implementation of maximin-support.
    This implementation uses a sequence of max flow problems instead of ILP,
    making it more efficient and numerically stable.
    """

    # Test 1: Simple case - 3 voters, 2 candidates
    # Verifies basic functionality with minimal input
    profile = Profile(num_cand=2)
    profile.add_voters(
        [{0}, {1}, {0, 1}]  # Voter 1 approves A  # Voter 2 approves B  # Voter 3 approves A and B
    )
    result = abcrules.compute("maximin-support", profile, committeesize=2, algorithm="max-flow")
    assert result == [{0, 1}]

    # Test 2: 5 voters, 4 candidates
    # Tests with overlapping approval sets
    profile = Profile(num_cand=4)
    profile.add_voters(
        [
            {0, 1},  # Voter 1 approves A, B
            {0, 2},  # Voter 2 approves A, C
            {1, 2},  # Voter 3 approves B, C
            {2},  # Voter 4 approves C
            {1, 3},  # Voter 5 approves B, D
        ]
    )
    result = abcrules.compute("maximin-support", profile, committeesize=2, algorithm="max-flow")
    assert result == [{1, 2}]

    # Test 3: 7 voters, 5 candidates, committee size 3
    # Tests larger committee size with complex approval patterns
    profile = Profile(num_cand=5)
    profile.add_voters(
        [
            {0, 1},  # Voter 1 approves A, B
            {0, 2},  # Voter 2 approves A, C
            {1, 2, 3},  # Voter 3 approves B, C, D
            {1, 3},  # Voter 4 approves B, D
            {2, 4},  # Voter 5 approves C, E
            {3, 4},  # Voter 6 approves D, E
            {2, 3},  # Voter 7 approves C, D
        ]
    )
    result = abcrules.compute("maximin-support", profile, committeesize=3, algorithm="max-flow")
    assert result == [{1, 2, 3}]

    # Test 4: 6 voters with duplicate approval sets
    # Tests symmetry and tie-breaking with identical voter groups
    profile = Profile(num_cand=4)
    profile.add_voters(
        [
            {0, 1},  # Voter 1 approves A, B
            {0, 1},  # Voter 2 approves A, B
            {1, 2},  # Voter 3 approves B, C
            {1, 2},  # Voter 4 approves B, C
            {2, 3},  # Voter 5 approves C, D
            {2, 3},  # Voter 6 approves C, D
        ]
    )
    result = abcrules.compute("maximin-support", profile, committeesize=2, algorithm="max-flow")
    assert result == [{1, 2}]

    # Test 5: Sparse approval sets
    # Tests behavior when some voters have minimal approvals
    profile = Profile(num_cand=3)
    profile.add_voters(
        [
            {0},  # Voter 1 approves A only
            {0, 1},  # Voter 2 approves A, B
            {2},  # Voter 3 approves C only
        ]
    )
    result = abcrules.compute("maximin-support", profile, committeesize=2, algorithm="max-flow")
    assert result == [{0, 1}]

    result = abcrules.compute(
        "maximin-support", profile, committeesize=2, resolute=False, algorithm="max-flow"
    )
    assert {0, 1} in result
    assert {0, 2} in result

    # Test 6: Large example with 16 voters, 7 candidates, and block structure
    # Tests scalability and handling of voter blocks with overlapping preferences
    profile = Profile(num_cand=7)
    profile.add_voters(
        [
            # Block 1 (prefers A/B/C)
            {0, 1},
            {0, 2},
            {1, 2},
            {0, 1, 2},
            # Block 2 (prefers C/D/E)
            {2, 3},
            {2, 4},
            {3, 4},
            {2, 3, 4},
            # Block 3 (prefers E/F/G)
            {4, 5},
            {4, 6},
            {5, 6},
            {4, 5, 6},
            # Mixed voters
            {1, 3},
            {2, 5},
            {3, 6},
            {0, 4, 6},
        ]
    )
    result = abcrules.compute("maximin-support", profile, committeesize=4, algorithm="max-flow")
    assert {2, 3, 4, 6} in result

    # Test 7: Committee size = 1
    # Edge case: minimal committee size
    profile = Profile(num_cand=3)
    profile.add_voters([{0}, {1}, {2}])
    result = abcrules.compute(
        "maximin-support", profile, committeesize=1, resolute=False, algorithm="max-flow"
    )
    assert len(result) == 3
    assert all(len(committee) == 1 for committee in result)

    # Test 8: Committee size equals number of candidates
    # Edge case: maximal committee size
    profile = Profile(num_cand=3)
    profile.add_voters([{0, 1, 2}])
    result = abcrules.compute("maximin-support", profile, committeesize=3, algorithm="max-flow")
    assert result == [{0, 1, 2}]

    # Test 9: Single voter
    # Edge case: minimal number of voters
    profile = Profile(num_cand=4)
    profile.add_voters([{0, 1, 2}])
    result = abcrules.compute(
        "maximin-support", profile, committeesize=2, resolute=False, algorithm="max-flow"
    )
    assert len(result) == 3
    assert all(len(committee) == 2 for committee in result)

    # Test 10: All voters approve all candidates
    # Edge case: complete agreement among voters
    profile = Profile(num_cand=3)
    profile.add_voters(
        [
            {0, 1, 2},
            {0, 1, 2},
            {0, 1, 2},
        ]
    )
    result = abcrules.compute("maximin-support", profile, committeesize=2, algorithm="max-flow")
    assert len(result) > 0
    assert all(len(committee) == 2 for committee in result)

    # Test 11: Disjoint approval sets
    # Tests behavior when voters have completely different preferences
    profile = Profile(num_cand=4)
    profile.add_voters([{0}, {1}, {2}, {3}])
    result = abcrules.compute(
        "maximin-support", profile, committeesize=2, resolute=False, algorithm="max-flow"
    )
    assert len(result) > 0
    assert all(len(committee) == 2 for committee in result)

    # Test 12: Resolute parameter - return only one committee
    # Tests that resolute=True returns exactly one winning committee
    profile = Profile(num_cand=4)
    profile.add_voters(
        [
            {0, 1},
            {0, 1},
            {1, 2},
            {1, 2},
            {2, 3},
            {2, 3},
        ]
    )
    result = abcrules.compute("maximin-support", profile, committeesize=2, algorithm="max-flow")
    assert result == [{1, 2}]

    # Test 13: Resolute=False - may return multiple committees
    # Tests that resolute=False returns all winning committees
    profile = Profile(num_cand=4)
    profile.add_voters(
        [
            {0, 1},
            {0, 1},
            {1, 2},
            {1, 2},
            {2, 3},
            {2, 3},
        ]
    )
    result = abcrules.compute(
        "maximin-support", profile, committeesize=2, resolute=False, algorithm="max-flow"
    )
    assert result == ({1, 2})

    # Test 14: Consistency check - compare max-flow with default implementation
    # Verifies that max-flow produces the same results as ILP-based implementations
    profile = Profile(num_cand=4)
    profile.add_voters(
        [
            {0, 1},
            {0, 2},
            {1, 2},
            {2},
            {1, 3},
        ]
    )
    committeesize = 2
    result_maxflow = abcrules.compute(
        "maximin-support", profile, committeesize, resolute=False, algorithm="max-flow"
    )
    # Compute using default algorithm (ILP-based - algorithm == "fastest" )
    result_default = abcrules.compute("maximin-support", profile, committeesize, resolute=False)
    # Both should produce the same set of winning committees
    assert set(frozenset(c) for c in result_maxflow) == set(frozenset(c) for c in result_default)

    # Test 15: Invalid input - committee size = 0
    # Tests error handling for invalid committee size
    profile = Profile(num_cand=3)
    profile.add_voters([{0, 1}])
    with pytest.raises((ValueError, AssertionError)):
        abcrules.compute("maximin-support", profile, committeesize=0, algorithm="max-flow")

    # Test 16: Invalid input - committee size > number of candidates
    # Tests error handling when committee size exceeds available candidates
    profile = Profile(num_cand=3)
    profile.add_voters([{0, 1}])
    with pytest.raises((ValueError, AssertionError)):
        abcrules.compute("maximin-support", profile, committeesize=5, algorithm="max-flow")

    # Test 17: Invalid input - negative committee size
    # Tests error handling for negative committee size
    profile = Profile(num_cand=3)
    profile.add_voters([{0, 1}])
    with pytest.raises((ValueError, AssertionError)):
        result_default = abcrules.compute(
            "maximin-support", profile, committeesize=-1, algorithm="max-flow"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
