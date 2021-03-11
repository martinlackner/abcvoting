"""Proposition A.4
from the survey: "Approval-Based Multi-Winner Voting:
Axioms, Algorithms, and Applications"
by Martin Lackner and Piotr Skowron
"""

from abcvoting import abcrules
from abcvoting.preferences import Profile
from abcvoting import misc


print(misc.header("Proposition A.4", "*"))

num_cand = 6
a, b, c, d, e, f = range(6)  # a = 0, b = 1, c = 2, ...
cand_names = "abcdef"

manipulations = [
    ("cc", True, 2, [{a, b}] + [{a}] * 3 + [{c}], {b}, [{a, c}], [{a, b}]),
    ("sav", False, 1, [{a, b, c}, {d, e}], {a}, [{d}, {e}], [{a}]),
    (
        "revseqpav",
        False,
        2,
        [{a, b, c}, {b, d}, {c, b}, {a, d, e}, {b, e}],
        {a},
        [{b, d}, {b, e}],
        [{a, b}],
    ),
    (
        "seqphragmen",
        False,
        2,
        [{0, 1, 2}, {0, 1}, {1, 5}, {2, 4}, {1, 4, 5}, {1, 3, 5}],
        {2},
        [{1, 5}],
        [{1, 2}],
    ),
    (
        "seqpav",
        False,
        3,
        [{0, 1}, {1, 3}, {2, 5}, {0, 1, 5}, {1, 5}, {1, 2}],
        {0},
        [{1, 2, 5}],
        [{0, 1, 5}],
    ),
    (
        "minimaxav",
        False,
        3,
        [{0, 1, 2}, {1, 3}, {0, 1, 4}, {0, 1, 3}, {0, 1, 4}, {0, 1}],
        {2},
        [{0, 1, 3}],
        [{0, 1, 2}],
    ),
    (
        "pav",
        False,
        3,
        [{2, 3, 4}, {0, 1}, {1, 5}, {0, 2, 3}, {1, 2, 5}, {2, 4, 5}],
        {4},
        [{1, 2, 5}],
        [{1, 2, 4}],
    ),
    (
        "rule-x",
        False,
        3,
        [{1, 2, 3}, {0, 1}, {1, 3}, {2, 3}, {3, 4}, {3, 4}],
        {2},
        [{1, 3, 4}],
        [{1, 2, 3}],
    ),
    ("greedy-monroe", True, 2, [{0, 1}, {0, 2, 5}, {0, 2, 3}, {4, 5}], {1}, [{0, 2}], [{0, 1}]),
    (
        "monroe",
        False,
        3,
        [
            {1, 3},
            {0, 1, 2},
            {1, 4},
            {3, 4},
            {4, 5},
            {1, 2, 4},
            {2, 3, 4},
            {1, 2},
            {0, 5},
            {1, 2, 3},
            {0, 5},
            {0, 3},
        ],
        {5},
        [{0, 1, 4}],
        [{1, 3, 5}],
    ),
    (
        "seqcc",
        False,
        3,
        [
            {1, 4, 5},
            {0, 1},
            {3, 4, 5},
            {3, 4},
            {1, 5},
            {2, 3},
            {0, 1, 2},
            {0, 2},
            {2, 3},
            {0, 1, 4},
            {0, 4, 5},
            {1, 2, 3},
        ],
        {2},
        [{0, 1, 3}],
        [{1, 2, 4}],
    ),
    # ("minimaxphragmen", True, 3, [{a, b}] + [{b, c, d}] * 3, [a}, {{b, c, d}}, {{a, b, c}]),
    # this does not work because minimax-Phragmen does not support a specified tiebreaking
    # between committees ([a, b, c] should be prefered to [b, c, d])
]

for manip in manipulations:
    (rule_id, resolute, committeesize, approval_sets, modvote, commsfirst, commsafter) = manip

    print(misc.header(abcrules.get_longname(rule_id), "-"))

    profile = Profile(num_cand, cand_names=cand_names)
    profile.add_voters(approval_sets)
    truepref = profile[0].approved
    print(profile.str_compact())

    committees = abcrules.compute(rule_id, profile, committeesize, resolute=resolute)
    print("original winning committees:\n" + misc.str_sets_of_candidates(committees, cand_names))

    # verify correctness
    assert committees == commsfirst

    print(
        "Manipulation by voter 0: "
        + misc.str_set_of_candidates(approval_sets[0], cand_names)
        + " --> "
        + misc.str_set_of_candidates(modvote, cand_names)
    )
    if not all(cand in truepref for cand in modvote):
        print(" (not a subset!)")

    approval_sets[0] = modvote
    profile = Profile(num_cand, cand_names=cand_names)
    profile.add_voters(approval_sets)

    committees = abcrules.compute(rule_id, profile, committeesize, resolute=resolute)
    print(
        "\nwinning committees after manipulation:\n"
        + misc.str_sets_of_candidates(committees, cand_names)
    )

    # verify correctness
    assert committees == commsafter

    # verify that this is a counterexample to inclusion-strategyproofness
    # with the Kelly (cautious) set extension
    for commfirst in commsfirst:
        for commafter in commsafter:
            for cand in set(commfirst) & set(truepref):
                assert cand in commafter
            assert set(commfirst) & set(truepref) < set(commafter) & set(truepref)
