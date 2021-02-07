"""Proposition A.3
from the survey: "Approval-Based Multi-Winner Voting:
Axioms, Algorithms, and Applications"
by Martin Lackner and Piotr Skowron
"""

import sys

sys.path.insert(0, "..")
from abcvoting import abcrules
from abcvoting.preferences import Profile
from abcvoting import misc


print(misc.header("Proposition A.3", "*"))

num_cand = 8
a, b, c, d, e, f, g, h = range(num_cand)  # a = 0, b = 1, c = 2, ...
cand_names = "abcdefgh"

monotonicity_instances = [
    (
        "seqphragmen",
        3,  # from Xavier Mora, Maria Oliver (2015)
        [{0, 1}] * 10 + [{2}] * 3 + [{3}] * 12 + [{0, 1, 2}] * 21 + [{2, 3}] * 6,
        True,
        {0, 1},
        [{0, 1, 3}],
        [{0, 2, 3}, {1, 2, 3}],
    ),
    (
        "seqphragmen",
        3,  # from Xavier Mora, Maria Oliver (2015)
        [{2}] * 7 + [{0, 1}] * 4 + [{0, 1, 2}] + [{0, 1, 3}] * 16 + [{2, 3}] * 4,
        False,
        {0, 1},
        [{0, 1, 2}],
        [{0, 2, 3}, {1, 2, 3}],
    ),
    (
        "rule-x",
        3,
        [{1, 3}, {0, 1}, {1, 3, 4}, {0, 4}, {2, 3, 4}, {2, 4}, {2, 3, 4}, {0, 2, 4}, {1, 2, 3}],
        True,
        [0],
        [{0, 3, 4}],
        [{1, 2, 4}],
    ),
    (
        "rule-x",
        3,
        [{1}, {1, 4, 5}, {1, 4}, {2}, {1, 4}, {2, 5}, {5}],
        False,
        {1, 4, 5},
        [{1, 4, 5}],
        [{1, 2, 5}, {1, 4, 5}, {2, 4, 5}],
    ),
    (
        "rule-x-without-2nd-phase",
        4,
        [
            {0, 1, 2},
            {0, 6},
            {3, 4},
            {1, 3, 5},
            {0, 5},
            {0, 1, 2},
            {7},
            {0, 7},
            {1, 7},
            {1, 3},
            {3, 4, 5},
            {2, 4, 7},
        ],
        True,
        {4},
        [{0, 1, 4}],
        [{0, 1, 4}, {0, 3, 7}],
    ),
    (
        "revseqpav",
        3,
        [
            {0, 4},
            {1, 2, 3},
            {3, 4},
            {2, 4},
            {1, 3, 4},
            {2, 4},
            {0, 1, 2},
            {2, 3, 4},
            {0, 3, 4},
            {1, 3},
            {0, 4},
            {0, 3, 4},
            {0, 1},
            {0, 3},
            {0, 1, 3},
            {2, 4},
            {1, 2, 3},
            {1, 2},
        ],
        False,
        {2, 3},
        [{2, 3, 4}],
        [{1, 3, 4}],
    ),
    (
        "greedy-monroe",
        3,
        [{1, 2, 3}, {0, 2, 5}, {0, 3, 4}, {2, 4}, {0, 1}, {3, 5}, {3, 5}, {1, 4}, {1, 5}],
        True,
        {4},
        [{1, 4, 5}],
        [{1, 2, 3}],
    ),
    ("greedy-monroe", 2, [{3}, {2}, {1}, {0, 2}], False, {1, 2}, [{1, 2}], [{0, 2}]),
    (
        "pav",
        4,  # from Sanchez-Fernandez and Fisteus (2019)
        [{0, 1}, {0, 1, 2}, {4, 5}, {4, 5}]
        + [{0, 4}, {1, 4}, {2, 4}, {0, 5}, {1, 5}, {2, 5}, {0, 6}, {1, 6}, {2, 6}] * 3
        + [{3}] * 100,
        False,
        {2, 3},
        [{0, 1, 2, 3}],
        [{3, 4, 5, 6}],
    ),
    (
        "cc",
        3,  # from Sanchez-Fernandez and Fisteus (2019)
        [{a}, {a, d}, {a, e}, {c, d}, {c, e}, {b}] * 2 + [{d}],
        False,
        {b, c},
        [{a, b, c}],
        [{a, b, c}, {b, d, e}],
    ),
    (
        "monroe",
        4,  # from Sanchez-Fernandez and Fisteus (2019)
        [{a, e}] * 5
        + [{a, g}] * 4
        + [{b, e}] * 5
        + [{b, h}] * 4
        + [{c, f}] * 5
        + [{c, g}] * 4
        + [{d, f}] * 3
        + [{d, h}] * 3,
        True,
        {e},
        [{e, f, g, h}],
        [{a, b, c, d}, {e, f, g, h}],
    ),
    (
        "monroe",
        3,
        [{a}, {a, d}, {a, e}] * 2 + [{b}, {c, d}] * 4 + [{b, e}] + [{c, e}] * 3,
        False,
        {b, c},
        [{a, b, c}],
        [{a, b, c}, {b, d, e}],
    ),
    (
        "seqpav",
        3,
        [{1, 2}, {1, 3}, {4, 5}, {0, 4}, {2, 5}, {0, 1}, {1, 5}, {0, 4}],
        False,
        {4, 5},
        [{1, 4, 5}],
        [{0, 1, 5}, {1, 4, 5}],
    ),
    (
        "seqpav",
        4,  # from Sanchez-Fernandez and Fisteus (2019)
        [{a, b, d}] * 7 + [{a, b, e}] * 4 + [{a, c, d}] * 3 + [{a, c, e}] * 5,
        True,
        {c, d},
        [{a, b, c, d}],
        [{a, b, d, e}],
    ),
    (
        "seqcc",
        3,
        [{0}, {0}, {0, 1, 2}, {0, 3}, {0, 3}, {1}, {1}, {1, 3}, {2}, {2}, {2}, {3}],
        True,
        {1, 2},
        [{0, 1, 2}],
        [{0, 1, 2}, {0, 2, 3}, {1, 2, 3}],
    ),
    (
        "seqcc",
        3,
        [{4}, {0}, {0}, {0, 1, 2}, {0, 3}, {0, 3}, {1}, {1}, {1, 3}, {2}, {2}, {2}, {3}],
        False,
        {1, 2},
        [{0, 1, 2}],
        [{0, 1, 2}, {0, 2, 3}, {1, 2, 3}],
    ),
    (
        "optphragmen",
        6,  # from Sanchez-Fernandez and Fisteus (2019)
        [{1, 2, 3, 4, 5}] * 13 + [{0, 6}, {0}] * 2 + [{6}] * 1,
        True,
        {0, 1, 2, 3, 4, 5},
        [{0, 1, 2, 3, 4, 5}, {1, 2, 3, 4, 5, 6}],
        [
            {0, 1, 2, 3, 4, 6},
            {0, 1, 2, 3, 5, 6},
            {0, 1, 2, 4, 5, 6},
            {0, 1, 3, 4, 5, 6},
            {0, 2, 3, 4, 5, 6},
        ],
    ),
    (
        "optphragmen",
        6,  # from Sanchez-Fernandez and Fisteus (2019)
        [{7}] + [{1, 2, 3, 4, 5}] * 13 + [{0, 6}, {0}] * 2 + [{6}],
        False,
        {0, 1, 2, 3, 4, 5},
        [{0, 1, 2, 3, 4, 5}, {1, 2, 3, 4, 5, 6}],
        [
            {0, 1, 2, 3, 4, 6},
            {0, 1, 2, 3, 5, 6},
            {0, 1, 2, 4, 5, 6},
            {0, 1, 3, 4, 5, 6},
            {0, 2, 3, 4, 5, 6},
        ],
    ),
    (
        "minimaxav",
        5,  # from Sanchez-Fernandez and Fisteus (2019)
        [{5}, {1}, {2}, {3}, {1, 4, 6}, {1, 4, 7}, {1, 5, 6}, {1, 5, 7}, {1, 6}],
        False,
        {1, 2, 3, 4},
        [{1, 2, 3, 4, 5}],
        [{1, 2, 3, 6, 7}],
    ),
]


for inst in monotonicity_instances:
    (
        rule_id,
        committeesize,
        approval_sets,
        addvoter,
        new_approval_set,
        commsfirst,
        commsafter,
    ) = inst

    print(misc.header(abcrules.rules[rule_id].longname, "-"))

    profile = Profile(num_cand, cand_names=cand_names)
    profile.add_voters(approval_sets)
    original_approval_set = set(approval_sets[0])
    print(profile.str_compact())

    # irresolute if possible
    if False in abcrules.rules[rule_id].resolute:
        resolute = False
    else:
        resolute = True

    committees = abcrules.compute(rule_id, profile, committeesize, resolute=resolute)
    print("original winning committees:\n" + misc.str_sets_of_candidates(committees, cand_names))
    # verify correctness
    assert committees == commsfirst
    some_variant = any(
        all(cand in committee for cand in new_approval_set) for committee in commsfirst
    )
    all_variant = all(
        all(cand in committee for cand in new_approval_set) for committee in commsfirst
    )
    assert some_variant or all_variant
    if all_variant:
        assert not all(
            all(cand in committee for cand in new_approval_set) for committee in commsafter
        )
    else:
        assert not any(
            all(cand in committee for cand in new_approval_set) for committee in commsafter
        )

    if addvoter:
        print("additional voter: " + misc.str_set_of_candidates(new_approval_set, cand_names))
        approval_sets.append(new_approval_set)
    else:
        approval_sets[0] = list(set(new_approval_set) | set(approval_sets[0]))
        print(
            "change of voter 0: "
            + misc.str_set_of_candidates(list(original_approval_set), cand_names)
            + " --> "
            + misc.str_set_of_candidates(approval_sets[0], cand_names)
        )

    profile = Profile(num_cand, cand_names=cand_names)
    profile.add_voters(approval_sets)

    committees = abcrules.compute(rule_id, profile, committeesize, resolute=resolute)
    print(
        "\nwinning committees after the modification:\n"
        + misc.str_sets_of_candidates(committees, cand_names)
    )

    # verify correctness
    assert committees == commsafter

    print(abcrules.rules[rule_id].shortname + " fails ", end="")
    if addvoter:
        if len(new_approval_set) == 1:
            print("candidate", end="")
        else:
            print("support", end="")
        print(" monotonicity with additional voters")
    else:
        if len(set(new_approval_set) - original_approval_set) == 1:
            print("candidate", end="")
        else:
            print("support", end="")
        print(" monotonicity without additional voters")
    print()
