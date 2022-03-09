"""
Proposition A.3.

From "Multi-Winner Voting with Approval Preferences"
by Martin Lackner and Piotr Skowron
https://arxiv.org/abs/2007.01795
"""

from abcvoting import abcrules
from abcvoting.preferences import Profile
from abcvoting import misc
from collections import namedtuple


print(misc.header("Proposition A.3", "*"))

num_cand = 8
a, b, c, d, e, f, g, h = range(num_cand)  # a = 0, b = 1, c = 2, ...
cand_names = "abcdefgh"

MonotonicityInstance = namedtuple(
    "MonotonicityInstance",
    "rule_id committeesize approval_sets with_additional_voter mod_approval_set "
    "committees_first committees_after",
)

monotonicity_instances = [
    MonotonicityInstance(
        # from Xavier Mora, Maria Oliver (2015)
        rule_id="seqphragmen",
        committeesize=3,
        approval_sets=[{0, 1}] * 10 + [{2}] * 3 + [{3}] * 12 + [{0, 1, 2}] * 21 + [{2, 3}] * 6,
        with_additional_voter=True,
        mod_approval_set={0, 1},
        committees_first=[{0, 1, 3}],
        committees_after=[{0, 2, 3}],
    ),
    MonotonicityInstance(  # from Sanchez-Fernandez and Fisteus (2019)
        rule_id="minimaxphragmen",
        committeesize=6,
        approval_sets=[{1, 2, 3, 4, 5}] * 13 + [{0, 6}, {0}] * 2 + [{6}] * 1,
        with_additional_voter=True,
        mod_approval_set={0, 1, 2, 3, 4, 5},
        committees_first=[{0, 1, 2, 3, 4, 5}, {1, 2, 3, 4, 5, 6}],
        committees_after=[
            {0, 1, 2, 3, 4, 6},
            {0, 1, 2, 3, 5, 6},
            {0, 1, 2, 4, 5, 6},
            {0, 1, 3, 4, 5, 6},
            {0, 2, 3, 4, 5, 6},
        ],
    ),
    MonotonicityInstance(
        # from Sanchez-Fernandez and Fisteus (2019)
        rule_id="seqpav",
        committeesize=4,
        approval_sets=[{a, b, d}] * 7 + [{a, b, e}] * 4 + [{a, c, d}] * 3 + [{a, c, e}] * 5,
        with_additional_voter=True,
        mod_approval_set={c, d},
        committees_first=[{a, b, c, d}],
        committees_after=[{a, b, d, e}],
    ),
    MonotonicityInstance(
        rule_id="seqcc",
        committeesize=3,
        approval_sets=[
            {0},
            {0},
            {0},
            {2, 3, 0},
            {1},
            {1, 2},
            {1, 2},
            {1, 3},
            {2},
            {2},
            {3},
            {3},
        ],
        with_additional_voter=True,
        mod_approval_set={0, 3},
        committees_first=[{0, 2, 3}],
        committees_after=[{0, 1, 2}],
    ),
    MonotonicityInstance(
        rule_id="greedy-monroe",
        committeesize=3,
        approval_sets=[
            {1, 2, 3},
            {0, 2, 5},
            {0, 3, 4},
            {2, 4},
            {0, 1},
            {3, 5},
            {3, 5},
            {1, 4},
            {1, 5},
        ],
        with_additional_voter=True,
        mod_approval_set={4},
        committees_first=[{1, 4, 5}],
        committees_after=[{1, 2, 3}],
    ),
    MonotonicityInstance(
        rule_id="rule-x",
        committeesize=3,
        approval_sets=[
            {1, 3},
            {0, 1},
            {1, 3, 4},
            {0, 4},
            {2, 3, 4},
            {2, 4},
            {2, 3, 4},
            {0, 2, 4},
            {1, 2, 3},
        ],
        with_additional_voter=True,
        mod_approval_set={0},
        committees_first=[{0, 3, 4}],
        committees_after=[{1, 2, 4}],
    ),
    MonotonicityInstance(
        rule_id="rule-x-without-phragmen-phase",
        committeesize=4,
        approval_sets=[
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
        with_additional_voter=True,
        mod_approval_set={4},
        committees_first=[{0, 1, 4}],
        committees_after=[{0, 3, 7}],
    ),
    MonotonicityInstance(
        # from Sanchez-Fernandez and Fisteus (2019)
        rule_id="monroe",
        committeesize=4,
        approval_sets=[{a, e}] * 5
        + [{a, g}] * 4
        + [{b, e}] * 5
        + [{b, h}] * 4
        + [{c, f}] * 5
        + [{c, g}] * 4
        + [{d, f}] * 3
        + [{d, h}] * 3,
        with_additional_voter=True,
        mod_approval_set={e},
        committees_first=[{e, f, g, h}],
        committees_after=[{a, b, c, d}, {e, f, g, h}],
    ),
    MonotonicityInstance(
        # from Sanchez-Fernandez and Fisteus (2019)
        rule_id="pav",
        committeesize=4,
        approval_sets=[{0, 1}, {0, 1, 2}, {4, 5}, {4, 5}]
        + [{0, 4}, {1, 4}, {2, 4}, {0, 5}, {1, 5}, {2, 5}, {0, 6}, {1, 6}, {2, 6}] * 3
        + [{3}] * 100,
        with_additional_voter=False,
        mod_approval_set={2, 3},
        committees_first=[{0, 1, 2, 3}],
        committees_after=[{3, 4, 5, 6}],
    ),
    MonotonicityInstance(
        # from Sanchez-Fernandez and Fisteus (2019)
        rule_id="cc",
        committeesize=3,
        approval_sets=[{a}, {a, d}, {a, e}, {c, d}, {c, e}, {b}] * 2 + [{d}],
        with_additional_voter=False,
        mod_approval_set={b, c},
        committees_first=[{a, b, c}],
        committees_after=[{a, b, c}, {b, d, e}],
    ),
    MonotonicityInstance(
        # from Sanchez-Fernandez and Fisteus (2019)
        rule_id="monroe",
        committeesize=3,
        approval_sets=[{a}, {a, d}, {a, e}] * 2 + [{b}, {c, d}] * 4 + [{b, e}] + [{c, e}] * 3,
        with_additional_voter=False,
        mod_approval_set={b, c},
        committees_first=[{a, b, c}],
        committees_after=[{a, b, c}, {b, d, e}],
    ),
    MonotonicityInstance(  # from Sanchez-Fernandez and Fisteus (2019)
        rule_id="minimaxphragmen",
        committeesize=6,
        approval_sets=[{7}] + [{1, 2, 3, 4, 5}] * 13 + [{0, 6}, {0}] * 2 + [{6}],
        with_additional_voter=False,
        mod_approval_set={0, 1, 2, 3, 4, 5},
        committees_first=[{0, 1, 2, 3, 4, 5}, {1, 2, 3, 4, 5, 6}],
        committees_after=[
            {0, 1, 2, 3, 4, 6},
            {0, 1, 2, 3, 5, 6},
            {0, 1, 2, 4, 5, 6},
            {0, 1, 3, 4, 5, 6},
            {0, 2, 3, 4, 5, 6},
        ],
    ),
    MonotonicityInstance(  # from Sanchez-Fernandez and Fisteus (2019)
        rule_id="minimaxav",
        committeesize=5,
        approval_sets=[{5}, {1}, {2}, {3}, {1, 4, 6}, {1, 4, 7}, {1, 5, 6}, {1, 5, 7}, {1, 6}],
        with_additional_voter=False,
        mod_approval_set={1, 2, 3, 4},
        committees_first=[{1, 2, 3, 4, 5}],
        committees_after=[{1, 2, 3, 6, 7}],
    ),
    MonotonicityInstance(  # from Xavier Mora, Maria Oliver (2015)
        rule_id="seqphragmen",
        committeesize=3,
        approval_sets=[{2}] * 7 + [{0, 1}] * 4 + [{0, 1, 2}] + [{0, 1, 3}] * 16 + [{2, 3}] * 4,
        with_additional_voter=False,
        mod_approval_set={0, 1},
        committees_first=[{0, 1, 2}],
        committees_after=[{0, 2, 3}],
    ),
    MonotonicityInstance(
        rule_id="seqpav",
        committeesize=3,
        approval_sets=[{2, 3}, {0, 2}, {0, 3}, {0, 5}, {1, 2}, {1, 5}, {1, 5}, {2, 4}],
        with_additional_voter=False,
        mod_approval_set={0, 5},
        committees_first=[{0, 2, 5}],
        committees_after=[{0, 1, 2}],
    ),
    MonotonicityInstance(
        rule_id="revseqpav",
        committeesize=3,
        approval_sets=[
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
        with_additional_voter=False,
        mod_approval_set={2, 3},
        committees_first=[{2, 3, 4}],
        committees_after=[{1, 3, 4}],
    ),
    MonotonicityInstance(
        rule_id="seqcc",
        committeesize=3,
        approval_sets=[
            {4},
            {0},
            {0, 3},
            {1},
            {1},
            {1},
            {0, 2},
            {0, 2},
            {1, 2, 3},
            {2},
            {2},
            {3},
            {3},
        ],
        with_additional_voter=False,
        mod_approval_set={1, 3},
        committees_first=[{1, 2, 3}],
        committees_after=[{0, 1, 2}],
    ),
    MonotonicityInstance(
        rule_id="rule-x",
        committeesize=3,
        approval_sets=[{1}, {0, 1, 4}, {1, 4}, {2}, {1, 4}, {0, 2}, {0}],
        with_additional_voter=False,
        mod_approval_set={0, 4},
        committees_first=[{0, 1, 4}],
        committees_after=[{0, 1, 2}],
    ),
    MonotonicityInstance(
        rule_id="greedy-monroe",
        committeesize=2,
        approval_sets=[{3}, {2}, {1}, {0, 2}],
        with_additional_voter=False,
        mod_approval_set={1, 2},
        committees_first=[{1, 2}],
        committees_after=[{0, 2}],
    ),
]


for inst in monotonicity_instances:
    print(misc.header(abcrules.get_rule(inst.rule_id).longname, "-"))

    profile = Profile(num_cand, cand_names=cand_names)
    profile.add_voters(inst.approval_sets)
    original_approval_set = set(inst.approval_sets[0])
    print(profile.str_compact())

    committees = abcrules.compute(inst.rule_id, profile, inst.committeesize)
    print("original winning committees:\n" + misc.str_sets_of_candidates(committees, cand_names))
    # verify correctness
    assert (
        committees == inst.committees_first
    ), f"({inst.rule_id}) {committees} != {inst.committees_first}"
    some_variant = any(
        all(cand in committee for cand in inst.mod_approval_set)
        for committee in inst.committees_first
    )
    all_variant = all(
        all(cand in committee for cand in inst.mod_approval_set)
        for committee in inst.committees_first
    )
    assert some_variant or all_variant
    if all_variant:
        assert not all(
            all(cand in committee for cand in inst.mod_approval_set)
            for committee in inst.committees_after
        )
    else:
        assert not any(
            all(cand in committee for cand in inst.mod_approval_set)
            for committee in inst.committees_after
        )

    if inst.with_additional_voter:
        print("additional voter: " + misc.str_set_of_candidates(inst.mod_approval_set, cand_names))
        new_approval_set = inst.mod_approval_set
        profile.add_voter(new_approval_set)
    else:
        new_approval_set = list(set(inst.mod_approval_set) | set(inst.approval_sets[0]))
        print(
            "change of voter 0: "
            + misc.str_set_of_candidates(list(original_approval_set), cand_names)
            + " --> "
            + misc.str_set_of_candidates(new_approval_set, cand_names)
        )
        profile[0] = new_approval_set

    committees = abcrules.compute(inst.rule_id, profile, inst.committeesize)
    print(
        "\nwinning committees after the modification:\n"
        + misc.str_sets_of_candidates(committees, cand_names)
    )

    # verify correctness
    assert (
        committees == inst.committees_after
    ), f"({inst.rule_id}) {committees} != {inst.committees_after}"

    print(f"{abcrules.get_rule(inst.rule_id).shortname} fails ", end="")
    if inst.with_additional_voter:
        if len(inst.mod_approval_set) == 1:
            print("candidate", end="")
        else:
            print("support", end="")
        print(" monotonicity with additional voters")
    else:
        if len(inst.mod_approval_set) == 1:
            print("candidate", end="")
        else:
            print("support", end="")
        print(" monotonicity without additional voters")
    print()
