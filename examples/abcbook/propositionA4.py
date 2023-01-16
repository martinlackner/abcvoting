"""
Proposition A.4.

From "Multi-Winner Voting with Approval Preferences"
by Martin Lackner and Piotr Skowron
http://dx.doi.org/10.1007/978-3-031-09016-5
"""

from abcvoting import abcrules
from abcvoting.preferences import Profile
from abcvoting import misc
from collections import namedtuple


print(misc.header("Proposition A.4", "*"))

num_cand = 6
a, b, c, d, e, f = range(6)  # a = 0, b = 1, c = 2, ...
cand_names = "abcdef"

ManipulationInstance = namedtuple(
    "ManipulationInstance",
    "rule_id committeesize approval_sets manipulated_vote committees_first committees_after",
)

manipulations = [
    ManipulationInstance(
        rule_id="cc",
        committeesize=2,
        approval_sets=[{a, b}] + [{a}] * 3 + [{c}],
        manipulated_vote={b},
        committees_first=[{a, c}],
        committees_after=[{a, b}],
    ),
    ManipulationInstance(
        rule_id="pav",
        committeesize=3,
        approval_sets=[{2, 3, 4}, {0, 1}, {1, 5}, {0, 2, 3}, {1, 2, 5}, {2, 4, 5}],
        manipulated_vote={4},
        committees_first=[{1, 2, 5}],
        committees_after=[{1, 2, 4}],
    ),
    ManipulationInstance(
        rule_id="seqpav",
        committeesize=3,
        approval_sets=[{0, 1}, {1, 3}, {2, 5}, {0, 1, 5}, {1, 5}, {1, 2}],
        manipulated_vote={0},
        committees_first=[{1, 2, 5}],
        committees_after=[{0, 1, 5}],
    ),
    ManipulationInstance(
        rule_id="seqcc",
        committeesize=3,
        approval_sets=[
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
        manipulated_vote={2},
        committees_first=[{0, 1, 3}],
        committees_after=[{1, 2, 4}],
    ),
    ManipulationInstance(
        rule_id="revseqpav",
        committeesize=2,
        approval_sets=[{a, b, c}, {b, d}, {c, b}, {a, d, e}, {b, e}],
        manipulated_vote={a},
        committees_first=[{b, d}],
        committees_after=[{a, b}],
    ),
    ManipulationInstance(
        rule_id="monroe",
        committeesize=3,
        approval_sets=[
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
        manipulated_vote={5},
        committees_first=[{0, 1, 4}],
        committees_after=[{1, 3, 5}],
    ),
    ManipulationInstance(
        rule_id="greedy-monroe",
        committeesize=2,
        approval_sets=[{0, 1}, {0, 2, 5}, {0, 2, 3}, {4, 5}],
        manipulated_vote={1},
        committees_first=[{0, 2}],
        committees_after=[{0, 1}],
    ),
    ManipulationInstance(
        rule_id="seqphragmen",
        committeesize=2,
        approval_sets=[{0, 1, 2}, {0, 1}, {1, 5}, {2, 4}, {1, 4, 5}, {1, 3, 5}],
        manipulated_vote={2},
        committees_first=[{1, 5}],
        committees_after=[{1, 2}],
    ),
    ManipulationInstance(
        rule_id="leximaxphragmen",
        committeesize=3,
        approval_sets=[{a, b}] + [{b, c, d}] + [{b, c, d}] + [{b, c, d}],
        manipulated_vote={a},
        committees_first=[{b, c, d}],
        committees_after=[{a, b, c}],
    ),
    ManipulationInstance(
        rule_id="equal-shares",
        committeesize=3,
        approval_sets=[{1, 2, 3}, {0, 1}, {1, 3}, {2, 3}, {3, 4}, {3, 4}],
        manipulated_vote={2},
        committees_first=[{1, 3, 4}],
        committees_after=[{1, 2, 3}],
    ),
    ManipulationInstance(
        rule_id="minimaxav",
        committeesize=3,
        approval_sets=[{0, 1, 2}, {1, 3}, {0, 1, 4}, {0, 1, 3}, {0, 1, 4}, {0, 1}],
        manipulated_vote={2},
        committees_first=[{0, 1, 3}],
        committees_after=[{0, 1, 2}],
    ),
    ManipulationInstance(
        rule_id="sav",
        committeesize=1,
        approval_sets=[{a, b, c}, {d, e}],
        manipulated_vote={a},
        committees_first=[{d}],
        committees_after=[{a}],
    ),
]

for inst in manipulations:
    print(misc.header(abcrules.Rule(inst.rule_id).longname, "-"))

    profile = Profile(num_cand, cand_names=cand_names)
    profile.add_voters(inst.approval_sets)
    truepref = profile[0].approved
    print(profile.str_compact())

    parameters = {}
    if inst.rule_id == "leximaxphragmen":
        parameters["lexicographic_tiebreaking"] = True

    committees = abcrules.compute(
        inst.rule_id, profile, inst.committeesize, resolute=True, **parameters
    )
    committee1 = committees[0]
    print("original winning committee:\n " + misc.str_set_of_candidates(committee1, cand_names))

    # verify correctness
    assert (
        committee1 in inst.committees_first
    ), f"({inst.rule_id}) {committees[0]} not in {inst.committees_first}"

    print(
        "\nManipulation by voter 0: "
        + misc.str_set_of_candidates(inst.approval_sets[0], cand_names)
        + " --> "
        + misc.str_set_of_candidates(inst.manipulated_vote, cand_names)
    )
    if not all(cand in truepref for cand in inst.manipulated_vote):
        print(" (not a subset!)")

    inst.approval_sets[0] = inst.manipulated_vote
    profile = Profile(num_cand, cand_names=cand_names)
    profile.add_voters(inst.approval_sets)

    if "brute-force" in abcrules.Rule(inst.rule_id).available_algorithms:
        algorithm = "brute-force"  # correct tie-breaking between candidates
    else:
        algorithm = "fastest"
    committees = abcrules.compute(
        inst.rule_id, profile, inst.committeesize, resolute=True, algorithm=algorithm, **parameters
    )
    committee2 = committees[0]
    print(
        "\nwinning committee after manipulation:\n "
        + misc.str_set_of_candidates(committee2, cand_names)
    )

    # verify correctness
    assert (
        committee2 in inst.committees_after
    ), f"({inst.rule_id}) {committees[0]} not in {inst.committees_after}"

    # verify that this is a counterexample to inclusion-strategyproofness (resolute)
    for cand in set(committee1) & set(truepref):
        assert cand in committee2
    assert committee1 & set(truepref) < committee2 & set(truepref)

    print()
