"""
Approval-based committee (ABC) rules implemented as constraint
 satisfaction programs with OR-Tools.
"""

from ortools.sat.python import cp_model
from abcvoting.misc import sorted_committees
from abcvoting import scores
import functools


def _optimize_rule_ortools(
    set_opt_model_func,
    profile,
    committeesize,
    resolute,
    max_num_of_committees,
    name,
    committeescorefct,
):
    """Compute ABC rules, which are given in the form of an integer optimization problem,
    using the OR-Tools CP-SAT Solver.

    Parameters
    ----------
    set_opt_model_func : callable
        sets constraints and objective and adds additional variables, see examples below for its
        signature
    profile : abcvoting.preferences.Profile
        approval sets of voters
    committeesize : int
        number of chosen alternatives
    resolute : bool
    max_num_of_committees : int
        maximum number of committees this method returns, value can be None
    name : str
        name of the model, used for error messages
    committeescorefct : callable
        a function used to compute the score of a committee

    Returns
    -------
    committees : list of set
        a list of winning committees,
        each of them represented as set of integers from `0` to `num_cand` - 1

    """

    maxscore = None
    committees = []

    model = cp_model.CpModel()

    # `in_committee` is a binary variable indicating whether `cand` is in the committee
    in_committee = [model.NewBoolVar(f"cand{cand}_in_committee") for cand in profile.candidates]

    set_opt_model_func(
        model,
        profile,
        in_committee,
        committeesize,
    )

    while True:

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status not in [cp_model.OPTIMAL, cp_model.INFEASIBLE]:
            raise RuntimeError(
                f"OR-Tools returned an unexpected status code: {status}"
                f"Warning: solutions may be incomplete or not optimal (model {name})."
            )
        elif status == cp_model.INFEASIBLE:
            if len(committees) == 0:
                # we are in the first round of searching for committees
                # and Gurobi didn't find any
                raise RuntimeError("OR-Tools found no solution (INFEASIBLE)  (model {name})")
            break

        committee = set(
            cand for cand in profile.candidates if solver.Value(in_committee[cand]) >= 1
        )
        if len(committee) != committeesize:
            raise RuntimeError(
                f"_optimize_rule_ortools produced a committee with "
                f"fewer than `committeesize` members (model {name})."
            )

        objective_value = committeescorefct(profile, committee)  # exact value

        if maxscore is None:
            maxscore = objective_value
        elif objective_value > maxscore:
            raise RuntimeError(
                "OR-Tools found a solution better than a previous optimum. This "
                f"should not happen (previous optimal score: {maxscore}, "
                f"new optimal score: {objective_value}, model {name}))."
            )
        elif objective_value < maxscore:
            # no longer optimal
            break

        committees.append(committee)

        if resolute:
            break
        if max_num_of_committees is not None and len(committees) >= max_num_of_committees:
            return committees

        # find a new committee that has not been found before
        model.Add(sum(in_committee[cand] for cand in committee) <= committeesize - 1)

    return committees


def _ortools_cc(profile, committeesize, resolute, max_num_of_committees):
    def set_opt_model_func(
        model,
        profile,
        in_committee,
        committeesize,
    ):
        num_voters = len(profile)
        satisfaction = [
            model.NewBoolVar(name=f"satisfaction-of-{voter_id}") for voter_id in range(num_voters)
        ]

        model.Add(sum(in_committee[cand] for cand in profile.candidates) == committeesize)

        for voter_id in range(num_voters):
            model.Add(
                satisfaction[voter_id]
                <= sum(in_committee[cand] for cand in profile[voter_id].approved)
            )
            # satisfaction is boolean

        # maximizing the negative distance makes code more similar to the other methods here
        if profile.has_unit_weights():
            model.Maximize(sum(satisfaction[voter_id] for voter_id in range(num_voters)))
        else:
            model.Maximize(
                sum(
                    satisfaction[voter_id] * profile[voter_id].weight
                    for voter_id in range(num_voters)
                )
            )

    if not all(isinstance(voter.weight, int) for voter in profile):
        raise TypeError(
            f"_ortools_cc requires integer weights "
            f"(encountered {[voter.weight for voter in profile]}."
        )

    committees = _optimize_rule_ortools(
        set_opt_model_func,
        profile,
        committeesize,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
        name="CC",
        committeescorefct=functools.partial(scores.thiele_score, "cc"),
    )
    return sorted_committees(committees)


def _ortools_lexcc(profile, committeesize, resolute, max_num_of_committees):
    pass
    # TODO: write


def _ortools_monroe(profile, committeesize, resolute, max_num_of_committees):
    def set_opt_model_func(
        model,
        profile,
        in_committee,
        committeesize,
    ):
        num_voters = len(profile)

        # optimization goal: variable "satisfaction"
        satisfaction = model.NewIntVar(lb=0, ub=num_voters, name="satisfaction")

        model.Add(sum(in_committee[cand] for cand in profile.candidates) == committeesize)

        # a partition of voters into committeesize many sets
        partition = {
            (cand, voter): model.NewBoolVar(name=f"partition{cand}-{voter}")
            for cand in profile.candidates
            for voter in range(num_voters)
        }

        for i in range(len(profile)):
            # every voter has to be part of a voter partition set
            model.Add(sum(partition[(cand, i)] for cand in profile.candidates) == 1)
        for cand in profile.candidates:
            # every voter set in the partition has to contain
            # at least (num_voters // committeesize) candidates
            model.Add(
                sum(partition[(cand, j)] for j in range(len(profile)))
                >= (num_voters // committeesize - num_voters * (1 - in_committee[cand]))
            )
            # every voter set in the partition has to contain
            # at most ceil(num_voters/committeesize) candidates
            model.Add(
                sum(partition[(cand, j)] for j in range(len(profile)))
                <= (
                    num_voters // committeesize
                    + bool(num_voters % committeesize)
                    + num_voters * (1 - in_committee[cand])
                )
            )
            # if in_committee[i] = 0 then partition[(i,j) = 0
            model.Add(
                sum(partition[(cand, j)] for j in range(len(profile)))
                <= num_voters * in_committee[cand]
            )

        # constraint for objective variable "satisfaction"
        model.Add(
            sum(
                partition[(cand, j)] * (cand in profile[j].approved)
                for j in range(len(profile))
                for cand in profile.candidates
            )
            >= satisfaction
        )

        # optimization objective
        model.Maximize(satisfaction)

    committees = _optimize_rule_ortools(
        set_opt_model_func,
        profile,
        committeesize,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
        name="Monroe",
        committeescorefct=scores.monroescore,
    )
    return sorted_committees(committees)


def _ortools_minimaxav(profile, committeesize, resolute, max_num_of_committees):
    def set_opt_model_func(
        model,
        profile,
        in_committee,
        committeesize,
    ):
        max_hamming_distance = model.NewIntVar(
            lb=0, ub=profile.num_cand, name="max_hamming_distance"
        )
        model.Add(sum(in_committee[cand] for cand in profile.candidates) == committeesize)

        for voter in profile:
            not_approved = [cand for cand in profile.candidates if cand not in voter.approved]
            # maximum Hamming distance is greater of equal than the Hamming distances
            # between individual voters and the committee
            model.Add(
                max_hamming_distance
                >= sum(1 - in_committee[cand] for cand in voter.approved)
                + sum(in_committee[cand] for cand in not_approved)
            )

        # maximizing the negative distance makes code more similar to the other methods here
        model.Maximize(-max_hamming_distance)

    committees = _optimize_rule_ortools(
        set_opt_model_func,
        profile,
        committeesize,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
        name="Minimax AV",
        committeescorefct=lambda profile, committee: -scores.minimaxav_score(profile, committee),
        # negative because _optimize_rule_mip maximizes while minimaxav minimizes
    )
    return sorted_committees(committees)
