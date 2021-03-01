"""
Approval-based committee (ABC) rules implemented as constraint
 satisfaction programs with OR-Tools.
"""

from ortools.sat.python import cp_model
from abcvoting.misc import sorted_committees


def _optimize_rule_ortools(set_opt_model_func, profile, committeesize, resolute):
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

    Returns
    -------
    committees : list of sets
        a list of winning committees, each of them represented as set of integers

    """

    maxscore = None
    committees = []

    # TODO add a max iterations parameter with fancy default value which works in almost all
    #  cases to avoid endless hanging computations, e.g. when CI runs the tests
    while True:
        model = cp_model.CpModel()

        # `in_committee` is a binary variable indicating whether `cand` is in the committee
        in_committee = [
            model.NewBoolVar(f"cand{cand}_in_committee") for cand in profile.candidates
        ]

        set_opt_model_func(
            model,
            profile,
            in_committee,
            committeesize,
            committees,
        )

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status not in [cp_model.OPTIMAL, cp_model.INFEASIBLE]:
            raise RuntimeError(
                f"OR-Tools returned an unexpected status code: {status}"
                "Warning: solutions may be incomplete or not optimal."
            )
        elif status == cp_model.INFEASIBLE:
            if len(committees) == 0:
                # we are in the first round of searching for committees
                # and Gurobi didn't find any
                raise RuntimeError("OR-Tools found no solution (INFEASIBLE)")
            break
        objective_value = solver.ObjectiveValue()

        if maxscore is None:
            maxscore = objective_value
        elif objective_value > maxscore:
            raise RuntimeError(
                "OR-Tools found a solution better than a previous optimum. This "
                f"should not happen (previous optimal score: {maxscore}, "
                f"new optimal score: {objective_value})."
            )
        elif objective_value < maxscore:
            # no longer optimal
            break

        committee = set(
            cand for cand in profile.candidates if solver.Value(in_committee[cand]) >= 1
        )
        if len(committee) != committeesize:
            raise RuntimeError(
                "_optimize_rule_ortools produced a committee with "
                "fewer than `committeesize` members."
            )
        committees.append(committee)

        if resolute:
            break

    return committees


def _ortools_cc(profile, committeesize, resolute):
    def set_opt_model_func(
        model,
        profile,
        in_committee,
        committeesize,
        previously_found_committees,
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

        # find a new committee that has not been found before
        for committee in previously_found_committees:
            model.Add(sum(in_committee[cand] for cand in committee) <= committeesize - 1)

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
            f"_ortools_cc requires integer weights (encountered {[voter.weight for voter in profile]}."
        )

    committees = _optimize_rule_ortools(
        set_opt_model_func,
        profile,
        committeesize,
        resolute=resolute,
    )
    return sorted_committees(committees)


def _ortools_monroe(profile, committeesize, resolute):
    def set_opt_model_func(
        model,
        profile,
        in_committee,
        committeesize,
        previously_found_committees,
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

        # find a new committee that has not been found before
        for committee in previously_found_committees:
            model.Add(sum(in_committee[cand] for cand in committee) <= committeesize - 1)

        # optimization objective
        model.Maximize(satisfaction)

    committees = _optimize_rule_ortools(
        set_opt_model_func,
        profile,
        committeesize,
        resolute=resolute,
    )
    return sorted_committees(committees)


def _ortools_minimaxav(profile, committeesize, resolute):
    def set_opt_model_func(
        model,
        profile,
        in_committee,
        committeesize,
        previously_found_committees,
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

        # find a new committee that has not been found before
        for committee in previously_found_committees:
            model.Add(sum(in_committee[cand] for cand in committee) <= committeesize - 1)

        # maximizing the negative distance makes code more similar to the other methods here
        model.Maximize(-max_hamming_distance)

    committees = _optimize_rule_ortools(
        set_opt_model_func,
        profile,
        committeesize,
        resolute=resolute,
    )
    return sorted_committees(committees)
