"""
Approval-based committee (ABC) rules implemented as constraint
 satisfaction programs with OR-Tools.
"""

from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
from abcvoting.misc import sorted_committees


def _optimize_rule_ortools(
    set_opt_model_func, profile, committeesize, scorefct, resolute, solver_id
):
    """Compute rules, which are given in the form of an optimization problem, using Gurobi.

    Parameters
    ----------
    set_opt_model_func : callable
        sets constraints and objective and adds additional variables, see examples below for its
        signature
    profile : abcvoting.preferences.Profile
        approval sets of voters
    committeesize : int
        number of chosen alternatives
    scorefct : callable
    resolute : bool
    solver_id : str

    Returns
    -------
    committees : list of sets
        a list of chosen committees, each of them represented as list with candidates named from
        `0` to `num_cand`, profile.cand_names is ignored

    """

    maxscore = None
    committees = []

    if solver_id == "cp":
        cp_formulation = True
    elif solver_id in ["gurobi", "scip", "glpk_mi", "cbc"]:
        cp_formulation = False
    else:
        raise ValueError(f"Solver {solver_id} not known in OR Tools.")

    if solver_id == "glpk_mi":
        solver_id = "glpk_mip"

    # TODO add a max iterations parameter with fancy default value which works in almost all
    #  cases to avoid endless hanging computations, e.g. when CI runs the tests
    while True:
        if cp_formulation:
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
                scorefct,
                cp_formulation=cp_formulation,
            )

            solver = cp_model.CpSolver()
            status = solver.Solve(model)

            if status not in [cp_model.OPTIMAL, cp_model.INFEASIBLE]:
                raise RuntimeError(
                    f"OR Tools returned an unexpected status code: {status}"
                    "Warning: solutions may be incomplete or not optimal."
                )
            elif status == cp_model.INFEASIBLE:
                if len(committees) == 0:
                    # we are in the first round of searching for committees
                    # and Gurobi didn't find any
                    raise RuntimeError("OR Tools found no solution (INFEASIBLE)")
                break
            objective_value = solver.ObjectiveValue()
        else:
            solver = pywraplp.Solver.CreateSolver(solver_id)

            # `in_committee` is a binary variable indicating whether `cand` is in the committee
            in_committee = [
                solver.BoolVar(f"cand{cand}_in_committee") for cand in profile.candidates
            ]

            set_opt_model_func(
                solver,
                profile,
                in_committee,
                committeesize,
                committees,
                scorefct,
                cp_formulation=cp_formulation,
            )

            status = solver.Solve()

            if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.INFEASIBLE]:
                raise RuntimeError(
                    f"OR Tools returned an unexpected status code: {status}"
                    "Warning: solutions may be incomplete or not optimal."
                )
            elif status == pywraplp.Solver.INFEASIBLE:
                if len(committees) == 0:
                    # we are in the first round of searching for committees
                    # and Gurobi didn't find any
                    raise RuntimeError("OR Tools found no solution (INFEASIBLE)")
                break
            objective_value = solver.Objective().Value()

        if maxscore is None:
            maxscore = objective_value
        elif objective_value > maxscore:
            raise RuntimeError(
                "OR Tools found a solution better than a previous optimum. This "
                f"should not happen (previous optimal score: {maxscore}, "
                f"new optimal score: {objective_value})."
            )
        elif objective_value < maxscore:
            # no longer optimal
            break

        if cp_formulation:
            committee = set(
                cand for cand in profile.candidates if solver.Value(in_committee[cand]) >= 1
            )
        else:
            committee = set(
                cand for cand in profile.candidates if in_committee[cand].solution_value() >= 1
            )
        assert len(committee) == committeesize
        committees.append(committee)

        if resolute:
            break

    return committees


def __ortools_minimaxav(profile, committeesize, resolute, solver_id):
    def set_opt_model_func(
        model,
        profile,
        in_committee,
        committeesize,
        previously_found_committees,
        scorefct,
        cp_formulation=True,
    ):

        if cp_formulation:
            max_hamming_distance = model.NewIntVar(
                lb=0, ub=2 * committeesize, name="max_hamming_distance"
            )
        else:
            max_hamming_distance = model.IntVar(
                lb=0, ub=2 * committeesize, name="max_hamming_distance"
            )
        model.Add(sum(in_committee[cand] for cand in profile.candidates) == committeesize)

        for voter in profile:
            not_approved = [cand for cand in profile.candidates if cand not in voter.approved]
            # maximum hamming distance is greater of equal than the Hamming distances
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
        scorefct=None,
        resolute=resolute,
        solver_id=solver_id,
    )
    return sorted_committees(committees)
