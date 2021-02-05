"""
Approval-based committee (ABC) rules implemented as constraint
 satisfaction programs with OR-Tools.
"""

import mip
from abcvoting.misc import sorted_committees


ACCURACY = 1e-9


def _optimize_rule_mip(set_opt_model_func, profile, committeesize, scorefct, resolute, solver_id):
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

    if solver_id == "gurobi":
        solver_id = "GRB"
    elif solver_id in ["cbc"]:
        solver_id = "CBC"
    else:
        raise ValueError(f"Solver {solver_id} not known in MIP.")

    # TODO add a max iterations parameter with fancy default value which works in almost all
    #  cases to avoid endless hanging computations, e.g. when CI runs the tests
    while True:
        model = mip.Model(solver_name=solver_id)

        # `in_committee` is a binary variable indicating whether `cand` is in the committee
        in_committee = [
            model.add_var(var_type=mip.BINARY, name=f"cand{cand}_in_committee")
            for cand in profile.candidates
        ]

        set_opt_model_func(
            model,
            profile,
            in_committee,
            committeesize,
            committees,
            scorefct,
        )

        #
        # emphasis is optimality:
        # activates procedures that produce improved lower bounds, focusing in pruning the search
        # tree even if the production of the first feasible solutions is delayed.
        model.emphasis = 2
        model.max_gap = ACCURACY
        model.max_mip_gap = ACCURACY

        status = model.optimize()

        if status not in [mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.INFEASIBLE]:
            raise RuntimeError(
                f"OR Tools returned an unexpected status code: {status}"
                "Warning: solutions may be incomplete or not optimal."
            )
        elif status == mip.OptimizationStatus.INFEASIBLE:
            if len(committees) == 0:
                # we are in the first round of searching for committees
                # and Gurobi didn't find any
                raise RuntimeError("OR Tools found no solution (INFEASIBLE)")
            break
        objective_value = model.objective_value

        if maxscore is None:
            maxscore = objective_value
        elif objective_value > maxscore + ACCURACY:
            raise RuntimeError(
                "OR Tools found a solution better than a previous optimum. This "
                f"should not happen (previous optimal score: {maxscore}, "
                f"new optimal score: {objective_value})."
            )
        elif objective_value < maxscore - ACCURACY:
            # no longer optimal
            break

        committee = set(cand for cand in profile.candidates if in_committee[cand].x >= 1)
        assert len(committee) == committeesize
        committees.append(committee)

        if resolute:
            break

    return committees


def _mip_thiele_methods(profile, committeesize, scorefct, resolute, solver_id):
    def set_opt_model_func(
        model, profile, in_committee, committeesize, previously_found_committees, scorefct
    ):
        # utility[(voter, l)] contains (intended binary) variables counting the number of approved
        # candidates in the selected committee by `voter`. This utility[(voter, l)] is true for
        # exactly the number of candidates in the committee approved by `voter` for all
        # l = 1...committeesize.
        #
        # If scorefct(l) > 0 for l >= 1, we assume that scorefct is monotonic decreasing and
        # therefore in combination with the objective function the following interpreation is
        # valid:
        # utility[(voter, l)] indicates whether `voter` approves at least l candidates in the
        # committee (this is the case for scorefct "pav", "slav" or "geom").
        utility = {}

        for i, voter in enumerate(profile):
            for l in range(1, committeesize + 1):
                utility[(voter, l)] = model.add_var(lb=0.0, ub=1.0, name=f"utility-v{i}-l{l}")
                # should be binary. this is guaranteed since the objective
                # is maximal if all utilitity-values are either 0 or 1.
                # using vtype=gb.GRB.BINARY does not change result, but makes things slower a bit

        # constraint: the committee has the required size
        model += mip.xsum(in_committee) == committeesize

        # constraint: utilities are consistent with actual committee
        for voter in profile:
            model += mip.xsum(utility[voter, l] for l in range(1, committeesize + 1)) == mip.xsum(
                in_committee[cand] for cand in voter.approved
            )

        # find a new committee that has not been found yet by excluding previously found committees
        for committee in previously_found_committees:
            model += mip.xsum(in_committee[cand] for cand in committee) <= committeesize - 1

        # objective: the PAV score of the committee
        model.objective = mip.maximize(
            mip.xsum(
                float(scorefct(l)) * voter.weight * utility[(voter, l)]
                for voter in profile
                for l in range(1, committeesize + 1)
            )
        )

    score_values = [scorefct(l) for l in range(1, committeesize + 1)]
    if not all(
        first > second or first == second == 0
        for first, second in zip(score_values, score_values[1:])
    ):
        raise ValueError("scorefct must be monotonic decreasing")

    committees = _optimize_rule_mip(
        set_opt_model_func,
        profile,
        committeesize,
        scorefct=scorefct,
        resolute=resolute,
        solver_id=solver_id,
    )
    return sorted_committees(committees)


def __mip_minimaxav(profile, committeesize, resolute, solver_id):
    def set_opt_model_func(
        model, profile, in_committee, committeesize, previously_found_committees, scorefct
    ):
        max_hamming_distance = model.add_var(
            var_type=mip.INTEGER, lb=0, ub=2 * committeesize, name="max_hamming_distance"
        )

        model += mip.xsum(in_committee[cand] for cand in profile.candidates) == committeesize

        for voter in profile:
            not_approved = [cand for cand in profile.candidates if cand not in voter.approved]
            # maximum hamming distance is greater of equal than the Hamming distances
            # between individual voters and the committee
            model += max_hamming_distance >= mip.xsum(
                1 - in_committee[cand] for cand in voter.approved
            ) + mip.xsum(in_committee[cand] for cand in not_approved)

        # find a new committee that has not been found before
        for committee in previously_found_committees:
            model += mip.xsum(in_committee[cand] for cand in committee) <= committeesize - 1

        # maximizing the negative distance makes code more similar to the other methods here
        model.objective = mip.maximize(-max_hamming_distance)

    committees = _optimize_rule_mip(
        set_opt_model_func,
        profile,
        committeesize,
        scorefct=None,
        resolute=resolute,
        solver_id=solver_id,
    )
    return sorted_committees(committees)
