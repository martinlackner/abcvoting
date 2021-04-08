"""
Approval-based committee (ABC) rules implemented as (mixed) integer linear programs (ILPs)
 with Python MIP.
"""

import mip
from abcvoting.misc import sorted_committees
from abcvoting.output import output, DEBUG


ACCURACY = 1e-9


def _optimize_rule_mip(set_opt_model_func, profile, committeesize, scorefct, resolute, solver_id):
    """Compute rules, which are given in the form of an optimization problem, using Python MIP.

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

    if solver_id not in ["gurobi", "cbc"]:
        raise ValueError(f"Solver {solver_id} not known in Python MIP.")

    # TODO add a max iterations parameter with fancy default value which works in almost all
    #  cases to avoid endless hanging computations, e.g. when CI runs the tests
    while True:
        model = mip.Model(solver_name=solver_id)

        # note: verbose = 1 causes issues with unittests, seems as if output is printed too late
        # and anyway the output does not seem to be very helpful
        model.verbose = 0

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

        # emphasis is optimality:
        # activates procedures that produce improved lower bounds, focusing in pruning the search
        # tree even if the production of the first feasible solutions is delayed.
        model.emphasis = 2
        model.opt_tol = ACCURACY
        model.integer_tol = ACCURACY

        status = model.optimize()

        if status not in [mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.INFEASIBLE]:
            raise RuntimeError(
                f"Python MIP returned an unexpected status code: {status}"
                "Warning: solutions may be incomplete or not optimal."
            )
        elif status == mip.OptimizationStatus.INFEASIBLE:
            if len(committees) == 0:
                # we are in the first round of searching for committees
                # and Gurobi didn't find any
                raise RuntimeError("Python MIP found no solution (INFEASIBLE)")
            break
        objective_value = model.objective_value

        if maxscore is None:
            maxscore = objective_value
        elif objective_value > maxscore + ACCURACY:
            raise RuntimeError(
                "Python MIP found a solution better than a previous optimum. This "
                f"should not happen (previous optimal score: {maxscore}, "
                f"new optimal score: {objective_value})."
            )
        elif objective_value < maxscore - ACCURACY:
            # no longer optimal
            break

        committee = set(
            cand for cand in profile.candidates if in_committee[cand].x >= 1 - ACCURACY
        )
        if len(committee) != committeesize:
            raise RuntimeError(
                "_optimize_rule_mip produced a committee with "
                "fewer than `committeesize` members."
            )
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
                utility[(voter, l)] = model.add_var(var_type=mip.BINARY, name=f"utility-v{i}-l{l}")
                # TODO: could be faster with lb=0.0, ub=1.0, var_type=mip.CONTINUOUS

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


def _mip_monroe(profile, committeesize, resolute, solver_id):
    def set_opt_model_func(
        model, profile, in_committee, committeesize, previously_found_committees, scorefct
    ):
        num_voters = len(profile)

        # optimization goal: variable "satisfaction"
        satisfaction = model.add_var(ub=num_voters, var_type=mip.INTEGER, name="satisfaction")

        model += mip.xsum(in_committee[cand] for cand in profile.candidates) == committeesize

        # a partition of voters into `committeesize` many sets
        partition = {}
        for cand in profile.candidates:
            for voter in range(len(profile)):
                partition[(cand, voter)] = model.add_var(var_type=mip.BINARY, name="partition")
        for voter in range(len(profile)):
            # every voter has to be part of a voter partition set
            model += mip.xsum(partition[(cand, voter)] for cand in profile.candidates) == 1
        for cand in profile.candidates:
            # every voter set in the partition has to contain
            # at least (num_voters // committeesize) candidates
            model += mip.xsum(partition[(cand, voter)] for voter in range(len(profile))) >= (
                num_voters // committeesize - num_voters * (1 - in_committee[cand])
            )
            # every voter set in the partition has to contain
            # at most ceil(num_voters/committeesize) candidates
            model += mip.xsum(partition[(cand, voter)] for voter in range(len(profile))) <= (
                num_voters // committeesize
                + bool(num_voters % committeesize)
                + num_voters * (1 - in_committee[cand])
            )
            # if in_committee[i] = 0 then partition[(i,j) = 0
            model += (
                mip.xsum(partition[(cand, voter)] for voter in range(len(profile)))
                <= num_voters * in_committee[cand]
            )

        # constraint for objective variable "satisfaction"
        model += (
            mip.xsum(
                partition[(cand, voter)] * (cand in profile[voter].approved)
                for voter in range(len(profile))
                for cand in profile.candidates
            )
            >= satisfaction
        )

        # find a new committee that has not been found before
        for committee in previously_found_committees:
            model += mip.xsum(in_committee[cand] for cand in committee) <= committeesize - 1

        # optimization objective
        model.objective = mip.maximize(satisfaction)

    committees = _optimize_rule_mip(
        set_opt_model_func,
        profile,
        committeesize,
        scorefct=None,
        resolute=resolute,
        solver_id=solver_id,
    )
    return sorted_committees(committees)


def _mip_minimaxphragmen(profile, committeesize, resolute, solver_id):
    """ILP for Phragmen's minimax rule (minimax-Phragmen), using Python MIP.

    Minimizes the maximum load.

    Warning: does not include the lexicographic optimization as specified
    in Markus Brill, Rupert Freeman, Svante Janson and Martin Lackner.
    Phragmen's Voting Methods and Justified Representation.
    https://arxiv.org/abs/2102.12305
    Instead: minimizes the maximum load (without consideration of the
             second-, third-, ...-largest load
    """

    def set_opt_model_func(
        model, profile, in_committee, committeesize, previously_found_committees, scorefct
    ):
        load = {}
        for cand in profile.candidates:
            for voter in profile:
                load[(voter, cand)] = model.add_var(lb=0.0, ub=1.0, var_type=mip.CONTINUOUS)

        # constraint: the committee has the required size
        model += mip.xsum(in_committee[cand] for cand in profile.candidates) == committeesize

        for cand in profile.candidates:
            for voter in profile:
                if cand not in voter.approved:
                    load[(voter, cand)] = 0

        # a candidate's load is distributed among his approvers
        for cand in profile.candidates:
            model += (
                mip.xsum(
                    voter.weight * load[(voter, cand)]
                    for voter in profile
                    if cand in profile.candidates
                )
                >= in_committee[cand]
            )

        # find a new committee that has not been found before
        for committee in previously_found_committees:
            model += mip.xsum(in_committee[cand] for cand in committee) <= committeesize - 1

        loadbound = model.add_var(
            lb=0, ub=committeesize, var_type=mip.CONTINUOUS, name="loadbound"
        )
        for voter in profile:
            model += mip.xsum(load[(voter, cand)] for cand in voter.approved) <= loadbound

        # maximizing the negative distance makes code more similar to the other methods here
        model.objective = mip.maximize(-loadbound)

    committees = _optimize_rule_mip(
        set_opt_model_func,
        profile,
        committeesize,
        scorefct=None,
        resolute=resolute,
        solver_id=solver_id,
    )
    return sorted_committees(committees)


def _mip_minimaxav(profile, committeesize, resolute, solver_id):
    def set_opt_model_func(
        model, profile, in_committee, committeesize, previously_found_committees, scorefct
    ):
        max_hamming_distance = model.add_var(
            var_type=mip.INTEGER,
            lb=0,
            ub=profile.num_cand,
            name="max_hamming_distance",
        )

        model += mip.xsum(in_committee[cand] for cand in profile.candidates) == committeesize

        for voter in profile:
            not_approved = [cand for cand in profile.candidates if cand not in voter.approved]
            # maximum Hamming distance is greater of equal than the Hamming distances
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
