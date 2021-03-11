"""
Approval-based committee (ABC) rules implemented as a integer linear
programs (ILPs) with CVXPY.
"""

from abcvoting.misc import sorted_committees

try:
    import cvxpy as cp

    cvxpy_available = True
except ImportError:
    cvxpy_available = False

try:
    import numpy as np

    numpy_available = True
except ImportError:
    numpy_available = False


CVXPY_ACCURACY = 1e-7


def cvxpy_thiele_methods(profile, committeesize, scorefct_id, resolute, solver_id):
    """Compute thiele method using CVXPY. This is similar to `_gurobi_thiele_methods()`,
    where `gurobipy` is used as interface to Gurobi. This method supports Gurobi too, but also
    other solvers.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        approval sets of voters
    committeesize : int
        number of chosen alternatives
    scorefct_id : str
        must be one of: 'pav'
    resolute : bool
        return only one result
    solver_id : str
        must be one of: 'glpk_mi', 'cbc', 'scip', 'cvxpy_gurobi'
        'cvxpy_gurobi' uses Gurobi in the background, similar to
        `abcrules_gurobi._gurobi_thiele_methods()`, but using the CVXPY interface instead of
        gurobipy.

    Returns
    -------
    committees : list of lists
        a list of chosen committees, each of them represented as list with candidates named from
        `0` to `num_cand`, profile.cand_names is ignored

    """
    if solver_id in ["glpk_mi", "cbc", "scip"]:
        solver = getattr(cp, solver_id.upper())
    elif solver_id == "gurobi":
        solver = cp.GUROBI
    else:
        raise ValueError(f"Unknown solver_id for usage with CVXPY: {solver_id}")

    committees = []
    maxscore = None

    # TODO should we use functions for abcvoting.scores? Does it make it slower?
    if scorefct_id == "pav":
        scorefct_value = np.tile(1 / np.arange(1, committeesize + 1), (len(profile), 1))
    elif scorefct_id == "av":
        raise ValueError("scorefct must be monotonic decreasing")
    else:
        raise NotImplementedError(f"invalid scorefct_id: {scorefct_id}")

    # TODO does this make things slower in case of weights == 1 to multiply weights? We could
    #  skip it then of course...
    weights1d = np.array([voter.weight for voter in profile])
    # for some reason CVXPY doesn't like the broadcasting, so we need a 2d array
    weights = np.tile(weights1d[np.newaxis].T, (1, committeesize))

    while True:
        in_committee = cp.Variable(profile.num_cand, boolean=True)

        # utility[i, j] indicates whether voter i approves at least j candidates in the
        # committee, i.e. in row i the first l values are true if i approves l candidates in the
        # committee and all other values are false.
        # explicitly setting boolean=True is not necessary, can be skipped and is then implicit
        # also true as done in abcrules_gurobi._gurobi_thiele_methods()
        utility = cp.Variable((len(profile), committeesize), boolean=True)

        # left-hand-side and right-hand-side of the equality constraints:
        lhs = cp.sum(utility, axis=1)
        rhs = cp.hstack(
            [cp.sum([in_committee[cand] for cand in voter.approved]) for voter in profile]
        )

        constraints = [cp.sum(in_committee) == committeesize, lhs == rhs]

        if solver_id == "glpk_mi":
            # weird workaround necessary... :(
            # see https://github.com/cvxgrp/cvxpy/issues/1112#issuecomment-730360543
            constraints = [
                cp.sum(in_committee) <= committeesize,
                cp.sum(in_committee) >= committeesize,
                lhs <= rhs,
                lhs >= rhs,
            ]

        # find a new committee that has not been found yet by excluding previously found committees
        for committee in committees:
            constraints.append(cp.sum(in_committee[committee]) <= committeesize - 1)

        # I don't really understand why, but the * does not seem to be supported here...
        score = cp.sum(cp.multiply(cp.multiply(scorefct_value, weights), utility))

        objective = cp.Maximize(score)

        problem = cp.Problem(objective, constraints)

        cvxpy_workaround_infisible = False
        try:
            problem.solve(solver=solver)
        except KeyError:
            # TODO this is a workaround for https://github.com/cvxgrp/cvxpy/issues/1191
            cvxpy_workaround_infisible = True

        if problem.status in (cp.INFEASIBLE, cp.UNBOUNDED) or cvxpy_workaround_infisible:
            if len(committees) == 0:
                raise RuntimeError("no solutions found")
            break
        elif problem.status != cp.OPTIMAL:
            raise RuntimeError(
                f"Solver returned status {problem.status}. At the moment abcvoting "
                "can't handle this error."
            )

        if maxscore is None:
            maxscore = problem.value

        # TODO is there a way to find accuracy for all solvers?
        # 1e-7 is based on CVXOPT's default value:
        # https://www.cvxpy.org/tutorial/advanced/index.html
        # We might miss committees if the value is too high or get wrong solutions if it's too low.
        if maxscore - problem.value > CVXPY_ACCURACY:
            # no longer optimal
            break

        committee = np.arange(profile.num_cand)[in_committee.value.astype(np.bool)]

        committees.append(committee.tolist())

        if resolute:
            break

    return sorted_committees(committees)
