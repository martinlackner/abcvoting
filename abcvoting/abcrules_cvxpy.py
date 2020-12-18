"""
Approval-based committee (ABC) rules implemented as a integer linear
programs (ILPs) with CVXPY.
"""

from __future__ import print_function

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


def cvxpy_thiele_methods(profile, committeesize, scorefct_str, resolute, algorithm):
    """

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        preferences of voters
    committeesize : int
        number of chosen alternatives
    scorefct_str : str
        must be one of: 'pav'
    resolute : bool
        return only one result
    algorithm : str
        must be one of: 'glpk_mi', 'cbc', 'scip', 'cvxpy_gurobi'
        'cvxpy_gurobi' uses Gurobi in the background, similar to
        `abcrules_gurobi.__gurobi_thiele_methods()`, but using the CVXPY interface instead of
        gurobipy.

    Returns
    -------
    committees : list of lists
        FIXME what should this actually return? is range(0) ok for candidates or does profile
        include names for the candidates?

    """
    if algorithm in ['glpk_mi', 'cbc', 'scip']:
        solver = getattr(cp, algorithm.upper())
    elif algorithm == 'cvxpy_gurobi':
        solver = cp.GUROBI
    else:
        raise ValueError(f"Unknown algorithm for usage with CVXPY: {algorithm}")

    committees = []
    maxscore = None

    if scorefct_str == 'pav':
        scorefct_value = np.tile(
            1 / np.arange(1, committeesize + 1),
            (len(profile), 1)
        )
    else:
        raise NotImplemented(f"invalid scorefct_str: {scorefct_str}")

    # TODO does this make things slower in case of weights == 1 to multiply weights? We could
    #  skip it then of course...
    weights1d = np.array([pref.weight for pref in profile])
    # for some reason CVXPY doesn't like the broadcasting, so we need a 2d array
    weights = np.tile(weights1d[np.newaxis].T, (1, committeesize))

    while True:
        in_committee = cp.Variable(profile.num_cand, boolean=True)

        # utility[i, j] indicates whether voter i approves at least j candidates in the
        # committee, i.e. in row i the first l values are true if i approves l candidates in the
        # committee and all other values are false.
        # explicitly setting boolean=True is not necessary, can be skipped and is then implicit
        # also true as done in abcrules_gurobi.__gurobi_thiele_methods()
        utility = cp.Variable((len(profile), committeesize), boolean=True)

        # left-hand-side and right-hand-side of the equality constraints:
        lhs = cp.sum(utility, axis=1)
        rhs = cp.hstack([cp.sum([in_committee[c] for c in pref]) for pref in profile])

        constraints = [cp.sum(in_committee) == committeesize,
                       lhs == rhs]

        if algorithm == 'glpk_mi':
            # weird workaround necessary... :(
            # see https://github.com/cvxgrp/cvxpy/issues/1112#issuecomment-730360543
            constraints = [cp.sum(in_committee) <= committeesize,
                           cp.sum(in_committee) >= committeesize,
                           lhs <= rhs,
                           lhs >= rhs]

        # find a new committee that has not been found before, by making previously found
        # committees invalid
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
            # TODO how to deal with OPTIMAL_INACCURATE, INFEASIBLE_INACCURATE, UNBOUNDED_INACCURATE?
            raise RuntimeError("something bad happened")

        if maxscore is None:
            maxscore = problem.value

        if maxscore - problem.value > 1e-13:   # TODO replace with a reasonable value for accuracy!
            # no longer optimal
            break

        committee = np.arange(profile.num_cand)[in_committee.value.astype(np.bool)]

        committees.append(committee.tolist())

        # TODO this is the right way if we don't want to compute all solutions, but see below...
        # if resolute:
        #    break

    # TODO this fixes test.abcrules.test_tiebreaking_order, is there a better solution?
    if resolute:
        return [sorted(committees)[0]]

    return committees
