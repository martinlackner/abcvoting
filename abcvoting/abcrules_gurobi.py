"""
Approval-based committee (ABC) rules implemented as a integer linear
programs (ILPs) with Gurobi (https://www.gurobi.com/)
"""

from __future__ import print_function
from abcvoting.misc import sorted_committees

try:
    import gurobipy as gb

    available = True
except ImportError:
    available = False

GUROBI_ACCURACY = 1e-9


def _optimize_rule_gurobi(
    set_opt_model_func, profile, committeesize, scorefct, resolute, verbose=False
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
    verbose : bool

    Returns
    -------
    committees : list of lists
        a list of chosen committees, each of them represented as list with candidates named from
        `0` to `num_cand`, profile.cand_names is ignored

    """

    maxscore = None
    committees = []

    # TODO add a max iterations parameter with fancy default value which works in almost all
    #  cases to avoid endless hanging computations, e.g. when CI runs the tests
    while True:
        m = gb.Model()

        # `in_committee` is a binary variable indicating whether `cand` is in the committee
        in_committee = m.addVars(
            profile.num_cand, vtype=gb.GRB.BINARY, name="in_committee"
        )

        set_opt_model_func(
            m,
            profile,
            in_committee,
            committeesize,
            committees,
            profile.candidates,
            scorefct,
        )

        m.setParam("OutputFlag", False)
        m.setParam("FeasibilityTol", GUROBI_ACCURACY)
        m.setParam("PoolSearchMode", 0)

        m.optimize()

        if m.Status not in [2, 3, 4]:
            # m.Status == 2 implies solution found
            # m.Status in [3, 4] implies infeasible --> no more solutions
            # otherwise ...
            raise RuntimeError(
                f"Gurobi returned an unexpected status code: {m.Status}"
                "Warning: solutions may be incomplete or not optimal."
            )
        elif m.Status != 2:
            if len(committees) == 0:
                # we are in the first round of searching committees and Gurobi doesn't find anything
                raise RuntimeError("Gurobi found no solution")
            break

        if maxscore is None:
            maxscore = m.objVal
        elif m.objVal > maxscore + GUROBI_ACCURACY:
            raise RuntimeError(
                "Gurobi found a solution better than a previous optimum. This "
                f"should not happen (previous optimal score: {maxscore}, "
                f"new optimal score: {m.objVal})."
            )
        elif m.objVal < maxscore - GUROBI_ACCURACY:
            # no longer optimal
            break

        committee = [
            cand
            for cand in profile.candidates
            if in_committee[cand].Xn >= 1 - GUROBI_ACCURACY
        ]
        assert len(committee) == committeesize
        committees.append(committee)

        if resolute:
            break

    # optional output
    if verbose:
        print("optimal score: " + str(maxscore))
        print("(i.e., all scores are <= " + str(maxscore) + ")")
    # end of optional output

    return committees


def __gurobi_thiele_methods(profile, committeesize, scorefct, resolute):
    def set_opt_model_func(
        m, profile, in_committee, committeesize, committees, cands, scorefct
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

        for voter in profile:
            for l in range(1, committeesize + 1):
                # TODO Should we use vtype=gb.GRB.BINARY? Does it make it faster to use ub=1.0?
                utility[(voter, l)] = m.addVar(ub=1.0)

        # constraint: the committee has the required size
        m.addConstr(gb.quicksum(in_committee) == committeesize)

        # constraint: utilities are consistent with actual committee
        for voter in profile:
            m.addConstr(
                gb.quicksum(utility[voter, l] for l in range(1, committeesize + 1))
                == gb.quicksum(in_committee[cand] for cand in voter.approved)
            )

        # find a new committee that has not been found yet by excluding previously found committees
        for committee in committees:
            m.addConstr(
                gb.quicksum(in_committee[cand] for cand in cands if cand in committee)
                <= committeesize - 1
            )

        # objective: the PAV score of the committee
        m.setObjective(
            gb.quicksum(
                float(scorefct(l)) * voter.weight * utility[(voter, l)]
                for voter in profile
                for l in range(1, committeesize + 1)
            ),
            gb.GRB.MAXIMIZE,
        )

    score_values = [scorefct(l) for l in range(1, committeesize + 1)]
    if not all(
        first > second or first == second == 0
        for first, second in zip(score_values, score_values[1:])
    ):
        raise ValueError("scorefct must be monotonic decreasing")

    committees = _optimize_rule_gurobi(
        set_opt_model_func, profile, committeesize, scorefct, resolute
    )
    return sorted_committees(committees)


def __gurobi_monroe(profile, committeesize, resolute):
    def set_opt_model_func(
        m, profile, in_committee, committeesize, committees, cands, scorefct
    ):
        num_voters = len(profile)

        # optimization goal: variable "satisfaction"
        satisfaction = m.addVar(vtype=gb.GRB.INTEGER, name="satisfaction")

        m.addConstr(gb.quicksum(in_committee[cand] for cand in cands) == committeesize)

        # a partition of voters into committeesize many sets
        partition = m.addVars(
            profile.num_cand, len(profile), vtype=gb.GRB.INTEGER, lb=0, name="partition"
        )
        for i in range(len(profile)):
            # every voter has to be part of a voter partition set
            m.addConstr(
                gb.quicksum(partition[(cand, i)] for cand in cands) == profile[i].weight
            )
        for cand in cands:
            # every voter set in the partition has to contain
            # at least (num_voters // committeesize) candidates
            m.addConstr(
                gb.quicksum(partition[(cand, j)] for j in range(len(profile)))
                >= (num_voters // committeesize - num_voters * (1 - in_committee[cand]))
            )
            # every voter set in the partition has to contain
            # at most ceil(num_voters/committeesize) candidates
            m.addConstr(
                gb.quicksum(partition[(cand, j)] for j in range(len(profile)))
                <= (
                    num_voters // committeesize
                    + bool(num_voters % committeesize)
                    + num_voters * (1 - in_committee[cand])
                )
            )
            # if in_committee[i] = 0 then partition[(i,j) = 0
            m.addConstr(
                gb.quicksum(partition[(cand, j)] for j in range(len(profile)))
                <= num_voters * in_committee[cand]
            )

        # constraint for objective variable "satisfaction"
        m.addConstr(
            gb.quicksum(
                partition[(cand, j)] * (cand in profile[j].approved)
                for j in range(len(profile))
                for cand in cands
            )
            >= satisfaction
        )

        # find a new committee that has not been found before
        for committee in committees:
            m.addConstr(
                gb.quicksum(in_committee[cand] for cand in cands if cand in committee)
                <= committeesize - 1
            )

        # optimization objective
        m.setObjective(satisfaction, gb.GRB.MAXIMIZE)

    committees = _optimize_rule_gurobi(
        set_opt_model_func, profile, committeesize, scorefct=None, resolute=resolute
    )
    return sorted_committees(committees)


def __gurobi_optphragmen(profile, committeesize, resolute, verbose):
    """opt-Phragmen

    Warning: does not include the lexicographic optimization as specified
    in Markus Brill, Rupert Freeman, Svante Janson and Martin Lackner.
    Phragmen's Voting Methods and Justified Representation.
    http://martin.lackner.xyz/publications/phragmen.pdf

    Instead: minimizes the maximum load (without consideration of the
             second-, third-, ...-largest load
    """

    def set_opt_model_func(
        m, profile, in_committee, committeesize, committees, cands, scorefct
    ):
        load = {}
        for cand in cands:
            for voter in profile:
                load[(voter, cand)] = m.addVar(ub=1.0, lb=0.0)

        # constraint: the committee has the required size
        m.addConstr(gb.quicksum(in_committee[cand] for cand in cands) == committeesize)

        for cand in cands:
            for voter in profile:
                if cand not in voter.approved:
                    m.addConstr(load[(voter, cand)] == 0)

        # a candidate's load is distributed among his approvers
        for cand in cands:
            m.addConstr(
                gb.quicksum(
                    voter.weight * load[(voter, cand)]
                    for voter in profile
                    if cand in cands
                )
                == in_committee[cand]
            )

        # find a new committee that has not been found before
        for committee in committees:
            m.addConstr(
                gb.quicksum(in_committee[cand] for cand in cands if cand in committee)
                <= committeesize - 1
            )

        loadbound = m.addVar(name="loadbound")
        for voter in profile:
            m.addConstr(
                gb.quicksum(load[(voter, cand)] for cand in voter.approved) <= loadbound
            )

        # maximizing the negative distance makes code more similar to the other methods here
        m.setObjective(-loadbound, gb.GRB.MAXIMIZE)

    committees = _optimize_rule_gurobi(
        set_opt_model_func,
        profile,
        committeesize,
        scorefct=None,
        resolute=resolute,
        verbose=verbose,
    )
    return sorted_committees(committees)


def __gurobi_minimaxav(profile, committeesize, resolute):
    def set_opt_model_func(
        m, profile, in_committee, committeesize, committees, cands, scorefct
    ):
        num_voters = len(profile)
        # optimization goal: variable "sum_difference"
        max_hamdistance = m.addVar(vtype=gb.GRB.INTEGER, name="max_hamdistance")

        m.addConstr(gb.quicksum(in_committee[cand] for cand in cands) == committeesize)

        # the single differences between the committee and the voters
        difference = m.addVars(
            profile.num_cand, num_voters, vtype=gb.GRB.INTEGER, name="diff"
        )

        for i in cands:
            for j in range(num_voters):
                if i in profile[j].approved:
                    # constraint for the case that the candidate is approved
                    m.addConstr(difference[i, j] == 1 - in_committee[i])
                else:
                    # constraint for the case that the candidate isn't approved
                    m.addConstr(difference[i, j] == in_committee[i])

        for j in range(num_voters):
            # maximum hamming distance is greater of equal than any individual one
            m.addConstr(max_hamdistance >= gb.quicksum(difference[i, j] for i in cands))

        # find a new committee that has not been found before
        for committee in committees:
            m.addConstr(
                gb.quicksum(in_committee[cand] for cand in cands if cand in committee)
                <= committeesize - 1
            )

        # maximizing the negative distance makes code more similar to the other methods here
        m.setObjective(-max_hamdistance, gb.GRB.MAXIMIZE)

    committees = _optimize_rule_gurobi(
        set_opt_model_func, profile, committeesize, scorefct=None, resolute=resolute
    )
    return sorted_committees(committees)
