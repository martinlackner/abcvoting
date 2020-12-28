"""
Approval-based committee (ABC) rules implemented as a integer linear
programs (ILPs) with Gurobi (https://www.gurobi.com/)
"""

from __future__ import print_function
try:
    import gurobipy as gb
    available = True
except ImportError:
    available = False

GUROBI_ACCURACY = 1e-9


def _optimize_rule_gurobi(set_opt_model_func, profile, committeesize, scorefct,
                          resolute, verbose=False):
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

    cands = list(range(profile.num_cand))
    maxscore = None
    committees = []

    # TODO add a max iterations parameter with fancy default value which works in almost all
    #  cases to avoid endless hanging computations, e.g. when CI runs the tests
    while True:
        m = gb.Model()

        # a binary variable indicating whether c is in the committee
        in_committee = m.addVars(profile.num_cand,
                                 vtype=gb.GRB.BINARY,
                                 name="in_committee")

        set_opt_model_func(m, profile, in_committee, committeesize, committees, cands, scorefct)

        m.setParam('OutputFlag', False)
        m.setParam('FeasibilityTol', GUROBI_ACCURACY)
        m.setParam('PoolSearchMode', 0)

        m.optimize()

        if m.Status not in [2, 3, 4]:
            # m.Status == 2 implies solution found
            # m.Status in [3, 4] implies infeasible --> no more solutions
            # otherwise ...
            print("Warning (_optimize_rule_gurobi): solutions may be incomplete or not optimal.")
            print("(Gurobi return code", m.Status, ")")
        if m.Status != 2:
            if len(committees) == 0:
                raise RuntimeError("Gurobi found no solution")
            break

        if maxscore is None:
            maxscore = m.objVal
        elif m.objVal > maxscore + GUROBI_ACCURACY:
            raise RuntimeError("Gurobi found a solution better than a previous optimum. This "
                               f"should not happen (previous optimal score: {maxscore}, "
                               f"new optimal score: {m.objVal}).")
        elif m.objVal < maxscore - GUROBI_ACCURACY:
            # no longer optimal
            break

        committee = [c for c in cands if in_committee[c].Xn >= 1-GUROBI_ACCURACY]
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
    def set_opt_model_func(m, profile, in_committee, committeesize, committees, cands, scorefct):
        # utility[(pref, num_appr)] contains (intended binary) variables indicating
        # whether pref approves at least num_appr candidates in the committee

        utility = {}
        for pref in profile:
            for num_appr in range(1, committeesize + 1):
                # TODO Should we use vtype=gb.GRB.BINARY? Does it make it faster to use ub=1.0?
                utility[(pref, num_appr)] = m.addVar(ub=1.0)

        # constraint: the committee has the required size
        m.addConstr(gb.quicksum(in_committee) == committeesize)

        # constraint: utilities are consistent with actual committee
        for pref in profile:
            m.addConstr(gb.quicksum(utility[pref, num_appr]
                                    for num_appr in range(1, committeesize + 1)) ==
                        gb.quicksum(in_committee[c] for c in pref))

        # find a new committee that has not been found yet by excluding previously found committees
        for comm in committees:
            m.addConstr(
                gb.quicksum(in_committee[c] for c in cands if c in comm) <= committeesize - 1)

        # objective: the PAV score of the committee
        m.setObjective(
            gb.quicksum(float(scorefct(num_appr)) * pref.weight * utility[(pref, num_appr)]
                        for pref in profile
                        for num_appr in range(1, committeesize + 1)),
            gb.GRB.MAXIMIZE)
    return _optimize_rule_gurobi(set_opt_model_func, profile, committeesize, scorefct, resolute)


def __gurobi_monroe(profile, committeesize, resolute):
    def set_opt_model_func(
            m, profile, in_committee, committeesize, committees, cands, scorefct):
        num_voters = len(profile)

        # optimization goal: variable "satisfaction"
        satisfaction = m.addVar(vtype=gb.GRB.INTEGER, name="satisfaction")

        m.addConstr(gb.quicksum(in_committee[c] for c in cands)
                    == committeesize)

        # a partition of voters into committeesize many sets
        partition = m.addVars(profile.num_cand, len(profile),
                              vtype=gb.GRB.INTEGER, lb=0, name="partition")
        for i in range(len(profile)):
            # every voter has to be part of a voter partition set
            m.addConstr(gb.quicksum(partition[(c, i)]
                                    for c in cands)
                        == profile[i].weight)
        for c in cands:
            # every voter set in the partition has to contain
            # at least (num_voters // committeesize) candidates
            m.addConstr(gb.quicksum(partition[(c, j)]
                                    for j in range(len(profile)))
                        >= (num_voters // committeesize
                            - num_voters * (1 - in_committee[c])))
            # every voter set in the partition has to contain
            # at most ceil(num_voters/committeesize) candidates
            m.addConstr(gb.quicksum(partition[(c, j)]
                                    for j in range(len(profile)))
                        <= (num_voters // committeesize
                            + bool(num_voters % committeesize)
                            + num_voters * (1 - in_committee[c])))
            # if in_committee[i] = 0 then partition[(i,j) = 0
            m.addConstr(gb.quicksum(partition[(c, j)]
                                    for j in range(len(profile)))
                        <= num_voters * in_committee[c])

        # constraint for objective variable "satisfaction"
        m.addConstr(gb.quicksum(partition[(c, j)] *
                                (c in profile[j])
                                for j in range(len(profile))
                                for c in cands)
                    >= satisfaction)

        # find a new committee that has not been found before
        for comm in committees:
            m.addConstr(
                gb.quicksum(in_committee[c] for c in cands if c in comm) <= committeesize - 1)

        # optimization objective
        m.setObjective(satisfaction, gb.GRB.MAXIMIZE)
    return _optimize_rule_gurobi(set_opt_model_func, profile, committeesize, scorefct=None,
                                 resolute=resolute)


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
            m, profile, in_committee, committeesize, committees, cands, scorefct):
        load = {}
        for c in cands:
            for pref in profile:
                load[(pref, c)] = m.addVar(ub=1.0, lb=0.0)

        # constraint: the committee has the required size
        m.addConstr(gb.quicksum(in_committee[c] for c in cands) == committeesize)

        for c in cands:
            for pref in profile:
                if c not in pref:
                    m.addConstr(load[(pref, c)] == 0)

        # a candidate's load is distributed among his approvers
        for c in cands:
            m.addConstr(gb.quicksum(pref.weight * load[(pref, c)]
                                    for pref in profile if c in cands)
                        == in_committee[c])

        # find a new committee that has not been found before
        for comm in committees:
            m.addConstr(
                gb.quicksum(in_committee[c] for c in cands if c in comm) <= committeesize - 1)

        loadbound = m.addVar(name="loadbound")
        for pref in profile:
            m.addConstr(gb.quicksum(load[(pref, c)]
                                    for c in pref)
                        <= loadbound)

        # maximizing the negative distance makes code more similar to the other methods here
        m.setObjective(-loadbound, gb.GRB.MAXIMIZE)

    return _optimize_rule_gurobi(set_opt_model_func, profile, committeesize, scorefct=None,
                                 resolute=resolute, verbose=verbose)


def __gurobi_minimaxav(profile, committeesize, resolute):
    def set_opt_model_func(
            m, profile, in_committee, committeesize, committees, cands, scorefct):
        num_voters = len(profile)
        # optimization goal: variable "sum_difference"
        max_hamdistance = m.addVar(vtype=gb.GRB.INTEGER, name="max_hamdistance")

        m.addConstr(gb.quicksum(in_committee[c] for c in cands)
                    == committeesize)

        # the single differences between the committee and the voters
        difference = m.addVars(profile.num_cand, num_voters, vtype=gb.GRB.INTEGER,
                               name="diff")

        for i in cands:
            for j in range(num_voters):
                if i in profile[j]:
                    # constraint for the case that the candidate is approved
                    m.addConstr(difference[i, j] == 1 - in_committee[i])
                else:
                    # constraint for the case that the candidate isn't approved
                    m.addConstr(difference[i, j] == in_committee[i])

        for j in range(num_voters):
            # maximum hamming distance is greater of equal than any individual one
            m.addConstr(max_hamdistance >= gb.quicksum(difference[i, j]
                                                       for i in cands
                                                       )
                        )

        # find a new committee that has not been found before
        for comm in committees:
            m.addConstr(
                gb.quicksum(in_committee[c] for c in cands if c in comm) <= committeesize - 1)

        # maximizing the negative distance makes code more similar to the other methods here
        m.setObjective(-max_hamdistance, gb.GRB.MAXIMIZE)

    return _optimize_rule_gurobi(set_opt_model_func, profile, committeesize, scorefct=None,
                                 resolute=resolute)
