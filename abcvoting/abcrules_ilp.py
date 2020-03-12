"""
Approval-based committee (ABC) rules implemented as a integer linear
programs (ILPs) with Gurobi (https://www.gurobi.com/)
"""

from __future__ import print_function
try:
    import gurobipy as gb
except ImportError:
    None    # Gurobi not available


def __gurobi_thiele_methods(profile, committeesize,
                            scorefct, resolute=False):

    m = gb.Model()
    cands = list(range(profile.num_cand))

    # a binary variable indicating whether c is in the committee
    in_committee = m.addVars(profile.num_cand,
                             vtype=gb.GRB.BINARY,
                             name="in_comm")

    # a (intended binary) variable indicating
    # whether v approves at least l candidates in the committee
    utility = {}
    for pref in profile:
        for l in range(1, committeesize + 1):
            utility[(pref, l)] = m.addVar(ub=1.0)

    # constraint: the committee has the required size
    m.addConstr(gb.quicksum(in_committee[c] for c in cands) == committeesize)

    # constraint: utilities are consistent with actual committee
    for pref in profile:
        m.addConstr(gb.quicksum(utility[pref, l]
                                for l in range(1, committeesize + 1)) ==
                    gb.quicksum(in_committee[c] for c in pref))

    # objective: the PAV score of the committee
    m.setObjective(
        gb.quicksum(float(scorefct(l)) * pref.weight * utility[(pref, l)]
                    for pref in profile
                    for l in range(1, committeesize + 1)),
        gb.GRB.MAXIMIZE)

    m.setParam('OutputFlag', False)

    if resolute:
        m.setParam('PoolSearchMode', 0)
    else:
        # output all optimal committees
        m.setParam('PoolSearchMode', 2)
        # abort after (roughly) 100 optimal solutions
        m.setParam('PoolSolutions', 100)
        # ignore suboptimal committees
        m.setParam('PoolGap', 0)

    m.optimize()

    if m.Status != 2:
        print("Warning (__gurobi_thiele_methods):"+
              " solutions may be incomplete or not optimal.")
        print("(Gurobi return code", m.Status, ")")

    # extract committees from model
    committees = []
    if resolute:
        committees.append([c for c in cands if in_committee[c].Xn >= 0.99])
    else:
        for sol in range(m.SolCount):
            m.setParam('SolutionNumber', sol)
            committees.append([c for c in cands if in_committee[c].Xn >= 0.99])

    return committees


def __gurobi_monroe(profile, committeesize, resolute):
    
    num_voters = len(profile)
    cands = list(range(profile.num_cand))

    # Alternative: split voters -> generate new profile with all weights = 1
    # prof2 = Profile(profile.num_cand)
    # for v in profile:
    #    for _ in range(v.weight):
    #        prof2.add_preference(DichotomousPreference(v,
    #                                                   profile.num_cand))
    # total_weight = profile.voters_num()

    m = gb.Model()

    # optimization goal: variable "satisfaction"
    satisfaction = m.addVar(vtype=gb.GRB.INTEGER, name="satisfaction")

    # a list of committee members
    in_committee = m.addVars(profile.num_cand, vtype=gb.GRB.BINARY,
                             name="in_comm")
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

    m.update()

    # constraint for objective variable "satisfaction"
    m.addConstr(gb.quicksum(partition[(c, j)] *
                            (c in profile[j])
                            for j in range(len(profile))
                            for c in cands)
                >= satisfaction)

    # optimization objective
    m.setObjective(satisfaction, gb.GRB.MAXIMIZE)

    m.setParam('OutputFlag', False)

    if resolute:
        m.setParam('PoolSearchMode', 0)
    else:
        # output all optimal committees
        m.setParam('PoolSearchMode', 2)
        # abort after (roughly) 100 optimal solutions
        m.setParam('PoolSolutions', 100)
        # ignore suboptimal committees
        m.setParam('PoolGap', 0)

    m.optimize()

    if m.Status != 2:
        print("Warning (Monroe): solutions may be incomplete or not optimal.")
        print("(Gurobi return code", m.Status, ")")

    # extract committees from model
    committees = []
    if resolute:
        committees.append([c for c in cands if in_committee[c].Xn >= 0.99])
    else:
        for sol in range(m.SolCount):
            m.setParam('SolutionNumber', sol)
            committees.append([c for c in cands if in_committee[c].Xn >= 0.99])

    return committees


# opt-Phragmen
#
# Warning: does not include the tie-breaking mechanism as specified
# in Markus Brill, Rupert Freeman, Svante Janson and Martin Lackner.
# Phragmen's Voting Methods and Justified Representation.
# http://martin.lackner.xyz/publications/phragmen.pdf
#
# Instead: minimizes the maximum load
def __gurobi_optphragmen(profile, committeesize, resolute=False):

    cands = list(range(profile.num_cand))

    m = gb.Model()
    m.setParam('OutputFlag', False)

    # a binary variable indicating whether c is in the committee
    in_committee = m.addVars(profile.num_cand, vtype=gb.GRB.BINARY,
                             name="in_comm")

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

    loadbound = m.addVar(name="loadbound")
    for pref in profile:
        m.addConstr(gb.quicksum(load[(pref, c)]
                                for c in pref)
                    <= loadbound)
    m.setObjective(loadbound, gb.GRB.MINIMIZE)

    m.setParam('OutputFlag', False)

    if resolute:
        m.setParam('PoolSearchMode', 0)
    else:
        # output all optimal committees
        m.setParam('PoolSearchMode', 2)
        # abort after (roughly) 100 optimal solutions
        m.setParam('PoolSolutions', 100)
        # ignore suboptimal committees
        m.setParam('PoolGap', 0)

    m.optimize()

    if m.Status != 2:
        print("Warning (opt-Phragmen): solutions may be "
              + "incomplete or not optimal.")
        print("(Gurobi return code", m.Status, ")")

    # extract committees from model
    committees = []
    if resolute:
        committees.append([c for c in cands if in_committee[c].Xn >= 0.99])
    else:
        for sol in range(m.SolCount):
            m.setParam('SolutionNumber', sol)
            committees.append([c for c in cands if in_committee[c].Xn >= 0.99])

    return committees


def __gurobi_minimaxav(profile, committeesize, resolute=False):

    num_voters = len(profile)
    cands = list(range(profile.num_cand))

    m = gb.Model()

    # optimization goal: variable "sum_difference"
    max_hamdistance = m.addVar(vtype=gb.GRB.INTEGER, name="max_hamdistance")

    # a list of committee members
    in_committee = m.addVars(profile.num_cand, vtype=gb.GRB.BINARY,
                             name="in_comm")
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

    # optimization objective
    m.setObjective(max_hamdistance, gb.GRB.MINIMIZE)

    # m.setParam('OutputFlag', False)

    m.setParam('OutputFlag', False)

    if resolute:
        m.setParam('PoolSearchMode', 0)
    else:
        # output all optimal committees
        m.setParam('PoolSearchMode', 2)
        # abort after (roughly) 100 optimal solutions
        m.setParam('PoolSolutions', 1000)
        # ignore suboptimal committees
        m.setParam('PoolGap', 0)

    m.optimize()

    if m.Status != 2:
        print("Warning (Minimax AV): solutions may be incomplete"
              + " or not optimal.")
        print("(Gurobi return code", m.Status, ")")

    # extract committees from model
    committees = []
    if resolute:
        committees.append([c for c in cands if in_committee[c].Xn >= 0.99])
    else:
        for sol in range(m.SolCount):
            m.setParam('SolutionNumber', sol)
            committees.append([c for c in cands if in_committee[c].Xn >= 0.99])

    return committees
