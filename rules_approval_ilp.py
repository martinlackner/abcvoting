# Approval-based multiwinner rules implemented as an integer linear
# program (ILP) with Gurobi

# Author: Martin Lackner, Stefan Forster


from committees import enough_approved_candidates, sort_committees
import score_functions as sf
try:
    import gurobipy as gb
except ImportError:
    None    # Gurobi not available


def compute_thiele_methods_ilp(profile, committeesize,
                               scorefct_str, resolute=False):

    enough_approved_candidates(profile, committeesize)
    scorefct = sf.get_scorefct(scorefct_str, committeesize)

    m = gb.Model()
    cands = list(range(profile.num_cand))

    # a binary variable indicating whether c is in the committee
    in_committee = m.addVars(profile.num_cand,
                             vtype=gb.GRB.BINARY,
                             name="in_comm")

    # a (intended binary) variable indicating
    # whether v approves at least l candidates in the committee
    utility = {}
    for v in profile.preferences:
        for l in range(1, committeesize + 1):
            utility[(v, l)] = m.addVar(ub=1.0)

    # constraint: the committee has the required size
    m.addConstr(gb.quicksum(in_committee[c] for c in cands) == committeesize)

    # constraint: utilities are consistent with actual committee
    for v in profile.preferences:
        m.addConstr(gb.quicksum(utility[v, l]
                                for l in range(1, committeesize + 1)) ==
                    gb.quicksum(in_committee[c] for c in v.approved))

    # objective: the PAV score of the committee
    m.setObjective(gb.quicksum(float(scorefct(l)) * v.weight * utility[(v, l)]
                               for v in profile.preferences
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
        print "Warning (" + scorefct_str + "):",
        print "solutions may be incomplete or not optimal."
        print "(Gurobi return code", m.Status, ")"

    # extract committees from model
    committees = []
    if resolute:
        committees.append([c for c in cands if in_committee[c].Xn >= 0.99])
    else:
        for sol in range(m.SolCount):
            m.setParam('SolutionNumber', sol)
            committees.append([c for c in cands if in_committee[c].Xn >= 0.99])

    committees = sort_committees(committees)

    return committees


def compute_monroe_ilp(profile, committeesize, resolute=False):
    enough_approved_candidates(profile, committeesize)

    # Monroe is only defined for unit weights
    if not profile.has_unit_weights():
        raise NotImplementedError("Monroe is only defined for" +
                                  " unit weights (weight=1)")

    num_voters = len(profile.preferences)
    cands = range(profile.num_cand)

    # Alternative: split voters -> generate new profile with all weights = 1
    # prof2 = Profile(profile.num_cand)
    # for v in profile:
    #    for _ in range(v.weight):
    #        prof2.add_preference(DichotomousPreference(v.approved,
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
    partition = m.addVars(profile.num_cand, len(profile.preferences),
                          vtype=gb.GRB.INTEGER, lb=0, name="partition")
    for i in range(len(profile.preferences)):
        # every voter has to be part of a voter partition set
        m.addConstr(gb.quicksum(partition[(j, i)]
                                for j in cands)
                    == profile.preferences[i].weight)
    for i in cands:
        # every voter set in the partition has to contain
        # at least (num_voters // committeesize) candidates
        m.addConstr(gb.quicksum(partition[(i, j)]
                                for j in range(len(profile.preferences)))
                    >= (num_voters // committeesize
                        - num_voters * (1 - in_committee[i])))
        # every voter set in the partition has to contain
        # at most ceil(num_voters/committeesize) candidates
        m.addConstr(gb.quicksum(partition[(i, j)]
                                for j in range(len(profile.preferences)))
                    <= (num_voters // committeesize
                        + bool(num_voters % committeesize)
                        + num_voters * (1 - in_committee[i])))
        # if in_committee[i] = 0 then partition[(i,j) = 0
        m.addConstr(gb.quicksum(partition[(i, j)]
                                for j in range(len(profile.preferences)))
                    <= num_voters * in_committee[i])

    # constraint for objective variable "satisfaction"
    m.addConstr(gb.quicksum(partition[(i, j)] *
                            (i in profile.preferences[j].approved)
                            for j in range(len(profile.preferences))
                            for i in cands)
                >= satisfaction)

    # optimization objective
    m.setObjective(satisfaction, gb.GRB.MAXIMIZE)

    # m.setParam('OutputFlag', False)

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
        print "Warning (Monroe): solutions may be incomplete or not optimal."
        print "(Gurobi return code", m.Status, ")"

    # extract committees from model
    committees = []
    if resolute:
        committees.append([c for c in cands if in_committee[c].Xn >= 0.99])
    else:
        for sol in range(m.SolCount):
            m.setParam('SolutionNumber', sol)
            committees.append([c for c in cands if in_committee[c].Xn >= 0.99])

    # if len(committees)>10:
    #    print "Warning (Monroe): more than 10 committees found;",
    #    print "returning first 10"
    #    committees = committees[:10]

    committees = sort_committees(committees)

    return committees


def compute_maximin_ilp(profile, committeesize, resolute=False):
    enough_approved_candidates(profile, committeesize)

    voters = profile.preferences
    num_voters = len(voters)
    cands = range(profile.num_cand)

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
            if i in voters[j].approved:
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
        print "Warning (Maximin AV): solutions may be incomplete or not optimal."
        print "(Gurobi return code", m.Status, ")"

    # extract committees from model
    committees = []
    if resolute:
        committees.append([c for c in cands if in_committee[c].Xn >= 0.99])
    else:
        for sol in range(m.SolCount):
            m.setParam('SolutionNumber', sol)
            committees.append([c for c in cands if in_committee[c].Xn >= 0.99])

    committees = sort_committees(committees)

    return committees


def compute_minimax_ilp(profile, committeesize, resolute=False):
    enough_approved_candidates(profile, committeesize)

    voters = profile.preferences
    num_voters = len(voters)
    cands = range(profile.num_cand)

    m = gb.Model()

    # optimization goal: variable "sum_difference"
    sum_differences = m.addVar(vtype=gb.GRB.INTEGER, name="sum_diff")

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
            if i in voters[j].approved:
                # constraint for the case that the candidate is approved
                m.addConstr(difference[i, j] == 1 - in_committee[i])
            else:
                # constraint for the case that the candidate isn't approved
                m.addConstr(difference[i, j] == in_committee[i])

    # sum_differences
    m.addConstr(sum_differences == gb.quicksum(difference[i, j]
                                               for i in cands
                                               for j in range(num_voters)
                                               )
                )

    # optimization objective
    m.setObjective(sum_differences, gb.GRB.MINIMIZE)

    # m.setParam('OutputFlag', False)

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
        print "Warning (Minimax AV): solutions may be incomplete or not optimal."
        print "(Gurobi return code", m.Status, ")"

    # extract committees from model
    committees = []
    if resolute:
        committees.append([c for c in cands if in_committee[c].Xn >= 0.99])
    else:
        for sol in range(m.SolCount):
            m.setParam('SolutionNumber', sol)
            committees.append([c for c in cands if in_committee[c].Xn >= 0.99])

    committees = sort_committees(committees)

    return committees
