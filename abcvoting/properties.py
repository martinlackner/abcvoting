"""
Properties of committees (i.e., sets of candidates)
"""

import itertools
import math
from abcvoting.output import output
from abcvoting.preferences import Profile


try:
    import gurobipy as gb

    gurobipy_available = True
except ImportError:
    gurobipy_available = False


def check_pareto_optimality(profile, committee, algorithm="brute-force"):
    """Test whether a committee satisfies Pareto optimality.
    
    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        approval sets of voters
    committee : set / list / tuple
        set of candidates
    algorithm : string
        describe which algorithm to use

    Returns
    -------
    None

    Reference
    ---------
    Lackner, M., Skowron, P. (2021)
    Survey "Approval-Based Committee Voting"
    """

    committee = _validate_input(profile, committee)

    if algorithm == "brute-force":
        result = _check_pareto_optimality_brute_force(profile, committee)
    elif algorithm == "gurobi":
        result = _check_pareto_optimality_gurobi(profile, committee)
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for check_pareto_optimality"
        )

    if result:
        output.info(f"Committee {str_set_of_candidates(committee)} is Pareto optimal.")
    else:
        output.info(f"Committee {str_set_of_candidates(committee)} is not Pareto optimal.")
        # TODO: detailed ouput with output.details() showing
        #  - which committee is Pareto dominating
        #  - which voters have a better satisfaction with the new committee

    return result


def check_EJR(profile, committee, algorithm="brute-force"):
    """Test whether a committee satisfies EJR.
    
    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        approval sets of voters
    committee : set / list / tuple
        set of candidates
    algorithm : string
        describe which algorithm to use

    Returns
    -------
    None
    
    Reference
    ---------
    Aziz, H., Brill, M., Conitzer, V., Elkind, E., Freeman, R., & Walsh, T. (2017). 
    Justified representation in approval-based committee voting. 
    Social Choice and Welfare, 48(2), 461-485.
    """

    committee = _validate_input(profile, committee)

    if algorithm == "brute-force":
        result = _check_EJR_brute_force(profile, committee)
    elif algorithm == "gurobi":
        result = _check_EJR_gurobi(profile, committee)
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for check_EJR"
        )

    if result:
        output.info(f"Committee {str_set_of_candidates(committee)} satisfies EJR.")
    else:
        output.info(f"Committee {str_set_of_candidates(committee)} does not satisfy EJR.")

    return result


def check_PJR(profile, committee, algorithm="brute-force"):
    """Test whether a committee satisfies PJR.
    
    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        approval sets of voters
    committee : set / list / tuple
        set of candidates
    algorithm : string
        describe which algorithm to use

    Returns
    -------
    None

    Reference
    ---------
    Sánchez-Fernández, L., Elkind, E., Lackner, M., Fernández, N., Fisteus, J., Val, P. B., & Skowron, P. (2017, February). 
    Proportional justified representation. 
    In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 31, No. 1).
    """

    committee = _validate_input(profile, committee)

    if algorithm == "brute-force":
        result = _check_PJR_brute_force(profile, committee)
    elif algorithm == "gurobi":
        result = _check_PJR_gurobi(profile, committee)
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for check_PJR"
        )

    if result:
        output.info(f"Committee {str_set_of_candidates(committee)} satisfies PJR.")
    else:
        output.info(f"Committee {str_set_of_candidates(committee)} does not satisfy PJR.")

    return result


def dominates(comm1, comm2, profile):
    """Test whether a committee comm1 dominates another committee comm2. That is, test whether
    each voter in the profile has at least as many approved candidates in comm1 as in comm2, 
    and there is at least one voter with strictly more approved candidates in comm1. 
    
    Parameters
    ----------
    comm1 : set
        set of candidates
    comm2 : set
        set of candidates
    profile : abcvoting.preferences.Profile
        approval sets of voters

    Returns
    -------
    bool
    """

    # flag to check whether there are at least as many approved candidates 
    # in dominating committee as in input committee
    condition_at_least_as_many = True

    # iterate through all voters
    for voter in profile:
        # check if there are at least as many approved candidates in comm1 as in comm2
        condition_at_least_as_many = condition_at_least_as_many and (len(voter.approved & comm1) >= len(voter.approved & comm2))
        # check if domination has already failed
        if not condition_at_least_as_many:
            return False

    # if not yet returned by now, then check for condition whether there is a voter with strictly
    # more preferred candidates in dominating committee than in input committee
    for voter in profile:
        # check if for some voter, there are strictly more preferred candidates in comm1 than in comm2
        if len(voter.approved & comm1) > len(voter.approved & comm2):
            return True

    # If function has still not returned by now, then it means that comm1 does not dominate comm2
    return False


def _check_pareto_optimality_brute_force(profile, committee):
    """Test using brute-force whether a committee is Pareto optimal. That is, there is no other committee
    which dominates it. 
    
    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        approval sets of voters
    committee : set
        set of candidates

    Returns
    -------
    bool
    """
    # iterate through all possible combinations of number_candidates taken committee_size at a time
    for comm in itertools.combinations(range(profile.num_cand), len(committee)):
        if dominates(set(comm), committee, profile):
            # if a generated committee dominates the "query" committee, then it is not Pareto optimal
            return False 

    # if function has not returned up until now, then no other committee dominates it. Thus Pareto optimal
    return True


def _check_pareto_optimality_gurobi(profile, committee):
    """Test, by an ILP and the Gurobi solver, whether a committee is Pareto optimal. That is, there is no 
    other committee which dominates it.
    
    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        approval sets of voters
    committee : set
        set of candidates

    Returns
    -------
    bool
    """
    if not gurobipy_available:
        raise ImportError("Gurobi (gurobipy) not available.")

    # array to store number of approved candidates per voter in the query committee
    num_apprvd_cands_query = []

    # loop through all voters and store how many approved candidates they have in query committee
    for voter in profile:
        num_apprvd_cands_query.append(len(voter.approved & committee))

    # create the model to be optimized
    model = gb.Model()

    # binary variables: indicate whether voter i approves of 
    # at least l candidates in the dominating committee
    utility = {}
    for voter in profile:
        for l in range(1, len(committee) + 1):
            utility[(voter, l)] = model.addVar(vtype=gb.GRB.BINARY)

    # binary variables: indicate whether a candidate is inside the dominating committee
    in_committee = model.addVars(profile.num_cand, vtype=gb.GRB.BINARY, name="in_committee")

    # binary variables: determine for which voter(s) the condition of having strictly
    # more preferred candidates in dominating committee will be satisfied
    condition_strictly_more = model.addVars(range(len(profile)), vtype=gb.GRB.BINARY, name="condition_strictly_more")

    # constraint: utility actually matches the number of approved candidates in the committee, for all voters
    for voter in profile:
        model.addConstr(
            gb.quicksum(utility[(voter, l)] for l in range(1, len(committee) + 1))
            == gb.quicksum(in_committee[cand] for cand in voter.approved)
        )

    # constraint: all voters should have at least as many preferred candidates
    # in the dominating committee as in the query committee
    for voter, i in zip(profile, range(len(profile))):
        model.addConstr(gb.quicksum(utility[(voter, l)] for l in range(1, len(committee) + 1)) >= num_apprvd_cands_query[i])

    # constraint: the condition of having strictly more approved candidates in 
    # dominating commitee will be satisifed for at least one voter
    model.addConstr(gb.quicksum(condition_strictly_more) >= 1)

    # loop through all variables in the condition_strictly_more array (there is one for each voter)
    # if it has value 1, then the condition of having strictly more preferred candidates on the dominating committee 
    # has to be satisfied for this voter
    for voter, i in zip(profile, range(len(profile))):
        model.addConstr(
            (condition_strictly_more[i] == 1) 
            >> (gb.quicksum(utility[(voter, l)] for l in range(1, len(committee) + 1)) >= num_apprvd_cands_query[i] + 1)
        )

    # constraint: committee has the right size
    model.addConstr(gb.quicksum(in_committee) == len(committee))

    # set the objective function
    model.setObjective(
        gb.quicksum(
            utility[(voter, l)]
            for voter in profile
            for l in range(1, len(committee) + 1)
        ), 
        gb.GRB.MAXIMIZE
    )

    # optimize the model
    model.optimize()

    # return value based on status code
    # status code 2 means model was solved to optimality, thus a dominating committee was found
    if model.Status == 2:
        return False
    # status code 3 means that model is infeasible, thus no dominating committee was found 
    elif model.Status == 3:
        return True
    else:
        raise RuntimeError(f"Gurobi returned an unexpected status code: {model.Status}")


def _check_EJR_brute_force(profile, committee):
    """Test using brute-force whether a committee satisfies EJR.
    
    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        approval sets of voters
    committee : set
        set of candidates

    Returns
    -------
    bool
    """
    # list of candidates with less than ell approved candidates in committee
    voters_less_than_ell_approved_candidates = []

    # should check for ell from 1 until committee size
    ell_upper_bound = len(committee) + 1

    # loop through all possible ell
    for ell in range(1, ell_upper_bound):
        # clear list of candidates to consider
        voters_less_than_ell_approved_candidates.clear()

        # compute minimum group size for this ell
        group_size = math.ceil(ell * (len(profile) / len(committee)))

        # compute list of candidates to consider
        for i in range(len(profile)):
            if len(profile[i].approved & committee) < ell:
                voters_less_than_ell_approved_candidates.append(i)

        # check if an ell-cohesive group can be formed with considered candidates
        if len(voters_less_than_ell_approved_candidates) < group_size:
            # if not possible then simply continue with next ell
            continue

        # check all possible combinations of considered voters,
        # taken (possible group size) at a time
        for combination in itertools.combinations(voters_less_than_ell_approved_candidates, group_size):
            # to calculate the cut of approved candidates for the considered voters
            cut = set()

            # initialize the cut to be the approval set of the first candidate in current combination
            cut = profile[combination[0]].approved

            # calculate the cut over all voters for current combination
            # (also can skip first voter in combination, but inexpensive enough...)
            for j in combination:
                cut = cut & profile[j].approved

            # if size of cut is >= ell, then combination is an ell-cohesive group
            if len(cut) >= ell:
                # we have found combination to be an ell-cohesive set, with no voter having
                # at least ell approved candidates in committee. Thus EJR fails
                return False

    # if function has not returned by now, then it means that for all ell, 
    # no ell-cohesive group was found among candidates with less than ell
    # approved candidates in committee. Thus committee satisfies EJR
    return True


def _check_EJR_gurobi(profile, committee):
    """Test, by an ILP and the Gurobi solver, whether a committee satisfies EJR.
    
    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        approval sets of voters
    committee : set
        set of candidates

    Returns
    -------
    bool
    """

    if not gurobipy_available:
        raise ImportError("Gurobi (gurobipy) not available.")

    # compute matrix-dictionary for voters approval
    # approval_matrix[(voter, cand)] = 1 if cand is in voter's approval set
    # approval_matrix[(voter, cand)] = 0 otherwise
    approval_matrix = {}

    for voter in profile:
        for cand in range(profile.num_cand):
            if cand in voter.approved:
                approval_matrix[(voter, cand)] = 1
            else:
                approval_matrix[(voter, cand)] = 0

    # create the model to be optimized
    model = gb.Model()

    # integer variable: ell
    ell = model.addVar(vtype=gb.GRB.INTEGER, name="ell")

    # binary variables: indicate whether a voter is inside the ell-cohesive group
    in_group = model.addVars(len(profile), vtype=gb.GRB.BINARY, name="in_group")

    # constraints: ell has to be between 1 and committeesize inclusive
    model.addConstr(ell >= 1)
    model.addConstr(ell <= len(committee))

    # model the value ceil(ell * (len(profile) / len(committee))) using auxiliary
    # integer variable min_group_size
    # source: https://support.gurobi.com/hc/en-us/community/posts/360054499471-Linear-program-with-ceiling-or-floor-functions-HOW-
    min_group_size = model.addVar(vtype=gb.GRB.INTEGER, name="min_group_size")
    model.addConstr(min_group_size >= (ell * (len(profile) / len(committee))))
    model.addConstr(min_group_size <= (ell * (len(profile) / len(committee))) + 1 - 1e-8)

    # constraint: size of ell-cohesive group should be appropriate wrt ell
    model.addConstr(gb.quicksum(in_group) >= min_group_size)

    # constraints based on binary indicator variables:
    # if voter is in ell-cohesive group, then the voter should have
    # strictly less than ell approved candidates in committee
    for i in range(len(profile)):
        model.addConstr(
            (in_group[i] == 1)
            >> (len(profile[i].approved & committee) + 1 <= ell)
        )

    in_cut = model.addVars(profile.num_cand, vtype=gb.GRB.BINARY, name="in_cut")

    # the voters in group should agree on at least ell candidates
    model.addConstr(gb.quicksum(in_cut) >= ell)

    # if a candidate is in the cut, then matrix[voter, cand] = 1 for all voters inside the group
    # thus the following product should match the sum:
    model.addConstr(
        gb.quicksum(in_cut) * gb.quicksum(in_group)
        ==
        sum((sum((approval_matrix[(voter, cand)] * in_group[voter_index]) for voter, voter_index in zip(profile, range(len(profile)))) * in_cut[cand] for cand in range(profile.num_cand))
    ))

    # -----------------------------------------------------------------------------------------
    # The following lines can be alternatively used, instead of the constraint above
    # for much improved readability, but worse run time
    # -----------------------------------------------------------------------------------------

    # add auxiliary binary variables for each voter-cand pair
    # in_group_in_cut = model.addVars(range(len(profile)), range(profile.num_cand), vtype=gb.GRB.BINARY, name="in_group_in_cut")

    # # compute value of auxiliary variables
    # for voter in range(len(profile)):
    #     for cand in range(profile.num_cand):
    #         model.addConstr(in_group_in_cut[voter, cand] == gb.and_(in_group[voter], in_cut[cand]))

    # # if both in_group[voter] = 1 AND in_cut[cand] = 1
    # # then the entry approval_matrix[voter, cand] = 1
    # for voter, voter_index in zip(profile, range(len(profile))):
    #     for cand in range(profile.num_cand):
    #         model.addGenConstrIndicator(in_group_in_cut[voter_index, cand], 1, approval_matrix[voter, cand], gb.GRB.EQUAL, 1)
    
    # -----------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------

    # add objective function
    model.setObjective(ell, gb.GRB.MINIMIZE)

    # optimize the model
    model.optimize()

    # return value based on status code
    # status code 2 means model was solved to optimality, thus an ell-cohesive group
    # that satisfies the condition of EJR was found
    if model.Status == 2:
        return False
    # status code 3 means that model is infeasible, thus no ell-cohesive group
    # that satisfies the condition of EJR was found 
    elif model.Status == 3:
        return True
    else:
        raise RuntimeError(f"Gurobi returned an unexpected status code: {model.Status}")


def _check_PJR_brute_force(profile, committee):
    """Test using brute-force whether a committee satisfies PJR.
    
    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        approval sets of voters
    committee : set
        set of candidates

    Returns
    -------
    bool
    """

    # list of candidates with less than ell approved candidates in committee
    # will not consider voters with >= ell approved candidates in committee, 
    # because this voter will immediately satisfy the condition of PJR
    voters_less_than_ell_approved_candidates = []

    # loop through all ell from 1 to committee size
    for ell in range(1, len(committee) + 1):
        # clear list of candidates to consider
        voters_less_than_ell_approved_candidates.clear()

        # compute minimum group size for this ell
        group_size = math.ceil(ell * (len(profile) / len(committee)))

        # compute list of candidates to consider
        for i in range(len(profile)):
            if len(profile[i].approved & committee) < ell:
                voters_less_than_ell_approved_candidates.append(i)

        # check if an ell-cohesive group can be formed with considered candidates
        if len(voters_less_than_ell_approved_candidates) < group_size:
            # if not possible then simply continue with next ell
            continue

        # check all possible combinations of considered voters,
        # taken (possible group size) at a time
        for combination in itertools.combinations(voters_less_than_ell_approved_candidates, group_size):
            # to calculate the cut of approved candidates for the voters in this considered group
            cut = set()

            # initialize the cut to be the approval set of the first candidate in current combination
            cut = profile[combination[0]].approved

            # calculate the cut over all voters for current combination
            # (also can skip first voter in combination, but inexpensive enough...)
            for j in combination:
                cut = cut & profile[j].approved

            # if size of cut is >= ell, then combination is an ell-cohesive group
            if len(cut) >= ell:
                # now calculate union of approved candidates over all voters inside this ell-cohesive group
                approved_cands_union = set()
                for j in combination:
                    approved_cands_union = approved_cands_union.union(profile[j].approved)

                # if intersection of approved_cands_union with committee yields a set of size
                # strictly less than ell, then this ell-cohesive group violates PJR
                if len(approved_cands_union & committee) < ell:
                    return False

    # if function has not returned by now, then it means that for all ell, 
    # no ell-cohesive group was found among candidates with less than ell
    # approved candidates in committee. Thus committee satisfies PJR
    return True


def _check_PJR_gurobi(profile, committee):
    """Test, by an ILP and the Gurobi solver, whether a committee satisfies PJR.
    
    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        approval sets of voters
    committee : set
        set of candidates

    Returns
    -------
    bool
    """

    if not gurobipy_available:
        raise ImportError("Gurobi (gurobipy) not available.")

    # compute matrix-dictionary for voters approval
    # approval_matrix[(voter, cand)] = 1 if cand is in voter's approval set
    # approval_matrix[(voter, cand)] = 0 otherwise
    approval_matrix = {}

    for voter in profile:
        for cand in range(profile.num_cand):
            if cand in voter.approved:
                approval_matrix[(voter, cand)] = 1
            else:
                approval_matrix[(voter, cand)] = 0

    # compute deterministically array of binary variables that
    # indicate whether a candidate is inside the input committee
    in_committee = []
    for cand in range(profile.num_cand):
        if cand in committee:
            in_committee.append(1)
        else:
            in_committee.append(0)

    # create the model to be optimized
    model = gb.Model()

    # integer variable: ell
    ell = model.addVar(vtype=gb.GRB.INTEGER, name="ell")

    # binary variables: indicate whether a voter is inside the ell-cohesive group
    in_group = model.addVars(len(profile), vtype=gb.GRB.BINARY, name="in_group")

    # constraints: ell has to be between 1 and committeesize inclusive
    model.addConstr(ell >= 1)
    model.addConstr(ell <= len(committee))

    # model the value ceil(ell * (len(profile) / len(committee))) using auxiliary
    # integer variable min_group_size
    # source: https://support.gurobi.com/hc/en-us/community/posts/360054499471-Linear-program-with-ceiling-or-floor-functions-HOW-
    min_group_size = model.addVar(vtype=gb.GRB.INTEGER, name="min_group_size")
    model.addConstr(min_group_size >= (ell * (len(profile) / len(committee))))
    model.addConstr(min_group_size <= (ell * (len(profile) / len(committee))) + 1 - 1e-8)

    # constraint: size of ell-cohesive group should be appropriate wrt ell
    model.addConstr(gb.quicksum(in_group) >= min_group_size)

    # binary variables: indicate whether a candidate is in the interesection 
    # of approved candidates over voters inside the group
    in_cut = model.addVars(profile.num_cand, vtype=gb.GRB.BINARY, name="in_cut")

    # the voters in group should agree on at least ell candidates
    model.addConstr(gb.quicksum(in_cut) >= ell)

    # if a candidate is in the cut, then matrix[voter, cand] = 1 for all voters inside the group
    # thus the following product should match the sum:
    model.addConstr(
        gb.quicksum(in_cut) * gb.quicksum(in_group)
        ==
        sum((sum((approval_matrix[(voter, cand)] * in_group[voter_index]) for voter, voter_index in zip(profile, range(len(profile)))) * in_cut[cand] for cand in range(profile.num_cand))
    ))

    # binary variables: indicate whether a candidate is inside the union
    # of approved candidates, taken over voters that are in the ell-cohesive group
    in_union = model.addVars(profile.num_cand, vtype=gb.GRB.BINARY, name="in_union")

    # compute the in_union variables, depending on the values of in_cut
    for voter_index in range(len(profile)):
        for cand in profile[voter_index].approved:
            model.addConstr((in_group[voter_index] == 1) >> (in_union[cand] == 1))

    # -----------------------------------------------------------------------------------------
    # The following constraint can be omitted from the model for improved run time.
    # The solver will tend to have as few candidates in union as possible,
    # because this increases the chance of finding an ell-cohesive group
    # that satisfies the condition of PJR
    # -----------------------------------------------------------------------------------------
    # constraint to ensure that no in_union variable is wrongly set to 1
    # for cand in range(profile.num_cand):
    #     model.addConstr((gb.quicksum(in_group[voter_index] * approval_matrix[voter, cand] for voter_index, voter in zip(range(len(profile)), profile))) - in_union[cand] >= 0)
    # -----------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------
    
    # constraint to ensure that the intersection between candidates that are in union
    # intersected with the input committee, has size strictly less than ell
    model.addConstr(gb.quicksum(in_union[cand] * in_committee[cand] for cand in range(profile.num_cand)) + 1 <= ell)

    # add objective function
    model.setObjective(ell, gb.GRB.MINIMIZE)

    # optimize the model
    model.optimize()

    # return value based on status code
    # status code 2 means model was solved to optimality, thus an ell-cohesive group
    # that satisfies the condition of PJR was found
    if model.Status == 2:
        return False
    # status code 3 means that model is infeasible, thus no ell-cohesive group
    # that satisfies the condition of PJR was found 
    elif model.Status == 3:
        return True
    else:
        raise RuntimeError(f"Gurobi returned an unexpected status code: {model.Status}")


def _validate_input(profile, committee):
    """Test whether input is valid.
    
    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        approval sets of voters
    committee : set / list / tuple
        set of candidates

    Returns
    -------
    committee : set
        set of candidates
    """

    # check the data types
    if not isinstance(profile, Profile):
        raise TypeError(f"Argument \"profile\" should be instance of {Profile}")

    if not (isinstance(committee, set) or isinstance(committee, list) or isinstance(committee, tuple)):
        raise TypeError(f"Argument \"committee\" should be instance of either {set}, {list} or {tuple}")

    # if argument committee is not of type set, first convert to set
    if not isinstance(committee, set):
        committee = set(committee)

    # check if candidates appearing in committee actually exist within the profile
    # and also whether they are the correct datatype (int)
    for cand in committee:
        if not isinstance(cand, int):
            raise TypeError(f"Elements of \"committee\" should be instance of {int}")

        if cand < 0:
            raise TypeError(f"Elements of \"committee\" should be non-negative integers")

        if cand >= profile.num_cand:
            raise TypeError(f"Candidate {cand} in \"committee\" does not appear in \"profile\"")

    # return the argument committee (possibly changed data type)
    return committee