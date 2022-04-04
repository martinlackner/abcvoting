"""
Properties of committees.
"""

import itertools
import math
from abcvoting.output import output, WARNING
from abcvoting.misc import str_set_of_candidates, CandidateSet, dominate

try:
    import gurobipy as gb
except ImportError:
    gb = None


ACCURACY = 1e-8  # 1e-9 causes problems (some unit tests fail)


def _set_gurobi_model_parameters(model):
    model.setParam("OutputFlag", False)
    model.setParam("FeasibilityTol", ACCURACY)
    model.setParam("OptimalityTol", ACCURACY)
    model.setParam("IntFeasTol", ACCURACY)
    model.setParam("MIPGap", ACCURACY)
    model.setParam("PoolSearchMode", 0)
    model.setParam("MIPFocus", 2)  # focus more attention on proving optimality
    model.setParam("IntegralityFocus", 1)


def full_analysis(profile, committee):
    """
    Test all implemented properties for the given committee.

    Returns a dictionary with the following keys: "pareto", "jr", "pjr", and "ejr".
    The values are `True` or `False`, depending on whether this property is satisfied.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.

    Returns
    -------
    dict
    """
    results = {}

    # temporarily no output
    current_verbosity = output.verbosity
    output.set_verbosity(WARNING)

    results["pareto"] = check_pareto_optimality(profile, committee)
    results["jr"] = check_JR(profile, committee)
    results["pjr"] = check_PJR(profile, committee)
    results["ejr"] = check_EJR(profile, committee)

    description = {
        "pareto": "Pareto optimality",
        "jr": "Justified representation (JR)",
        "pjr": "Proportional justified representation (PJR)",
        "ejr": "Extended justified representation (EJR)",
    }

    # restore output verbosity
    output.set_verbosity(current_verbosity)

    for prop, value in results.items():
        output.info(f"{description[prop]:50s} : {value}")


def check_pareto_optimality(profile, committee, algorithm="fastest"):
    """
    Test whether a committee satisfies Pareto optimality.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.
    algorithm : str, optional
        The algorithm to be used.

    Returns
    -------
    bool

    References
    ----------
    Multi-Winner Voting with Approval Preferences.
    Martin Lackner and Piotr Skowron.
    <https://arxiv.org/abs/2007.01795>

    Examples
    --------
    .. doctest::

        >>> from abcvoting.preferences import Profile
        >>> from abcvoting.output import output, DETAILS

        >>> profile = Profile(3)
        >>> profile.add_voters([[0], [0, 2], [1, 2], [1, 2]])
        >>> print(profile)
        profile with 4 voters and 3 candidates:
         voter 0:   {0},
         voter 1:   {0, 2},
         voter 2:   {1, 2},
         voter 3:   {1, 2}

        >>> output.set_verbosity(DETAILS)  # enable output for check_pareto_optimality
        >>> result = check_pareto_optimality(profile, committee={0, 1})
        Committee {0, 1} is not Pareto optimal.
        (It is dominated by the committee {0, 2}.)
        >>> result = check_pareto_optimality(profile, committee={0, 2})
        Committee {0, 2} is Pareto optimal.

    .. testcleanup::

        output.set_verbosity()

    We see that the committee {0, 2} is Pareto optimal, but not the committee {0, 1}.
    """

    # check that `committee` is a valid input
    committee = CandidateSet(committee, num_cand=profile.num_cand)

    if algorithm == "fastest":
        if gb:
            algorithm = "gurobi"
        else:
            algorithm = "brute-force"

    if algorithm == "brute-force":
        result, detailed_information = _check_pareto_optimality_brute_force(profile, committee)
    elif algorithm == "gurobi":
        result, detailed_information = _check_pareto_optimality_gurobi(profile, committee)
    else:
        raise NotImplementedError(
            "Algorithm " + str(algorithm) + " not specified for check_pareto_optimality"
        )

    if result:
        output.info(f"Committee {str_set_of_candidates(committee)} is Pareto optimal.")
    else:
        output.info(f"Committee {str_set_of_candidates(committee)} is not Pareto optimal.")
        dominating_committee = detailed_information["dominating_committee"]
        output.details(
            f"(It is dominated by the committee {str_set_of_candidates(dominating_committee)}.)"
        )

    return result


def check_EJR(profile, committee, algorithm="fastest"):
    """
    Test whether a committee satisfies Extended Justified Representation (EJR).

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.
    algorithm : str, optional
        The algorithm to be used.

    Returns
    -------
    bool

    References
    ----------
    Aziz, H., Brill, M., Conitzer, V., Elkind, E., Freeman, R., & Walsh, T. (2017).
    Justified representation in approval-based committee voting.
    Social Choice and Welfare, 48(2), 461-485.
    https://arxiv.org/abs/1407.8269

    Examples
    --------
    .. doctest::

        >>> from abcvoting.preferences import Profile
        >>> from abcvoting.output import output, DETAILS

        >>> profile = Profile(5)
        >>> profile.add_voters([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [3, 4], [3, 4]])
        >>> print(profile)
        profile with 6 voters and 5 candidates:
         voter 0:   {0, 1, 2},
         voter 1:   {0, 1, 2},
         voter 2:   {0, 1, 2},
         voter 3:   {0, 1, 2},
         voter 4:   {3, 4},
         voter 5:   {3, 4}

        >>> output.set_verbosity(DETAILS)  # enable output for check_EJR
        >>> result = check_EJR(profile, committee={0, 3, 4})
        Committee {0, 3, 4} does not satisfy EJR.
         (The 2-cohesive group of voters {0, 1, 2, 3} (66.7% of all voters)
         jointly approve the candidates {0, 1, 2}, but none of them approves 2
         candidates in the committee.)

        >>> result = check_EJR(profile, committee={1, 2, 3})
        Committee {1, 2, 3} satisfies EJR.

    .. testcleanup::

        output.set_verbosity()
    """

    # check that `committee` is a valid input
    committee = CandidateSet(committee, num_cand=profile.num_cand)

    if algorithm == "fastest":
        if gb:
            algorithm = "gurobi"
        else:
            algorithm = "brute-force"

    if algorithm == "brute-force":
        result, detailed_information = _check_EJR_brute_force(profile, committee)
    elif algorithm == "gurobi":
        result, detailed_information = _check_EJR_gurobi(profile, committee)
    else:
        raise NotImplementedError("Algorithm " + str(algorithm) + " not specified for check_EJR")

    if result:
        output.info(f"Committee {str_set_of_candidates(committee)} satisfies EJR.")
    else:
        output.info(f"Committee {str_set_of_candidates(committee)} does not satisfy EJR.")
        ell = detailed_information["ell"]
        cands = detailed_information["joint_candidates"]
        cohesive_group = detailed_information["cohesive_group"]
        output.details(
            f"(The {ell}-cohesive group of voters {str_set_of_candidates(cohesive_group)}"
            f" ({len(cohesive_group)/len(profile)*100:.1f}% of all voters)"
            f" jointly approve the candidates {str_set_of_candidates(cands)}, but none of them "
            f"approves {ell} candidates in the committee.)",
            indent=" ",
        )

    return result


def check_PJR(profile, committee, algorithm="fastest"):
    """
    Test whether a committee satisfies Proportional Justified Representation (PJR).

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.
    algorithm : str, optional
        The algorithm to be used.

    Returns
    -------
    bool

    References
    ----------
    Sánchez-Fernández, L., Elkind, E., Lackner, M., Fernández, N., Fisteus, J., Val, P. B., &
    Skowron, P. (2017).
    Proportional justified representation.
    In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 31, No. 1).
    https://arxiv.org/abs/1611.09928

    Examples
    --------
    .. doctest::

        >>> from abcvoting.preferences import Profile
        >>> from abcvoting.output import output, DETAILS

        >>> profile = Profile(5)
        >>> profile.add_voters([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [3, 4], [3, 4]])
        >>> print(profile)
        profile with 6 voters and 5 candidates:
         voter 0:   {0, 1, 2},
         voter 1:   {0, 1, 2},
         voter 2:   {0, 1, 2},
         voter 3:   {0, 1, 2},
         voter 4:   {3, 4},
         voter 5:   {3, 4}

        >>> output.set_verbosity(DETAILS)  # enable output for check_PJR
        >>> result = check_PJR(profile, committee={0, 3, 4})
        Committee {0, 3, 4} does not satisfy PJR.
        (The 2-cohesive group of voters {0, 1, 2, 3} (66.7% of all voters)
        jointly approve the candidates {0, 1, 2}, but they approve fewer than
        2 candidates in the committee.)

        >>> result = check_PJR(profile, committee={1, 2, 3})
        Committee {1, 2, 3} satisfies PJR.

    .. testcleanup::

        output.set_verbosity()
    """

    # check that `committee` is a valid input
    committee = CandidateSet(committee, num_cand=profile.num_cand)

    if algorithm == "fastest":
        if gb:
            algorithm = "gurobi"
        else:
            algorithm = "brute-force"

    if algorithm == "brute-force":
        result, detailed_information = _check_PJR_brute_force(profile, committee)
    elif algorithm == "gurobi":
        result, detailed_information = _check_PJR_gurobi(profile, committee)
    else:
        raise NotImplementedError("Algorithm " + str(algorithm) + " not specified for check_PJR")

    if result:
        output.info(f"Committee {str_set_of_candidates(committee)} satisfies PJR.")
    else:
        output.info(f"Committee {str_set_of_candidates(committee)} does not satisfy PJR.")
        ell = detailed_information["ell"]
        cands = detailed_information["joint_candidates"]
        cohesive_group = detailed_information["cohesive_group"]
        output.details(
            f"(The {ell}-cohesive group of voters {str_set_of_candidates(cohesive_group)}"
            f" ({len(cohesive_group)/len(profile)*100:.1f}% of all voters)"
            f" jointly approve the candidates {str_set_of_candidates(cands)}, but they "
            f"approve fewer than {ell} candidates in the committee.)"
        )

    return result


def check_JR(profile, committee):
    """
    Test whether a committee satisfies Justified Representation (JR).

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.

    Returns
    -------
    bool

    References
    ----------
    Aziz, H., Brill, M., Conitzer, V., Elkind, E., Freeman, R., & Walsh, T. (2017).
    Justified representation in approval-based committee voting.
    Social Choice and Welfare, 48(2), 461-485.
    https://arxiv.org/abs/1407.8269

    Examples
    --------
    .. doctest::

        >>> from abcvoting.preferences import Profile
        >>> from abcvoting.output import output, DETAILS

        >>> profile = Profile(4)
        >>> profile.add_voters([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [3], [3]])
        >>> print(profile)
        profile with 6 voters and 4 candidates:
         voter 0:   {0, 1, 2},
         voter 1:   {0, 1, 2},
         voter 2:   {0, 1, 2},
         voter 3:   {0, 1, 2},
         voter 4:   {3},
         voter 5:   {3}

        >>> output.set_verbosity(DETAILS)  # enable output for check_JR
        >>> result = check_JR(profile, committee={0, 1, 2})
        Committee {0, 1, 2} does not satisfy JR.
        (The 1-cohesive group of voters {4, 5} (33.3% of all voters) jointly
        approve candidate 3, but none of them approve a candidate in the
        committee.)

        >>> result = check_JR(profile, committee={1, 2, 3})
        Committee {1, 2, 3} satisfies JR.

    .. testcleanup::

        output.set_verbosity()
    """

    # check that `committee` is a valid input
    committee = CandidateSet(committee, num_cand=profile.num_cand)

    result, detailed_information = _check_JR(profile, committee)

    if result:
        output.info(f"Committee {str_set_of_candidates(committee)} satisfies JR.")
    else:
        output.info(f"Committee {str_set_of_candidates(committee)} does not satisfy JR.")
        cand = detailed_information["joint_candidate"]
        cohesive_group = detailed_information["cohesive_group"]
        output.details(
            f"(The 1-cohesive group of voters {str_set_of_candidates(cohesive_group)}"
            f" ({len(cohesive_group)/len(profile)*100:.1f}% of all voters)"
            f" jointly approve candidate {profile.cand_names[cand]}, but none of them "
            "approve a candidate in the committee.)"
        )

    return result


def _check_pareto_optimality_brute_force(profile, committee):
    """
    Test using brute-force whether a committee is Pareto optimal.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.

    Returns
    -------
    bool
    """
    # iterate through all possible committees
    for other_committee in itertools.combinations(profile.candidates, len(committee)):
        if dominate(profile, other_committee, committee):
            # if a generated committee dominates the "query" committee,
            # then it is not Pareto optimal
            detailed_information = {"dominating_committee": other_committee}
            return False, detailed_information
    # If function has not returned up until now, then no other committee dominates it.
    # It is thus Pareto optimal.
    detailed_information = {}
    return True, detailed_information


def _check_pareto_optimality_gurobi(profile, committee):
    """
    Test, by an ILP and the Gurobi solver, whether a committee is Pareto optimal.

    That is, there is no other committee which dominates it.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.

    Returns
    -------
    bool
    """
    if not gb:
        raise ImportError("Gurobi (gurobipy) not available.")

    # array to store number of approved candidates per voter in the query committee
    num_apprvd_cands_query = [len(voter.approved & committee) for voter in profile]

    model = gb.Model()

    # binary variables: indicate whether voter i approves of
    # at least x candidates in the dominating committee
    utility = {}
    for voter in profile:
        for x in range(1, len(committee) + 1):
            utility[(voter, x)] = model.addVar(vtype=gb.GRB.BINARY)

    # binary variables: indicate whether a candidate is inside the dominating committee
    in_committee = model.addVars(profile.num_cand, vtype=gb.GRB.BINARY, name="in_committee")

    # binary variables: determine for which voter(s) the condition of having strictly
    # more preferred candidates in dominating committee will be satisfied
    condition_strictly_more = model.addVars(
        range(len(profile)), vtype=gb.GRB.BINARY, name="condition_strictly_more"
    )

    # constraint: utility actually matches the number of approved candidates
    # in the dominating committee, for all voters
    for voter in profile:
        model.addConstr(
            gb.quicksum(utility[(voter, x)] for x in range(1, len(committee) + 1))
            == gb.quicksum(in_committee[cand] for cand in voter.approved)
        )

    # constraint: all voters should have at least as many preferred candidates
    # in the dominating committee as in the query committee
    for i, voter in enumerate(profile):
        model.addConstr(
            gb.quicksum(utility[(voter, x)] for x in range(1, len(committee) + 1))
            >= num_apprvd_cands_query[i]
        )

    # constraint: the condition of having strictly more approved candidates in
    # dominating committee will be satisfied for at least one voter
    model.addConstr(gb.quicksum(condition_strictly_more) >= 1)

    # loop through all variables in the condition_strictly_more array (there is one for each voter)
    # if it has value 1, then the condition of having strictly more preferred candidates on the
    # dominating committee has to be satisfied for this voter
    for i, voter in enumerate(profile):
        model.addConstr(
            (condition_strictly_more[i] == 1)
            >> (
                gb.quicksum(utility[(voter, x)] for x in range(1, len(committee) + 1))
                >= num_apprvd_cands_query[i] + 1
            )
        )

    # constraint: committee has the right size
    model.addConstr(gb.quicksum(in_committee) == len(committee))

    # set the objective function
    model.setObjective(
        gb.quicksum(
            utility[(voter, x)] for voter in profile for x in range(1, len(committee) + 1)
        ),
        gb.GRB.MAXIMIZE,
    )

    _set_gurobi_model_parameters(model)
    model.optimize()

    # return value based on status code
    # status code 2 means model was solved to optimality, thus a dominating committee was found
    if model.Status == 2:
        committee = set(cand for cand in profile.candidates if in_committee[cand].Xn >= 0.9)
        detailed_information = {"dominating_committee": committee}
        return False, detailed_information

    # status code 3 means that model is infeasible, thus no dominating committee was found
    if model.Status == 3:
        detailed_information = {}
        return True, detailed_information

    raise RuntimeError(f"Gurobi returned an unexpected status code: {model.Status}")


def _check_EJR_brute_force(profile, committee):
    """
    Test using brute-force whether a committee satisfies EJR.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.

    Returns
    -------
    bool
    """

    # should check for ell from 1 until committee size
    ell_upper_bound = len(committee) + 1

    # loop through all possible ell
    for ell in range(1, ell_upper_bound):
        # list of candidates with less than ell approved candidates in committee
        voters_less_than_ell_approved_candidates = []

        # compute minimum group size for this ell
        group_size = math.ceil(ell * (len(profile) / len(committee)))

        # compute list of candidates to consider
        for i, voter in enumerate(profile):
            if len(voter.approved & committee) < ell:
                voters_less_than_ell_approved_candidates.append(i)

        # check if an ell-cohesive group can be formed with considered candidates
        if len(voters_less_than_ell_approved_candidates) < group_size:
            # if not possible then simply continue with next ell
            continue

        # check all possible combinations of considered voters,
        # taken (possible group size) at a time
        for combination in itertools.combinations(
            voters_less_than_ell_approved_candidates, group_size
        ):
            # to calculate the cut of approved candidates for the considered voters
            # initialize the cut to be the approval set of the first candidate in current
            # combination
            cut = set(profile[combination[0]].approved)

            # calculate the cut over all voters for current combination
            # (also can skip first voter in combination, but inexpensive enough...)
            for j in combination:
                cut = cut & profile[j].approved

            # if size of cut is >= ell, then combination is an ell-cohesive group
            if len(cut) >= ell:
                # we have found combination to be an ell-cohesive set, with no voter having
                # at least ell approved candidates in committee. Thus EJR fails
                detailed_information = {
                    "cohesive_group": voters_less_than_ell_approved_candidates,
                    "ell": ell,
                    "joint_candidates": cut,
                }
                return False, detailed_information

    # if function has not returned by now, then it means that for all ell,
    # no ell-cohesive group was found among candidates with less than ell
    # approved candidates in committee. Thus committee satisfies EJR
    detailed_information = {}
    return True, detailed_information


def _check_EJR_gurobi(profile, committee):
    """
    Test, by an ILP and the Gurobi solver, whether a committee satisfies EJR.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.

    Returns
    -------
    bool
    """

    if not gb:
        raise ImportError("Gurobi (gurobipy) not available.")

    # compute matrix-dictionary for voters approval
    # approval_matrix[(voter, cand)] = 1 if cand is in voter's approval set
    # approval_matrix[(voter, cand)] = 0 otherwise
    approval_matrix = {}
    for voter in profile:
        for cand in profile.candidates:
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

    # constraint: size of ell-cohesive group should be appropriate wrt. ell
    model.addConstr(gb.quicksum(in_group) >= ell * len(profile) / len(committee))

    # constraints based on binary indicator variables:
    # if voter is in ell-cohesive group, then the voter should have
    # strictly less than ell approved candidates in committee
    for vi, voter in enumerate(profile):
        model.addConstr((in_group[vi] == 1) >> (len(voter.approved & committee) + 1 <= ell))

    in_cut = model.addVars(profile.num_cand, vtype=gb.GRB.BINARY, name="in_cut")

    # the voters in group should agree on at least ell candidates
    model.addConstr(gb.quicksum(in_cut) >= ell)

    # if a candidate is in the cut, then `approval_matrix[(voter, cand)]` *must be* 1 for all
    # voters inside the group
    for voter_index, voter in enumerate(profile):
        for cand in profile.candidates:
            if approval_matrix[(voter, cand)] == 0:
                model.addConstr(in_cut[cand] + in_group[voter_index] <= 1.5)  # not both true

    # model.setObjective(ell, gb.GRB.MINIMIZE)

    _set_gurobi_model_parameters(model)
    model.optimize()

    # return value based on status code
    # status code 2 means model was solved to optimality, thus an ell-cohesive group
    # that satisfies the condition of EJR was found
    if model.Status == 2:
        cohesive_group = set(vi for vi, _ in enumerate(profile) if in_group[vi].Xn >= 0.9)
        joint_candidates = set(cand for cand in profile.candidates if in_cut[cand].Xn >= 0.9)
        detailed_information = {
            "cohesive_group": cohesive_group,
            "ell": round(ell.Xn),
            "joint_candidates": joint_candidates,
        }
        return False, detailed_information

    # status code 3 means that model is infeasible, thus no ell-cohesive group
    # that satisfies the condition of EJR was found
    if model.Status == 3:
        detailed_information = {}
        return True, detailed_information

    raise RuntimeError(f"Gurobi returned an unexpected status code: {model.Status}")


def _check_PJR_brute_force(profile, committee):
    """
    Test using brute-force whether a committee satisfies PJR.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.

    Returns
    -------
    bool
    """

    # considering ell-cohesive groups
    for ell in range(1, len(committee) + 1):
        # list of candidates with less than ell approved candidates in committee
        # will not consider voters with >= ell approved candidates in committee,
        # because this voter will immediately satisfy the condition of PJR
        voters_less_than_ell_approved_candidates = []

        # compute minimum group size for this ell
        group_size = math.ceil(ell * (len(profile) / len(committee)))

        # compute list of candidates to consider
        for vi, voter in enumerate(profile):
            if len(voter.approved & committee) < ell:
                voters_less_than_ell_approved_candidates.append(vi)

        # check if an ell-cohesive group can be formed with considered candidates
        if len(voters_less_than_ell_approved_candidates) < group_size:
            # if not possible then simply continue with next ell
            continue

        # check all possible combinations of considered voters,
        # taken (possible group size) at a time
        for group in itertools.combinations(voters_less_than_ell_approved_candidates, group_size):
            # initialize the cut to be the approval set of the first voter in the group
            cut = set(profile[group[0]].approved)

            # calculate the cut over all voters for current group
            for j in group:
                cut = cut & profile[j].approved

            # if size of cut is >= ell, then group is an ell-cohesive group
            if len(cut) >= ell:
                # now calculate union of approved candidates over all voters inside
                # this ell-cohesive group
                approved_cands_union = set()
                for j in group:
                    approved_cands_union = approved_cands_union.union(profile[j].approved)

                # if intersection of approved_cands_union with committee yields a set of size
                # strictly less than ell, then this ell-cohesive group violates PJR
                if len(approved_cands_union & committee) < ell:
                    detailed_information = {
                        "cohesive_group": voters_less_than_ell_approved_candidates,
                        "ell": ell,
                        "joint_candidates": cut,
                    }
                    return False, detailed_information

    # if function has not returned by now, then it means that for all ell,
    # no ell-cohesive group was found among candidates with less than ell
    # approved candidates in committee. Thus committee satisfies PJR
    detailed_information = {}
    return True, detailed_information


def _check_PJR_gurobi(profile, committee):
    """
    Test with a Gurobi ILP whether a committee satisfies PJR.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.

    Returns
    -------
    bool
    """
    if not gb:
        raise ImportError("Gurobi (gurobipy) not available.")

    # compute matrix-dictionary for voters approval
    # approval_matrix[(voter, cand)] = 1 if cand is in voter's approval set
    # approval_matrix[(voter, cand)] = 0 otherwise
    approval_matrix = {}

    for voter in profile:
        for cand in profile.candidates:
            if cand in voter.approved:
                approval_matrix[(voter, cand)] = 1
            else:
                approval_matrix[(voter, cand)] = 0

    # compute deterministically array of binary variables that
    # indicate whether a candidate is inside the input committee
    in_committee = []
    for cand in profile.candidates:
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

    # constraint: size of ell-cohesive group should be appropriate wrt ell
    model.addConstr(gb.quicksum(in_group) >= ell * len(profile) / len(committee))

    # binary variables: indicate whether a candidate is in the intersection
    # of approved candidates over voters inside the group
    in_cut = model.addVars(profile.num_cand, vtype=gb.GRB.BINARY, name="in_cut")

    # the voters in group should agree on at least ell candidates
    model.addConstr(gb.quicksum(in_cut) >= ell)

    # if a candidate is in the cut, then `approval_matrix[(voter, cand)]` *must be* 1 for all
    # voters inside the group
    for voter_index, voter in enumerate(profile):
        for cand in profile.candidates:
            if approval_matrix[(voter, cand)] == 0:
                model.addConstr(in_cut[cand] + in_group[voter_index] <= 1.5)  # not both true

    # binary variables: indicate whether a candidate is inside the union
    # of approved candidates, taken over voters that are in the ell-cohesive group
    in_union = model.addVars(profile.num_cand, vtype=gb.GRB.BINARY, name="in_union")

    # compute the in_union variables, depending on the values of in_cut
    for vi, voter in enumerate(profile):
        for cand in voter.approved:
            model.addConstr((in_group[vi] == 1) >> (in_union[cand] == 1))

    # constraint to ensure that the intersection between candidates that are in union
    # intersected with the input committee, has size strictly less than ell
    model.addConstr(
        gb.quicksum(in_union[cand] * in_committee[cand] for cand in profile.candidates) + 1 <= ell
    )

    # model.setObjective(ell, gb.GRB.MINIMIZE)

    _set_gurobi_model_parameters(model)
    model.optimize()

    # return value based on status code
    # status code 2 means model was solved to optimality, thus an ell-cohesive group
    # that satisfies the condition of PJR was found
    if model.Status == 2:
        cohesive_group = set(vi for vi, _ in enumerate(profile) if in_group[vi].Xn >= 0.9)
        joint_candidates = set(cand for cand in profile.candidates if in_cut[cand].Xn >= 0.9)
        detailed_information = {
            "cohesive_group": cohesive_group,
            "ell": round(ell.Xn),
            "joint_candidates": joint_candidates,
        }
        return False, detailed_information

    # status code 3 means that model is infeasible, thus no ell-cohesive group
    # that satisfies the condition of PJR was found
    if model.Status == 3:
        detailed_information = {}
        return True, detailed_information

    raise RuntimeError(f"Gurobi returned an unexpected status code: {model.Status}")


def _check_JR(profile, committee):
    """
    Test whether a committee satisfies JR.

    Uses the polynomial-time algorithm proposed by Aziz et.al (2017).

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.

    Returns
    -------
    bool
    """

    for cand in profile.candidates:
        group = set()
        for vi, voter in enumerate(profile):
            # if current candidate appears in this voter's ballot AND
            # this voter's approval ballot does NOT intersect with input committee
            if (cand in voter.approved) and (len(voter.approved & committee) == 0):
                group.add(vi)

        if len(group) * len(committee) >= len(profile):  # |group| >= num_voters / |committee|
            detailed_information = {"cohesive_group": group, "joint_candidate": cand}
            return False, detailed_information

    # if function has not yet returned by now, then this means no such candidate
    # exists. Then input committee must satisfy JR wrt the input profile
    detailed_information = {}
    return True, detailed_information
