"""
Axiomatic properties of committees.
"""

import gurobipy as gb
import itertools
import math
from fractions import Fraction
from abcvoting.output import output, WARNING
from abcvoting.misc import str_set_of_candidates, CandidateSet, dominate, powerset


ACCURACY = 1e-8  # 1e-9 causes problems (some unit tests fail)
PROPERTY_NAMES = [
    "pareto",
    "jr",
    "pjr",
    "ejr",
    "ejr+",
    "fjr",
    "priceability",
    "stable-priceability",
    "core",
]


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

    Returns a dictionary with the following keys: "pareto", "jr", "pjr", "ejr", "ejr+",
    "fjr", "priceability", "stable-priceability" and "core".
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

    for property_name in PROPERTY_NAMES:
        results[property_name] = check(property_name, profile, committee)

    description = {
        "pareto": "Pareto optimality",
        "jr": "Justified representation (JR)",
        "pjr": "Proportional justified representation (PJR)",
        "ejr": "Extended justified representation (EJR)",
        "ejr+": "EJR+",
        "fjr": "Full justified representation (FJR)",
        "priceability": "Priceability",
        "stable-priceability": "Stable Priceability",
        "core": "The core",
    }

    # restore output verbosity
    output.set_verbosity(current_verbosity)

    for prop, value in results.items():
        output.info(f"{description[prop]:50s} : {value}")

    return results


def check(property_name, profile, committee, quota=None, algorithm="fastest"):
    """
    Test whether a committee satisfies a given property.

    Parameters
    ----------
    property_name : str
        Name of a property.
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.
    quota : Fraction or float, optional
        The quota, i.e., size of a group, to deserve one committee member.

        Does apply only to some axiomatic properties. If the chosen property does not use a
        quota and `quota` is set to a value, `check()` raises a ValueError.
    algorithm : str, optional
        The algorithm to be used.

    Returns
    -------
    bool
    """

    if property_name not in PROPERTY_NAMES:
        raise ValueError(f"Property {property_name} not known.")

    if property_name == "pareto":
        if quota is not None:
            raise ValueError("check_pareto_optimality() does not use the parameter `quota`.")
        return check_pareto_optimality(profile, committee, algorithm=algorithm)
    elif property_name == "jr":
        return check_JR(profile, committee, quota=quota)
    elif property_name == "pjr":
        return check_PJR(profile, committee, quota=quota, algorithm=algorithm)
    elif property_name == "ejr":
        return check_EJR(profile, committee, quota=quota, algorithm=algorithm)
    elif property_name == "ejr+":
        return check_EJR_plus(profile, committee, quota=quota)
    elif property_name == "fjr":
        return check_FJR(profile, committee, quota=quota, algorithm=algorithm)
    elif property_name == "priceability":
        if quota is not None:
            raise ValueError("check_priceability() does not use the parameter `quota`.")
        return check_priceability(profile, committee, algorithm=algorithm)
    elif property_name == "stable-priceability":
        if quota is not None:
            raise ValueError("check_stable_priceability() does not use the parameter `quota`.")
        return check_stable_priceability(profile, committee, algorithm=algorithm)
    elif property_name == "core":
        return check_core(profile, committee, quota=quota, algorithm=algorithm)
    else:
        raise NotImplementedError(f"Property {property_name} not implemented.")


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
        The algorithm to be used. The available algorithms are
        `gurobi` and `brute-force`.

    Returns
    -------
    bool

    References
    ----------
    Multi-Winner Voting with Approval Preferences.
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>

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
        algorithm = "gurobi"

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


def check_EJR(profile, committee, quota=None, algorithm="fastest"):
    """
    Test whether a committee satisfies Extended Justified Representation (EJR).

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.
    quota : Fraction or float, optional
        The quota, i.e., size of a group, to deserve one committee member.

        Defaults to n/k, i.e., the number of voters divided by the committee size.
    algorithm : str, optional
        The algorithm to be used. The available algorithms are
        `gurobi` and `brute-force`.

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
        >>> profile.add_voters([[0, 1], [0, 1], [0, 1], [0, 1], [2, 3], [2, 3]])
        >>> print(profile)
        profile with 6 voters and 4 candidates:
         voter 0:   {0, 1},
         voter 1:   {0, 1},
         voter 2:   {0, 1},
         voter 3:   {0, 1},
         voter 4:   {2, 3},
         voter 5:   {2, 3}

        >>> output.set_verbosity(DETAILS)  # enable output for check_EJR
        >>> result = check_EJR(profile, committee={0, 2, 3})
        Committee {0, 2, 3} does not satisfy EJR.
         (The 2-cohesive group of voters {0, 1, 2, 3} (66.7% of all voters)
         jointly approve the candidates {0, 1}, but none of them approves 2
         candidates in the committee.)

        >>> result = check_EJR(profile, committee={0, 1, 2})
        Committee {0, 1, 2} satisfies EJR.

    .. testcleanup::

        output.set_verbosity()
    """

    if quota is None:
        standard_quota = True
        quota = Fraction(profile.total_weight(), len(committee))
    else:
        standard_quota = False

    # check that `committee` is a valid input
    committee = CandidateSet(committee, num_cand=profile.num_cand)

    if algorithm == "fastest":
        algorithm = "gurobi"

    if algorithm == "brute-force":
        result, detailed_information = _check_EJR_brute_force(profile, committee, quota)
    elif algorithm == "gurobi":
        result, detailed_information = _check_EJR_gurobi(profile, committee, quota)
    else:
        raise NotImplementedError("Algorithm " + str(algorithm) + " not specified for check_EJR")

    message = f"Committee {str_set_of_candidates(committee)} "
    if result:
        message += "satisfies EJR"
    else:
        message += "does not satisfy EJR"
    if standard_quota:
        message += "."
    else:
        message += f" (for quota = {quota})."
    output.info(message)

    if not result:
        ell = detailed_information["ell"]
        cands = detailed_information["joint_candidates"]
        cohesive_group = detailed_information["cohesive_group"]
        fractional_size = sum(profile[vi].weight for vi in cohesive_group) / profile.total_weight()
        fractional_size = sum(profile[vi].weight for vi in cohesive_group) / profile.total_weight()
        output.details(
            f"(The {ell}-cohesive group of voters {str_set_of_candidates(cohesive_group)}"
            f" ({fractional_size * 100:.1f}% of all voters"
            f"{' (by weight)' if not profile.has_unit_weights() else ''})"
            f" jointly approve the candidates {str_set_of_candidates(cands)}, but none of them "
            f"approves {ell} candidates in the committee.)",
            indent=" ",
        )

    return result


def check_PJR(profile, committee, quota=None, algorithm="fastest"):
    """
    Test whether a committee satisfies Proportional Justified Representation (PJR).

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.
    quota : Fraction or float, optional
        The quota, i.e., size of a group, to deserve one committee member.

        Defaults to n/k, i.e., the number of voters divided by the committee size.
    algorithm : str, optional
        The algorithm to be used. The available algorithms are
        `gurobi` and `brute-force`.

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

        >>> profile = Profile(4)
        >>> profile.add_voters([[0, 1], [0, 1], [0, 1], [0, 1], [2, 3], [2, 3]])
        >>> print(profile)
        profile with 6 voters and 4 candidates:
         voter 0:   {0, 1},
         voter 1:   {0, 1},
         voter 2:   {0, 1},
         voter 3:   {0, 1},
         voter 4:   {2, 3},
         voter 5:   {2, 3}

        >>> output.set_verbosity(DETAILS)  # enable output for check_PJR
        >>> result = check_PJR(profile, committee={0, 2, 3})
        Committee {0, 2, 3} does not satisfy PJR.
        (The 2-cohesive group of voters {0, 1, 2, 3} (66.7% of all voters)
        jointly approve the candidates {0, 1}, but they approve fewer than 2
        candidates in the committee.)

        >>> result = check_PJR(profile, committee={0, 1, 2})
        Committee {0, 1, 2} satisfies PJR.

    .. testcleanup::

        output.set_verbosity()
    """

    if quota is None:
        standard_quota = True
        quota = Fraction(profile.total_weight(), len(committee))
    else:
        standard_quota = False

    # check that `committee` is a valid input
    committee = CandidateSet(committee, num_cand=profile.num_cand)

    if algorithm == "fastest":
        algorithm = "gurobi"

    if algorithm == "brute-force":
        result, detailed_information = _check_PJR_brute_force(profile, committee, quota)
    elif algorithm == "gurobi":
        result, detailed_information = _check_PJR_gurobi(profile, committee, quota)
    else:
        raise NotImplementedError("Algorithm " + str(algorithm) + " not specified for check_PJR")

    message = f"Committee {str_set_of_candidates(committee)} "
    if result:
        message += "satisfies PJR"
    else:
        message += "does not satisfy PJR"
    if standard_quota:
        message += "."
    else:
        message += f" (for quota = {quota})."
    output.info(message)

    if not result:
        ell = detailed_information["ell"]
        cands = detailed_information["joint_candidates"]
        cohesive_group = detailed_information["cohesive_group"]
        fractional_size = sum(profile[vi].weight for vi in cohesive_group) / profile.total_weight()
        output.details(
            f"(The {ell}-cohesive group of voters {str_set_of_candidates(cohesive_group)}"
            f" ({fractional_size * 100:.1f}% of all voters"
            f"{' (by weight)' if not profile.has_unit_weights() else ''})"
            f" jointly approve the candidates {str_set_of_candidates(cands)}, but they "
            f"approve fewer than {ell} candidates in the committee.)"
        )

    return result


def check_JR(profile, committee, quota=None):
    """
    Test whether a committee satisfies Justified Representation (JR).

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.
    quota : Fraction or float, optional
        The quota, i.e., size of a group, to deserve one committee member.

        Defaults to n/k, i.e., the number of voters divided by the committee size.

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

    if quota is None:
        standard_quota = True
        quota = Fraction(profile.total_weight(), len(committee))
    else:
        standard_quota = False

    # check that `committee` is a valid input
    committee = CandidateSet(committee, num_cand=profile.num_cand)

    result, detailed_information = _check_JR(profile, committee, quota=quota)

    message = f"Committee {str_set_of_candidates(committee)} "
    if result:
        message += "satisfies JR"
    else:
        message += "does not satisfy JR"
    if standard_quota:
        message += "."
    else:
        message += f" (for quota = {quota})."
    output.info(message)

    if not result:
        cand = detailed_information["joint_candidate"]
        cohesive_group = detailed_information["cohesive_group"]
        fractional_size = sum(profile[vi].weight for vi in cohesive_group) / profile.total_weight()
        output.details(
            f"(The 1-cohesive group of voters {str_set_of_candidates(cohesive_group)}"
            f" ({fractional_size * 100:.1f}% of all voters"
            f"{' (by weight)' if not profile.has_unit_weights() else ''})"
            f" jointly approve candidate {profile.cand_names[cand]}, but none of them"
            " approve a candidate in the committee.)"
        )

    return result


def _check_JR(profile, committee, quota):
    """
    Test whether a committee satisfies JR.

    Uses the polynomial-time algorithm proposed by Aziz et.al (2017).

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.
    quota : Fraction or float, optional
        The quota, i.e., size of a group, to deserve one committee member.

    Returns
    -------
    bool
    """

    for cand in profile.candidates:
        group = set()
        group_weight = 0
        for vi, voter in enumerate(profile):
            # if current candidate appears in this voter's ballot AND
            # this voter's approval ballot does NOT intersect with input committee
            if (cand in voter.approved) and (len(voter.approved & committee) == 0):
                group.add(vi)
                group_weight += voter.weight

        if group_weight >= quota:
            detailed_information = {"cohesive_group": group, "joint_candidate": cand}
            return False, detailed_information

    # if function has not yet returned by now, then this means no such candidate
    # exists. Then input committee must satisfy JR wrt the input profile
    detailed_information = {}
    return True, detailed_information


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

    # constraint: the condition of having strictly more approved candidates in
    # dominating committee will be satisfied for at least one voter
    model.addConstr(condition_strictly_more.sum() >= 1)

    # constraint: all voters should have at least as many preferred candidates
    # in the dominating committee as in the query committee.
    # for the voter with the condition_strictly_more variable set to 1, it should have
    # at least one more preferred candidate in the dominating committee.
    for i, voter in enumerate(profile):
        model.addConstr(
            gb.quicksum(utility[(voter, x)] for x in range(1, len(committee) + 1))
            >= num_apprvd_cands_query[i] + condition_strictly_more[i]
        )

    # constraint: committee has the right size
    model.addConstr(in_committee.sum() == len(committee))

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
        committee = {cand for cand in profile.candidates if in_committee[cand].Xn >= 0.9}
        detailed_information = {"dominating_committee": committee}
        return False, detailed_information

    # status code 3 means that model is infeasible, thus no dominating committee was found
    if model.Status == 3:
        detailed_information = {}
        return True, detailed_information

    raise RuntimeError(f"Gurobi returned an unexpected status code: {model.Status}")


def _check_EJR_brute_force(profile, committee, quota):
    """
    Test using brute-force whether a committee satisfies EJR.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.
    quota : Fraction or float, optional
        The quota, i.e., size of a group, to deserve one committee member.

    Returns
    -------
    bool
    """

    # largest possible ell such that ell-cohesive groups can exist
    ell_upper_bound = int(profile.total_weight() / quota)

    # loop through all possible ell
    for ell in range(1, ell_upper_bound + 1):
        # list of voters with less than ell approved candidates in committee
        voters_less_than_ell_approved_candidates = []
        voters_less_than_ell_approved_candidates_weight = 0

        # compute minimum group size for this ell
        min_group_size = math.ceil(ell * quota)

        # compute list of voters to consider
        for i, voter in enumerate(profile):
            if len(voter.approved & committee) < ell:
                voters_less_than_ell_approved_candidates.append(i)
                voters_less_than_ell_approved_candidates_weight += voter.weight

        # check if an ell-cohesive group can be formed with considered voters
        if voters_less_than_ell_approved_candidates_weight < min_group_size:
            # if not possible then simply continue with next ell
            continue

        # check all possible combinations of considered voters,
        # taken (possible group size) at a time
        # todo: to support weighted profile, this iterator would need to be adjusted
        for combination in itertools.combinations(
            voters_less_than_ell_approved_candidates, min_group_size
        ):
            # compute set of candidates approved by all voters in combination
            cut = set.intersection(*(profile[vi].approved for vi in combination))

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
    # no ell-cohesive group was found among voters with less than ell
    # approved candidates in committee. Thus committee satisfies EJR
    detailed_information = {}
    return True, detailed_information


def _check_EJR_gurobi(profile, committee, quota):
    """
    Test, by an ILP and the Gurobi solver, whether a committee satisfies EJR.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.
    quota : Fraction or float, optional
        The quota, i.e., size of a group, to deserve one committee member.

    Returns
    -------
    bool
    """

    # create the model to be optimized
    model = gb.Model()

    # integer variable: ell
    ell = model.addVar(vtype=gb.GRB.INTEGER, name="ell")

    # binary variables: indicate whether a voter is inside the ell-cohesive group
    in_group = model.addVars(len(profile), vtype=gb.GRB.BINARY, name="in_group")

    model.addConstr(ell >= 1)
    # largest possible ell such that ell-cohesive groups can exist
    model.addConstr(ell <= int(profile.total_weight() / quota))

    # constraint: size of ell-cohesive group should be appropriate wrt. ell
    model.addConstr(
        gb.quicksum(voter.weight * in_group[vi] for vi, voter in enumerate(profile)) >= ell * quota
    )

    # constraints based on binary indicator variables:
    # if voter is in ell-cohesive group, then the voter should have
    # strictly less than ell approved candidates in committee
    for vi, voter in enumerate(profile):
        model.addConstr((in_group[vi] == 1) >> (len(voter.approved & committee) + 1 <= ell))

    in_cut = model.addVars(profile.num_cand, vtype=gb.GRB.BINARY, name="in_cut")

    # the voters in group should agree on at least ell candidates
    model.addConstr(in_cut.sum() >= ell)

    # candidates in cut should be approved by all voters in group
    for vi, voter in enumerate(profile):
        for cand in profile.candidates:
            if cand not in voter.approved:
                # in_group[vi] implies not in_cut[cand]
                model.addConstr(in_cut[cand] <= 1 - in_group[vi])

    _set_gurobi_model_parameters(model)
    model.optimize()

    # return value based on status code
    # optimality means that an ell-cohesive group
    # that satisfies the condition of EJR was found
    if model.status == gb.GRB.OPTIMAL:
        cohesive_group = {vi for vi, _ in enumerate(profile) if in_group[vi].Xn >= 0.9}
        joint_candidates = {cand for cand in profile.candidates if in_cut[cand].Xn >= 0.9}
        detailed_information = {
            "cohesive_group": cohesive_group,
            "ell": round(ell.Xn),
            "joint_candidates": joint_candidates,
        }
        return False, detailed_information

    # infeasible means that no ell-cohesive group
    # that satisfies the condition of EJR was found
    if model.status == gb.GRB.INFEASIBLE:
        detailed_information = {}
        return True, detailed_information

    raise RuntimeError(f"Gurobi returned an unexpected status code: {model.status}")


def _check_PJR_brute_force(profile, committee, quota):
    """
    Test using brute-force whether a committee satisfies PJR.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.
    quota : Fraction or float, optional
        The quota, i.e., size of a group, to deserve one committee member.

    Returns
    -------
    bool
    """

    # largest possible ell such that ell-cohesive groups can exist
    ell_upper_bound = int(profile.total_weight() / quota)

    # considering ell-cohesive groups
    for ell in range(1, ell_upper_bound + 1):
        # list of voters with less than ell approved candidates in committee
        # will not consider voters with >= ell approved candidates in committee,
        # because this voter will immediately satisfy the condition of PJR
        voters_less_than_ell_approved_candidates = []
        voters_less_than_ell_approved_candidates_weight = 0

        # compute minimum group size for this ell
        min_group_size = math.ceil(ell * quota)

        # compute list of voters to consider
        for vi, voter in enumerate(profile):
            if len(voter.approved & committee) < ell:
                voters_less_than_ell_approved_candidates.append(vi)
                voters_less_than_ell_approved_candidates_weight += voter.weight

        # check if an ell-cohesive group can be formed with considered voters
        if voters_less_than_ell_approved_candidates_weight < min_group_size:
            # if not possible then simply continue with next ell
            continue

        # check all possible combinations of considered voters,
        # taken (possible group size) at a time
        # todo: to support weighted profile, this iterator would need to be adjusted
        for group in itertools.combinations(
            voters_less_than_ell_approved_candidates, min_group_size
        ):
            # find set of candidates that are approved by all voters in group
            cut = set.intersection(*(profile[j].approved for j in group))

            # if size of cut is >= ell, then group is an ell-cohesive group
            if len(cut) >= ell:
                # now calculate union of approved candidates over all voters inside
                # this ell-cohesive group
                approved_cands_union = set()
                for j in group:
                    approved_cands_union = approved_cands_union.union(profile[j].approved)

                # if intersection of `approved_cands_union` with committee yields a set of size
                # strictly less than ell, then this ell-cohesive group violates PJR
                if len(approved_cands_union & committee) < ell:
                    detailed_information = {
                        "cohesive_group": voters_less_than_ell_approved_candidates,
                        "ell": ell,
                        "joint_candidates": cut,
                    }
                    return False, detailed_information

    # if function has not returned by now, then it means that for all ell,
    # no ell-cohesive group was found among voters with less than ell
    # approved candidates in committee. Thus committee satisfies PJR
    detailed_information = {}
    return True, detailed_information


def _check_PJR_gurobi(profile, committee, quota):
    """
    Test with a Gurobi ILP whether a committee satisfies PJR.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.
    quota : Fraction or float, optional
        The quota, i.e., size of a group, to deserve one committee member.

    Returns
    -------
    bool
    """

    # create the model to be optimized
    model = gb.Model()

    # integer variable: ell
    ell = model.addVar(vtype=gb.GRB.INTEGER, name="ell")

    # binary variables: indicate whether a voter is inside the ell-cohesive group
    in_group = model.addVars(len(profile), vtype=gb.GRB.BINARY, name="in_group")

    model.addConstr(ell >= 1)
    # largest possible ell such that ell-cohesive groups can exist
    model.addConstr(ell <= int(profile.total_weight() / quota))

    # constraint: size of ell-cohesive group should be appropriate wrt ell
    model.addConstr(
        gb.quicksum(voter.weight * in_group[vi] for vi, voter in enumerate(profile)) >= ell * quota
    )

    # binary variables: indicate whether a candidate is in the intersection
    # of approved candidates over voters inside the group
    in_cut = model.addVars(profile.num_cand, vtype=gb.GRB.BINARY, name="in_cut")

    # the voters in group should agree on at least ell candidates
    model.addConstr(in_cut.sum() >= ell)

    # candidates in cut should be approved by all voters in group
    for vi, voter in enumerate(profile):
        for cand in profile.candidates:
            if cand not in voter.approved:
                # in_group[vi] implies not in_cut[cand]
                model.addConstr(in_cut[cand] <= 1 - in_group[vi])

    # binary variables: indicate whether a candidate is inside the union
    # of approved candidates, taken over voters that are in the ell-cohesive group
    in_union = model.addVars(profile.num_cand, vtype=gb.GRB.BINARY, name="in_union")

    # compute the in_union variables, depending on the values of in_group
    for vi, voter in enumerate(profile):
        for cand in voter.approved:
            model.addConstr(in_union[cand] >= in_group[vi])

    # constraint to ensure that the intersection between candidates that are in union
    # intersected with the input committee, has size strictly less than ell
    model.addConstr(gb.quicksum(in_union[cand] for cand in committee) + 1 <= ell)

    _set_gurobi_model_parameters(model)
    model.optimize()

    # return value based on status code
    if model.status == gb.GRB.OPTIMAL:
        cohesive_group = {vi for vi, _ in enumerate(profile) if in_group[vi].Xn >= 0.9}
        joint_candidates = {cand for cand in profile.candidates if in_cut[cand].Xn >= 0.9}
        detailed_information = {
            "cohesive_group": cohesive_group,
            "ell": round(ell.Xn),
            "joint_candidates": joint_candidates,
        }
        return False, detailed_information

    if model.status == gb.GRB.INFEASIBLE:
        detailed_information = {}
        return True, detailed_information

    raise RuntimeError(f"Gurobi returned an unexpected status code: {model.status}")


def check_EJR_plus(profile, committee, quota=None):
    """
    Test whether a committee satisfies EJR+.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.
    quota : Fraction or float, optional
        The quota, i.e., size of a group, to deserve one committee member.

        Defaults to n/k, i.e., the number of voters divided by the committee size.

    Returns
    -------
    bool

    References
    ----------
    Brill, M., & Peters, J. (2023).
    Robust and Verifiable Proportionality Axioms for Multiwinner Voting.
    https://arxiv.org/abs/2302.01989
    """

    if quota is None:
        standard_quota = True
        quota = Fraction(profile.total_weight(), len(committee))
    else:
        standard_quota = False

    # check that `committee` is a valid input
    committee = CandidateSet(committee, num_cand=profile.num_cand)

    result, detailed_information = _check_EJR_plus(profile, committee, quota)

    message = f"Committee {str_set_of_candidates(committee)} "
    if result:
        message += "satisfies EJR+"
    else:
        message += "does not satisfy EJR+"
    if standard_quota:
        message += "."
    else:
        message += f" (for quota = {quota})."
    output.info(message)

    if not result:
        cand = detailed_information["joint_candidate"]
        cohesive_group = detailed_information["cohesive_group"]
        ell = detailed_information["ell"]
        fractional_size = sum(profile[vi].weight for vi in cohesive_group) / profile.total_weight()
        output.details(
            f"(The group of voters {str_set_of_candidates(cohesive_group)}"
            f" ({fractional_size * 100:.1f}% of all voters"
            f"{' (by weight)' if not profile.has_unit_weights() else ''})"
            f" deserves {ell} candidates,"
            f" and jointly approve candidate {profile.cand_names[cand]} which is not part of the"
            f" committee, but no member approves at least {ell} members of the committee.)"
        )

    return result


def _check_EJR_plus(profile, committee, quota):
    """
    Test whether a committee satisfies EJR+.

    Uses the polynomial-time algorithm proposed by Brill and Peters (2023).

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.
    quota : Fraction or float, optional
        The quota, i.e., size of a group, to deserve one committee member.

    Returns
    -------
    bool
    """

    # largest possible ell such that ell-cohesive groups can exist
    ell_upper_bound = int(profile.total_weight() / quota)

    for cand in profile.candidates:
        if cand in committee:
            continue
        supporters_by_utility = {ell: set() for ell in range(ell_upper_bound + 1)}
        for vi, voter in enumerate(profile):
            if cand in voter.approved:
                utility = len(voter.approved & committee)
                supporters_by_utility[utility].add(vi)

        group = set()
        for ell in range(ell_upper_bound):
            # group of supporters of cand with utility <= ell
            group |= supporters_by_utility[ell]
            if sum(profile[vi].weight for vi in group) >= (ell + 1) * quota:
                # EJR+ requires someone to get utility at least ell + 1, but no one does
                detailed_information = {
                    "cohesive_group": group,
                    "joint_candidate": cand,
                    "ell": ell + 1,
                }
                return False, detailed_information

    # if function has not yet returned by now, then this means no such candidate
    # has sufficiently many sufficiently unsatisfied supporters.
    # Then input committee must satisfy EJR+
    detailed_information = {}
    return True, detailed_information


def check_priceability(profile, committee, algorithm="fastest", stable=False):
    """
    Test whether a committee satisfies Priceability.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.
    algorithm : str, optional
        The algorithm to be used. Only `gurobi` is available.
    stable : bool, default=False
        Whether to check for stable priceability.

    Returns
    -------
    bool

    References
    ----------
    Multi-Winner Voting with Approval Preferences.
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>
    """

    # check that `committee` is a valid input
    committee = CandidateSet(committee, num_cand=profile.num_cand)

    if algorithm == "fastest":
        algorithm = "gurobi"

    if algorithm == "gurobi":
        result = _check_priceability_gurobi(profile, committee, stable)
    else:
        raise NotImplementedError(f"Algorithm {algorithm} not specified for check_priceability")

    stable_str = "stable "
    if result:
        output.info(
            f"Committee {str_set_of_candidates(committee)} is {stable_str * stable}priceable."
        )
    else:
        output.info(
            f"Committee {str_set_of_candidates(committee)} is not {stable_str * stable}priceable."
        )

    return result


def check_stable_priceability(profile, committee, algorithm="fastest"):
    """
    Test whether a committee satisfies stable Priceability.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.
    algorithm : str, optional
        The algorithm to be used. Only `gurobi` is available.

    Returns
    -------
    bool

    References
    ----------
    Multi-Winner Voting with Approval Preferences.
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>
    """

    return check_priceability(profile, committee, algorithm, stable=True)


def _check_priceability_gurobi(profile, committee, stable=False):
    """
    Test, by an ILP and the Gurobi solver, whether a committee is priceable.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        approval sets of voters
    committee : set
        set of candidates

    Returns
    -------
    bool

    References
    ----------
    Multi-Winner Voting with Approval Preferences.
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>

    Market-Based Explanations of Collective Decisions.
    Dominik Peters, Grzegorz Pierczyski, Nisarg Shah, Piotr Skowron.
    <https://www.cs.toronto.edu/~nisarg/papers/priceability.pdf>
    """

    model = gb.Model()

    approved_candidates = [
        cand for cand in profile.candidates if any(cand in voter.approved for voter in profile)
    ]
    if len(approved_candidates) < len(committee):
        # there are fewer candidates that are approved by at least one voter than candidates
        # in the committee.
        # in this case, return True iff all approved candidates appear in the committee
        # note: the original definition of priceability does not work in this case.
        return all(cand in committee for cand in approved_candidates)

    budget = model.addVar(vtype=gb.GRB.CONTINUOUS)
    payment = {}
    for voter in profile:
        payment[voter] = {}
        for candidate in profile.candidates:
            payment[voter][candidate] = model.addVar(vtype=gb.GRB.CONTINUOUS)

    # condition 1 [from "Multi-Winner Voting with Approval Preferences", Definition 4.8]
    for voter in profile:
        model.addConstr(
            gb.quicksum(payment[voter][candidate] for candidate in profile.candidates)
            <= voter.weight * budget
        )

    # condition 2 [from "Multi-Winner Voting with Approval Preferences", Definition 4.8]
    for voter in profile:
        for candidate in profile.candidates:
            if candidate not in voter.approved:
                model.addConstr(payment[voter][candidate] == 0)

    # condition 3 [from "Multi-Winner Voting with Approval Preferences", Definition 4.8]
    for candidate in profile.candidates:
        if candidate in committee:
            model.addConstr(gb.quicksum(payment[voter][candidate] for voter in profile) == 1)
        else:
            model.addConstr(gb.quicksum(payment[voter][candidate] for voter in profile) == 0)

    if stable:
        # condition 4*
        # [from "Market-Based Explanations of Collective Decisions", Section 3.1, Equation (3)]
        for candidate in profile.candidates:
            if candidate not in committee:
                extrema = []
                for voter in profile:
                    if candidate in voter.approved:
                        extremum = model.addVar(vtype=gb.GRB.CONTINUOUS)
                        extrema.append(extremum)
                        r = model.addVar(vtype=gb.GRB.CONTINUOUS)
                        max_Payment = model.addVar(vtype=gb.GRB.CONTINUOUS)
                        model.addConstr(
                            r
                            == voter.weight * budget
                            - gb.quicksum(
                                payment[voter][committee_member] for committee_member in committee
                            )
                        )
                        model.addGenConstrMax(
                            max_Payment,
                            [payment[voter][committee_member] for committee_member in committee],
                        )
                        model.addGenConstrMax(extremum, [max_Payment, r])
                model.addConstr(gb.quicksum(extrema) <= 1)
    else:
        # condition 4 [from "Multi-Winner Voting with Approval Preferences", Definition 4.8]
        for candidate in profile.candidates:
            if candidate not in committee:
                model.addConstr(
                    gb.quicksum(
                        voter.weight * budget
                        - gb.quicksum(
                            payment[voter][committee_member] for committee_member in committee
                        )
                        for voter in profile
                        if candidate in voter.approved
                    )
                    <= 1
                )

    model.setObjective(budget)
    _set_gurobi_model_parameters(model)
    model.optimize()

    if model.Status == gb.GRB.OPTIMAL:
        # todo: adjust output for weighted profiles
        if profile.has_unit_weights():
            output.details(f"Budget: {budget.X}")

            column_widths = {
                candidate: max(
                    len(str(candidate)),
                    max(len(str(payment[voter][candidate].X)) for voter in payment),
                )
                for candidate in profile.candidates
            }
            column_widths["voter"] = len(str(len(profile)))
            output.details(
                " " * column_widths["voter"]
                + " | "
                + " | ".join(
                    str(i).rjust(column_widths[candidate])
                    for i, candidate in enumerate(profile.candidates)
                )
            )
            for i, voter in enumerate(profile):
                output.details(
                    str(i).rjust(column_widths["voter"])
                    + " | "
                    + " | ".join(
                        str(pay.X).rjust(column_widths[candidate])
                        for candidate, pay in payment[voter].items()
                    )
                )

        return True
    elif model.Status == gb.GRB.INFEASIBLE:
        output.details("No feasible budget and payment function")
        return False
    else:
        raise RuntimeError(f"Gurobi returned an unexpected status code: {model.Status}")


def check_FJR(profile, committee, quota=None, algorithm="fastest"):
    """
    Test whether a committee satisfies Full Justified Representation (FJR).

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.
    quota : Fraction or float, optional
        The quota, i.e., size of a group, to deserve one committee member.

        Defaults to n/k, i.e., the number of voters divided by the committee size.
    algorithm : str, optional
        The algorithm to be used. The available algorithms are
        `gurobi` and `brute-force`.

    Returns
    -------
    bool

    References
    ----------
    Multi-Winner Voting with Approval Preferences.
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>
    """

    if quota is None:
        standard_quota = True
        quota = Fraction(profile.total_weight(), len(committee))
    else:
        standard_quota = False

    # check that `committee` is a valid input
    committee = CandidateSet(committee, num_cand=profile.num_cand)

    if algorithm == "fastest":
        algorithm = "gurobi"

    if algorithm == "brute-force":
        result, detailed_information = _check_FJR_brute_force(profile, committee, quota)
    elif algorithm == "gurobi":
        result, detailed_information = _check_FJR_gurobi(profile, committee, quota)
    else:
        raise NotImplementedError("Algorithm " + str(algorithm) + " not specified for check_FJR")

    message = f"Committee {str_set_of_candidates(committee)} "
    if result:
        message += "satisfies FJR"
    else:
        message += "does not satisfy FJR"
    if standard_quota:
        message += "."
    else:
        message += f" (for quota = {quota})."
    output.info(message)

    if not result:
        ell = detailed_information["ell"]
        beta = detailed_information["beta"]
        cands = detailed_information["joint_candidates"]
        cohesive_group = detailed_information["cohesive_group"]
        fractional_size = sum(profile[vi].weight for vi in cohesive_group) / profile.total_weight()
        output.details(
            f"(The weakly cohesive group of voters {str_set_of_candidates(cohesive_group)}"
            f"({fractional_size * 100:.1f}% of all voters"
            f"{' (by weight)' if not profile.has_unit_weights() else ''})"
            f"each approve at least {beta} of the {ell} candidates {str_set_of_candidates(cands)},"
            f"but all approve at most {beta - 1} candidates in the committee.)",
            indent=" ",
        )
    return result


def _check_FJR_brute_force(profile, committee, quota):
    """Test using brute-force whether a committee satisfies FJR.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        approval sets of voters
    committee : set
        set of candidates
    quota : Fraction or float, optional
        The quota, i.e., size of a group, to deserve one committee member.

    Returns
    -------
    bool

    References
    ----------
    Multi-Winner Voting with Approval Preferences.
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>
    """

    # largest possible size of set T (and beta)
    set_upper_bound = int(profile.total_weight() / quota)

    committee_utility_at_most = {
        utility: set(
            vi for vi, voter in enumerate(profile) if len(voter.approved & committee) <= utility
        )
        for utility in range(set_upper_bound + 1)
    }

    for T in powerset(profile.approved_candidates(), max_size=set_upper_bound):
        T = set(T)
        T_utility_at_least = {
            utility: set(
                vi for vi, voter in enumerate(profile) if len(voter.approved & T) >= utility
            )
            for utility in range(len(T) + 1)
        }
        for beta in range(1, len(T) + 1):
            # find set of voters who have utility at least beta in T
            # and utility at most beta-1 in committee
            cohesive_group = T_utility_at_least[beta] & committee_utility_at_most[beta - 1]

            # is coalition large enough to deserve |T| seats?
            if sum(profile[vi].weight for vi in cohesive_group) >= len(T) * quota:
                detailed_information = {
                    "ell": len(T),
                    "beta": beta,
                    "joint_candidates": T,
                    "cohesive_group": cohesive_group,
                }
                return False, detailed_information

    detailed_information = {}
    return True, detailed_information


def _check_FJR_gurobi(profile, committee, quota):
    """Test, by an ILP and the Gurobi solver, whether a committee satisfies FJR.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        approval sets of voters
    committee : set
        set of candidates
    quota : Fraction or float, optional
        The quota, i.e., size of a group, to deserve one committee member.

    Returns
    -------
    bool

    References
    ----------
    Multi-Winner Voting with Approval Preferences.
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>
    """

    # largest possible size of set T (and beta)
    set_upper_bound = int(profile.total_weight() / quota)

    model = gb.Model()

    set_of_voters = model.addVars(range(len(profile)), vtype=gb.GRB.BINARY)
    set_of_candidates = model.addVars(range(profile.num_cand), vtype=gb.GRB.BINARY)
    beta = model.addVar(lb=1, ub=set_upper_bound, vtype=gb.GRB.INTEGER)
    model.addConstr(beta <= set_of_candidates.sum())

    # coalition large enough to deserve |set_of_candidates| seats
    model.addConstr(
        set_of_candidates.sum() * quota
        <= gb.quicksum(voter.weight * set_of_voters[vi] for vi, voter in enumerate(profile))
    )
    model.addConstr(set_of_voters.sum() >= 1)
    for i, voter in enumerate(profile):
        # if i is in set_of_voters, then:
        # (a) i has utility at most beta-1 in committee
        utility_in_committee = len(committee & voter.approved)
        model.addConstr(utility_in_committee * set_of_voters[i] <= beta - 1)
        # (b) i has utility at least beta in set_of_candidates
        approved_in_set_of_candidates = [
            (c in voter.approved) * set_of_candidates[i] for i, c in enumerate(profile.candidates)
        ]
        model.addConstr(
            gb.quicksum(approved_in_set_of_candidates)
            >= beta - set_upper_bound * (1 - set_of_voters[i])
        )

    _set_gurobi_model_parameters(model)
    model.optimize()

    if model.Status == gb.GRB.OPTIMAL:
        T = [c for c in profile.candidates if set_of_candidates[c].Xn > 0.9]
        detailed_information = {
            "ell": len(T),
            "beta": beta.Xn,
            "joint_candidates": T,
            "cohesive_group": [i for i in range(len(profile)) if set_of_voters[i].Xn > 0.9],
        }
        return False, detailed_information
    elif model.Status == gb.GRB.INFEASIBLE:
        detailed_information = {}
        return True, detailed_information
    else:
        raise RuntimeError(f"Gurobi returned an unexpected status code: {model.Status}")


def check_core(profile, committee, quota=None, algorithm="fastest"):
    """
    Test whether a committee is in the core.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee : iterable of int
        A committee.
    quota : Fraction or float, optional
        The quota, i.e., size of a group, to deserve one committee member.

        Defaults to n/k, i.e., the number of voters divided by the committee size.
    algorithm : str, optional
        The algorithm to be used. The available algorithms are
        `gurobi` and `brute-force`.

    Returns
    -------
    bool

    References
    ----------
    Multi-Winner Voting with Approval Preferences.
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>
    """

    if quota is None:
        standard_quota = True
        quota = Fraction(profile.total_weight(), len(committee))
    else:
        standard_quota = False

    # check that `committee` is a valid input
    committee = CandidateSet(committee, num_cand=profile.num_cand)

    if algorithm == "fastest":
        algorithm = "gurobi"

    if algorithm == "brute-force":
        result, detailed_information = _check_core_brute_force(profile, committee, quota)
    elif algorithm == "gurobi":
        result, detailed_information = _check_core_gurobi(profile, committee, quota)
    else:
        raise NotImplementedError(f"Algorithm {algorithm} not specified for check_core")

    message = f"Committee {str_set_of_candidates(committee)} "
    if result:
        message += "is in the core"
    else:
        message += "is not in the core"
    if standard_quota:
        message += "."
    else:
        message += f" (for quota = {quota})."
    output.info(message)

    if not result:
        objection = detailed_information["objection"]
        coalition = detailed_information["coalition"]
        fractional_size = sum(profile[vi].weight for vi in coalition) / profile.total_weight()
        output.details(
            f"(The group of voters {str_set_of_candidates(coalition)}"
            f" ({fractional_size * 100:.1f}% of all voters"
            f"{' (by weight)' if not profile.has_unit_weights() else ''})"
            f" can block the outcome by proposing {str_set_of_candidates(objection)},"
            f" in which each group member approves strictly more candidates.)"
        )

    return result


def _check_core_brute_force(profile, committee, quota):
    """Test using brute-force whether a committee is in the core.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        approval sets of voters
    committee : set
        set of candidates
    quota : Fraction or float, optional
        The quota, i.e., size of a group, to deserve one committee member.

    Returns
    -------
    bool

    References
    ----------
    Multi-Winner Voting with Approval Preferences.
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>
    """

    max_num_of_candidates = int(profile.total_weight() / quota)

    for cands in powerset(profile.approved_candidates(), max_size=max_num_of_candidates):
        cands = set(cands)
        set_of_voters = [
            vi
            for vi, voter in enumerate(profile)
            if len(voter.approved & cands) > len(voter.approved & committee)
        ]  # set of voters that would profit from `cands`
        if not set_of_voters:
            continue
        if len(cands) * quota <= sum(profile[vi].weight for vi in set_of_voters):
            # a sufficient number of voters would profit from deviating to `cands`
            detailed_information = {"coalition": set_of_voters, "objection": cands}
            return False, detailed_information
    detailed_information = {}
    return True, detailed_information


def _check_core_gurobi(profile, committee, quota):
    """Test, by an ILP and the Gurobi solver, whether a committee is in the core.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        approval sets of voters
    committee : set
        set of candidates
    quota : Fraction or float, optional
        The quota, i.e., size of a group, to deserve one committee member.

    Returns
    -------
    bool

    References
    ----------
    Multi-Winner Voting with Approval Preferences.
    Martin Lackner and Piotr Skowron.
    <http://dx.doi.org/10.1007/978-3-031-09016-5>
    """

    model = gb.Model()

    set_of_voters = model.addVars(range(len(profile)), vtype=gb.GRB.BINARY)
    set_of_candidates = model.addVars(range(profile.num_cand), vtype=gb.GRB.BINARY)

    # set_of_voters is large enough to afford set_of_candidates
    model.addConstr(
        set_of_candidates.sum() * quota
        <= gb.quicksum(voter.weight * set_of_voters[vi] for vi, voter in enumerate(profile))
    )
    model.addConstr(set_of_voters.sum() >= 1)

    # every voter in set_of_voters prefers set_of_candidates to committee
    for i, voter in enumerate(profile):
        approved = [
            (c in voter.approved) * set_of_candidates[i] for i, c in enumerate(profile.candidates)
        ]
        model.addConstr(
            (set_of_voters[i] == 1)
            >> (gb.quicksum(approved) >= len(voter.approved & committee) + 1)
        )

    _set_gurobi_model_parameters(model)
    model.optimize()

    if model.Status == gb.GRB.OPTIMAL:
        coalition = {vi for vi, _ in enumerate(profile) if set_of_voters[vi].Xn >= 0.9}
        objection = {cand for cand in profile.candidates if set_of_candidates[cand].Xn >= 0.9}
        detailed_information = {
            "coalition": coalition,
            "objection": objection,
        }
        return False, detailed_information
    elif model.Status == gb.GRB.INFEASIBLE:
        detailed_information = {}
        return True, detailed_information
    else:
        raise RuntimeError(f"Gurobi returned an unexpected status code: {model.Status}")
