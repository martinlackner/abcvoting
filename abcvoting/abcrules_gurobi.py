"""ABC rules implemented as integer linear programs (ILPs) with Gurobi."""

import functools
import gurobipy as gb
import itertools
import math
from abcvoting.misc import sorted_committees
from abcvoting import scores
from abcvoting import misc
from abcvoting.output import output


ACCURACY = 1e-8  # 1e-9 causes problems (some unit tests fail)
CMP_ACCURACY = 10 * ACCURACY  # when comparing float numbers obtained from a MIP
LEXICOGRAPHIC_BLOCK_SIZE = (
    10  # The maximal number of candidates optimized in one step while lexicographic tiebreaking
)


def _set_gurobi_model_parameters(model):
    model.setParam("OutputFlag", False)
    model.setParam("FeasibilityTol", ACCURACY)
    model.setParam("OptimalityTol", ACCURACY)
    model.setParam("IntFeasTol", ACCURACY)
    model.setParam("MIPGap", ACCURACY)
    model.setParam("PoolSearchMode", 0)
    model.setParam("MIPFocus", 2)  # focus more attention on proving optimality
    model.setParam("IntegralityFocus", 1)


def _optimize_rule_gurobi(
    set_opt_model_func,
    profile,
    committeesize,
    resolute,
    max_num_of_committees,
    name="None",
    committeescorefct=None,
    lexicographic_tiebreaking=False,
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
    resolute : bool
    max_num_of_committees : int
        maximum number of committees this method returns, value can be None
    name : str
        name of the model, used for error messages
    committeescorefct : callable
        a function used to compute the score of a committee
    lexicographic_tiebreaking : bool
        Require lexicographic tiebreaking among tied committees.

    Returns
    -------
    committees : list of set
        a list of winning committees,
        each of them represented as set of integers from `0` to `num_cand` - 1

    maxscore : float
        best objective value returned by ILP

    """

    maxscore = None
    committees = []

    model = gb.Model()

    # `in_committee` is a binary variable indicating whether `cand` is in the committee
    in_committee = model.addVars(profile.num_cand, vtype=gb.GRB.BINARY, name="in_committee")

    set_opt_model_func(model, in_committee)
    _set_gurobi_model_parameters(model)

    while True:
        model.optimize()

        if model.Status not in [gb.GRB.OPTIMAL, gb.GRB.INFEASIBLE]:
            raise RuntimeError(
                f"Gurobi returned an unexpected status code: {model.Status}\n"
                f"Warning: solutions may be incomplete or not optimal (model {name})."
            )
        if model.Status != gb.GRB.OPTIMAL:
            if len(committees) == 0:
                # we are in the first round of searching for committees
                # and Gurobi didn't find any
                raise RuntimeError(f"Gurobi found no solution (model {name})")
            break

        committee = {
            cand
            for cand in profile.candidates
            if in_committee[cand].Xn >= 0.9
            # this should be >= 1 - ACCURACY, but apparently it is not necessarily the case that
            # integers are only ACCURACY apart from either 0 or 1
        }
        if len(committee) != committeesize:
            raise RuntimeError(
                "_optimize_rule_gurobi() produced a committee with "
                f"fewer than `committeesize` members (model {name}).\n"
                + "\n".join(
                    f"({v.varName}, {v.x})" for v in model.getVars() if "in_committee" in v.varName
                )
            )

        if committeescorefct is None:
            objective_value = model.objVal  # numeric value from MIP
        else:
            objective_value = committeescorefct(profile, committee)  # exact value

        if maxscore is None:
            maxscore = objective_value
        elif (committeescorefct is not None and objective_value > maxscore) or (
            committeescorefct is None and objective_value > maxscore + CMP_ACCURACY
        ):
            raise RuntimeError(
                "Gurobi found a solution better than a previous optimum. This "
                f"should not happen (previous optimal score: {maxscore}, "
                f"new optimal score: {model.objVal}, model {name})."
            )
        elif (committeescorefct is not None and objective_value < maxscore) or (
            committeescorefct is None and objective_value < maxscore - CMP_ACCURACY
        ):
            # no longer optimal
            break

        if lexicographic_tiebreaking:
            lex_model = model.copy()
            lex_model.addConstr(lex_model.getObjective() == maxscore)

            in_committee_lex = [
                lex_model.getVarByName(f"in_committee[{cand}]") for cand in range(profile.num_cand)
            ]

            for step_start in range(0, profile.num_cand, LEXICOGRAPHIC_BLOCK_SIZE):
                step_end = min(step_start + LEXICOGRAPHIC_BLOCK_SIZE, profile.num_cand)
                current_block = [in_committee_lex[idx] for idx in range(step_start, step_end)]

                lex_objective_expr = gb.quicksum(
                    (2 ** (len(current_block) - idx - 1)) * current_block[idx]
                    for idx in range(len(current_block))
                )

                lex_model.setObjective(lex_objective_expr, gb.GRB.MAXIMIZE)
                lex_model.optimize()

                if lex_model.Status != gb.GRB.OPTIMAL:
                    raise RuntimeError("Lex optimization failed.")

                # Bound the variables of the current block
                for var in current_block:
                    val = var.X
                    if val >= 0.9:
                        var.lb = 1
                        var.ub = 1
                    else:
                        var.lb = 0
                        var.ub = 0

            # Recompute Comittee after lexicographic tiebreaking
            committee = {cand for cand in profile.candidates if in_committee_lex[cand].X == 1}

        committees.append(committee)

        if resolute:
            break
        if max_num_of_committees is not None and len(committees) >= max_num_of_committees:
            return committees, maxscore

        # find a new committee that has not been found yet by excluding previously found committees
        model.addConstr(gb.quicksum(in_committee[cand] for cand in committee) <= committeesize - 1)

    return committees, maxscore


def _gurobi_thiele_methods(
    scorefct_id,
    profile,
    committeesize,
    resolute,
    max_num_of_committees,
    lexicographic_tiebreaking=False,
):
    def set_opt_model_func(model, in_committee):
        # utility[(voter, x)] contains (intended binary) variables counting the number of approved
        # candidates in the selected committee by `voter`. This utility[(voter, x)] is true for
        # exactly the number of candidates in the committee approved by `voter` for all
        # x = 1...committeesize.
        #
        # If marginal_scorefct(x) > 0 for x >= 1, we assume that marginal_scorefct is monotonic
        # decreasing and therefore in combination with the objective function the following
        # interpretation is valid:
        # utility[(voter, x)] indicates whether `voter` approves at least x candidates in the
        # committee (this is the case for scorefct_id "pav", "slav" or "geom").

        utility = {}

        max_in_committee = {}
        for i, voter in enumerate(profile):
            # maximum number of approved candidates that this voter can have in a committee
            max_in_committee[voter] = min(len(voter.approved), committeesize)
            for x in range(1, max_in_committee[voter] + 1):
                utility[(voter, x)] = model.addVar(vtype=gb.GRB.BINARY, name=f"utility({i, x})")

        # constraint: the committee has the required size
        model.addConstr(in_committee.sum() == committeesize)

        # constraint: utilities are consistent with actual committee
        for voter in profile:
            model.addConstr(
                gb.quicksum(utility[voter, x] for x in range(1, max_in_committee[voter] + 1))
                == gb.quicksum(in_committee[cand] for cand in voter.approved)
            )

        # objective: the Thiele score of the committee
        model.setObjective(
            gb.quicksum(
                float(marginal_scorefct(x)) * voter.weight * utility[(voter, x)]
                for voter in profile
                for x in range(1, max_in_committee[voter] + 1)
            ),
            gb.GRB.MAXIMIZE,
        )

    marginal_scorefct = scores.get_marginal_scorefct(scorefct_id, committeesize)

    score_values = [marginal_scorefct(x) for x in range(1, committeesize + 1)]
    if not all(
        first > second or first == second == 0
        for first, second in zip(score_values, score_values[1:])
    ):
        raise ValueError("The score function must be monotonic decreasing")
    min_score_value = min(val for val in score_values if val > 0)
    if min_score_value < ACCURACY:
        output.warning(
            f"Thiele scoring function {scorefct_id} can take smaller values "
            f"(min={min_score_value}) than Gurobi accuracy ({ACCURACY})."
        )

    committees, _ = _optimize_rule_gurobi(
        set_opt_model_func=set_opt_model_func,
        profile=profile,
        committeesize=committeesize,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
        name=scorefct_id,
        committeescorefct=functools.partial(scores.thiele_score, scorefct_id),
        lexicographic_tiebreaking=lexicographic_tiebreaking,
    )
    return sorted_committees(committees)


def _gurobi_lexcc(
    profile, committeesize, resolute, max_num_of_committees, lexicographic_tiebreaking=False
):
    def set_opt_model_func(model, in_committee):
        # utility[(voter, x)] contains (intended binary) variables counting the number of approved
        # candidates in the selected committee by `voter`. This utility[(voter, x)] is true for
        # exactly the number of candidates in the committee approved by `voter` for all
        # x = 1...committeesize.

        utility = {}
        iteration = len(satisfaction_constraints)
        scorefcts = [scores.get_marginal_scorefct(f"atleast{i + 1}") for i in range(iteration + 1)]

        max_in_committee = {}
        for i, voter in enumerate(profile):
            # maximum number of approved candidates that this voter can have in a committee
            max_in_committee[voter] = min(len(voter.approved), committeesize)
            for x in range(1, max_in_committee[voter] + 1):
                utility[(voter, x)] = model.addVar(vtype=gb.GRB.BINARY, name=f"utility({i, x})")

        # constraint: the committee has the required size
        model.addConstr(in_committee.sum() == committeesize)

        # constraint: utilities are consistent with actual committee
        for voter in profile:
            model.addConstr(
                gb.quicksum(utility[voter, x] for x in range(1, max_in_committee[voter] + 1))
                == gb.quicksum(in_committee[cand] for cand in voter.approved)
            )

        # additional constraints from previous iterations
        for prev_iteration in range(iteration):
            model.addConstr(
                gb.quicksum(
                    float(scorefcts[prev_iteration](x)) * voter.weight * utility[(voter, x)]
                    for voter in profile
                    for x in range(1, max_in_committee[voter] + 1)
                )
                >= satisfaction_constraints[prev_iteration] - ACCURACY
            )

        # objective: the at-least-y score of the committee in iteration y
        model.setObjective(
            gb.quicksum(
                float(scorefcts[iteration](x)) * voter.weight * utility[(voter, x)]
                for voter in profile
                for x in range(1, max_in_committee[voter] + 1)
            ),
            gb.GRB.MAXIMIZE,
        )

    # proceed in `committeesize` many iterations to achieve lexicographic tie-breaking
    satisfaction_constraints = []
    for iteration in range(1, committeesize):
        # in iteration x maximize the number of voters that have at least x approved candidates
        # in the committee
        committees, _ = _optimize_rule_gurobi(
            set_opt_model_func=set_opt_model_func,
            profile=profile,
            committeesize=committeesize,
            resolute=True,
            max_num_of_committees=None,
            name=f"lexcc-atleast{iteration}",
            committeescorefct=functools.partial(scores.thiele_score, f"atleast{iteration}"),
        )
        satisfaction_constraints.append(
            scores.thiele_score(f"atleast{iteration}", profile, committees[0])
        )
    iteration = committeesize
    committees, _ = _optimize_rule_gurobi(
        set_opt_model_func=set_opt_model_func,
        profile=profile,
        committeesize=committeesize,
        resolute=resolute,
        lexicographic_tiebreaking=lexicographic_tiebreaking,
        max_num_of_committees=max_num_of_committees,
        name="lexcc-final",
        committeescorefct=functools.partial(scores.thiele_score, f"atleast{committeesize}"),
    )
    satisfaction_constraints.append(
        scores.thiele_score(f"atleast{iteration}", profile, committees[0])
    )
    detailed_info = {"opt_score_vector": satisfaction_constraints}
    return sorted_committees(committees), detailed_info


def _gurobi_monroe(
    profile, committeesize, resolute, max_num_of_committees, lexicographic_tiebreaking=False
):
    def set_opt_model_func(model, in_committee):
        num_voters = len(profile)

        # optimization goal: variable "satisfaction"
        satisfaction = model.addVar(ub=num_voters, vtype=gb.GRB.INTEGER, name="satisfaction")

        model.addConstr(
            gb.quicksum(in_committee[cand] for cand in profile.candidates) == committeesize
        )

        # a partition of voters into committeesize many sets
        partition = model.addVars(
            profile.num_cand, len(profile), vtype=gb.GRB.INTEGER, lb=0, name="partition"
        )
        for i in range(len(profile)):
            # every voter has to be part of a voter partition set
            model.addConstr(gb.quicksum(partition[(cand, i)] for cand in profile.candidates) == 1)
        for cand in profile.candidates:
            # every voter set in the partition has to contain
            # at least (num_voters // committeesize) candidates
            model.addConstr(
                gb.quicksum(partition[(cand, j)] for j in range(len(profile)))
                >= (num_voters // committeesize - num_voters * (1 - in_committee[cand]))
            )
            # every voter set in the partition has to contain
            # at most ceil(num_voters/committeesize) candidates
            model.addConstr(
                gb.quicksum(partition[(cand, j)] for j in range(len(profile)))
                <= (
                    num_voters // committeesize
                    + bool(num_voters % committeesize)
                    + num_voters * (1 - in_committee[cand])
                )
            )
            # if in_committee[i] = 0 then partition[(i,j) = 0
            model.addConstr(
                gb.quicksum(partition[(cand, j)] for j in range(len(profile)))
                <= num_voters * in_committee[cand]
            )

        # constraint for objective variable "satisfaction"
        model.addConstr(
            gb.quicksum(
                partition[(cand, j)] * (cand in profile[j].approved)
                for j in range(len(profile))
                for cand in profile.candidates
            )
            >= satisfaction
        )

        # optimization objective
        model.setObjective(satisfaction, gb.GRB.MAXIMIZE)

    committees, _ = _optimize_rule_gurobi(
        set_opt_model_func=set_opt_model_func,
        profile=profile,
        committeesize=committeesize,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
        name="Monroe",
        committeescorefct=scores.monroescore,
        lexicographic_tiebreaking=lexicographic_tiebreaking,
    )
    return sorted_committees(committees)


def _gurobi_minimaxphragmen(
    profile, committeesize, resolute, max_num_of_committees, lexicographic_tiebreaking=False
):
    """ILP for Phragmen's minimax rule (minimax-Phragmen), using Gurobi.

    Minimizes the maximum load.

    Warning: does not include the lexicographic optimization as specified
    in Markus Brill, Rupert Freeman, Svante Janson and Martin Lackner.
    Phragmen's Voting Methods and Justified Representation.
    https://arxiv.org/abs/2102.12305
    Instead: minimizes the maximum load (without consideration of the
             second-, third-, ...-largest load
    """

    def set_opt_model_func(model, in_committee):
        load = {}
        for cand in profile.candidates:
            for i, voter in enumerate(profile):
                load[(voter, cand)] = model.addVar(ub=1.0, lb=0.0, name=f"load{i}-{cand}")

        # constraint: the committee has the required size
        model.addConstr(
            gb.quicksum(in_committee[cand] for cand in profile.candidates) == committeesize
        )

        for cand in profile.candidates:
            for voter in profile:
                if cand not in voter.approved:
                    load[(voter, cand)] = 0

        # a candidate's load is distributed among his approvers
        for cand in profile.candidates:
            model.addConstr(
                gb.quicksum(
                    voter.weight * load[(voter, cand)]
                    for voter in profile
                    if cand in profile.candidates
                )
                >= in_committee[cand]
            )

        loadbound = model.addVar(lb=0, ub=committeesize, name="loadbound")
        for voter in profile:
            model.addConstr(
                gb.quicksum(load[(voter, cand)] for cand in voter.approved) <= loadbound
            )

        # maximizing the negative distance makes code more similar to the other methods here
        model.setObjective(-loadbound, gb.GRB.MAXIMIZE)

    # check if a sufficient number of candidates is approved
    if len(profile.approved_candidates()) < committeesize:
        # An insufficient number of candidates is approved:
        # Committees consist of all approved candidates plus
        # a correct number of unapproved candidates
        approved_candidates = profile.approved_candidates()
        remaining_candidates = [
            cand for cand in profile.candidates if cand not in approved_candidates
        ]
        num_missing_candidates = committeesize - len(approved_candidates)

        if resolute:
            return [approved_candidates | set(remaining_candidates[:num_missing_candidates])]

        committees = [
            approved_candidates | set(extra)
            for extra in itertools.combinations(remaining_candidates, num_missing_candidates)
        ]

        if max_num_of_committees:
            return committees[:max_num_of_committees]
        else:
            return committees

    committees, _ = _optimize_rule_gurobi(
        set_opt_model_func=set_opt_model_func,
        profile=profile,
        committeesize=committeesize,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
        name="minimax-Phragmen",
        lexicographic_tiebreaking=lexicographic_tiebreaking,
    )
    return sorted_committees(committees)


def _gurobi_leximaxphragmen(
    profile, committeesize, resolute, max_num_of_committees, lexicographic_tiebreaking
):
    def set_opt_model_func(model, in_committee):
        load = {}
        loadbound_constraint = {}
        for cand in profile.candidates:
            for i, voter in enumerate(profile):
                load[(voter, cand)] = model.addVar(ub=1.0, lb=0.0, name=f"load{i}-{cand}")

        for i, _ in enumerate(profile):
            for j, _ in enumerate(profile):
                loadbound_constraint[(i, j)] = model.addVar(
                    vtype=gb.GRB.BINARY, name=f"loadbound_constraint({i, j})"
                )

        for i, _ in enumerate(profile):
            model.addConstr(
                gb.quicksum(loadbound_constraint[(i, j)] for j, _ in enumerate(profile)) == 1
            )
            model.addConstr(
                gb.quicksum(loadbound_constraint[(j, i)] for j, _ in enumerate(profile)) == 1
            )

        # constraint: the committee has the required size
        model.addConstr(
            gb.quicksum(in_committee[cand] for cand in profile.candidates) == committeesize
        )

        for cand in profile.candidates:
            for voter in profile:
                if cand not in voter.approved:
                    load[(voter, cand)] = 0

        # a candidate's load is distributed among his approvers
        for cand in profile.candidates:
            model.addConstr(
                gb.quicksum(
                    voter.weight * load[(voter, cand)]
                    for voter in profile
                    if cand in profile.candidates
                )
                >= in_committee[cand]
            )

        for i, _ in enumerate(loadbounds):
            for j, voter in enumerate(profile):
                model.addConstr(
                    gb.quicksum(load[(voter, cand)] for cand in voter.approved)
                    <= loadbounds[i]
                    + (1 - loadbound_constraint[(i, j)]) * committeesize
                    + ACCURACY
                    # constraint applies only if loadbound_constraint[(i, voter)] == 1
                )

        newloadbound = model.addVar(lb=0, ub=committeesize, name="new loadbound")
        for j, voter in enumerate(profile):
            model.addConstr(
                gb.quicksum(load[(voter, cand)] for cand in voter.approved)
                <= newloadbound
                + gb.quicksum(
                    loadbound_constraint[(i, j)] * committeesize for i in range(len(loadbounds))
                )
            )

        # maximizing the negative distance makes code more similar to the other methods here
        model.setObjective(-newloadbound, gb.GRB.MAXIMIZE)

    # check if a sufficient number of candidates is approved
    approved_candidates = profile.approved_candidates()
    if len(approved_candidates) < committeesize:
        # An insufficient number of candidates is approved:
        # Committees consist of all approved candidates plus
        #  a correct number of unapproved candidates
        remaining_candidates = [
            cand for cand in profile.candidates if cand not in approved_candidates
        ]
        num_missing_candidates = committeesize - len(approved_candidates)

        if resolute:
            return [approved_candidates | set(remaining_candidates[:num_missing_candidates])]

        return [
            approved_candidates | set(extra)
            for extra in itertools.combinations(remaining_candidates, num_missing_candidates)
        ]

    loadbounds = []
    for iteration in range(len(profile) - 1):
        # in interation we enforce a new loadbound.
        # first for all voters, then for all except one, then for all except two, etc.
        committees, neg_loadbound = _optimize_rule_gurobi(
            set_opt_model_func=set_opt_model_func,
            profile=profile,
            committeesize=committeesize,
            resolute=True,
            max_num_of_committees=None,
            name=f"leximaxphragmen-iteration{iteration}",
        )
        if math.isclose(neg_loadbound, 0, rel_tol=CMP_ACCURACY, abs_tol=CMP_ACCURACY):
            # all other voters have a load of zero, no further loadbounds constraints required
            break
        loadbounds.append(-neg_loadbound)

    committees, _ = _optimize_rule_gurobi(
        set_opt_model_func=set_opt_model_func,
        profile=profile,
        committeesize=committeesize,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
        name="leximaxphragmen-final",
        lexicographic_tiebreaking=lexicographic_tiebreaking,
    )

    return sorted_committees(committees)


def _gurobi_maximin_support_scorefct(profile, base_committee):
    """Uses an LP to compute the maximin support values obtained when adding any
    candidate to the committee.

    Based on the LP described in the proof of Theorem 4.2 of
    L. Sánchez-Fernández et al.
    "The maximin support method: an extension of the D'Hondt method to
    approval-based multiwinner elections"
    Mathematical Programming (2022)
    """

    scores = [0] * profile.num_cand
    remaining_candidates = [cand for cand in profile.candidates if cand not in base_committee]

    for added_cand in remaining_candidates:
        committee = set(base_committee) | {added_cand}

        model = gb.Model()
        _set_gurobi_model_parameters(model)

        minimum = model.addVar(lb=0, name="minimum")  # named "s" in the paper

        f = model.addVars(len(profile), profile.num_cand, lb=0, name="fractional_assignment")
        for vi, voter in enumerate(profile):
            if voter.approved & committee:
                model.addConstr(
                    gb.quicksum(f[vi, cand] for cand in voter.approved & committee) == voter.weight
                )

        for cand in committee:
            model.addConstr(
                gb.quicksum(
                    f[vi, cand] for vi, voter in enumerate(profile) if cand in voter.approved
                )
                >= minimum
            )

        model.setObjective(minimum, gb.GRB.MAXIMIZE)
        model.optimize()
        if model.status != gb.GRB.OPTIMAL:
            raise RuntimeError(
                f"Gurobi returned an unexpected status code: {model.Status}"
                " while computing the maximin support score."
            )

        scores[added_cand] = minimum.x

    return scores


def _gurobi_minimaxav(
    profile, committeesize, resolute, max_num_of_committees, lexicographic_tiebreaking=False
):
    def set_opt_model_func(model, in_committee):
        max_hamming_distance = model.addVar(
            lb=0,
            ub=profile.num_cand,
            vtype=gb.GRB.INTEGER,
            name="max_hamming_distance",
        )

        model.addConstr(
            gb.quicksum(in_committee[cand] for cand in profile.candidates) == committeesize
        )

        for voter in profile:
            not_approved = [cand for cand in profile.candidates if cand not in voter.approved]
            # maximum Hamming distance is greater of equal than the Hamming distances
            # between individual voters and the committee
            model.addConstr(
                max_hamming_distance
                >= gb.quicksum(1 - in_committee[cand] for cand in voter.approved)
                + gb.quicksum(in_committee[cand] for cand in not_approved)
            )

        # maximizing the negative distance makes code more similar to the other methods here
        model.setObjective(-max_hamming_distance, gb.GRB.MAXIMIZE)

    committees, _ = _optimize_rule_gurobi(
        set_opt_model_func=set_opt_model_func,
        profile=profile,
        committeesize=committeesize,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
        name="Minimax AV",
        committeescorefct=lambda profile, committee: -scores.minimaxav_score(profile, committee),
        # negative because _optimize_rule_gurobi maximizes while minimaxav minimizes
        lexicographic_tiebreaking=lexicographic_tiebreaking,
    )
    return sorted_committees(committees)


def _gurobi_lexminimaxav(
    profile, committeesize, resolute, max_num_of_committees, lexicographic_tiebreaking=False
):
    def set_opt_model_func(model, in_committee):
        voteratmostdistances = {}

        for i, voter in enumerate(profile):
            for dist in range(profile.num_cand + 1):
                voteratmostdistances[(i, dist)] = model.addVar(
                    vtype=gb.GRB.BINARY, name=f"atmostdistance({i, dist})"
                )
                if dist >= len(voter.approved) + committeesize:
                    # distances are always <= len(voter.approved) + committeesize
                    voteratmostdistances[(i, dist)] = 1
                if dist < abs(len(voter.approved) - committeesize):
                    # distances are never < abs(len(voter.approved) - committeesize)
                    voteratmostdistances[(i, dist)] = 0

        # constraint: the committee has the required size
        model.addConstr(in_committee.sum() == committeesize)

        # constraint: distances are consistent with actual committee
        for i, voter in enumerate(profile):
            not_approved = [cand for cand in profile.candidates if cand not in voter.approved]
            for dist in range(profile.num_cand + 1):
                if isinstance(voteratmostdistances[(i, dist)], int):
                    # trivially satisfied
                    continue
                model.addConstr(
                    (voteratmostdistances[(i, dist)] == 1)
                    >> (
                        gb.quicksum(1 - in_committee[cand] for cand in voter.approved)
                        + gb.quicksum(in_committee[cand] for cand in not_approved)
                        <= dist
                    )
                )

        # additional constraints from previous iterations
        for dist, num_voters_achieving_distance in hammingdistance_constraints.items():
            model.addConstr(
                gb.quicksum(voteratmostdistances[(i, dist)] for i, _ in enumerate(profile))
                >= num_voters_achieving_distance - ACCURACY
            )

        new_distance = min(hammingdistance_constraints.keys()) - 1
        # objective: maximize number of voters achieving at most distance `new_distance`
        model.setObjective(
            gb.quicksum(voteratmostdistances[(i, new_distance)] for i, _ in enumerate(profile)),
            gb.GRB.MAXIMIZE,
        )

    # compute minimaxav as baseline and then improve on it
    committees = _gurobi_minimaxav(
        profile, committeesize, resolute=True, max_num_of_committees=None
    )
    maxdistance = scores.minimaxav_score(profile, committees[0])
    # all voters have at most this distance
    hammingdistance_constraints = {maxdistance: len(profile)}
    for distance in range(maxdistance - 1, -1, -1):
        # in iteration `distance` we maximize the number of voters that have at
        # most a Hamming distance of `distance` to the committee
        if distance == 0:
            # last iteration
            _resolute = resolute
            _max_num_of_committees = max_num_of_committees
        else:
            _resolute = True
            _max_num_of_committees = None
        committees, _ = _optimize_rule_gurobi(
            set_opt_model_func=set_opt_model_func,
            profile=profile,
            committeesize=committeesize,
            resolute=_resolute,
            max_num_of_committees=_max_num_of_committees,
            lexicographic_tiebreaking=True,
            name=f"lexminimaxav-atmostdistance{distance}",
            committeescorefct=functools.partial(
                scores.num_voters_with_upper_bounded_hamming_distance, distance
            ),
        )
        num_voters_achieving_distance = scores.num_voters_with_upper_bounded_hamming_distance(
            distance, profile, committees[0]
        )
        hammingdistance_constraints[distance] = num_voters_achieving_distance
    committees = sorted_committees(committees)
    detailed_info = {
        "hammingdistance_constraints": hammingdistance_constraints,
        "opt_distances": [misc.hamming(voter.approved, committees[0]) for voter in profile],
    }
    return committees, detailed_info
