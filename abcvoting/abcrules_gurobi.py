"""
Approval-based committee (ABC) rules implemented as a integer linear
programs (ILPs) with Gurobi (https://www.gurobi.com/)
"""

from abcvoting.misc import sorted_committees
from abcvoting import scores
import functools

try:
    import gurobipy as gb

    gurobipy_available = True
except ImportError:
    gurobipy_available = False


ACCURACY = 1e-8  # 1e-9 causes problems (some unit tests fail)
CMP_ACCURACY = 10 * ACCURACY  # when comparing float numbers obtained from a MIP


def _optimize_rule_gurobi(
    set_opt_model_func,
    profile,
    committeesize,
    resolute,
    max_num_of_committees,
    name="None",
    committeescorefct=None,
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

    Returns
    -------
    committees : list of sets
        a list of chosen committees, each of them represented as list with candidates named from
        `0` to `num_cand`, profile.cand_names is ignored

    """

    if not gurobipy_available:
        raise ImportError("Gurobi (gurobipy) not available.")

    maxscore = None
    committees = []

    while True:
        model = gb.Model()

        # `in_committee` is a binary variable indicating whether `cand` is in the committee
        in_committee = model.addVars(profile.num_cand, vtype=gb.GRB.BINARY, name="in_committee")

        set_opt_model_func(model, in_committee)

        # find a new committee that has not been found yet by excluding previously found committees
        for committee in committees:
            model.addConstr(
                gb.quicksum(in_committee[cand] for cand in committee) <= committeesize - 1
            )

        model.setParam("OutputFlag", False)
        model.setParam("FeasibilityTol", ACCURACY)
        model.setParam("OptimalityTol", ACCURACY)
        model.setParam("IntFeasTol", ACCURACY)
        model.setParam("MIPGap", ACCURACY)
        model.setParam("PoolSearchMode", 0)
        model.setParam("MIPFocus", 2)  # focus more attention on proving optimality
        model.setParam("IntegralityFocus", 1)

        model.optimize()

        if model.Status not in [2, 3, 4]:
            # model.Status == 2 implies solution found
            # model.Status in [3, 4] implies infeasible --> no more solutions
            # otherwise ...
            raise RuntimeError(
                f"Gurobi returned an unexpected status code: {model.Status}"
                f"Warning: solutions may be incomplete or not optimal (model {name})."
            )
        elif model.Status != 2:
            if len(committees) == 0:
                # we are in the first round of searching for committees
                # and Gurobi didn't find any
                raise RuntimeError(f"Gurobi found no solution (model {name})")
            break

        committee = set(
            cand for cand in profile.candidates if in_committee[cand].Xn >= 1 - ACCURACY
        )
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

        committees.append(committee)

        if resolute:
            break
        if max_num_of_committees is not None and len(committees) >= max_num_of_committees:
            return committees

    return committees


def _gurobi_thiele_methods(
    scorefct_id,
    profile,
    committeesize,
    resolute,
    max_num_of_committees,
):
    def set_opt_model_func(model, in_committee):
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

        max_in_committee = {}
        for i, voter in enumerate(profile):
            # maximum number of approved candidates that this voter can have in a committee
            max_in_committee[voter] = min(len(voter.approved), committeesize)
            for l in range(1, max_in_committee[voter] + 1):
                utility[(voter, l)] = model.addVar(vtype=gb.GRB.BINARY, name=f"utility({i,l})")

        # constraint: the committee has the required size
        model.addConstr(gb.quicksum(in_committee) == committeesize)

        # constraint: utilities are consistent with actual committee
        for voter in profile:
            model.addConstr(
                gb.quicksum(utility[voter, l] for l in range(1, max_in_committee[voter] + 1))
                == gb.quicksum(in_committee[cand] for cand in voter.approved)
            )

        # objective: the Thiele score of the committee
        model.setObjective(
            gb.quicksum(
                float(scorefct(l)) * voter.weight * utility[(voter, l)]
                for voter in profile
                for l in range(1, max_in_committee[voter] + 1)
            ),
            gb.GRB.MAXIMIZE,
        )

    scorefct = scores.get_scorefct(scorefct_id, committeesize)

    score_values = [scorefct(l) for l in range(1, committeesize + 1)]
    if not all(
        first > second or first == second == 0
        for first, second in zip(score_values, score_values[1:])
    ):
        raise ValueError("scorefct must be monotonic decreasing")
    min_score_value = min(val for val in score_values if val > 0)
    if min_score_value < ACCURACY:
        raise ValueError(
            f"Thiele scoring function {scorefct_id} can take smaller values "
            f"(min={min_score_value}) than Gurobi accuracy ({ACCURACY})."
        )

    committees = _optimize_rule_gurobi(
        set_opt_model_func=set_opt_model_func,
        profile=profile,
        committeesize=committeesize,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
        name=scorefct_id,
        committeescorefct=functools.partial(scores.thiele_score, scorefct_id),
    )
    return sorted_committees(committees)


def _gurobi_lexcc(profile, committeesize, resolute, max_num_of_committees):
    def set_opt_model_func(model, in_committee):
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
        iteration = len(satisfaction_constraints)
        scorefcts = [scores.get_scorefct(f"atleast{i+1}") for i in range(iteration + 1)]

        max_in_committee = {}
        for i, voter in enumerate(profile):
            # maximum number of approved candidates that this voter can have in a committee
            max_in_committee[voter] = min(len(voter.approved), committeesize)
            for l in range(1, max_in_committee[voter] + 1):
                utility[(voter, l)] = model.addVar(vtype=gb.GRB.BINARY, name=f"utility({i, l})")

        # constraint: the committee has the required size
        model.addConstr(gb.quicksum(in_committee) == committeesize)

        # constraint: utilities are consistent with actual committee
        for voter in profile:
            model.addConstr(
                gb.quicksum(utility[voter, l] for l in range(1, max_in_committee[voter] + 1))
                == gb.quicksum(in_committee[cand] for cand in voter.approved)
            )

        # additional constraints from previous iterations
        for prev_iteration in range(0, iteration):
            model.addConstr(
                gb.quicksum(
                    float(scorefcts[prev_iteration](l)) * voter.weight * utility[(voter, l)]
                    for voter in profile
                    for l in range(1, max_in_committee[voter] + 1)
                )
                >= satisfaction_constraints[prev_iteration] - ACCURACY
            )

        # objective: the at-least-x score of the committee in iteration x
        model.setObjective(
            gb.quicksum(
                float(scorefcts[iteration](l)) * voter.weight * utility[(voter, l)]
                for voter in profile
                for l in range(1, max_in_committee[voter] + 1)
            ),
            gb.GRB.MAXIMIZE,
        )

    # proceed in `committeesize` many iterations to achieve lexicographic tie-breaking
    satisfaction_constraints = []
    for iteration in range(1, committeesize):
        # in iteration x maximize the number of voters that have at least x approved candidates
        # in the committee
        committees = _optimize_rule_gurobi(
            set_opt_model_func=set_opt_model_func,
            profile=profile,
            committeesize=committeesize,
            resolute=resolute,
            max_num_of_committees=max_num_of_committees,
            name=f"lexcc-atleast{iteration}",
            committeescorefct=functools.partial(scores.thiele_score, f"atleast{iteration}"),
        )
        satisfaction_constraints.append(
            scores.thiele_score(f"atleast{iteration}", profile, committees[0])
        )
    iteration = committeesize
    committees = _optimize_rule_gurobi(
        set_opt_model_func=set_opt_model_func,
        profile=profile,
        committeesize=committeesize,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
        name=f"lexcc-final",
        committeescorefct=functools.partial(scores.thiele_score, f"atleast{committeesize}"),
    )
    satisfaction_constraints.append(
        scores.thiele_score(f"atleast{iteration}", profile, committees[0])
    )
    detailed_info = {"opt_score_vector": satisfaction_constraints}
    return sorted_committees(committees), detailed_info


def _gurobi_monroe(profile, committeesize, resolute, max_num_of_committees):
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

    committees = _optimize_rule_gurobi(
        set_opt_model_func=set_opt_model_func,
        profile=profile,
        committeesize=committeesize,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
        name="Monroe",
        committeescorefct=scores.monroescore,
    )
    return sorted_committees(committees)


def _gurobi_minimaxphragmen(profile, committeesize, resolute, max_num_of_committees):
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

    committees = _optimize_rule_gurobi(
        set_opt_model_func=set_opt_model_func,
        profile=profile,
        committeesize=committeesize,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
        name="minimax-Phragmen",
    )
    return sorted_committees(committees)


def _gurobi_minimaxav(profile, committeesize, resolute, max_num_of_committees):
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

    committees = _optimize_rule_gurobi(
        set_opt_model_func=set_opt_model_func,
        profile=profile,
        committeesize=committeesize,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
        name="Minimax AV",
        committeescorefct=lambda profile, committee: scores.minimaxav_score(profile, committee)
        * -1,  # negative because _optimize_rule_mip maximizes while minimaxav minimizes
    )
    return sorted_committees(committees)
