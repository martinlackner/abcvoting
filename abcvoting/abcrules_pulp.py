"""ABC rules implemented as integer linear programs (ILPs) with Pulp."""

import functools
import pulp
import itertools
import math
from abcvoting.misc import sorted_committees
from abcvoting import scores
from abcvoting import misc
from abcvoting.output import output


ACCURACY = 1e-5  # 1e-9 causes problems (some unit tests fail)
CMP_ACCURACY = 10 * ACCURACY  # when comparing float numbers obtained from a MIP
LEXICOGRAPHIC_BLOCK_SIZE = (
    10  # The maximal number of candidates optimized in one step while lexicographic tiebreaking
)


def _optimize_rule_pulp(
    set_opt_model_func,
    profile,
    committeesize,
    resolute,
    max_num_of_committees,
    solver_id,
    name="None",
    committeescorefct=None,
    lexicographic_tiebreaking=False,
):
    maxscore = None
    committees = []

    prob = pulp.LpProblem(name=name, sense=pulp.LpMaximize)
    in_committee = {
        cand: pulp.LpVariable(f"in_committee[{cand}]", cat="Binary") for cand in profile.candidates
    }

    set_opt_model_func(prob, in_committee)

    while True:
        try:
            prob.solve(get_solver(solver_id))
            if pulp.LpStatus[prob.status] != "Optimal":
                raise pulp.PulpSolverError("Status not Optimal")
        except pulp.PulpSolverError:
            if len(committees) == 0:
                raise RuntimeError(f"Solver found no solution (model {name})")
            break

        committee = {cand for cand in profile.candidates if pulp.value(in_committee[cand]) >= 0.9}

        if len(committee) != committeesize:
            raise RuntimeError(
                f"_optimize_rule_pulp() produced a committee with incorrect size "
                f"(model {name}).\n"
                + "\n".join(f"({v.name}, {pulp.value(v)})" for v in in_committee)
            )

        if committeescorefct is None:
            objective_value = pulp.value(prob.objective)
        else:
            objective_value = committeescorefct(profile, committee)

        if maxscore is None:
            maxscore = objective_value
        elif (committeescorefct and objective_value > maxscore) or (
            not committeescorefct and objective_value > maxscore + CMP_ACCURACY
        ):
            raise RuntimeError(
                "Solver found a solution better than a previous optimum. This "
                f"should not happen (previous: {maxscore}, new: {objective_value}, model {name})"
            )
        elif (committeescorefct and objective_value < maxscore) or (
            not committeescorefct and objective_value < maxscore - CMP_ACCURACY
        ):
            break  # no longer optimal

        if lexicographic_tiebreaking:
            # Build new problem to lex-optimize
            lex_prob = pulp.LpProblem(name + "_lex", pulp.LpMaximize)
            in_committee_lex = [
                pulp.LpVariable(f"in_committee_lex[{cand}]", cat="Binary")
                for cand in profile.candidates
            ]

            # Add Blocking Constraints
            for comm in committees:
                lex_prob += (
                    pulp.lpSum(in_committee_lex[cand] for cand in comm) <= committeesize - 1
                )

            # Add all Modelspecific Constraints
            set_opt_model_func(lex_prob, in_committee_lex)

            # Make old Objective to Constraint
            lex_prob += (lex_prob.objective == maxscore), "FixObjectiveValue"

            for step_start in range(0, profile.num_cand, LEXICOGRAPHIC_BLOCK_SIZE):
                step_end = min(step_start + LEXICOGRAPHIC_BLOCK_SIZE, profile.num_cand)
                current_block = [in_committee_lex[i] for i in range(step_start, step_end)]

                lex_objective_expr = pulp.lpSum(
                    2 ** (len(current_block) - idx - 1) * current_block[idx]
                    for idx in range(len(current_block))
                )
                lex_prob.setObjective(lex_objective_expr)
                lex_prob.solve(get_solver(solver_id))

                if pulp.LpStatus[lex_prob.status] != "Optimal":
                    raise RuntimeError("Lex optimization failed.")

                # Fix current block
                for var in current_block:
                    val = pulp.value(var)
                    if val >= 0.9:
                        var.lowBound = 1
                        var.upBound = 1
                    else:
                        var.lowBound = 0
                        var.upBound = 0

            committee = {
                cand for cand in profile.candidates if pulp.value(in_committee_lex[cand]) >= 0.9
            }

        committees.append(committee)

        if resolute:
            break
        if max_num_of_committees is not None and len(committees) >= max_num_of_committees:
            return committees, maxscore

        # Block previously found committee
        prob += pulp.lpSum(in_committee[cand] for cand in committee) <= committeesize - 1

    return (
        sorted_committees(committees) if lexicographic_tiebreaking else committees,
        maxscore,
    )


def get_solver(solver_id):
    if solver_id == "highs":
        return pulp.HiGHS(msg=False)
    else:
        raise ValueError(f"Solver {solver_id} not known in Python Pulp.")


def _pulp_thiele_methods(
    scorefct_id,
    profile,
    committeesize,
    resolute,
    max_num_of_committees,
    solver_id,
    lexicographic_tiebreaking=False,
):
    def set_opt_model_func(prob, in_committee):
        utility = {}

        max_in_committee = {}
        for i, voter in enumerate(profile):
            max_in_committee[voter] = min(len(voter.approved), committeesize)
            for x in range(1, max_in_committee[voter] + 1):
                utility[(voter, x)] = pulp.LpVariable(f"utility({i},{x})", cat="Binary")

        # Constraint: committee size
        prob += pulp.lpSum(in_committee) == committeesize

        # Constraint: sum of utility equals number of approved candidates in committee
        for voter in profile:
            prob += pulp.lpSum(
                utility[(voter, x)] for x in range(1, max_in_committee[voter] + 1)
            ) == pulp.lpSum(in_committee[cand] for cand in voter.approved)

        # Objective: Thiele score
        prob += pulp.lpSum(
            float(marginal_scorefct(x)) * voter.weight * utility[(voter, x)]
            for voter in profile
            for x in range(1, max_in_committee[voter] + 1)
        )

    marginal_scorefct = scores.get_marginal_scorefct(scorefct_id, committeesize)

    score_values = [marginal_scorefct(x) for x in range(1, committeesize + 1)]
    if not all(
        first > second or first == second == 0
        for first, second in zip(score_values, score_values[1:])
    ):
        raise ValueError("The score function must be monotonic decreasing")

    min_score_value = min((val for val in score_values if val > 0), default=0)
    if min_score_value < ACCURACY:
        output.warning(
            f"Thiele scoring function {scorefct_id} can take smaller values "
            f"(min={min_score_value}) than solver accuracy ({ACCURACY})."
        )

    # Call the generic pulp optimizer
    committees, _ = _optimize_rule_pulp(
        set_opt_model_func,
        profile,
        committeesize,
        resolute,
        max_num_of_committees,
        solver_id=solver_id,
        name=scorefct_id,
        committeescorefct=functools.partial(scores.thiele_score, scorefct_id),
        lexicographic_tiebreaking=lexicographic_tiebreaking,
    )

    return sorted_committees(committees)


def _pulp_lexcc(
    profile,
    committeesize,
    resolute,
    max_num_of_committees,
    solver_id,
    lexicographic_tiebreaking=False,
):
    def set_opt_model_func(prob, in_committee):
        utility = {}
        iteration = len(satisfaction_constraints)
        scorefcts = [scores.get_marginal_scorefct(f"atleast{i + 1}") for i in range(iteration + 1)]

        max_in_committee = {}
        for i, voter in enumerate(profile):
            max_in_committee[voter] = min(len(voter.approved), committeesize)
            for x in range(1, max_in_committee[voter] + 1):
                utility[(voter, x)] = pulp.LpVariable(f"utility({i},{x})", cat="Binary")

        # Committee size constraint
        prob += pulp.lpSum(in_committee) == committeesize

        # Consistency constraint: sum of utility matches approved candidates in committee
        for voter in profile:
            prob += pulp.lpSum(
                utility[(voter, x)] for x in range(1, max_in_committee[voter] + 1)
            ) == pulp.lpSum(in_committee[c] for c in voter.approved)

        # Add satisfaction constraints from previous iterations
        for prev_iter in range(iteration):
            prob += (
                pulp.lpSum(
                    float(scorefcts[prev_iter](x)) * voter.weight * utility[(voter, x)]
                    for voter in profile
                    for x in range(1, max_in_committee[voter] + 1)
                )
                >= satisfaction_constraints[prev_iter] - ACCURACY
            )

        # Set objective for current iteration
        prob += pulp.lpSum(
            float(scorefcts[iteration](x)) * voter.weight * utility[(voter, x)]
            for voter in profile
            for x in range(1, max_in_committee[voter] + 1)
        )

    # Run committeesize iterations of lexicographic optimization
    satisfaction_constraints = []
    for iteration in range(1, committeesize):
        committees, _ = _optimize_rule_pulp(
            set_opt_model_func=set_opt_model_func,
            profile=profile,
            committeesize=committeesize,
            resolute=True,
            max_num_of_committees=None,
            solver_id=solver_id,
            name=f"lexcc-atleast{iteration}",
            committeescorefct=functools.partial(scores.thiele_score, f"atleast{iteration}"),
        )
        satisfaction_constraints.append(
            scores.thiele_score(f"atleast{iteration}", profile, committees[0])
        )

    # Final iteration
    iteration = committeesize
    committees, _ = _optimize_rule_pulp(
        set_opt_model_func=set_opt_model_func,
        profile=profile,
        committeesize=committeesize,
        resolute=resolute,
        lexicographic_tiebreaking=lexicographic_tiebreaking,
        max_num_of_committees=max_num_of_committees,
        solver_id=solver_id,
        name="lexcc-final",
        committeescorefct=functools.partial(scores.thiele_score, f"atleast{iteration}"),
    )
    satisfaction_constraints.append(
        scores.thiele_score(f"atleast{iteration}", profile, committees[0])
    )

    detailed_info = {"opt_score_vector": satisfaction_constraints}
    return sorted_committees(committees), detailed_info


def _pulp_monroe(
    profile,
    committeesize,
    resolute,
    max_num_of_committees,
    solver_id,
    lexicographic_tiebreaking=False,
):
    def set_opt_model_func(prob, in_committee):
        num_voters = len(profile)
        candidates = profile.candidates

        # Satisfaction variable
        satisfaction = pulp.LpVariable(
            "satisfaction", lowBound=0, upBound=num_voters, cat="Integer"
        )

        # Constraint: committee size
        prob += pulp.lpSum(in_committee[c] for c in candidates) == committeesize

        # Partition variables: how voters are assigned to candidates
        partition = {
            (c, i): pulp.LpVariable(f"partition({c},{i})", lowBound=0, cat="Integer")
            for c in candidates
            for i in range(num_voters)
        }

        # Each voter is assigned to exactly one candidate
        for i in range(num_voters):
            prob += pulp.lpSum(partition[c, i] for c in candidates) == 1

        # Capacity constraints
        min_q = num_voters // committeesize
        max_q = min_q + (1 if num_voters % committeesize != 0 else 0)

        for c in candidates:
            # Minimum quota if candidate is selected
            prob += pulp.lpSum(
                partition[c, j] for j in range(num_voters)
            ) >= min_q - num_voters * (1 - in_committee[c])
            # Maximum quota
            prob += pulp.lpSum(
                partition[c, j] for j in range(num_voters)
            ) <= max_q + num_voters * (1 - in_committee[c])
            # Block partitions if candidate not selected
            prob += (
                pulp.lpSum(partition[c, j] for j in range(num_voters))
                <= num_voters * in_committee[c]
            )

        # Satisfaction expression: voters assigned to approved candidates
        prob += (
            pulp.lpSum(
                partition[(c, j)] * int(c in profile[j].approved)
                for j in range(num_voters)
                for c in candidates
            )
            >= satisfaction
        )

        # Objective: maximize satisfaction
        prob += satisfaction

    committees, _ = _optimize_rule_pulp(
        set_opt_model_func=set_opt_model_func,
        profile=profile,
        committeesize=committeesize,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
        solver_id=solver_id,
        name="Monroe",
        committeescorefct=scores.monroescore,
        lexicographic_tiebreaking=lexicographic_tiebreaking,
    )

    return sorted_committees(committees)


def _pulp_minimaxphragmen(
    profile,
    committeesize,
    resolute,
    max_num_of_committees,
    solver_id,
    lexicographic_tiebreaking=False,
):
    def set_opt_model_func(prob, in_committee):
        load = {}
        for cand in profile.candidates:
            for i, voter in enumerate(profile):
                if cand in voter.approved:
                    load[(voter, cand)] = pulp.LpVariable(
                        f"load({i},{cand})", lowBound=0, upBound=1
                    )
                else:
                    load[(voter, cand)] = 0  # not a variable, fixed to 0

        # Committee size constraint
        prob += pulp.lpSum(in_committee[cand] for cand in profile.candidates) == committeesize

        # Candidate's load is distributed among approvers
        for cand in profile.candidates:
            prob += (
                pulp.lpSum(
                    voter.weight * load[(voter, cand)]
                    for voter in profile
                    if cand in voter.approved
                )
                >= in_committee[cand]
            )

        # Loadbound variable for maximum voter load
        loadbound = pulp.LpVariable("loadbound", lowBound=0, upBound=committeesize)

        # Constraint: Each voter's total load must be below loadbound
        for voter in profile:
            prob += pulp.lpSum(load[(voter, cand)] for cand in voter.approved) <= loadbound

        # Objective: minimize maximum load => maximize negative loadbound
        prob += -loadbound

    # Handle insufficient approved candidates
    if len(profile.approved_candidates()) < committeesize:
        approved_candidates = profile.approved_candidates()
        remaining_candidates = [c for c in profile.candidates if c not in approved_candidates]
        num_missing = committeesize - len(approved_candidates)

        if resolute:
            return [approved_candidates | set(remaining_candidates[:num_missing])]

        committees = [
            approved_candidates | set(extra)
            for extra in itertools.combinations(remaining_candidates, num_missing)
        ]
        if max_num_of_committees:
            return committees[:max_num_of_committees]
        else:
            return committees

    committees, _ = _optimize_rule_pulp(
        set_opt_model_func=set_opt_model_func,
        profile=profile,
        committeesize=committeesize,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
        solver_id=solver_id,
        name="minimax-Phragmen",
        lexicographic_tiebreaking=lexicographic_tiebreaking,
    )

    return sorted_committees(committees)


def _pulp_leximaxphragmen(
    profile,
    committeesize,
    resolute,
    max_num_of_committees,
    solver_id,
    lexicographic_tiebreaking,
):
    loadbounds = []

    def set_opt_model_func(prob, in_committee):
        num_voters = len(profile)
        candidates = profile.candidates
        load = {}
        loadbound_constraint = {}

        # Define load variables
        for cand in candidates:
            for i, voter in enumerate(profile):
                if cand in voter.approved:
                    load[(voter, cand)] = pulp.LpVariable(
                        f"load({i},{cand})", lowBound=0.0, upBound=1.0
                    )
                else:
                    load[(voter, cand)] = 0  # constant

        # Define binary constraint indicator variables
        for i in range(num_voters):
            for j in range(num_voters):
                loadbound_constraint[(i, j)] = pulp.LpVariable(
                    f"loadbound_constraint({i},{j})", cat="Binary"
                )

        # Each column and row in constraint matrix has exactly one 1
        for i in range(num_voters):
            prob += pulp.lpSum(loadbound_constraint[(i, j)] for j in range(num_voters)) == 1
            prob += pulp.lpSum(loadbound_constraint[(j, i)] for j in range(num_voters)) == 1

        # Committee size
        prob += pulp.lpSum(in_committee[c] for c in candidates) == committeesize

        # Load constraints per candidate
        for cand in candidates:
            prob += (
                pulp.lpSum(
                    voter.weight * load[(voter, cand)]
                    for voter in profile
                    if cand in voter.approved
                )
                >= in_committee[cand]
            )

        # Apply prior loadbound constraints
        for i, bound in enumerate(loadbounds):
            for j, voter in enumerate(profile):
                prob += (
                    pulp.lpSum(load[(voter, cand)] for cand in voter.approved)
                    <= bound + (1 - loadbound_constraint[(i, j)]) * committeesize + 1e-6
                )

        # Final loadbound (current lex level)
        newloadbound = pulp.LpVariable("newloadbound", lowBound=0.0, upBound=committeesize)
        for j, voter in enumerate(profile):
            prob += pulp.lpSum(
                load[(voter, cand)] for cand in voter.approved
            ) <= newloadbound + pulp.lpSum(
                loadbound_constraint[(i, j)] * committeesize for i in range(len(loadbounds))
            )

        # Maximize the negative of the new loadbound (i.e., minimize loadbound)
        prob += -newloadbound

    # Check for insufficient approved candidates
    approved_candidates = profile.approved_candidates()
    if len(approved_candidates) < committeesize:
        remaining_candidates = [c for c in profile.candidates if c not in approved_candidates]
        num_missing = committeesize - len(approved_candidates)

        if resolute:
            return [approved_candidates | set(remaining_candidates[:num_missing])]

        return [
            approved_candidates | set(extra)
            for extra in itertools.combinations(remaining_candidates, num_missing)
        ]

    # Iterative refinement: enforce tighter bounds
    for iteration in range(len(profile) - 1):
        committees, neg_loadbound = _optimize_rule_pulp(
            set_opt_model_func=set_opt_model_func,
            profile=profile,
            committeesize=committeesize,
            resolute=True,
            max_num_of_committees=None,
            solver_id=solver_id,
            name=f"leximaxphragmen-iteration{iteration}",
        )
        if math.isclose(neg_loadbound, 0, rel_tol=1e-4, abs_tol=1e-4):
            break
        loadbounds.append(-neg_loadbound)

    # Final optimization with all loadbound constraints
    committees, _ = _optimize_rule_pulp(
        set_opt_model_func=set_opt_model_func,
        profile=profile,
        committeesize=committeesize,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
        name="leximaxphragmen-final",
        lexicographic_tiebreaking=lexicographic_tiebreaking,
    )

    return sorted_committees(committees)


def _pulp_minimaxav(
    profile,
    committeesize,
    resolute,
    max_num_of_committees,
    solver_id,
    lexicographic_tiebreaking=False,
):
    def set_opt_model_func(prob, in_committee):
        # Create the max Hamming distance variable
        max_hamming_distance = pulp.LpVariable(
            "max_hamming_distance", lowBound=0, upBound=profile.num_cand, cat="Integer"
        )

        # Committee size constraint
        prob += pulp.lpSum(in_committee[cand] for cand in profile.candidates) == committeesize

        # Constraints for max Hamming distance for each voter
        for voter in profile:
            not_approved = [cand for cand in profile.candidates if cand not in voter.approved]
            approved = voter.approved

            prob += max_hamming_distance >= (
                pulp.lpSum(1 - in_committee[cand] for cand in approved)
                + pulp.lpSum(in_committee[cand] for cand in not_approved)
            )

        # Objective: minimize the max Hamming distance (by maximizing negative)
        prob += -max_hamming_distance

    committees, _ = _optimize_rule_pulp(
        set_opt_model_func=set_opt_model_func,
        profile=profile,
        committeesize=committeesize,
        resolute=resolute,
        max_num_of_committees=max_num_of_committees,
        solver_id=solver_id,
        name="MinimaxAv",
        committeescorefct=lambda profile, committee: -scores.minimaxav_score(profile, committee),
        lexicographic_tiebreaking=lexicographic_tiebreaking,
    )
    return sorted_committees(committees)


def _pulp_lexminimaxav(
    profile,
    committeesize,
    resolute,
    max_num_of_committees,
    solver_id,
    lexicographic_tiebreaking=False,
):
    hammingdistance_constraints = {}

    def set_opt_model_func(prob, in_committee):
        voteratmostdistances = {}

        for i, voter in enumerate(profile):
            for dist in range(profile.num_cand + 1):
                if dist >= len(voter.approved) + committeesize:
                    voteratmostdistances[(i, dist)] = 1
                elif dist < abs(len(voter.approved) - committeesize):
                    voteratmostdistances[(i, dist)] = 0
                else:
                    voteratmostdistances[(i, dist)] = pulp.LpVariable(
                        f"atmostdistance({i},{dist})", cat="Binary"
                    )

        # committee size constraint
        prob += pulp.lpSum(in_committee[cand] for cand in profile.candidates) == committeesize

        # consistent distance constraints
        for i, voter in enumerate(profile):
            not_approved = [cand for cand in profile.candidates if cand not in voter.approved]
            for dist in range(profile.num_cand + 1):
                var = voteratmostdistances[(i, dist)]
                if isinstance(var, int):
                    continue  # trivially satisfied
                hamming_expr = pulp.lpSum(
                    1 - in_committee[cand] for cand in voter.approved
                ) + pulp.lpSum(in_committee[cand] for cand in not_approved)
                # distance var == 1 implies hamming_expr <= dist
                M = profile.num_cand + 1
                prob += hamming_expr <= dist + M * (1 - var)

        # add previous round constraints
        for dist, num_voters in hammingdistance_constraints.items():
            dist_vars = [
                voteratmostdistances[(i, dist)]
                for i in range(len(profile))
                if isinstance(voteratmostdistances[(i, dist)], pulp.LpVariable)
            ]
            static_vars = [
                voteratmostdistances[(i, dist)]
                for i in range(len(profile))
                if isinstance(voteratmostdistances[(i, dist)], int)
            ]
            total = pulp.lpSum(dist_vars) + sum(static_vars)
            prob += total >= num_voters - 1e-5  # tiny epsilon for float compatibility

        # new objective: number of voters with Hamming distance <= new_distance
        new_distance = min(hammingdistance_constraints.keys()) - 1
        new_obj = [
            (
                voteratmostdistances[(i, new_distance)]
                if isinstance(voteratmostdistances[(i, new_distance)], pulp.LpVariable)
                else voteratmostdistances[(i, new_distance)]
            )
            for i in range(len(profile))
        ]
        prob += pulp.lpSum(new_obj)

    # Step 1: Compute baseline from minimaxav
    committees = _pulp_minimaxav(
        profile, committeesize, resolute=True, max_num_of_committees=None, solver_id=solver_id
    )
    maxdistance = scores.minimaxav_score(profile, committees[0])
    hammingdistance_constraints = {maxdistance: len(profile)}

    # Step 2: Lexicographic optimization
    for distance in range(maxdistance - 1, -1, -1):
        _resolute = resolute if distance == 0 else True
        _max_num = max_num_of_committees if distance == 0 else None

        committees, _ = _optimize_rule_pulp(
            set_opt_model_func=set_opt_model_func,
            profile=profile,
            committeesize=committeesize,
            resolute=_resolute,
            max_num_of_committees=_max_num,
            solver_id=solver_id,
            lexicographic_tiebreaking=lexicographic_tiebreaking,
            name=f"lexminimaxav-atmostdistance{distance}",
            committeescorefct=functools.partial(
                scores.num_voters_with_upper_bounded_hamming_distance, distance
            ),
        )
        num_voters = scores.num_voters_with_upper_bounded_hamming_distance(
            distance, profile, committees[0]
        )
        hammingdistance_constraints[distance] = num_voters

    committees = sorted_committees(committees)
    detailed_info = {
        "hammingdistance_constraints": hammingdistance_constraints,
        "opt_distances": [misc.hamming(voter.approved, committees[0]) for voter in profile],
    }
    return committees, detailed_info
