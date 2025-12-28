"""
Test which ABC voting rules correspond to which apportionment methods.

Example from "A Mathematical Analysis of an Election System Proposed
by Gottlob Frege" by Paul Harrenstein, Marie‑Louise Lackner, Martin Lackner,
Example 7:
- Vote shares: p = (79/98, 7/98, 6/98, 3/98, 2/98, 1/98)
- Weights (multiplied by 980): (790, 70, 60, 30, 20, 10)
- House size k = 20
"""

from abcvoting import abcrules
from abcvoting.preferences import Profile, Voter
from collections import defaultdict

# Vote shares multiplied by 980 to get integer weights
weights = [790, 70, 60, 30, 20, 10]
num_parties = len(weights)
committeesize = 20

# For apportionment: create `committeesize` candidates per party
# Each voter approves all candidates of their party
candidates_per_party = committeesize
num_cand = num_parties * candidates_per_party

profile = Profile(num_cand)

for party_idx, weight in enumerate(weights):
    # Candidates for this party
    approved = set(range(party_idx * candidates_per_party, (party_idx + 1) * candidates_per_party))
    profile.add_voter(Voter(approved, weight=weight))

apportionment_results = {
    "Largest Remainder": (16, 2, 1, 1, 0, 0),
    "D'Hondt (Jefferson)": (18, 1, 1, 0, 0, 0),
    "Adams": (14, 2, 1, 1, 1, 1),
    "Sainte-Laguë (Webster)": (17, 1, 1, 1, 0, 0),
    "Huntington-Hill": (15, 1, 1, 1, 1, 1),
    "Quota method": (17, 2, 1, 0, 0, 0),
    "Frege's apportionment": (16, 1, 1, 1, 1, 0),
    "Majority": (20, 0, 0, 0, 0, 0),
}


def committee_to_seats(committee):
    """Convert a committee (set of candidate indices) to seat counts per party."""
    seats = [0] * num_parties
    for cand in committee:
        party = cand // candidates_per_party
        seats[party] += 1
    return tuple(seats)


print(f"Weights: {weights}")
print(f"Committee size: {committeesize}")
print()

# Check all main rules
results = {}
for rule_id in abcrules.MAIN_RULE_IDS:
    try:
        committees = abcrules.compute(rule_id, profile, committeesize, resolute=True)
        if committees:
            seats = committee_to_seats(committees[0])
            results[rule_id] = seats
    except Exception as e:
        results[rule_id] = f"Error: {e}"

# Group by seat allocation
print("Grouping by seat allocation:\n")
by_seats = defaultdict(list)
for rule_id, seats in results.items():
    if isinstance(seats, tuple):
        by_seats[seats].append(rule_id)

for seats, rules in sorted(by_seats.items(), key=lambda x: -x[0][0]):
    matches = [name for name, exp_seats in apportionment_results.items() if exp_seats == seats]
    match_str = f" = {', '.join(matches)}" if matches else ""
    print(f"{seats}{match_str}")
    for rule in sorted(rules):
        print(f"    - {rule}")
    print()

# Show errors
errors = {k: v for k, v in results.items() if isinstance(v, str)}
if errors:
    print("Rules with errors:")
    for rule_id, err in errors.items():
        print(f"  {rule_id}: {err}")

# compare with expected results from literature
rule_to_apportionment = {
    "av": "Majority",
    "pav": "D'Hondt (Jefferson)",
    "seqpav": "D'Hondt (Jefferson)",
    "revseqpav": "D'Hondt (Jefferson)",
    "seqphragmen": "D'Hondt (Jefferson)",
    "equal-shares": "D'Hondt (Jefferson)",
    "leximaxphragmen": "D'Hondt (Jefferson)",
    # "monroe": "Largest Remainder",
    # "greedy-monroe": "Largest Remainder",
    "consensus-rule": "Frege's apportionment",
    "adams": "Adams",
    "slav": "Sainte-Laguë (Webster)",
    "seqslav": "Sainte-Laguë (Webster)",
}

print("\nVerification of rule_to_apportionment mapping:\n")
for rule_id, apport_method in rule_to_apportionment.items():
    if rule_id not in results:
        print(f"  {rule_id}: NOT IN RESULTS (rule not computed)")
        continue

    computed = results[rule_id]

    if isinstance(computed, str):  # Error case
        print(f"  {rule_id}: {computed}")
        continue

    expected_seats = apportionment_results[apport_method]
    if computed == expected_seats:
        print(f"  {rule_id}: OK (matches {apport_method})")
    else:
        # Check what it actually matches
        actual_matches = [
            name for name, seats in apportionment_results.items() if seats == computed
        ]
        actual_str = ", ".join(actual_matches) if actual_matches else "nothing"
        print(
            f"  {rule_id}: MISMATCH - expected {apport_method} {expected_seats}, "
            f"got {computed} (matches {actual_str})"
        )
