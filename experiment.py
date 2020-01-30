# Experiments to compare how many voters each voting rule
# can take without needing too much time
#
# Author: Benjamin Krenn

import math
import random
import sys
import time

from genprofiles import random_IC_profile
from preferences import Profile
from rules_approval import compute_rule

random.seed(38572895)

rules = [
    "seqpav",
    "revseqpav",
    "av",
    "sav",
    "pav-ilp",
    "pav-noilp",
    "seqslav",
    "slav-ilp",
    "slav-noilp",
    "phrag",
    "monroe-ilp",
    "monroe-noilp",
    "greedy-monroe",
    "cc-ilp",
    "cc-noilp",
    "seqcc",
    "revseqcc",
    "minimaxav-noilp",
    "minimaxav-ilp",
    "optphrag",
    "rule-x",
    "phragmen-enestroem",
    ]


time_limit = .1
"""in seconds. Find the number of voters that the method needs 
around time_limit seconds."""
attempts = 5
""" How often to attempt each number of voters 
    (max time is taken)."""

cand_count = [10, 15, 20]
"""experiments for those numbers of candidates"""
committeesize = [math.ceil(c / 3.0) for c in cand_count]
"""The committee size also known as k"""
setsize = 3
"""size of the approval sets of each voter"""
resolute = True
"""only compute one committee with compute_rule"""

method = random_IC_profile
"""The method to generate random profiles"""
gen_params = {"setsize": setsize}
"""The parameters to generate random profiles.
num_voters and num_cand is provided directly not here."""

# Example for different method:
# method = random_2d_points_party_list_profile
# gen_params = {"num_parties":3, "partypointmode":"twogroups",
#               "voterpointmode":"normal", "sigma":0.7, "uniform":True}

logging = True

max_voters_in_time = {rule: {} for rule in rules}

for i in range(len(cand_count)):  # one time for each number of cands
    cands = cand_count[i]
    if logging:
        print("###############################################")
        print("round", i, "with", cands, "candidates.")
    k = committeesize[i]
    gen_params["num_cand"] = cands
    for rule in rules:  # for every rule
        if logging:
            print("###############################################")
            print("starting with rule", rule)
        prev_voters = 0  # lower bound for binary search
        if rule == "monroe-noilp":
            voters = 3
        else:
            voters = 10
        finished = False
        ex_time = 0
        max_v = sys.maxsize  # upper bound for binary search
        # is set so high because upper bound initially unknown

        while not finished:
            if prev_voters == voters:
                finished = True
                break
            if voters == 1:
                finished = True
                break
            if logging:
                print("attempting voters:", voters)
            max_time = 0
            failed = 0
            for _ in range(attempts):
                # repeat multiple times, take worst execution time
                appr_sets = method(num_voters=voters, **gen_params)
                profile = Profile(cands)
                profile.add_preferences(appr_sets)
                try:
                    start = time.time()
                    compute_rule(rule, profile, k, resolute=resolute)
                    end = time.time()
                    max_time = max(end - start, max_time)
                except:  # most likely to few voters and approved cands
                    failed += 1
            ex_time = max_time
            if failed == attempts:  # no attempt was successful
                prev_voters = voters
                voters = min(voters+1, max_v) # increase voters slowly
                continue
            if logging:
                print(round(ex_time, 4))

            if round(ex_time, 4) > time_limit:
                max_v = voters  # new upper bound
                if voters <= prev_voters+1:
                    # prev_voters always below time_limit
                    # voters always at least prev_voters
                    # therefore first higher voter count found
                    finished = True
                else:
                    voters = math.ceil((prev_voters + voters) / 2.0)

            elif round(ex_time, 4) == time_limit:
                max_v = voters
                finished = True
            else:
                prev_voters = voters
                if ex_time * 4 < time_limit:  # fast increase
                    voters *= 2
                elif ex_time * 2 < time_limit:  # medium increase
                    voters = math.ceil(voters*1.5)
                else:  # slow increase
                    voters = math.ceil(voters*1.1)
                if voters >= max_v:  # too high number not needed
                    if prev_voters == max_v:
                        voters = max_v
                        finished = True
                    else:
                        voters = math.ceil((prev_voters + max_v) / 2.0)

        max_voters_in_time[rule][cands] = voters
        if logging:
            print(voters, "with a runtime of", ex_time)

print("The results state at which number of voters the voting rules",
      "took more than", time_limit, "seconds.")
print("This is provided for every rule and each rule also has the",
      "experiments repeated with the following total candidate numbers:",
      cand_count)

# Result format:
# {rule: {10: voter_number10, 15: voter_number15, 20: voter_number20}
# where 10, 15 and 20 are the number of candidates.
print("###############################################")
print("Results:")
print("###############################################")
print(max_voters_in_time)
