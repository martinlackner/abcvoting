"""
Generates instances for unit tests.
"""

import random
from itertools import product
import os.path
from abcvoting import generate
from operator import itemgetter
from abcvoting import abcrules
from abcvoting import fileio


def generate_abc_yaml_testinstances(
    batchname,
    committeesizes,
    num_voters_values,
    num_cand_values,
    prob_distributions,
    av_neq_pav=False,
):
    random.seed(24121838)

    parameter_tuples = []
    for committeesize, num_voters, num_cand, prob_distribution in product(
        committeesizes, num_voters_values, num_cand_values, prob_distributions
    ):
        if committeesize >= num_cand:
            continue
        prob_model, kwargs = prob_distribution
        parameter_tuples.append((num_voters, num_cand, prob_model, kwargs, committeesize))
    parameter_tuples.sort(key=itemgetter(2))

    print(f"Generating {len(parameter_tuples)} instances for batch {batchname}...")
    num_instances = 0

    for index, (num_voters, num_cand, prob_model, kwargs, committeesize) in enumerate(
        parameter_tuples
    ):
        num_instances += 1

        # write instance to .abc.yaml file
        currdir = os.path.dirname(os.path.abspath(__file__))
        filename = currdir + f"/instance{batchname}{index:04d}.abc.yaml"

        print(f"generating {filename} ({prob_model} {kwargs})")
        while True:
            profile = generate.random_profile(prob_model, num_voters, num_cand, **kwargs)
            committees_av = abcrules.compute("av", profile, committeesize, resolute=False)
            committees_pav = abcrules.compute("pav", profile, committeesize, resolute=False)
            if not av_neq_pav:
                break
            intersection = set(tuple(sorted(committee)) for committee in committees_pav) & set(
                tuple(sorted(committee)) for committee in committees_av
            )
            if not intersection:
                break

        rule_instances = []
        for rule_id in abcrules.MAIN_RULE_IDS:
            rule = abcrules.Rule(rule_id)

            # if irresolute (resolute = False) is supported, then "result" should be
            # the list of committees returned for resolute=False.
            if False in rule.resolute_values:
                resolute = False
            else:
                resolute = True

            if rule_id == "rsd":
                committees = None  # result is random, not sensible for unit tests
            elif rule_id == "leximaxphragmen" and (num_cand > 7 or num_voters > 8):
                committees = None  # too slow
            else:
                committees = abcrules.compute(rule_id, profile, committeesize, resolute=resolute)

            for resolute in rule.resolute_values:
                rule_instances.append(
                    {"rule_id": rule_id, "resolute": resolute, "result": committees}
                )

        fileio.write_abcvoting_instance_to_yaml_file(
            filename,
            profile,
            committeesize=committeesize,
            description=(
                f"profile generated via prob_model={prob_model}, "
                f"num_voters={num_voters}, "
                f"num_cand={num_cand}, "
                f"kwargs={kwargs}"
            ),
            compute_instances=rule_instances,
        )

    print("Done.")


if __name__ == "__main__":
    batch = "S"
    committeesizes = [3, 4]
    num_voters_values = [8, 9]
    num_cand_values = [6]
    prob_distributions = [("IC", {"p": 0.5})]
    generate_abc_yaml_testinstances(
        batch,
        committeesizes,
        num_voters_values,
        num_cand_values,
        prob_distributions,
        av_neq_pav=True,
    )

    batch = "M"
    committeesizes = [3, 4]
    num_voters_values = [8, 9, 10, 11]
    num_cand_values = [6, 7]
    prob_distributions = [("IC fixed-size", {"setsize": 2}), ("IC fixed-size", {"setsize": 3})]
    generate_abc_yaml_testinstances(
        batch,
        committeesizes,
        num_voters_values,
        num_cand_values,
        prob_distributions,
        av_neq_pav=True,
    )

    batch = "L"
    committeesizes = [3, 4, 5, 6]
    num_voters_values = [8, 12, 15]
    num_cand_values = [6, 8, 9]
    prob_distributions = [
        ("IC fixed-size", {"setsize": 2}),
        ("IC fixed-size", {"setsize": 3}),
        ("Truncated Mallows", {"setsize": 2, "dispersion": 0.2}),
        ("Truncated Mallows", {"setsize": 3, "dispersion": 0.2}),
        ("Truncated Mallows", {"setsize": 4, "dispersion": 0.2}),
        ("Truncated Mallows", {"setsize": 2, "dispersion": 0.5}),
        ("Truncated Mallows", {"setsize": 3, "dispersion": 0.5}),
        ("Truncated Mallows", {"setsize": 4, "dispersion": 0.5}),
        ("Truncated Mallows", {"setsize": 2, "dispersion": 0.8}),
        ("Truncated Mallows", {"setsize": 3, "dispersion": 0.8}),
        ("Truncated Mallows", {"setsize": 4, "dispersion": 0.8}),
        ("Urn fixed-size", {"setsize": 2, "replace": 0.5}),
        ("Urn fixed-size", {"setsize": 3, "replace": 0.5}),
    ]
    approval_setsizes = [2, 3]
    generate_abc_yaml_testinstances(
        batch,
        committeesizes,
        num_voters_values,
        num_cand_values,
        prob_distributions,
        av_neq_pav=False,
    )

    batch = "VL"
    committeesizes = [6]
    num_voters_values = [24, 25]
    num_cand_values = [8, 9]
    prob_distributions = [
        ("IC", {"p": 0.3}),
        ("IC", {"p": 0.4}),
        ("IC", {"p": 0.5}),
        ("Truncated Mallows", {"setsize": 2, "dispersion": 0.5}),
        ("Truncated Mallows", {"setsize": 3, "dispersion": 0.5}),
        ("Truncated Mallows", {"setsize": 5, "dispersion": 0.5}),
        ("Truncated Mallows", {"setsize": 2, "dispersion": 0.8}),
        ("Truncated Mallows", {"setsize": 3, "dispersion": 0.8}),
        ("Truncated Mallows", {"setsize": 5, "dispersion": 0.8}),
        ("Urn fixed-size", {"setsize": 2, "replace": 0.5}),
        ("Urn fixed-size", {"setsize": 3, "replace": 0.5}),
        ("Urn fixed-size", {"setsize": 5, "replace": 0.5}),
    ]
    generate_abc_yaml_testinstances(
        batch,
        committeesizes,
        num_voters_values,
        num_cand_values,
        prob_distributions,
        av_neq_pav=False,
    )
