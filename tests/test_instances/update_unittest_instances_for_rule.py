"""
Updates test instance YAML files with recomputed results for a specific voting rule.

Usage:
    python update_unittest_instances_for_rule.py <rule_id>

Example:
    python update_unittest_instances_for_rule.py adams
"""

import sys
import os.path
import glob
from abcvoting import abcrules
from abcvoting import fileio


def update_rule_in_yaml_file(filename, rule_id):
    """
    Update or add results for a specific rule in a YAML file.

    Parameters
    ----------
        filename : str
            Path to the .abc.yaml file.

        rule_id : str
            The rule ID to update (e.g., "adams", "pav").
    """
    # Read the existing file
    profile, committeesize, compute_instances, data = fileio.read_abcvoting_yaml_file(filename)

    description = data.get("description", None)
    rule = abcrules.Rule(rule_id)

    # Determine which resolute value to use for computation
    # if irresolute (resolute = False) is supported, then "result" should be
    # the list of committees returned for resolute=False.
    if False in rule.resolute_values:
        result_resolute_value = False
    else:
        result_resolute_value = True

    # Compute result once for the canonical resolute value
    if rule_id == "rsd":
        # RSD is random, skip recomputation
        committees = None
    elif rule_id == "leximaxphragmen" and (profile.num_cand > 9 or len(profile) > 9):
        # Skip slow instances
        committees = None
    else:
        committees = abcrules.compute(
            rule_id, profile, committeesize, resolute=result_resolute_value
        )

    # Keep all existing compute instances except for the specified rule
    updated_compute_instances = []
    for compute_instance in compute_instances:
        if compute_instance["rule_id"] != rule_id:
            # Keep other rules unchanged
            clean_instance = {
                "rule_id": compute_instance["rule_id"],
                "resolute": compute_instance.get("resolute", False),
                "result": compute_instance.get("result"),
            }
            updated_compute_instances.append(clean_instance)

    # Add new instances for the specified rule (one for each supported resolute value)
    for resolute in rule.resolute_values:
        updated_compute_instances.append(
            {
                "rule_id": rule_id,
                "resolute": resolute,
                "result": committees,
            }
        )

    # Write back to file with updated results
    fileio.write_abcvoting_instance_to_yaml_file(
        filename,
        profile,
        committeesize=committeesize,
        description=description,
        compute_instances=updated_compute_instances,
    )


def update_all_yaml_files(rule_id):
    """
    Update all YAML files in the test_instances directory for a specific rule.

    Parameters
    ----------
        rule_id : str
            The rule ID to update (e.g., "adams", "pav").
    """
    # Verify the rule exists
    try:
        abcrules.Rule(rule_id)
    except ValueError:
        print(f"Error: Unknown rule_id '{rule_id}'")
        return

    # Get the directory containing this script
    currdir = os.path.dirname(os.path.abspath(__file__))

    # Find all .abc.yaml files
    yaml_files = glob.glob(os.path.join(currdir, "*.abc.yaml"))
    yaml_files.sort()

    print(f"Processing {len(yaml_files)} YAML files for rule '{rule_id}'...")

    processed_count = 0

    for filename in yaml_files:
        basename = os.path.basename(filename)
        try:
            update_rule_in_yaml_file(filename, rule_id)
            processed_count += 1
        except Exception as e:
            print(f"  ✗ Error processing {basename}: {e}")

    print(f"\nDone! Processed {processed_count} files.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_unittest_instances_for_rule.py <rule_id>")
        sys.exit(1)

    rule_id = sys.argv[1]
    update_all_yaml_files(rule_id)
