"""
Updates test instance YAML files with recomputed results for a specific voting rule.
Includes a timeout per instance to skip slow computations.

Usage:
    python update_unittest_instances_for_rule_with_timeout.py <rule_id> [timeout_seconds]

Example:
    python update_unittest_instances_for_rule_with_timeout.py leximaxphragmen 60
"""

import sys
import os.path
import glob
import signal
from abcvoting import abcrules
from abcvoting import fileio


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Computation timed out")


def compute_with_timeout(rule_id, profile, committeesize, resolute, timeout_seconds):
    """
    Compute committees with a timeout.
    Returns None if computation times out or fails.
    """
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        result = abcrules.compute(rule_id, profile, committeesize, resolute=resolute)
        signal.alarm(0)  # Cancel the alarm
        return result
    except TimeoutError:
        return None
    except Exception as e:
        signal.alarm(0)
        raise e


def update_rule_in_yaml_file(filename, rule_id, timeout_seconds):
    """
    Update or add results for a specific rule in a YAML file.
    Returns True if successful, False if timed out.
    """
    profile, committeesize, compute_instances, data = fileio.read_abcvoting_yaml_file(filename)
    description = data.get("description", None)
    rule = abcrules.Rule(rule_id)

    # Determine which resolute value to use for computation
    if False in rule.resolute_values:
        result_resolute_value = False
    else:
        result_resolute_value = True

    # Compute result with timeout
    if rule_id == "rsd":
        # RSD is random, skip recomputation
        committees = None
        timed_out = False
    else:
        committees = compute_with_timeout(
            rule_id, profile, committeesize, result_resolute_value, timeout_seconds
        )
        timed_out = committees is None

    if timed_out:
        return False

    # Keep all existing compute instances except for the specified rule
    updated_compute_instances = []
    for compute_instance in compute_instances:
        if compute_instance["rule_id"] != rule_id:
            clean_instance = {
                "rule_id": compute_instance["rule_id"],
                "resolute": compute_instance.get("resolute", False),
                "result": compute_instance.get("result"),
            }
            updated_compute_instances.append(clean_instance)

    # Add new instances for the specified rule
    for resolute in rule.resolute_values:
        updated_compute_instances.append(
            {
                "rule_id": rule_id,
                "resolute": resolute,
                "result": committees,
            }
        )

    # Write back to file
    fileio.write_abcvoting_instance_to_yaml_file(
        filename,
        profile,
        committeesize=committeesize,
        description=description,
        compute_instances=updated_compute_instances,
    )
    return True


def update_all_yaml_files(rule_id, timeout_seconds=60):
    """
    Update all YAML files in the test_instances directory for a specific rule.
    """
    try:
        abcrules.Rule(rule_id)
    except ValueError:
        print(f"Error: Unknown rule_id '{rule_id}'")
        return

    currdir = os.path.dirname(os.path.abspath(__file__))
    yaml_files = glob.glob(os.path.join(currdir, "*.abc.yaml"))
    yaml_files.sort()

    print(
        f"Processing {len(yaml_files)} YAML files for rule '{rule_id}' "
        f"(timeout: {timeout_seconds}s)..."
    )

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for i, filename in enumerate(yaml_files):
        basename = os.path.basename(filename)
        print(f"  [{i + 1}/{len(yaml_files)}] {basename}...", end=" ", flush=True)
        try:
            success = update_rule_in_yaml_file(filename, rule_id, timeout_seconds)
            if success:
                print("done")
                processed_count += 1
            else:
                print("TIMEOUT (skipped)")
                skipped_count += 1
        except Exception as e:
            print(f"ERROR: {e}")
            error_count += 1

    print(
        f"\nDone! Processed: {processed_count}, Skipped (timeout): "
        f"{skipped_count}, Errors: {error_count}"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python update_unittest_instances_for_rule_with_timeout.py "
            "<rule_id> [timeout_seconds]"
        )
        sys.exit(1)

    rule_id = sys.argv[1]
    timeout_seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    update_all_yaml_files(rule_id, timeout_seconds)
