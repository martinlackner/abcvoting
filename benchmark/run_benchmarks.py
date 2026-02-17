#!/usr/bin/env python3
"""
Benchmark runner for abcvoting dashboard.

Runs all ABC voting rules on test instances and outputs JSON results
for the dashboard generator.

Usage:
    python run_benchmarks.py
    python run_benchmarks.py --categories S M --timeout 60 --output results.json
"""

import argparse
import json
import os
import platform
import signal
import sys
import time
from datetime import datetime, timezone

# Add the project root to the path if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from abcvoting import fileio  # noqa: E402
from abcvoting.abcrules import MAIN_RULE_IDS, Rule, UnknownRuleIDError  # noqa: E402


# Configuration
DEFAULT_TIMEOUT = 300  # 5 minutes
DEFAULT_MAX_NUM_OF_COMMITTEES = 10
INSTANCE_DIR = "tests/test_instances"
GENERATED_INSTANCE_DIR = "benchmark/instances"
SIZE_CATEGORIES = ["S", "M", "L", "VL", "G"]  # G = generated benchmark instances


class TimeoutException(Exception):
    """Exception raised when a computation times out."""


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Computation timed out")


def tprint(*args, **kwargs):
    """Print with timestamp prefix."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}]", *args, **kwargs)


def collect_hardware_info():
    """Collect hardware information about the system."""
    cpu = None

    # Try /proc/cpuinfo first (Linux) - most reliable for CPU model name
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    cpu = line.split(":")[1].strip()
                    break
    except Exception:
        pass

    # Fallback to platform.processor()
    if not cpu:
        cpu = platform.processor() or "Unknown"

    # Try to get memory info
    ram_gb = None
    try:
        import psutil

        ram_gb = round(psutil.virtual_memory().total / (1000**3), 1)
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        ram_kb = int(line.split()[1])
                        ram_gb = round(ram_kb / (1000**2), 1)
                        break
        except Exception:
            ram_gb = None

    return {
        "cpu": cpu,
        "ram_gb": ram_gb,
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
    }


def get_abcvoting_version():
    """Get the installed abcvoting version."""
    try:
        from importlib.metadata import version

        return version("abcvoting")
    except Exception:
        return "unknown"


def load_instances_by_category(category, instance_dir=INSTANCE_DIR):
    """Load all instances for a size category (S, M, L, VL, G).

    Categories S, M, L, VL load from the test instances directory.
    Category G loads generated benchmark instances from benchmark/instances/.
    """
    instances = []

    # Get absolute path
    if not os.path.isabs(instance_dir):
        instance_dir = os.path.join(project_root, instance_dir)

    if category == "G":
        # Load generated benchmark instances from benchmark/instances/
        generated_dir = os.path.join(project_root, GENERATED_INSTANCE_DIR)
        if not os.path.exists(generated_dir):
            print(
                f"Warning: Generated instances directory not found: {generated_dir}",
                file=sys.stderr,
            )
            return instances

        # Load all .abc.yaml files from the generated instances directory
        for filename in sorted(os.listdir(generated_dir)):
            if filename.endswith(".abc.yaml"):
                filepath = os.path.join(generated_dir, filename)
                try:
                    profile, committeesize, _, _ = fileio.read_abcvoting_yaml_file(filepath)
                    instances.append((profile, committeesize, filename))
                except Exception as e:
                    print(f"Warning: Failed to load {filename}: {e}", file=sys.stderr)
    else:
        # Load instances from test directory with category prefix
        prefix = f"instance{category}"
        for filename in sorted(os.listdir(instance_dir)):
            if filename.startswith(prefix) and filename.endswith(".abc.yaml"):
                filepath = os.path.join(instance_dir, filename)
                try:
                    profile, committeesize, _, _ = fileio.read_abcvoting_yaml_file(filepath)
                    instances.append((profile, committeesize, filename))
                except Exception as e:
                    print(f"Warning: Failed to load {filename}: {e}", file=sys.stderr)

    return instances


def run_single_benchmark(
    rule,
    algorithm,
    profile,
    committeesize,
    resolute,
    timeout_seconds,
    max_num_of_committees=DEFAULT_MAX_NUM_OF_COMMITTEES,
    filename=None,
):
    """
    Run a single benchmark and return the runtime or None if it timed out.

    Parameters:
        rule: The Rule object
        algorithm: Algorithm to use
        profile: The profile to compute on
        committeesize: Target committee size
        resolute: Whether to use resolute mode
        timeout_seconds: Remaining time budget for this instance
        max_num_of_committees: Maximum number of committees to compute (default: 20)
        filename: Optional instance filename for error context

    Returns:
        float or None: Runtime in seconds, or None if timed out/failed
    """
    if timeout_seconds <= 0:
        return None

    # Set up timeout handler (Unix only)
    if hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds) + 1)  # Round up to ensure at least 1 second

    try:
        start_time = time.perf_counter()
        kwargs = {"algorithm": algorithm, "resolute": resolute}
        if not resolute and max_num_of_committees is not None:
            kwargs["max_num_of_committees"] = max_num_of_committees
        committees = rule.compute(profile, committeesize, **kwargs)
        if resolute and len(committees) != 1:
            raise RuntimeError(
                f"Expected exactly one committee in resolute mode, got {len(committees)}."
            )
        elif not resolute and len(committees) != max_num_of_committees:
            raise RuntimeError(
                f"Expected {max_num_of_committees} committees in irresolute mode, "
                f"got {len(committees)}."
            )
        elapsed = time.perf_counter() - start_time
        return elapsed
    except TimeoutException:
        return None
    except Exception as e:
        # Rule may not support this resolute value or algorithm
        context = f" (instance: {filename})" if filename else ""
        print(f"  Error{context}: {type(e).__name__}: {e}", file=sys.stderr)
        return None
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def run_benchmarks_for_rule(
    rule_id,
    instances,
    timeout_seconds,
    max_num_of_committees=DEFAULT_MAX_NUM_OF_COMMITTEES,
    verbose=False,
):
    """
    Run benchmarks for a single rule across all categories.

    Returns:
        dict: Results for this rule
    """
    rule = Rule(rule_id)
    # Get the library's predefined fastest algorithm
    library_fastest = rule.fastest_available_algorithm() if rule.available_algorithms else None

    result = {
        "shortname": rule.shortname,
        "longname": rule.longname,
        "fastest_algorithm": None,  # empirically measured fastest
        "library_fastest": library_fastest,  # predefined fastest from library
        "algorithms": {},
    }

    # Get available algorithms for this rule
    available_algorithms = rule.available_algorithms
    if not available_algorithms:
        tprint(f"  No available algorithms for {rule_id}", file=sys.stderr)
        return result

    # Track algorithm performance per mode: {mode: {algo: (finished, runtime)}}
    # More finished instances is better; among equal finished, lower runtime is better
    algorithm_stats_by_mode = {"resolute": {}, "irresolute": {}}

    for algorithm in available_algorithms:
        result["algorithms"][algorithm] = {"resolute": {}, "irresolute": {}}

        for resolute_mode, mode_name in [(True, "resolute"), (False, "irresolute")]:
            # Check if this rule supports this resolute mode
            if resolute_mode not in rule.resolute_values:
                continue

            total = len(instances)
            tprint(f"  - {rule_id}, {algorithm}, {mode_name} ({total} instances)")
            finished = 0
            cumulative_runtime = 0.0

            for profile, committeesize, filename in instances:
                remaining_time = timeout_seconds - cumulative_runtime
                # print(f" - {remaining_time}s left: starting with {filename}")
                runtime = run_single_benchmark(
                    rule,
                    algorithm,
                    profile,
                    committeesize,
                    resolute_mode,
                    remaining_time,
                    max_num_of_committees=max_num_of_committees,
                    filename=filename,
                )
                if runtime is not None and cumulative_runtime + runtime <= timeout_seconds:
                    finished += 1
                    cumulative_runtime += runtime
                else:
                    # Timed out - stop running more instances for this algorithm/mode
                    break

            result["algorithms"][algorithm][mode_name] = {
                "finished": finished,
                "total": total,
                "cumulative_runtime": round(cumulative_runtime, 4) if finished > 0 else None,
            }

            algorithm_stats_by_mode[mode_name][algorithm] = (finished, cumulative_runtime)
            tprint(f"    finished {finished}/{total} instances in {cumulative_runtime:.2f}s")

    # Determine fastest algorithm per mode (empirically measured)
    # Priority: 1) more finished instances, 2) lower runtime (for equal finished)
    for mode_name in ["resolute", "irresolute"]:
        mode_stats = algorithm_stats_by_mode[mode_name]
        if mode_stats:
            empirical_fastest = min(
                mode_stats, key=lambda algo: (-mode_stats[algo][0], mode_stats[algo][1])
            )
            result[f"fastest_algorithm_{mode_name}"] = empirical_fastest
        else:
            result[f"fastest_algorithm_{mode_name}"] = None

    # Overall fastest (for backwards compatibility) - use resolute if available
    result["fastest_algorithm"] = result.get("fastest_algorithm_resolute") or result.get(
        "fastest_algorithm_irresolute"
    )
    # Flag if empirical measurement differs from library's predefined order
    result["fastest_differs_from_library"] = (
        library_fastest is not None and result["fastest_algorithm"] != library_fastest
    )

    return result


def load_existing_results(output_file):
    """Load existing benchmark results from a JSON file if it exists."""
    if os.path.exists(output_file):
        try:
            with open(output_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            tprint(f"Warning: Could not load existing results from {output_file}: {e}")
    return None


def validate_metadata(existing_results, timeout_seconds, max_num_of_committees, categories):
    """
    Validate that existing results metadata matches current configuration.

    Returns:
        tuple: (is_valid, error_message) where error_message is None if valid
    """
    if not existing_results or "metadata" not in existing_results:
        return True, None

    existing_metadata = existing_results["metadata"]
    mismatches = []

    # Check timeout_seconds
    if existing_metadata.get("timeout_seconds") != timeout_seconds:
        mismatches.append(
            f"  timeout_seconds: existing={existing_metadata.get('timeout_seconds')}, "
            f"current={timeout_seconds}"
        )

    # Check max_num_of_committees
    if existing_metadata.get("max_num_of_committees") != max_num_of_committees:
        mismatches.append(
            f"  max_num_of_committees: existing={existing_metadata.get('max_num_of_committees')}, "
            f"current={max_num_of_committees}"
        )

    # Check categories
    existing_categories = existing_metadata.get("categories", [])
    if sorted(existing_categories) != sorted(categories):
        mismatches.append(f"  categories: existing={existing_categories}, current={categories}")

    if mismatches:
        error_msg = "Metadata mismatch between existing results and current configuration:\n"
        error_msg += "\n".join(mismatches)
        return False, error_msg

    return True, None


def save_results(results, output_file):
    """Save benchmark results to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def run_all_benchmarks(
    categories,
    timeout_seconds,
    instance_dir,
    output_file,
    max_num_of_committees=DEFAULT_MAX_NUM_OF_COMMITTEES,
    verbose=False,
    rule_ids=None,
    overwrite=False,
):
    """
    Run benchmarks for all rules (or specified rules).

    Parameters:
        categories: List of instance categories to run
        timeout_seconds: Cumulative timeout for all instances per algorithm/mode
        instance_dir: Directory containing test instances
        output_file: Path to the output JSON file (for incremental saves)
        max_num_of_committees: Maximum number of committees to compute (default: 20)
        verbose: Print verbose output
        rule_ids: List of rule IDs to run (None = all rules)
        overwrite: If True, re-run rules even if results already exist

    Returns:
        dict: Complete benchmark results
    """
    if rule_ids is None:
        rule_ids = MAIN_RULE_IDS

    # Try to load existing results
    existing_results = load_existing_results(output_file)

    # Validate metadata if existing results were loaded
    if existing_results:
        is_valid, error_msg = validate_metadata(
            existing_results, timeout_seconds, max_num_of_committees, categories
        )
        if not is_valid:
            print(f"Error: {error_msg}", file=sys.stderr)
            print(
                "\nTo resolve this, either:\n"
                "  1. Delete the existing results file and start fresh\n"
                "  2. Use matching configuration parameters\n"
                "  3. Specify a different output file with --output",
                file=sys.stderr,
            )
            sys.exit(1)

    results = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hardware": collect_hardware_info(),
            "timeout_seconds": timeout_seconds,
            "max_num_of_committees": max_num_of_committees,
            "abcvoting_version": get_abcvoting_version(),
            "categories": categories,
        },
        "results": {},
    }

    # Copy existing results for rules we're not re-running
    if existing_results and "results" in existing_results:
        results["results"] = existing_results["results"].copy()

    total_rules = len(rule_ids)
    # Load all instances from all categories
    tprint("Loading instances...")
    all_instances = []
    for category in categories:
        all_instances.extend(load_instances_by_category(category, instance_dir))

    tprint(f"Loaded {len(all_instances)} instances")

    for i, rule_id in enumerate(rule_ids, 1):
        # Skip if results already exist for this rule (unless overwrite is set)
        if rule_id in results["results"] and not overwrite:
            tprint(f"[{i}/{total_rules}] Skipping {rule_id} (already exists in results)")
            continue

        tprint(f"[{i}/{total_rules}] Running benchmarks for {rule_id}...")
        results["results"][rule_id] = run_benchmarks_for_rule(
            rule_id,
            all_instances,
            timeout_seconds,
            max_num_of_committees=max_num_of_committees,
            verbose=verbose,
        )

        # Save results after each rule
        save_results(results, output_file)
        tprint(f"  Results saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks for abcvoting dashboard")
    parser.add_argument(
        "--output",
        "-o",
        default="benchmark/data/benchmark_results.json",
        help="Output JSON file (default: benchmark/data/benchmark_results.json)",
    )
    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        choices=SIZE_CATEGORIES,
        default=["G"],
        help="Instance categories to run (default: G)",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Cumulative timeout for all instances in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--instance-dir",
        default=INSTANCE_DIR,
        help=f"Directory containing test instances (default: {INSTANCE_DIR})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output",
    )
    parser.add_argument(
        "--rules",
        "-r",
        nargs="+",
        help="Only run specific rules (by rule_id)",
    )
    parser.add_argument(
        "--max-committees",
        "-m",
        type=int,
        default=DEFAULT_MAX_NUM_OF_COMMITTEES,
        help=f"Maximum number of committees to compute (default: {DEFAULT_MAX_NUM_OF_COMMITTEES})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results for specified rules (default: skip existing)",
    )

    args = parser.parse_args()

    # Validate categories order
    categories = [c for c in SIZE_CATEGORIES if c in args.categories]

    # Validate rules if specified
    rule_ids = None
    if args.rules:
        for rule_id in args.rules:
            try:
                Rule(rule_id)
            except UnknownRuleIDError:
                print(f"Error: Unknown rule ID '{rule_id}'")
                sys.exit(1)
        rule_ids = args.rules

    print("Running benchmarks with:")
    print(f"  Categories: {', '.join(categories)}")
    print(f"  Timeout: {args.timeout}s cumulative")
    print(f"  Max committees: {args.max_committees}")
    print(f"  Instance directory: {args.instance_dir}")
    if rule_ids:
        print(f"  Rules: {', '.join(rule_ids)}")
    if args.overwrite:
        print("  Overwrite: enabled")
    print()

    # Warn if timeout protection is not available (Windows)
    if not hasattr(signal, "SIGALRM"):
        print(
            "WARNING: SIGALRM not available (Windows). Timeout protection is disabled.\n"
            "         Slow algorithms may run indefinitely.\n",
            file=sys.stderr,
        )

    run_all_benchmarks(
        categories,
        args.timeout,
        args.instance_dir,
        args.output,
        max_num_of_committees=args.max_committees,
        verbose=args.verbose,
        rule_ids=rule_ids,
        overwrite=args.overwrite,
    )

    tprint(f"Benchmarks complete. Results in {args.output}")


if __name__ == "__main__":
    main()
