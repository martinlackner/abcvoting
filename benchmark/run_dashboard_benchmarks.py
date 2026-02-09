#!/usr/bin/env python3
"""
Benchmark runner for abcvoting dashboard.

Runs all ABC voting rules on test instances and outputs JSON results
for the dashboard generator.

Usage:
    python run_dashboard_benchmarks.py --output benchmark_results.json
    python run_dashboard_benchmarks.py --categories S M --timeout 60 --output results.json
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
from abcvoting.abcrules import MAIN_RULE_IDS, Rule  # noqa: E402


# Configuration
DEFAULT_TIMEOUT = 300  # 5 minutes
INSTANCE_DIR = "tests/test_instances"
SIZE_CATEGORIES = ["S", "M", "L", "VL"]


class TimeoutException(Exception):
    """Exception raised when a computation times out."""

    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Computation timed out")


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

        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        ram_kb = int(line.split()[1])
                        ram_gb = round(ram_kb / (1024**2), 1)
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
    """Load all instances for a size category (S, M, L, VL)."""
    instances = []
    prefix = f"instance{category}"

    # Get absolute path
    if not os.path.isabs(instance_dir):
        instance_dir = os.path.join(project_root, instance_dir)

    for filename in sorted(os.listdir(instance_dir)):
        if filename.startswith(prefix) and filename.endswith(".abc.yaml"):
            filepath = os.path.join(instance_dir, filename)
            try:
                profile, committeesize, _, _ = fileio.read_abcvoting_yaml_file(filepath)
                instances.append((profile, committeesize, filename))
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}", file=sys.stderr)

    return instances


def get_status(finished, total):
    """Determine traffic light status based on completion ratio."""
    if total == 0:
        return "skipped"
    ratio = finished / total
    if ratio >= 0.9:
        return "green"
    elif ratio >= 0.5:
        return "yellow"
    else:
        return "red"


def run_single_benchmark(rule, algorithm, profile, committeesize, resolute, timeout_seconds):
    """
    Run a single benchmark and return the runtime or None if it timed out.

    Returns:
        float or None: Runtime in seconds, or None if timed out/failed
    """
    # Set up timeout handler (Unix only)
    if hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

    try:
        start_time = time.perf_counter()
        rule.compute(profile, committeesize, algorithm=algorithm, resolute=resolute)
        elapsed = time.perf_counter() - start_time
        return elapsed
    except TimeoutException:
        return None
    except Exception as e:
        # Rule may not support this resolute value or algorithm
        print(f"  Error: {e}", file=sys.stderr)
        return None
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def run_benchmarks_for_rule(rule_id, categories, timeout_seconds, instance_dir, verbose=False):
    """
    Run benchmarks for a single rule across all categories.

    Returns:
        dict: Results for this rule
    """
    rule = Rule(rule_id)
    result = {
        "shortname": rule.shortname,
        "longname": rule.longname,
        "fastest_algorithm": None,
        "algorithms": {},
    }

    # Get available algorithms for this rule
    available_algorithms = rule.available_algorithms
    if not available_algorithms:
        print(f"  No available algorithms for {rule_id}", file=sys.stderr)
        return result

    # Track fastest algorithm (by total runtime across all categories)
    algorithm_total_times = {}

    # Load all instances from all categories
    all_instances = []
    for category in categories:
        all_instances.extend(load_instances_by_category(category, instance_dir))

    for algorithm in available_algorithms:
        if verbose:
            print(f"  Algorithm: {algorithm}")

        result["algorithms"][algorithm] = {"resolute": {}, "irresolute": {}}

        for resolute_mode, mode_name in [(True, "resolute"), (False, "irresolute")]:
            # Check if this rule supports this resolute mode
            if resolute_mode not in rule.resolute_values:
                continue

            total = len(all_instances)
            finished = 0
            max_runtime = 0.0
            total_time_for_algorithm = 0

            if verbose:
                print(f"    {mode_name}: {total} instances")

            for profile, committeesize, filename in all_instances:
                runtime = run_single_benchmark(
                    rule, algorithm, profile, committeesize, resolute_mode, timeout_seconds
                )
                if runtime is not None:
                    finished += 1
                    max_runtime = max(max_runtime, runtime)
                    total_time_for_algorithm += runtime
                else:
                    # Timed out - stop running more instances for this algorithm/mode
                    break

            status = get_status(finished, total)
            result["algorithms"][algorithm][mode_name] = {
                "finished": finished,
                "total": total,
                "max_runtime": round(max_runtime, 4) if finished > 0 else None,
                "status": status,
            }

            algorithm_total_times[algorithm] = (
                algorithm_total_times.get(algorithm, 0) + total_time_for_algorithm
            )

    # Determine fastest algorithm
    if algorithm_total_times:
        result["fastest_algorithm"] = min(algorithm_total_times, key=algorithm_total_times.get)

    return result


def run_all_benchmarks(categories, timeout_seconds, instance_dir, verbose=False, rule_ids=None):
    """
    Run benchmarks for all rules (or specified rules).

    Parameters:
        categories: List of instance categories to run
        timeout_seconds: Timeout per instance
        instance_dir: Directory containing test instances
        verbose: Print verbose output
        rule_ids: List of rule IDs to run (None = all rules)

    Returns:
        dict: Complete benchmark results
    """
    if rule_ids is None:
        rule_ids = MAIN_RULE_IDS

    results = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hardware": collect_hardware_info(),
            "timeout_seconds": timeout_seconds,
            "abcvoting_version": get_abcvoting_version(),
            "categories": categories,
        },
        "results": {},
    }

    total_rules = len(rule_ids)
    for i, rule_id in enumerate(rule_ids, 1):
        print(f"[{i}/{total_rules}] Running benchmarks for {rule_id}...")
        results["results"][rule_id] = run_benchmarks_for_rule(
            rule_id, categories, timeout_seconds, instance_dir, verbose
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks for abcvoting dashboard")
    parser.add_argument(
        "--output",
        "-o",
        default="benchmark_results.json",
        help="Output JSON file (default: benchmark_results.json)",
    )
    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        choices=SIZE_CATEGORIES,
        default=SIZE_CATEGORIES,
        help=f"Instance categories to run (default: {' '.join(SIZE_CATEGORIES)})",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout per instance in seconds (default: {DEFAULT_TIMEOUT})",
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

    args = parser.parse_args()

    # Validate categories order
    categories = [c for c in SIZE_CATEGORIES if c in args.categories]

    # Validate and filter rules if specified
    rule_ids = None
    if args.rules:
        rule_ids = [r for r in args.rules if r in MAIN_RULE_IDS]
        if not rule_ids:
            print(f"Error: No valid rule IDs specified. Valid IDs: {', '.join(MAIN_RULE_IDS)}")
            sys.exit(1)

    print("Running benchmarks with:")
    print(f"  Categories: {', '.join(categories)}")
    print(f"  Timeout: {args.timeout}s per instance")
    print(f"  Instance directory: {args.instance_dir}")
    if rule_ids:
        print(f"  Rules: {', '.join(rule_ids)}")
    print()

    results = run_all_benchmarks(
        categories, args.timeout, args.instance_dir, args.verbose, rule_ids
    )

    # Write results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
