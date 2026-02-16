# Benchmark Tools

This directory contains scripts for benchmarking ABC voting rules.

## Running Benchmarks

```bash
# Activate the virtual environment
source venv/bin/activate

# Run benchmarks (all rules, all instance sizes)
python benchmark/run_benchmarks.py

# Run only small/medium instances with 60s timeout
python benchmark/run_benchmarks.py -c S M -t 60

# Run specific rules only
python benchmark/run_benchmarks.py -r pav cc

# Verbose output
python benchmark/run_benchmarks.py -v

# Custom output location
python benchmark/run_benchmarks.py -o custom_results.json
```

### Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output JSON file (default: `benchmark/data/benchmark_results.json`) |
| `-c, --categories` | Instance size categories: S, M, L, VL, G (default: G) |
| `-t, --timeout` | Cumulative timeout in seconds (default: 300) |
| `-m, --max-committees` | Maximum number of committees to compute (default: 10) |
| `-r, --rules` | Run specific rules only (by rule_id) |
| `-v, --verbose` | Print detailed output |

### Incremental Results

The benchmark runner supports incremental execution: if the output file already exists, it will skip rules that have already been benchmarked.

**Metadata validation:** When using an existing results file, the script validates that the current configuration matches the stored metadata. The following parameters must match:
- `timeout` (`-t`)
- `max-committees` (`-m`)
- `categories` (`-c`)

If there's a mismatch, the script exits with an error. To resolve this:
1. Delete the existing results file and start fresh
2. Use matching configuration parameters
3. Specify a different output file with `--output`

## Generating the Dashboard

```bash
# Generate dashboard (uses default input from benchmark/data/benchmark_results.json)
python benchmark/generate_dashboard.py

# Use custom input file
python benchmark/generate_dashboard.py -i custom_results.json
```

This creates an HTML dashboard at `benchmark/index.html`.

## Test Instances

Benchmarks use instances from `tests/test_instances/` with size prefixes:
- **S** - Small instances
- **M** - Medium instances
- **L** - Large instances
- **VL** - Very large instances
