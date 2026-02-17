#!/usr/bin/env python3
"""
Dashboard generator for abcvoting benchmarks.

Generates static HTML dashboard from JSON benchmark results.

Usage:
    python generate_dashboard.py
    python generate_dashboard.py --input results.json --output benchmark/
"""

import argparse
import json
import os
import shutil
import glob as glob_module
import yaml

# Directory containing generated benchmark instances
INSTANCES_DIR = os.path.join(os.path.dirname(__file__), "instances")


def collect_instance_statistics(instances_dir=None):
    """
    Collect statistics from benchmark instances for visualization.

    Returns:
        dict: Instance statistics including num_voters, num_cand, committeesize arrays
    """
    if instances_dir is None:
        instances_dir = INSTANCES_DIR

    stats = {
        "num_voters": [],
        "num_cand": [],
        "committeesize": [],
        "instances": [],  # List of (num_voters, num_cand, committeesize) tuples
    }

    if not os.path.exists(instances_dir):
        return stats

    instance_files = sorted(glob_module.glob(os.path.join(instances_dir, "*.abc.yaml")))

    for filepath in instance_files:
        try:
            with open(filepath) as f:
                data = yaml.safe_load(f)

            num_cand = data.get("num_cand", 0)
            committeesize = data.get("committeesize", 0)
            profile = data.get("profile", [])
            num_voters = len(profile)

            stats["num_voters"].append(num_voters)
            stats["num_cand"].append(num_cand)
            stats["committeesize"].append(committeesize)
            stats["instances"].append((num_voters, num_cand, committeesize))
        except Exception:
            continue

    return stats


def generate_css():
    """Generate inline CSS for the dashboard."""
    return """
<style>
    * {
        box-sizing: border-box;
    }
    body {
        font-family: system-ui, -apple-system, sans-serif;
        margin: 0;
        padding: 20px;
        background-color: #f5f5f5;
        color: #333;
    }
    .container {
        max-width: 1400px;
        margin: 0 auto;
    }
    header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        padding: 30px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    header h1 {
        margin: 0 0 15px 0;
        font-size: 2em;
    }
    .metadata {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        font-size: 0.9em;
        opacity: 0.9;
    }
    .metadata-item {
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .metadata-label {
        font-weight: bold;
    }
    .controls {
        background: white;
        padding: 15px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: flex;
        gap: 20px;
        align-items: center;
        flex-wrap: wrap;
    }
    .controls label {
        display: flex;
        align-items: center;
        gap: 8px;
        cursor: pointer;
    }
    .controls input[type="checkbox"] {
        width: 18px;
        height: 18px;
    }
    .section {
        background: white;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    details.section summary {
        cursor: pointer;
        list-style: none;
    }
    details.section summary::-webkit-details-marker {
        display: none;
    }
    details.section summary h2 {
        display: inline;
    }
    details.section summary h2::before {
        content: '▶';
        display: inline-block;
        margin-right: 8px;
        font-size: 0.8em;
    }
    details.section[open] summary h2::before {
        content: '▼';
    }
    details.section .explanation {
        margin-top: 15px;
    }
    .section h2 {
        margin-top: 0;
        color: #1a1a2e;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 10px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9em;
    }
    th, td {
        padding: 10px 12px;
        text-align: left;
        border-bottom: 1px solid #e0e0e0;
    }
    th {
        background-color: #f8f9fa;
        font-weight: 600;
        position: sticky;
        top: 0;
    }
    th.sortable {
        cursor: pointer;
        user-select: none;
    }
    th.sortable:hover {
        background-color: #e9ecef;
    }
    th.sortable::after {
        content: ' \\2195';
        opacity: 0.3;
    }
    th.sort-asc::after {
        content: ' \\2191';
        opacity: 1;
    }
    th.sort-desc::after {
        content: ' \\2193';
        opacity: 1;
    }
    tr:hover {
        background-color: #f8f9fa;
    }
    .cell-content {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    .cell-count {
        font-weight: 600;
    }
    .cell-time {
        font-size: 0.85em;
        color: #666;
    }
    .algorithm-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        background-color: #e9ecef;
        font-size: 0.85em;
        margin-left: 8px;
    }
    .fastest-badge {
        background-color: #d4edda;
        color: #155724;
    }
    .differs-badge {
        background-color: #fff3cd;
        color: #856404;
        cursor: help;
    }
    .nav-links {
        margin-top: 15px;
    }
    .nav-links a {
        color: rgba(255,255,255,0.9);
        text-decoration: none;
        margin-right: 20px;
    }
    .nav-links a:hover {
        text-decoration: underline;
    }
    .hidden {
        display: none;
    }
    .all-algorithms-row {
        background-color: #fafafa;
    }
    .all-algorithms-row .rule-cell {
        color: #999;
        font-weight: normal;
    }
    .all-algorithms-row .rule-cell small {
        display: none;
    }
    .explanation {
        line-height: 1.6;
    }
    .explanation h3 {
        color: #1a1a2e;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .explanation h3:first-child {
        margin-top: 0;
    }
    .explanation ul {
        margin: 10px 0;
        padding-left: 25px;
    }
    .explanation li {
        margin-bottom: 5px;
    }
    .explanation code {
        background-color: #f0f0f0;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.9em;
    }
    .charts-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-top: 20px;
    }
    .chart-box {
        background: #fafafa;
        border-radius: 8px;
        padding: 15px;
    }
    .chart-box h4 {
        margin: 0 0 10px 0;
        color: #333;
        font-size: 0.95em;
    }
    .chart-canvas {
        max-height: 300px;
    }
    @media (max-width: 900px) {
        .charts-container {
            grid-template-columns: 1fr;
        }
    }
    @media (max-width: 768px) {
        body {
            padding: 10px;
        }
        table {
            font-size: 0.8em;
        }
        th, td {
            padding: 6px 8px;
        }
    }
</style>
"""


def generate_javascript():
    """Generate inline JavaScript for the dashboard."""
    return """
<script>
    // Toggle between fastest algorithm and all algorithms
    function toggleAlgorithms() {
        const showAll = document.getElementById('showAllAlgorithms').checked;
        const allAlgoRows = document.querySelectorAll('.all-algorithms-row');
        allAlgoRows.forEach(row => {
            row.classList.toggle('hidden', !showAll);
        });
    }

    // Sort table by column (maintains algorithm grouping)
    function sortTable(tableId, columnIndex, isNumeric) {
        const table = document.getElementById(tableId);
        const tbody = table.querySelector('tbody');
        const allRows = Array.from(tbody.querySelectorAll('tr'));
        const header = table.querySelectorAll('th')[columnIndex];

        // Group rows: primary rows followed by their secondary algorithm rows
        const groups = [];
        let currentGroup = null;

        allRows.forEach(row => {
            if (!row.classList.contains('all-algorithms-row')) {
                // Primary row - start new group
                currentGroup = { primary: row, secondary: [] };
                groups.push(currentGroup);
            } else if (currentGroup) {
                // Secondary row - add to current group
                currentGroup.secondary.push(row);
            }
        });

        // Determine sort direction
        const isAsc = header.classList.contains('sort-asc');

        // Remove sort classes from all headers
        table.querySelectorAll('th').forEach(th => {
            th.classList.remove('sort-asc', 'sort-desc');
        });

        // Add appropriate sort class
        header.classList.add(isAsc ? 'sort-desc' : 'sort-asc');

        // Sort groups by primary row values
        groups.sort((a, b) => {
            let aVal = a.primary.cells[columnIndex].getAttribute('data-sort-value') ||
                       a.primary.cells[columnIndex].textContent.trim();
            let bVal = b.primary.cells[columnIndex].getAttribute('data-sort-value') ||
                       b.primary.cells[columnIndex].textContent.trim();

            if (isNumeric) {
                aVal = parseFloat(aVal) || 0;
                bVal = parseFloat(bVal) || 0;
            }

            if (aVal < bVal) return isAsc ? 1 : -1;
            if (aVal > bVal) return isAsc ? -1 : 1;
            return 0;
        });

        // Reorder rows maintaining group structure
        groups.forEach(group => {
            tbody.appendChild(group.primary);
            group.secondary.forEach(row => tbody.appendChild(row));
        });
    }

    // Initialize on page load
    document.addEventListener('DOMContentLoaded', function() {
        // Set up checkbox listener
        const checkbox = document.getElementById('showAllAlgorithms');
        if (checkbox) {
            checkbox.addEventListener('change', toggleAlgorithms);
        }
    });
</script>
"""


def generate_explanation_section(instance_stats):
    """Generate HTML for the benchmark explanation section with charts."""
    # Prepare data for charts
    instances_json = json.dumps(instance_stats.get("instances", []))

    # Calculate committee size distribution for bar chart
    committeesize_counts = {}
    for cs in instance_stats.get("committeesize", []):
        committeesize_counts[cs] = committeesize_counts.get(cs, 0) + 1

    # Sort by committee size
    sorted_sizes = sorted(committeesize_counts.keys())
    committeesize_labels = json.dumps([str(s) for s in sorted_sizes])
    committeesize_values = json.dumps([committeesize_counts[s] for s in sorted_sizes])

    return f"""
        <details class="section">
            <summary><h2>About the Benchmarks</h2></summary>
            <div class="explanation">
                <h3>Instance Generation</h3>
                <p>1000 benchmark instances are generated using various probability models from the abcvoting library:</p>
                <ul>
                    <li><strong>IC (Impartial Culture)</strong>: Each candidate is approved independently with probability p (p=0.3 or p=0.5)</li>
                    <li><strong>IC fixed-size</strong>: Each voter approves exactly k candidates uniformly at random (k=2, 3, or 4)</li>
                    <li><strong>Truncated Mallows</strong>: Voters have correlated preferences based on a central ranking with dispersion parameter</li>
                    <li><strong>Urn fixed-size</strong>: Polya-Eggenberger urn model with replacement parameter</li>
                </ul>

                <h3>Instance Parameters</h3>
                <ul>
                    <li><strong>Number of voters</strong>: Uniformly sampled from [5, 100]</li>
                    <li><strong>Number of candidates</strong>: Uniformly sampled from [5, 100]</li>
                    <li><strong>Committee size</strong>: Between <code>max(3, num_cand/10)</code> and <code>max(4, num_cand/3)</code></li>
                </ul>

                <h3>Instance Ordering</h3>
                <p>Instances are sorted by <strong>number of candidates</strong>, then by <strong>committee size</strong>,
                then by <strong>number of voters</strong>. This ordering prioritizes the parameters that most
                affect computational complexity for ABC voting rules.</p>

                <h3>Benchmark Execution</h3>
                <ul>
                    <li>Each rule/algorithm combination runs through instances in order</li>
                    <li>A <strong>cumulative timeout</strong> applies to all instances together (not per-instance)</li>
                    <li>When the cumulative runtime exceeds the timeout, remaining instances are skipped</li>
                    <li>The "Completed" column shows how many instances finished before timeout</li>
                </ul>

                <div class="charts-container">
                    <div class="chart-box">
                        <h4>Instance Distribution: Voters vs Candidates</h4>
                        <canvas id="scatterChart" class="chart-canvas"></canvas>
                    </div>
                    <div class="chart-box">
                        <h4>Committee Size Distribution</h4>
                        <canvas id="barChart" class="chart-canvas"></canvas>
                    </div>
                </div>
            </div>
        </details>

        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
        <script>
            (function() {{
                const instances = {instances_json};
                const committeesizeLabels = {committeesize_labels};
                const committeesizeValues = {committeesize_values};

                // Scatter plot: num_voters vs num_cand
                const scatterData = instances.map(function(inst) {{
                    return {{ x: inst[1], y: inst[0] }};  // x=num_cand, y=num_voters
                }});

                new Chart(document.getElementById('scatterChart'), {{
                    type: 'scatter',
                    data: {{
                        datasets: [{{
                            label: 'Instances',
                            data: scatterData,
                            backgroundColor: 'rgba(54, 162, 235, 0.6)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            pointRadius: 4,
                            pointHoverRadius: 6
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: true,
                        plugins: {{
                            legend: {{ display: false }}
                        }},
                        scales: {{
                            x: {{
                                title: {{ display: true, text: 'Number of Candidates' }},
                                min: 0,
                                max: 105,
                                ticks: {{ stepSize: 20 }}
                            }},
                            y: {{
                                title: {{ display: true, text: 'Number of Voters' }},
                                min: 0,
                                max: 105,
                                ticks: {{ stepSize: 20 }}
                            }}
                        }}
                    }}
                }});

                // Bar chart: committee size distribution
                new Chart(document.getElementById('barChart'), {{
                    type: 'bar',
                    data: {{
                        labels: committeesizeLabels,
                        datasets: [{{
                            label: 'Number of Instances',
                            data: committeesizeValues,
                            backgroundColor: 'rgba(75, 192, 192, 0.6)',
                            borderColor: 'rgba(75, 192, 192, 1)',
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: true,
                        plugins: {{
                            legend: {{ display: false }}
                        }},
                        scales: {{
                            x: {{
                                title: {{ display: true, text: 'Committee Size' }}
                            }},
                            y: {{
                                title: {{ display: true, text: 'Number of Instances' }},
                                beginAtZero: true
                            }}
                        }}
                    }}
                }});
            }})();
        </script>
    """


def format_runtime(runtime):
    """Format runtime for display."""
    if runtime is None:
        return "-"
    if runtime < 1:
        return f"{runtime * 1000:.0f}ms"
    elif runtime < 60:
        return f"{runtime:.2f}s"
    else:
        return f"{runtime / 60:.1f}m"


def ratio_to_color(ratio):
    """
    Convert a completion ratio (0-1) to an RGB color string.

    Uses a continuous gradient: red (0%) -> yellow (50%) -> green (100%).
    """
    # Clamp ratio to [0, 1]
    ratio = max(0.0, min(1.0, ratio))

    # Color stops: red -> yellow -> green
    # Red:    (255, 107, 107) at ratio=0
    # Yellow: (255, 215, 0)   at ratio=0.5
    # Green:  (144, 238, 144) at ratio=1

    if ratio <= 0.5:
        # Interpolate from red to yellow (ratio 0 to 0.5)
        t = ratio * 2  # Scale to 0-1
        r = int(255 + (255 - 255) * t)  # 255 -> 255
        g = int(107 + (215 - 107) * t)  # 107 -> 215
        b = int(107 + (0 - 107) * t)  # 107 -> 0
    else:
        # Interpolate from yellow to green (ratio 0.5 to 1)
        t = (ratio - 0.5) * 2  # Scale to 0-1
        r = int(255 + (144 - 255) * t)  # 255 -> 144
        g = int(215 + (238 - 215) * t)  # 215 -> 238
        b = int(0 + (144 - 0) * t)  # 0 -> 144

    return f"rgb({r}, {g}, {b})"


def generate_cells_html(cell_data):
    """Generate HTML for the two result cells (Completed, Runtime)."""
    if not cell_data:
        return (
            '<td style="background-color: #D3D3D3">-</td>'
            '<td style="background-color: #D3D3D3">-</td>'
        )

    finished = cell_data.get("finished", 0)
    total = cell_data.get("total", 0)
    runtime = cell_data.get("cumulative_runtime")

    # Completed instances cell
    completed_str = f"{finished}/{total}" if total > 0 else "-"

    # Calculate color based on completion ratio
    if total == 0:
        bg_color = "#D3D3D3"  # Gray for skipped
        runtime_str = "-"
        runtime_sort = 999999  # Sort to end
    else:
        ratio = finished / total
        bg_color = ratio_to_color(ratio)
        if finished < total:
            runtime_str = "timeout"
            runtime_sort = 999999  # Sort timeouts to end
        else:
            runtime_str = format_runtime(runtime)
            runtime_sort = runtime if runtime else 0

    return (
        f'<td style="background-color: {bg_color}" data-sort-value="{finished}">'
        f"{completed_str}</td>"
        f'<td style="background-color: {bg_color}" data-sort-value="{runtime_sort}">'
        f"{runtime_str}</td>"
    )


def generate_table_html(data, mode, table_id):
    """Generate HTML for a results table."""
    html = f"""
    <table id="{table_id}">
        <thead>
            <tr>
                <th class="sortable" onclick="sortTable('{table_id}', 0, false)">Rule</th>
                <th class="sortable" onclick="sortTable('{table_id}', 1, false)">Algorithm</th>
                <th class="sortable" onclick="sortTable('{table_id}', 2, true)">Completed</th>
                <th class="sortable" onclick="sortTable('{table_id}', 3, true)">Total Runtime</th>
            </tr>
        </thead>
        <tbody>
    """

    for rule_id, rule_data in data["results"].items():
        shortname = rule_data.get("shortname", rule_id)
        # Use mode-specific fastest, fall back to overall fastest for backwards compatibility
        fastest_algo = rule_data.get(f"fastest_algorithm_{mode}") or rule_data.get(
            "fastest_algorithm"
        )
        library_fastest = rule_data.get("library_fastest")
        algorithms = rule_data.get("algorithms", {})
        fastest_differs = library_fastest is not None and fastest_algo != library_fastest
        # No warning if both algorithms finished the same number of instances due to timeout (tie)
        if fastest_differs and library_fastest in algorithms:
            fastest_data = algorithms.get(fastest_algo, {}).get(mode, {})
            library_data = algorithms.get(library_fastest, {}).get(mode, {})
            fastest_finished = fastest_data.get("finished", 0)
            library_finished = library_data.get("finished", 0)
            total = fastest_data.get("total", 0)
            if fastest_finished == library_finished < total:
                fastest_differs = False

        # Sort algorithms to put the fastest one first
        algo_list = list(algorithms.items())
        algo_list.sort(key=lambda x: (x[0] != fastest_algo, x[0]))

        first_row = True
        for algo, algo_data in algo_list:
            mode_data = algo_data.get(mode, {})
            if not mode_data:
                continue

            is_fastest = algo == fastest_algo
            row_class = "" if first_row else "all-algorithms-row hidden"

            # Build algorithm badges
            badges = []
            if is_fastest:
                badges.append('<span class="algorithm-badge fastest-badge">fastest</span>')
                if fastest_differs:
                    badges.append(
                        f'<span class="algorithm-badge differs-badge" '
                        f"title=\"Library expects '{library_fastest}' to be fastest\">"
                        f'⚠ differs from "fastest"-default</span>'
                    )
            algo_badge = "".join(badges)

            longname = rule_data.get("longname", rule_id)
            html += f'<tr class="{row_class}">'
            # Always include rule name for proper sorting; style differs for secondary rows
            html += (
                f'<td class="rule-cell" data-sort-value="{shortname}">'
                f"<strong>{shortname}</strong><br><small>{longname}</small></td>"
            )
            html += f"<td>{algo}{algo_badge}</td>"
            html += generate_cells_html(mode_data)

            html += "</tr>"
            first_row = False

    html += """
        </tbody>
    </table>
    """

    return html


def generate_main_dashboard(data, output_dir, instance_stats=None):
    """Generate the main dashboard HTML page."""
    if instance_stats is None:
        instance_stats = collect_instance_statistics()

    metadata = data.get("metadata", {})
    hardware = metadata.get("hardware", {})

    timestamp = metadata.get("timestamp", "Unknown")
    if "T" in timestamp:
        timestamp = timestamp.split("T")[0]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>abcvoting Benchmark Dashboard</title>
    {generate_css()}
</head>
<body>
    <div class="container">
        <header>
            <h1>abcvoting Benchmark Dashboard</h1>
            <div class="metadata">
                <div class="metadata-item">
                    <span class="metadata-label">CPU:</span>
                    <span>{hardware.get("cpu", "Unknown")}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">RAM:</span>
                    <span>{hardware.get("ram_gb", "?")} GB</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">OS:</span>
                    <span>{hardware.get("os", "Unknown")}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Generated:</span>
                    <span>{timestamp}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Timeout:</span>
                    <span>{metadata.get("timeout_seconds", "?")}s</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">abcvoting:</span>
                    <span>v{metadata.get("abcvoting_version", "?")}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Categories:</span>
                    <span>{", ".join(metadata.get("categories", ["?"]))}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Instances:</span>
                    <span>{len(instance_stats.get("instances", []))}</span>
                </div>
            </div>
            <div class="nav-links">
                <a href="data/benchmark_results.json">Raw JSON Data</a>
            </div>
        </header>

        {generate_explanation_section(instance_stats)}

        <div class="controls">
            <label>
                <input type="checkbox" id="showAllAlgorithms">
                Show all algorithms (default: fastest only)
            </label>
        </div>

        <div class="section">
            <h2>Resolute Mode</h2>
            {generate_table_html(data, "resolute", "resolute-table")}
        </div>

        <div class="section">
            <h2>Irresolute Mode (max_num_of_committees={metadata.get("max_num_of_committees", "N/A")})</h2>
            {generate_table_html(data, "irresolute", "irresolute-table")}
        </div>
    </div>
    {generate_javascript()}
</body>
</html>
"""

    output_path = os.path.join(output_dir, "index.html")
    with open(output_path, "w") as f:
        f.write(html)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate HTML dashboard from benchmark results")
    parser.add_argument(
        "--input",
        "-i",
        default="benchmark/data/benchmark_results.json",
        help="Input JSON file with benchmark results (default: benchmark/data/benchmark_results.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="docs/benchmark",
        help="Output directory for HTML files (default: docs/benchmark)",
    )

    args = parser.parse_args()

    # Load and validate benchmark results
    try:
        with open(args.input) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {args.input}: {e}")
        return 1

    # Validate required structure
    if not isinstance(data, dict):
        print("Error: JSON root must be an object")
        return 1
    if "results" not in data:
        print("Error: JSON missing required 'results' key")
        return 1
    if not isinstance(data["results"], dict):
        print("Error: 'results' must be an object")
        return 1

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "data"), exist_ok=True)

    # Collect instance statistics for charts
    instance_stats = collect_instance_statistics()
    num_instances = len(instance_stats.get("instances", []))
    if num_instances > 0:
        print(f"Loaded statistics for {num_instances} instances")
    else:
        print("Warning: No instances found for charts (charts will be empty)")

    # Copy JSON data to output (if not already there)
    json_output = os.path.join(args.output, "data", "benchmark_results.json")
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(json_output)
    if input_path != output_path:
        shutil.copy(args.input, json_output)
        print(f"Copied JSON data to {json_output}")
    else:
        print(f"JSON data already at {json_output}")

    # Generate HTML pages
    main_page = generate_main_dashboard(data, args.output, instance_stats)
    print(f"Generated main dashboard: {main_page}")

    print(f"\nDashboard generated successfully in {args.output}/")
    print(f"Open {main_page} in a browser to view.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main() or 0)
