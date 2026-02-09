#!/usr/bin/env python3
"""
Dashboard generator for abcvoting benchmarks.

Generates static HTML dashboard from JSON benchmark results.

Usage:
    python generate_dashboard.py --input benchmark_results.json --output docs/benchmark/
"""

import argparse
import json
import os
import shutil


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
    .status-green {
        background-color: #90EE90;
    }
    .status-yellow {
        background-color: #FFD700;
    }
    .status-red {
        background-color: #FF6B6B;
    }
    .status-skipped {
        background-color: #D3D3D3;
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
    .legend {
        display: flex;
        gap: 20px;
        margin-bottom: 15px;
        flex-wrap: wrap;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .legend-color {
        width: 20px;
        height: 20px;
        border-radius: 4px;
    }
    .hidden {
        display: none;
    }
    .all-algorithms-row {
        background-color: #fafafa;
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

    // Sort table by column
    function sortTable(tableId, columnIndex, isNumeric) {
        const table = document.getElementById(tableId);
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const header = table.querySelectorAll('th')[columnIndex];

        // Determine sort direction
        const isAsc = header.classList.contains('sort-asc');

        // Remove sort classes from all headers
        table.querySelectorAll('th').forEach(th => {
            th.classList.remove('sort-asc', 'sort-desc');
        });

        // Add appropriate sort class
        header.classList.add(isAsc ? 'sort-desc' : 'sort-asc');

        // Sort rows
        rows.sort((a, b) => {
            let aVal = a.cells[columnIndex].getAttribute('data-sort-value') ||
                       a.cells[columnIndex].textContent.trim();
            let bVal = b.cells[columnIndex].getAttribute('data-sort-value') ||
                       b.cells[columnIndex].textContent.trim();

            if (isNumeric) {
                aVal = parseFloat(aVal) || 0;
                bVal = parseFloat(bVal) || 0;
            }

            if (aVal < bVal) return isAsc ? 1 : -1;
            if (aVal > bVal) return isAsc ? -1 : 1;
            return 0;
        });

        // Reorder rows
        rows.forEach(row => tbody.appendChild(row));
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


def generate_cells_html(cell_data):
    """Generate HTML for the two result cells (Completed, Runtime)."""
    if not cell_data:
        return '<td class="status-skipped">-</td><td class="status-skipped">-</td>'

    status = cell_data.get("status", "skipped")
    finished = cell_data.get("finished", 0)
    total = cell_data.get("total", 0)
    max_runtime = cell_data.get("max_runtime")

    # Completed instances cell
    completed_str = f"{finished}/{total}" if total > 0 else "-"

    # Runtime cell - show "timeout" if not all instances finished
    if total == 0:
        runtime_str = "-"
        runtime_sort = 999999  # Sort to end
    elif finished < total:
        runtime_str = "timeout"
        runtime_sort = 999999  # Sort timeouts to end
    else:
        runtime_str = format_runtime(max_runtime)
        runtime_sort = max_runtime if max_runtime else 0

    return (
        f'<td class="status-{status}" data-sort-value="{finished}">{completed_str}</td>'
        f'<td class="status-{status}" data-sort-value="{runtime_sort}">{runtime_str}</td>'
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
                <th class="sortable" onclick="sortTable('{table_id}', 3, true)">Runtime</th>
            </tr>
        </thead>
        <tbody>
    """

    for rule_id, rule_data in data["results"].items():
        shortname = rule_data.get("shortname", rule_id)
        fastest_algo = rule_data.get("fastest_algorithm")
        algorithms = rule_data.get("algorithms", {})

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
            algo_badge = (
                '<span class="algorithm-badge fastest-badge">fastest</span>' if is_fastest else ""
            )

            html += f'<tr class="{row_class}">'
            if first_row:
                longname = rule_data.get("longname", rule_id)
                html += f"<td><strong>{shortname}</strong><br><small>{longname}</small></td>"
            else:
                html += "<td></td>"

            html += f"<td>{algo}{algo_badge}</td>"
            html += generate_cells_html(mode_data)

            html += "</tr>"
            first_row = False

    html += """
        </tbody>
    </table>
    """

    return html


def generate_main_dashboard(data, output_dir):
    """Generate the main dashboard HTML page."""
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
            </div>
            <div class="nav-links">
                <a href="data/benchmark_results.json">Raw JSON Data</a>
            </div>
        </header>

        <div class="controls">
            <label>
                <input type="checkbox" id="showAllAlgorithms">
                Show all algorithms (default: fastest only)
            </label>
        </div>

        <div class="legend">
            <div class="legend-item">
                <div class="legend-color status-green"></div>
                <span>Green: ≥90% completed</span>
            </div>
            <div class="legend-item">
                <div class="legend-color status-yellow"></div>
                <span>Yellow: 50-89% completed</span>
            </div>
            <div class="legend-item">
                <div class="legend-color status-red"></div>
                <span>Red: <50% completed</span>
            </div>
            <div class="legend-item">
                <div class="legend-color status-skipped"></div>
                <span>Gray: Skipped</span>
            </div>
        </div>

        <div class="section">
            <h2>Resolute Mode</h2>
            {generate_table_html(data, "resolute", "resolute-table")}
        </div>

        <div class="section">
            <h2>Irresolute Mode</h2>
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
        required=True,
        help="Input JSON file with benchmark results",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="docs/benchmark",
        help="Output directory for HTML files (default: docs/benchmark)",
    )

    args = parser.parse_args()

    # Load benchmark results
    with open(args.input) as f:
        data = json.load(f)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "data"), exist_ok=True)

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
    main_page = generate_main_dashboard(data, args.output)
    print(f"Generated main dashboard: {main_page}")

    print(f"\nDashboard generated successfully in {args.output}/")
    print(f"Open {main_page} in a browser to view.")


if __name__ == "__main__":
    main()
