#!/usr/bin/env python3
"""
Plot benchmark results from plot_results directory.
Creates:
1. Line charts for each metric (TTFT, TPOT, E2E Latency, Max Tokens/s)
2. Grouped bar charts showing percentage improvement vs baseline
X-axis shows ISL-OSL combinations.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Default configuration
DEFAULT_RESULTS_DIR = Path("plot_results")
DEFAULT_OUTPUT_DIR = Path("benchmark_plots")

DEFAULT_BASELINE = "default-no-tuned"

# These will be set from command line args
RESULTS_DIR = DEFAULT_RESULTS_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

# Metrics to plot
METRICS = {
    "mean_ttft_ms": {
        "title": "Mean Time to First Token (TTFT)",
        "ylabel": "TTFT (ms)",
        "color": "#2E86AB",
        "better": "lower",
        "display_name": "Time to First Token (TTFT)"
    },
    "mean_tpot_ms": {
        "title": "Mean Time Per Output Token (TPOT)",
        "ylabel": "TPOT (ms)",
        "color": "#A23B72",
        "better": "lower",
        "display_name": "Time Per Output Token (TPOT)"
    },
    "mean_e2el_ms": {
        "title": "Mean End-to-End Latency",
        "ylabel": "E2E Latency (ms)",
        "color": "#F18F01",
        "better": "lower",
        "display_name": "End-to-End Latency"
    },
    "max_output_tokens_per_s": {
        "title": "Max Output Tokens Per Second",
        "ylabel": "Max Tokens/s",
        "color": "#C73E1D",
        "better": "higher",
        "display_name": "Max Output Tokens Per Second"
    }
}

# Global color mapping for configurations (to ensure consistency across plots)
_config_colors = {}


def get_config_colors(config_names: List[str]) -> Dict[str, Tuple]:
    """Get consistent colors for configurations."""
    global _config_colors
    sorted_names = sorted(config_names)
    
    # Generate colors using tab10 colormap
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_names)))
    
    for idx, name in enumerate(sorted_names):
        if name not in _config_colors:
            _config_colors[name] = colors[idx]
    
    return _config_colors


def extract_isl_osl(filename):
    """Extract ISL and OSL values from filename."""
    match = re.search(r'ISL-(\d+)-OSL-(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def load_benchmark_data():
    """Load all benchmark data from JSON files.
    
    Returns:
        tuple: (data dict, ref_num_requests, ref_concurrency)
    """
    data = {}
    ref_num_requests = None
    ref_concurrency = None
    
    for config_dir in sorted(RESULTS_DIR.iterdir()):
        if not config_dir.is_dir():
            continue
        
        config_name = config_dir.name
        data[config_name] = []
        
        for json_file in sorted(config_dir.glob("*.json")):
            isl, osl = extract_isl_osl(json_file.name)
            if isl is None:
                continue
            
            with open(json_file, 'r') as f:
                result = json.load(f)
            
            # Set reference values from the first valid result
            if ref_num_requests is None:
                ref_num_requests = result['num_prompts']
                ref_concurrency = result['max_concurrency']
            
            # Verify benchmark parameters match reference
            assert result['num_prompts'] == ref_num_requests, \
                f"Expected {ref_num_requests} prompts, got {result['num_prompts']}"
            assert result['max_concurrency'] == ref_concurrency, \
                f"Expected concurrency {ref_concurrency}, got {result['max_concurrency']}"
            
            entry = dict(result)  # Copy result to avoid modifying original
            entry.update({
                'isl': isl,
                'osl': osl,
                'isl_osl_label': f"{isl}-{osl}",
            })
            data[config_name].append(entry)
        
        # Sort by ISL then OSL
        data[config_name].sort(key=lambda x: (x['isl'], x['osl']))
    
    # Initialize color mapping
    get_config_colors(list(data.keys()))
    
    return data, ref_num_requests, ref_concurrency


def format_config_display_name(name: str) -> str:
    """Format configuration name for display."""
    return name.replace('_', ' ').replace('-', ' ').title()


def plot_metric(data, metric_key, metric_info, output_dir, num_requests, concurrency):
    """Create a line chart for a specific metric."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Get all unique ISL-OSL combinations (x-axis labels)
    all_labels = set()
    for config_data in data.values():
        for item in config_data:
            all_labels.add(item['isl_osl_label'])
    all_labels = sorted(all_labels, key=lambda x: tuple(map(int, str(x).split('-'))))
    
    # Markers for different configurations
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'd']
    
    # Plot lines for each configuration
    for idx, (config_name, config_data) in enumerate(sorted(data.items())):
        # Create a lookup for this config
        data_dict = {item['isl_osl_label']: item for item in config_data}
        
        # Get values for each label
        values = [data_dict.get(label, {}).get(metric_key, None) for label in all_labels]
        
        # Format config name for display
        display_name = format_config_display_name(config_name)
        
        # Get consistent color
        color = _config_colors.get(config_name, plt.cm.tab10(idx % 10))
        
        # Plot line with markers
        ax.plot(
            range(len(all_labels)),
            values,
            marker=markers[idx % len(markers)],
            linewidth=2.5,
            markersize=10,
            label=display_name,
            color=color,
            markerfacecolor='white',
            markeredgewidth=2,
            markeredgecolor=color
        )
    
    # Configure the chart
    ax.set_xlabel('ISL-OSL (Input/Output Sequence Length)', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_info['ylabel'], fontsize=12, fontweight='bold')
    # Add better indicator to title
    better_hint = f"({'Lower' if metric_info['better'] == 'lower' else 'Higher'} is better)"
    ax.set_title(
        f"{metric_info['title']}\n"
        f"({num_requests} requests, concurrency={concurrency}) - {better_hint}",
        fontsize=14, fontweight='bold', pad=20
    )
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    ax.legend(
        loc='best',
        fontsize=10,
        framealpha=0.9,
        title='Configuration',
        title_fontsize=11
    )
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add subtle background color
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / f"{metric_key}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def create_comparison_table(data, output_dir, num_requests, concurrency, custom_title=None):
    """Create a summary comparison table."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (metric_key, metric_info) in enumerate(METRICS.items()):
        ax = axes[idx]
        
        # Get all unique ISL-OSL combinations
        all_labels = set()
        for config_data in data.values():
            for item in config_data:
                all_labels.add(item['isl_osl_label'])
        all_labels = sorted(all_labels, key=lambda x: tuple(map(int, x.split('-'))))
        
        for cfg_idx, (config_name, config_data) in enumerate(sorted(data.items())):
            data_dict = {item['isl_osl_label']: item for item in config_data}
            values = [data_dict.get(label, {}).get(metric_key, None) for label in all_labels]
            
            display_name = format_config_display_name(config_name)
            
            # Get consistent color
            color = _config_colors.get(config_name, plt.cm.tab10(cfg_idx % 10))
            
            ax.plot(
                all_labels,
                values,
                marker='o',
                linewidth=2,
                markersize=8,
                label=display_name,
                color=color
            )
        
        ax.set_xlabel('ISL-OSL', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_info['ylabel'], fontsize=11, fontweight='bold')
        better_hint = f"({'Lower' if metric_info['better'] == 'lower' else 'Higher'} is better)"
        ax.set_title(f"{metric_info['title']}\n({better_hint})", fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=45, ha='right')
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
    
    # Use custom title if provided, otherwise use default
    if custom_title:
        title_text = f"{custom_title}\n({num_requests} requests, concurrency={concurrency})"
    else:
        title_text = f"Benchmark Comparison Overview\n({num_requests} requests, concurrency={concurrency})"
    
    fig.suptitle(title_text, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = output_dir / "comparison_overview.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# Percentage Improvement Bar Charts
# =============================================================================

def calculate_percentage_improvement(baseline_val: float, config_val: float, better: str) -> float:
    """
    Calculate percentage improvement.
    For 'lower' metrics: positive = improvement (baseline is higher)
    For 'higher' metrics: positive = improvement (config is higher)
    """
    if baseline_val is None or config_val is None or baseline_val == 0:
        return 0.0
    
    if better == "lower":
        # Lower is better: improvement = (baseline - config) / baseline * 100
        return ((baseline_val - config_val) / baseline_val) * 100
    else:
        # Higher is better: improvement = (config - baseline) / baseline * 100
        return ((config_val - baseline_val) / baseline_val) * 100


def calculate_improvements(data: Dict) -> Tuple[Dict, List[str], List[str]]:
    """
    Calculate percentage improvements for all metrics.
    Returns: ({metric_key: {config_name: [improvements per ISL-OSL]}}, all_labels, other_configs)
    """
    if DEFAULT_BASELINE not in data:
        print(f"Warning: Baseline configuration '{DEFAULT_BASELINE}' not found!")
        return {}, [], []
    
    baseline_data = data[DEFAULT_BASELINE]
    baseline_lookup = {}
    
    # Create baseline lookup for all metrics
    for item in baseline_data:
        label = item['isl_osl_label']
        baseline_lookup[label] = {k: item.get(k) for k in METRICS.keys()}
    
    all_labels = sorted(baseline_lookup.keys(), key=lambda x: tuple(map(int, x.split('-'))))
    other_configs = [c for c in sorted(data.keys()) if c != DEFAULT_BASELINE]
    
    if not other_configs:
        print("Warning: No other configurations found to compare against baseline")
        return {}, all_labels, []
    
    improvements = {}
    
    for metric_key, metric_info in METRICS.items():
        improvements[metric_key] = {}
        
        for config_name in other_configs:
            config_data = data[config_name]
            config_lookup = {item['isl_osl_label']: item.get(metric_key) for item in config_data}
            
            imp_values = []
            for label in all_labels:
                baseline_val = baseline_lookup[label].get(metric_key)
                config_val = config_lookup.get(label)
                
                if baseline_val is not None and config_val is not None:
                    imp = calculate_percentage_improvement(baseline_val, config_val, metric_info['better'])
                    imp_values.append(imp)
                else:
                    imp_values.append(0.0)
            
            improvements[metric_key][config_name] = imp_values
    
    return improvements, all_labels, other_configs


def plot_improvement_bar_chart(improvements: Dict, all_labels: List[str], 
                               other_configs: List[str], metric_key: str, 
                               metric_info: dict, output_dir: Path):
    """Create a grouped bar chart showing percentage improvements vs baseline."""
    
    metric_improvements = improvements.get(metric_key, {})
    if not metric_improvements:
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    n_labels = len(all_labels)
    n_configs = len(other_configs)
    
    # Set up bar positions
    x = np.arange(n_labels)
    width = 0.8 / n_configs  # Width of each bar
    
    # Plot bars for each configuration
    for idx, config_name in enumerate(other_configs):
        values = metric_improvements.get(config_name, [0] * n_labels)
        offset = width * (idx - (n_configs - 1) / 2)
        
        # Get consistent color (same as line charts)
        color = _config_colors.get(config_name, plt.cm.tab10(idx % 10))
        
        bars = ax.bar(x + offset, values, width, 
                     label=format_config_display_name(config_name),
                     color=color,
                     edgecolor='black',
                     linewidth=0.5)
        
        # Add value labels on all bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -10),
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=7, rotation=0)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Better direction hint
    better_text = "↑ Positive = Better" if metric_info['better'] == 'higher' else "↑ Positive = Better (Lower is better for this metric)"
    
    # Configure chart
    ax.set_xlabel('ISL-OSL (Input/Output Sequence Length)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'{metric_info["display_name"]} - Percentage Improvement vs Baseline\n'
        f'{better_text}',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    ax.legend(
        loc='best',
        fontsize=9,
        framealpha=0.9,
        title='Configuration',
        title_fontsize=10
    )
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    
    # Color the background
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    
    # Add subtle coloring for positive/negative regions
    y_min, y_max = ax.get_ylim()
    if y_max > 0:
        ax.axhspan(0, y_max, facecolor='green', alpha=0.05)
    if y_min < 0:
        ax.axhspan(y_min, 0, facecolor='red', alpha=0.05)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f"{metric_key}_improvement.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def print_improvement_summary(improvements: Dict, all_labels: List[str], 
                              other_configs: List[str]):
    """Print summary table of average improvements to CLI."""
    
    print("\n" + "=" * 120)
    print("AVERAGE PERCENTAGE IMPROVEMENT SUMMARY (vs Baseline: {})".format(
        format_config_display_name(DEFAULT_BASELINE)))
    print("=" * 120)
    print(f"{'Configuration':<55} ", end="")
    for metric_info in METRICS.values():
        print(f"{metric_info['title']:<16}", end="")
    print()
    print("-" * 120)
    
    for config_name in other_configs:
        display_name = format_config_display_name(config_name)
        print(f"{display_name:<55} ", end="")
        
        for metric_key in METRICS.keys():
            values = improvements.get(metric_key, {}).get(config_name, [])
            if values:
                avg_imp = sum(values) / len(values)
                print(f"{avg_imp:>+7.1f}%        ", end="")
            else:
                print(f"{'N/A':>16}", end="")
        print()
    
    print("=" * 120)
    print("Note: Positive % = Improvement, Negative % = Regression vs Baseline")
    print()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Plot benchmark results from a directory containing benchmark result subdirectories.'
    )
    parser.add_argument(
        '--results-dir', '-i',
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f'Directory containing benchmark result subdirectories (default: {DEFAULT_RESULTS_DIR})'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Directory to save output plots (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--baseline', '-b',
        type=str,
        default=DEFAULT_BASELINE,
        help=f'Baseline configuration name for percentage improvement calculations (default: {DEFAULT_BASELINE})'
    )
    parser.add_argument(
        '--overview-title', '-t',
        type=str,
        default=None,
        help='Custom title for the comparison overview chart (default: "Benchmark Comparison Overview")'
    )
    return parser.parse_args()


def main():
    """Main function to generate all plots."""
    global _config_colors, RESULTS_DIR, OUTPUT_DIR, DEFAULT_BASELINE
    
    args = parse_args()
    
    # Update global paths from command line args
    RESULTS_DIR = args.results_dir
    OUTPUT_DIR = args.output_dir
    DEFAULT_BASELINE = args.baseline
    
    _config_colors = {}  # Reset color mapping
    
    print(f"Results directory: {RESULTS_DIR.absolute()}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"Baseline config: {DEFAULT_BASELINE}")
    print()
    
    if not RESULTS_DIR.exists():
        print(f"Error: Results directory '{RESULTS_DIR}' does not exist!")
        return
    
    print("Loading benchmark data...")
    data, NUM_REQUESTS, CONCURRENCY = load_benchmark_data()
    
    print(f"\nFound {len(data)} configurations:")
    for config_name, config_data in sorted(data.items()):
        print(f"  - {config_name}: {len(config_data)} data points")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Generate individual metric charts (line charts)
    print("\n" + "=" * 60)
    print("Generating individual metric line charts...")
    print("=" * 60)
    for metric_key, metric_info in METRICS.items():
        plot_metric(data, metric_key, metric_info, OUTPUT_DIR, NUM_REQUESTS, CONCURRENCY)
    
    # Generate comparison overview
    print("\nGenerating comparison overview...")
    create_comparison_table(data, OUTPUT_DIR, NUM_REQUESTS, CONCURRENCY, custom_title=args.overview_title)
    
    # Generate percentage improvement bar charts
    print("\n" + "=" * 60)
    print("Generating percentage improvement bar charts...")
    print("=" * 60)
    improvements, all_labels, other_configs = calculate_improvements(data)
    
    if improvements and other_configs:
        # Print summary table
        print_improvement_summary(improvements, all_labels, other_configs)
        
        # Generate bar charts
        for metric_key, metric_info in METRICS.items():
            plot_improvement_bar_chart(improvements, all_labels, other_configs, 
                                       metric_key, metric_info, OUTPUT_DIR)
    else:
        print("Skipping improvement charts (baseline not found or no other configs)")
    
    print(f"\nAll plots saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
