#!/usr/bin/env python3
"""
GCG Experiment Log Parser and Visualization Tool

This script parses GCG optimization experiment logs and generates publication-quality graphs
showing the optimization progress over time.

Usage:
    python gcg_log_parser.py <log_file_path>
"""

import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ============================================================================
# CONFIGURATION VARIABLES - Adjust these to change graph layout
# ============================================================================

# Plot order configuration (0 = top, 1 = middle, 2 = bottom)
PLOT_ORDER = {
    'pg_loss': 0,           # Prompt Guard 2 Loss position
    'text_gen_loss': 1,     # Text Generation Loss position  
    'pg_score': 2           # Prompt Guard 2 Benign position (bottom)
}

# Background colors for objectives
OBJECTIVE_COLORS = {
    'text_gen': "#fbc1c1",  # Grey for text generation mode
    'pg': "#c1dfff"         # Light blue for PG mode
}

# Font sizes for publication
FONT_CONFIG = {
    'base_size': 20,        # Base font size (increased from 16)
    'title_size': 22,       # Title size (increased from 18)
    'label_size': 20,       # Axis label size (increased from 16)
    'tick_size': 18,        # Tick label size (increased from 14)
    'legend_size': 18       # Legend size (increased from 14)
}

# Figure dimensions
FIGURE_WIDTH = 12
FIGURE_HEIGHT = 14

# Maximum number of steps to display (None = display all)
MAX_STEPS = None  # Set to a number (e.g., 100) to limit the x-axis

# Whether to show legends on the loss graphs (PG Loss and Text Gen Loss)
SHOW_LOSS_LEGENDS = False  # Set to False to hide legends on loss graphs

# Note: The "Super Suffix Generated" red line appears when:
# - Eval Score = 0 AND PG Score > 0.9


def parse_configuration(lines: List[str]) -> Dict[str, str]:
    """
    Parse the configuration section from the log file.
    
    Args:
        lines: List of log file lines
        
    Returns:
        Dictionary containing configuration parameters
    """
    config = {}
    
    # Regular expressions for configuration extraction
    patterns = {
        'timestamp': r'Timestamp:\s*(.+)',
        'log_file': r'Log file:\s*(.+)',
        'model': r'(?<!PG )Model:\s*(.+)',
        'pg_model': r'PG Model:\s*(.+)',
        'prompt_file': r'Prompt file:\s*(.+)',
        'device': r'Device:\s*(.+)',
        'data_type': r'Data type:\s*(.+)',
        'probe_sampling': r'Probe sampling:\s*(.+)',
        'num_steps': r'Number of optimization steps:\s*(.+)',
        'target_direction': r'Target direction path:\s*(.+)',
        'target_layer': r'Target layer:\s*(.+)',
        'batch_size': r'Batch size:\s*(.+)',
        'search_width': r'Search width:\s*(.+)',
        'tokens_per_step': r'Tokens to replace per step:\s*(.+)',
        'top_k': r'Top-k:\s*(.+)',
        'buffer_size': r'Buffer size:\s*(.+)',
        'direction_weight': r'Direction weight:\s*(.+)',
        'use_last_token': r'Use last token:\s*(.+)',
        'maximize_similarity': r'Maximize similarity:\s*(.+)',
        'layer_for_direction': r'Layer for direction:\s*(.+)'
    }
    
    # Search through first 100 lines for configuration
    for line in lines[:min(100, len(lines))]:
        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                config[key] = match.group(1)
    
    # Print extracted configuration
    print("\n" + "="*80)
    print("EXTRACTED CONFIGURATION:")
    print("="*80)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*80 + "\n")
    
    return config


def parse_metrics(lines: List[str]) -> Tuple[List[int], List[float], List[float], 
                                              List[float], List[float], List[str]]:
    """
    Parse metrics and objectives from the log file.
    
    Args:
        lines: List of log file lines
        
    Returns:
        Tuple of (steps, eval_scores, pg_scores, text_gen_losses, pg_losses, objectives)
    """
    steps = []
    eval_scores = []
    pg_scores = []
    text_gen_losses = []
    pg_losses = []
    objectives = []
    
    current_step = 0
    current_objective = 'text_gen'  # Default objective
    
    # Regular expressions for data extraction
    step_pattern = r'Step (\d+): Generating candidates with \'(\w+)\' objective'
    metrics_pattern1 = r'Eval Score:\s*([\d.]+)\s*\|\s*PG Score:\s*([\d.]+)'
    metrics_pattern2 = r'text_gen Loss:\s*([\d.]+)\s*\|\s*PG Loss:\s*([\d.]+)'
    
    for i, line in enumerate(lines):
        # Check for step and objective
        step_match = re.search(step_pattern, line)
        if step_match:
            current_step = int(step_match.group(1))
            current_objective = step_match.group(2)
        
        # Check for metrics line 1 (Eval Score and PG Score)
        metrics_match1 = re.search(metrics_pattern1, line)
        if metrics_match1:
            eval_score = float(metrics_match1.group(1))
            pg_score = float(metrics_match1.group(2))
            
            # Look for the next line with losses
            if i + 1 < len(lines):
                metrics_match2 = re.search(metrics_pattern2, lines[i + 1])
                if metrics_match2:
                    text_gen_loss = float(metrics_match2.group(1))
                    pg_loss = float(metrics_match2.group(2))
                    
                    # Store all metrics
                    steps.append(current_step)
                    eval_scores.append(eval_score)
                    pg_scores.append(pg_score)
                    text_gen_losses.append(text_gen_loss)
                    pg_losses.append(pg_loss)
                    objectives.append(current_objective)
    
    print(f"Parsed {len(steps)} data points from log file")
    if steps:
        print(f"  Steps range: {min(steps)} to {max(steps)}")
        print(f"  Eval Score range: {min(eval_scores):.4f} to {max(eval_scores):.4f}")
        print(f"  PG Score range: {min(pg_scores):.4f} to {max(pg_scores):.4f}")
    
    return steps, eval_scores, pg_scores, text_gen_losses, pg_losses, objectives


def parse_log(log_file_path: str) -> Tuple[Dict, Tuple]:
    """
    Main function to parse the log file.
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        Tuple of (configuration, metrics)
    """
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    print(f"Read {len(lines)} lines from {log_file_path}")
    
    # Parse configuration
    config = parse_configuration(lines)
    
    # Parse metrics
    metrics = parse_metrics(lines)
    
    return config, metrics


def create_visualization(config: Dict, metrics: Tuple, output_path: str):
    """
    Create the visualization with three subplots.
    
    Args:
        config: Configuration dictionary
        metrics: Tuple of metrics data
        output_path: Path to save the figure
    """
    steps, eval_scores, pg_scores, text_gen_losses, pg_losses, objectives = metrics
    
    if not steps:
        print("No data points found to plot!")
        return
    
    # Convert to numpy arrays for easier manipulation
    steps = np.array(steps)
    eval_scores = np.array(eval_scores)
    pg_scores = np.array(pg_scores)
    text_gen_losses = np.array(text_gen_losses)
    pg_losses = np.array(pg_losses)
    
    # Apply MAX_STEPS filtering if configured
    if MAX_STEPS is not None:
        mask = steps <= MAX_STEPS
        steps = steps[mask]
        eval_scores = eval_scores[mask]
        pg_scores = pg_scores[mask]
        text_gen_losses = text_gen_losses[mask]
        pg_losses = pg_losses[mask]
        objectives = [obj for i, obj in enumerate(objectives) if i < len(steps)]
        print(f"Limiting display to first {MAX_STEPS} steps")
    
    # Set up the figure with larger font sizes for publication
    plt.rcParams.update({
        'font.size': FONT_CONFIG['base_size'],
        'axes.titlesize': FONT_CONFIG['title_size'],
        'axes.labelsize': FONT_CONFIG['label_size'],
        'xtick.labelsize': FONT_CONFIG['tick_size'],
        'ytick.labelsize': FONT_CONFIG['tick_size'],
        'legend.fontsize': FONT_CONFIG['legend_size'],
        'figure.titlesize': FONT_CONFIG['title_size'] + 2
    })
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), sharex=True)
    
    # Create a mapping of plot types to their axes based on PLOT_ORDER
    plot_axes = {}
    for plot_type, position in PLOT_ORDER.items():
        plot_axes[plot_type] = axes[position]
    
    # Add background shading based on objective
    for ax in axes:
        for i in range(len(steps)):
            if i < len(steps) - 1:
                width = steps[i + 1] - steps[i]
            else:
                width = 1  # Default width for last step
            
            # Use configured colors based on objective
            color = OBJECTIVE_COLORS[objectives[i]]
            ax.axvspan(steps[i], steps[i] + width, alpha=0.5, color=color, zorder=0)
    
    # Check for vertical red line position
    red_line_step = None
    # Find indices where eval_score == 0 AND pg_score > 0.9
    condition_met = (eval_scores == 0) & (pg_scores > 0.9)
    condition_indices = np.where(condition_met)[0]
    if len(condition_indices) > 0:
        red_line_step = steps[condition_indices[0]]
    
    # Plot Prompt Guard 2 Loss
    pg_loss_ax = plot_axes['pg_loss']
    pg_loss_ax.plot(steps, pg_losses, 'b-', linewidth=2, label='Prompt Guard 2 Loss')
    pg_loss_ax.set_ylabel('Prompt Guard 2 Loss', fontsize=FONT_CONFIG['label_size'], fontweight='bold')
    pg_loss_ax.grid(True, alpha=0.3)
    pg_loss_ax.set_ylim(bottom=0)
    
    # Add red line to PG Loss plot
    if red_line_step is not None:
        pg_loss_ax.axvline(x=red_line_step, color='red', linestyle='-', linewidth=2, 
                          alpha=0.7, label='Super Suffix Generated')
    
    # Conditionally show legend based on SHOW_LOSS_LEGENDS
    if SHOW_LOSS_LEGENDS:
        pg_loss_ax.legend(loc='upper center')
    
    # Plot Text Generation Loss
    text_gen_ax = plot_axes['text_gen_loss']
    text_gen_ax.plot(steps, text_gen_losses, 'orange', linewidth=2, label='Text Generation Loss')
    text_gen_ax.set_ylabel('Text Generation Loss', fontsize=FONT_CONFIG['label_size'], fontweight='bold')
    text_gen_ax.grid(True, alpha=0.3)
    text_gen_ax.set_ylim(bottom=0)
    
    # Add red line to Text Gen Loss plot
    if red_line_step is not None:
        text_gen_ax.axvline(x=red_line_step, color='red', linestyle='-', linewidth=2, 
                           alpha=0.7, label='Super Suffix Generated')
    
    # Conditionally show legend based on SHOW_LOSS_LEGENDS
    if SHOW_LOSS_LEGENDS:
        text_gen_ax.legend(loc='upper center')
    
    # Plot Prompt Guard 2 Benign
    pg_score_ax = plot_axes['pg_score']
    pg_score_ax.plot(steps, pg_scores, 'g-', linewidth=2, label='Prompt Guard 2 Benign')
    pg_score_ax.set_ylabel('Prompt Guard 2 Benign', fontsize=FONT_CONFIG['label_size'], fontweight='bold')
    pg_score_ax.grid(True, alpha=0.3)
    pg_score_ax.set_ylim(0, 1.1)
    
    # Add red line to PG Score plot
    if red_line_step is not None:
        pg_score_ax.axvline(x=red_line_step, color='red', linestyle='-', linewidth=2, 
                           alpha=0.7, label='Super Suffix Generated')
        print(f"Added vertical line at step {red_line_step} (first Eval Score = 0 and PG Score > 0.9)")
    
    # Set x-axis label on the bottom plot
    axes[2].set_xlabel('Step', fontsize=FONT_CONFIG['label_size'], fontweight='bold')
    
    # Create combined legend for PG Score plot with objectives and red line
    # This legend is always shown regardless of SHOW_LOSS_LEGENDS
    pg_patch = patches.Patch(color=OBJECTIVE_COLORS['pg'], label='PG Objective')
    text_gen_patch = patches.Patch(color=OBJECTIVE_COLORS['text_gen'], label='Text Gen Objective')
    
    # Collect handles for legend
    handles = [pg_score_ax.get_lines()[0], pg_patch, text_gen_patch]
    if red_line_step is not None:
        # Add the red line handle if it exists
        handles.append(pg_score_ax.get_lines()[1])
    
    pg_score_ax.legend(handles=handles, loc='lower center', ncol=2)
    
    # Set x-axis limits
    max_step_display = MAX_STEPS if MAX_STEPS is not None else max(steps) + 5
    for ax in axes:
        ax.set_xlim(0, max_step_display)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure as PDF
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"\nGraph saved to: {output_path}")
    
    plt.show()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Parse GCG experiment logs and generate visualization'
    )
    parser.add_argument(
        'log_file',
        type=str,
        help='Path to the log file to parse'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output path for the graph (default: <log_name>_graph.pdf in current directory)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file '{args.log_file}' does not exist!")
        sys.exit(1)
    
    # Determine output path - save to current working directory
    if args.output:
        output_path = args.output
    else:
        # Use just the basename and save to current directory
        output_name = log_path.stem + '_graph.pdf'
        output_path = output_name
    
    print(f"Processing log file: {args.log_file}")
    
    # Parse the log file
    config, metrics = parse_log(args.log_file)
    
    # Create visualization
    if metrics[0]:  # Check if we have data
        create_visualization(config, metrics, output_path)
    else:
        print("No metrics data found in log file!")
        sys.exit(1)


if __name__ == "__main__":
    main()