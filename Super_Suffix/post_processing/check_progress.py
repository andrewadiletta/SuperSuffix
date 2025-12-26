#!/usr/bin/env python3

import os
import re
import sys
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
LOG_DIR = '../logs'
PROMPT_FILE = 'prompts.txt'

# --- Regex Patterns ---
RE_MODEL = re.compile(r'Model: (.*)$', re.MULTILINE)
RE_PROMPT = re.compile(r'Input prompt: (.*)$', re.MULTILINE)
# --- NEW ---
# Add the exact completion marker to check for
COMPLETION_MARKER = "EXPERIMENT COMPLETED SUCCESSFULLY"

# --- MODIFIED FUNCTION ---
def get_completed_run(log_path):
    """
    Parses a log file and returns a (model, prompt) tuple.
    Returns None if the file is unreadable, missing data,
    OR if the "EXPERIMENT COMPLETED SUCCESSFULLY" marker is not found.
    """
    try:
        with open(log_path, 'r') as f:
            content = f.read()
    except Exception as e:
        tqdm.write(f"Warning: Could not read {log_path}: {e}")
        return None

    # We now require all three things to be present.
    model_match = RE_MODEL.search(content)
    prompt_match = RE_PROMPT.search(content)
    # Check for the completion marker string
    complete_match = COMPLETION_MARKER in content 

    if model_match and prompt_match and complete_match:
        model = model_match.group(1).strip()
        prompt = prompt_match.group(1).strip()
        return (model, prompt)
    
    # If any part is missing (model, prompt, or completion marker),
    # we treat it as an incomplete run.
    return None

def main():
    """
    Main function to scan logs and display completeness plot.
    """
    
    # --- Step 1: Read the master prompt list ---
    if not os.path.isfile(PROMPT_FILE):
        print(f"Error: Prompt file not found at '{PROMPT_FILE}'")
        sys.exit(1)
        
    try:
        with open(PROMPT_FILE, 'r') as f:
            all_prompts = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"Error reading {PROMPT_FILE}: {e}")
        sys.exit(1)

    if not all_prompts:
        print(f"Error: {PROMPT_FILE} is empty. No prompts to check against.")
        sys.exit(1)
        
    num_prompts = len(all_prompts)
    print(f"Loaded {num_prompts} prompts from {PROMPT_FILE}.")
    
    # --- Step 2: Scan logs for completed (model, prompt) pairs ---
    if not os.path.isdir(LOG_DIR):
        print(f"Error: Log directory not found at '{LOG_DIR}'")
        return

    log_files = [f for f in os.listdir(LOG_DIR) if f.endswith('.log')]
    if not log_files:
        print(f"No .log files found in '{LOG_DIR}'")
        return

    # This dict will store { 'model_name': {'prompt1', 'prompt2', ...} }
    completed_data = defaultdict(set)

    print(f"Scanning {len(log_files)} logs for completeness...")
    total_completed = 0
    for filename in tqdm(log_files, desc="Scanning logs"):
        log_path = os.path.join(LOG_DIR, filename)
        result = get_completed_run(log_path) # This now checks for completion
        
        if result:
            total_completed += 1
            model, prompt = result
            completed_data[model].add(prompt)
    print(f"Found {total_completed} completed runs across {len(completed_data)} models.")

    # --- Step 3: Prepare data for matplotlib ---
    if not completed_data:
        print("No completed log data found.")
        return

    sorted_models = sorted(completed_data.keys())
    num_models = len(sorted_models)
    
    # Create a 2D numpy array (matrix) initialized to 0 (missing)
    # Shape will be (num_models, num_prompts)
    data_matrix = np.zeros((num_models, num_prompts))
    
    # Populate the matrix
    # 1.0 = completed, 0.0 = missing
    for i, model in enumerate(sorted_models):
        for j, prompt in enumerate(all_prompts):
            if prompt in completed_data[model]:
                data_matrix[i, j] = 1.0

    print("Generating plot...")

    # --- Step 4: Generate the plot ---
    
    # Adjust fig Bsize to be wider if there are many prompts
    fig_width = max(15, num_prompts * 0.15)
    fig_height = max(8, num_models * 0.5)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # 'imshow' displays the data as an image
    # cmap='Greys': 0.0 -> white, 1.0 -> black
    # aspect='auto': Allows cells to be non-square
    ax.imshow(data_matrix, cmap='Greys', aspect='auto')

    # --- Configure Ticks and Labels ---
    
    # Set Y-axis labels to be the model names
    ax.set_yticks(np.arange(num_models))
    ax.set_yticklabels(sorted_models, fontsize=10)
    
    # Set X-axis labels to be the prompt index
    # To avoid clutter, only label every 5th prompt
    xtick_step = 5 if num_prompts > 50 else 1
    ax.set_xticks(np.arange(0, num_prompts, xtick_step))
    ax.set_xticklabels(np.arange(0, num_prompts, xtick_step), fontsize=9)
    
    ax.set_xlabel("Prompt Index", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.set_title("Experiment Completeness by Model and Prompt", fontsize=16)

    # Add a grid to separate the cells
    ax.set_xticks(np.arange(-.5, num_prompts, 1), minor=True)
    ax.set_yticks(np.arange(-.5, num_models, 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)

    plt.tight_layout()  # Adjusts plot to prevent labels from overlapping
    plt.savefig('experiment_completeness.png', dpi=300)
    print("Plot saved to experiment_completeness.png")

if __name__ == "__main__":
    main()