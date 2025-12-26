import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Regex Patterns for Log Parsing ---
RE_PRIMARY_PG = re.compile(r'\s*PG score: ([\d\.]+)')
RE_PRIMARY_EVAL = re.compile(r'\s*Eval score: ([\d\.]+) \(1=safe\)')
RE_SUPER = re.compile(r'\s*Eval Score: 0\.0+ \| PG Score: ([\d\.]+)')
COMPLETION_MARKER = "EXPERIMENT COMPLETED SUCCESSFULLY"



def parse_log_files(log_dir):
    """
    Parse log files to extract Primary and Super suffix PG scores.
    Returns two lists: primary_pg_scores and super_pg_scores
    """
    prompt_model_cache = []
    RE_MODEL = re.compile(r'Model: (.*)$', re.MULTILINE)
    RE_PROMPT = re.compile(r'Input prompt: (.*)$', re.MULTILINE)

        

    primary_pg_scores = []
    super_pg_scores = []
    
    if not os.path.isdir(log_dir):
        logger.warning(f"Log directory not found: {log_dir}")
        return primary_pg_scores, super_pg_scores
    
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
    if not log_files:
        logger.warning(f"No .log files found in '{log_dir}'")
        return primary_pg_scores, super_pg_scores
    
    logger.info(f"Parsing {len(log_files)} log files from '{log_dir}'...")
    
    for filename in tqdm(log_files, desc="Parsing log files"):
        log_path = os.path.join(log_dir, filename)
        
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Skip incomplete logs
        if COMPLETION_MARKER not in content:
            continue

        # Extract model and prompt information
        model_match = RE_MODEL.search(content)
        prompt_match = RE_PROMPT.search(content)

        if model_match and prompt_match:
            model = model_match.group(1).strip()
            prompt = prompt_match.group(1).strip()
            model_prompt_pair = (model, prompt)
            
            # Check if this model-prompt combination is already in cache
            if model_prompt_pair in prompt_model_cache:
                continue  # Skip this log file
            else:
                prompt_model_cache.append(model_prompt_pair)
        
        # --- Primary Suffix Scores (Line-by-line parsing) ---
        lines = content.splitlines()
        last_pg_score = None
        
        for line in lines:
            pg_match = RE_PRIMARY_PG.search(line)
            eval_match = RE_PRIMARY_EVAL.search(line)
            
            if pg_match:
                last_pg_score = float(pg_match.group(1))
            elif eval_match and last_pg_score is not None:
                eval_score = float(eval_match.group(1))
                if eval_score == 0.0:  # Success case
                    primary_pg_scores.append(last_pg_score)
                last_pg_score = None
        
        # --- Super Suffix Scores ---
        super_matches = RE_SUPER.findall(content)
        super_pg_scores.extend([float(pg) for pg in super_matches if float(pg) > 0.85])
    
    logger.info(f"Extracted {len(primary_pg_scores)} primary PG scores and {len(super_pg_scores)} super PG scores")
    
    return primary_pg_scores, super_pg_scores


def evaluate_prompts(pg_model, pg_tokenizer, prompts, device, label):
    """Evaluate a list of prompts and return their benign scores."""
    scores = []
    pg_model.eval()
    
    with torch.no_grad():
        for idx, prompt in enumerate(prompts, 1):
            # Format prompt as a chat message
            messages = [{"role": "user", "content": prompt}]
            
            # Tokenize
            inputs = pg_tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                padding=True
            ).to(device)

            inputs = pg_tokenizer(prompt, return_tensors="pt").to(device)
            
            # Get model predictions
            outputs = pg_model(**inputs)
            logits = outputs.logits
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Binary classification: [benign, jailbreak]
            # Index 0 is benign class - higher score means more benign
            score = probs[0][0].item()  # Score for benign class
            scores.append(score)
            
            logger.info(f"{label} - Prompt {idx}/{len(prompts)}: benign score = {score:.4f}")
    
    return scores


def print_example_table(prompts, scores, label, num_examples=4):
    """Print a table of example prompts with their scores."""
    print(f"\n{label} EXAMPLES:")
    print("=" * 100)
    print(f"{'#':<4} {'Benign Score':<15} {'Prompt':<80}")
    print("-" * 100)
    
    # Show first num_examples
    for i in range(min(num_examples, len(prompts))):
        prompt_text = prompts[i]
        # Truncate long prompts for display
        if len(prompt_text) > 75:
            prompt_text = prompt_text[:72] + "..."
        print(f"{i+1:<4} {scores[i]:<15.4f} {prompt_text:<80}")
    
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Evaluate prompts using Prompt Guard")
    parser.add_argument("--pg_model", type=str, default="meta-llama/Llama-Prompt-Guard-2-86M")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--good_file", type=str, default="../prompts/good_code.txt")
    parser.add_argument("--bad_file", type=str, default="../prompts/bad_code.txt")
    parser.add_argument("--log_dir", type=str, default="../logs", help="Directory containing log files")
    parser.add_argument("--output_plot", type=str, default="pg_scores_histogram.png")
    args = parser.parse_args()

    logger.info(f"Loading Prompt Guard model: {args.pg_model}")
    
    # Load model and tokenizer
    pg_model = AutoModelForSequenceClassification.from_pretrained(
        args.pg_model,
        torch_dtype=getattr(torch, args.dtype)
    ).to(args.device)
    
    pg_tokenizer = AutoTokenizer.from_pretrained(args.pg_model)
    
    # Define the specific Llama-2-style guard template
    guard_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
        "{% endif %}"
        "{% endfor %}"
    )
    
    # Check and set chat template for pg_tokenizer
    if pg_tokenizer.chat_template is None:
        logger.info(f"No chat template found for pg_tokenizer. Setting default guard template.")
        pg_tokenizer.chat_template = guard_template
    else:
        logger.info(f"Using existing chat template for pg_tokenizer.")
    
    # Read prompts from both files
    logger.info(f"Reading good prompts from: {args.good_file}")
    with open(args.good_file, 'r') as f:
        good_prompts = [line.strip() for line in f if line.strip()]
    logger.info(f"Found {len(good_prompts)} good prompts")
    
    logger.info(f"Reading bad prompts from: {args.bad_file}")
    with open(args.bad_file, 'r') as f:
        bad_prompts = [line.strip() for line in f if line.strip()]
    logger.info(f"Found {len(bad_prompts)} bad prompts")
    
    # Evaluate both sets of prompts
    good_scores = evaluate_prompts(pg_model, pg_tokenizer, good_prompts, args.device, "GOOD")
    bad_scores = evaluate_prompts(pg_model, pg_tokenizer, bad_prompts, args.device, "BAD")
    
    # Parse log files for Primary and Super suffix scores
    primary_scores, super_scores = parse_log_files(args.log_dir)
    
    # Print example tables
    print_example_table(good_prompts, good_scores, "GOOD CODE", num_examples=4)
    print_example_table(bad_prompts, bad_scores, "BAD CODE", num_examples=4)
    
    # Calculate statistics
    good_avg = sum(good_scores) / len(good_scores) if good_scores else 0
    bad_avg = sum(bad_scores) / len(bad_scores) if bad_scores else 0
    primary_avg = sum(primary_scores) / len(primary_scores) if primary_scores else None
    super_avg = sum(super_scores) / len(super_scores) if super_scores else None
    
    # Print results
    print("\n" + "="*60)
    print("GOOD CODE PROMPTS:")
    print(f"  Average Benign Score: {good_avg:.4f}")
    print(f"  Total prompts: {len(good_scores)}")
    print(f"  Min score: {min(good_scores):.4f}")
    print(f"  Max score: {max(good_scores):.4f}")
    print()
    print("BAD CODE PROMPTS:")
    print(f"  Average Benign Score: {bad_avg:.4f}")
    print(f"  Total prompts: {len(bad_scores)}")
    print(f"  Min score: {min(bad_scores):.4f}")
    print(f"  Max score: {max(bad_scores):.4f}")
    
    if primary_scores:
        print()
        print("PRIMARY SUFFIX (from logs):")
        print(f"  Average PG Score: {primary_avg:.4f}")
        print(f"  Total scores: {len(primary_scores)}")
        print(f"  Min score: {min(primary_scores):.4f}")
        print(f"  Max score: {max(primary_scores):.4f}")
    
    if super_scores:
        print()
        print("SUPER SUFFIX (from logs):")
        print(f"  Average PG Score: {super_avg:.4f}")
        print(f"  Total scores: {len(super_scores)}")
        print(f"  Min score: {min(super_scores):.4f}")
        print(f"  Max score: {max(super_scores):.4f}")
    
    print("="*60)
    
    # Create histogram with side-by-side bars (normalized by percentage)
    plt.figure(figsize=(16, 8))
    
    # Define bins
    bins = np.linspace(0, 1, 10)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    
    # Determine how many groups we have
    groups = ['Benign Code (No Suffix)', 'Malicious Code (No Suffix)']
    colors = ['green', 'red']
    all_scores = [good_scores, bad_scores]
    avg_scores = [good_avg, bad_avg]
    avg_colors = ['darkgreen', 'darkred']
    
    if primary_scores:
        groups.append('Primary Suffix')
        colors.append('blue')
        all_scores.append(primary_scores)
        avg_scores.append(primary_avg)
        avg_colors.append('darkblue')
    
    if super_scores:
        groups.append('Super Suffix')
        colors.append('purple')
        all_scores.append(super_scores)
        avg_scores.append(super_avg)
        avg_colors.append('darkviolet')
    
    num_groups = len(groups)
    bar_width = bin_width / (num_groups + 1)  # Add spacing between groups
    
    # Calculate histograms for each group (normalized to percentages)
    for i, (scores, color, label) in enumerate(zip(all_scores, colors, groups)):
        counts, _ = np.histogram(scores, bins=bins)
        # Normalize to percentage
        total_count = len(scores)
        percentages = (counts / total_count) * 100 if total_count > 0 else counts
        
        # Offset each group's bars
        offset = (i - num_groups/2 + 0.5) * bar_width
        positions = bin_centers + offset
        plt.bar(positions, percentages, width=bar_width, alpha=0.7, color=color, 
                label=label, edgecolor='black', linewidth=0.5)
    
    # Add vertical lines for averages
    for avg, color, label in zip(avg_scores, avg_colors, groups):
        if avg is not None:
            plt.axvline(avg, color=color, linestyle='--', linewidth=2, 
                       label=f'{label} Avg: {avg:.3f}', alpha=0.8)
    
    plt.xlabel('Benign Score (higher = more benign)', fontsize=13)
    plt.ylabel('Percentage (%)', fontsize=13)
    plt.title('Prompt Guard Benign Scores Distribution (Normalized)', fontsize=15, fontweight='bold')
    
    # Position legend at approximately 1/4 of the graph horizontally, centered vertically
    plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(0.25, 0.5), 
               framealpha=0.9, edgecolor='black')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.xlim(-0.05, 1.05)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(args.output_plot, dpi=300, bbox_inches='tight')
    logger.info(f"Histogram saved to: {args.output_plot}")
    print(f"\nHistogram saved to: {args.output_plot}")
    
    plt.close()


if __name__ == "__main__":
    main()