#!/usr/bin/env python3

import os
import sys
import re
import subprocess
import argparse
import tempfile
import time
import signal
import fcntl

# --- Configuration ---
SCRIPT_TO_RUN = "main.py"
#PROMPTS_FILE = os.path.join("prompts", "bad_code.txt")
PROMPTS_FILE = os.path.join("prompts", "harmbench.txt")
REPEATS_PER_PROMPT = 1

# NOTE: The following lists must be in corresponding order.
# e.g., MODELS[0] uses DIRECTION_FILES[0], DETECTION_FILES[0], and LAYERS[0]
MODELS = [
        "google/gemma-2b-it", 
        #"meta-llama/Llama-3.2-3B-instruct",
        #"meta-llama/Llama-3.1-8B-instruct", 
        #"microsoft/Phi-3-mini-128k-instruct",
        #"lmsys/vicuna-7b-v1.5",
        #"ibm-granite/granite-4.0-h-350M"
]
DIRECTION_FILES = [
    "premade_code_directions/gemma_code_layer_16.pt", 
    #"premade_code_directions/llama3B_code_layer_16.pt",
    #"premade_code_directions/llama8B_code_layer_26.pt",
    #"premade_code_directions/phi_code_layer_16.pt", 
    #"premade_code_directions/vicuna_code_layer_26.pt",
    #"premade_code_directions/granite_code_layer_16.pt"
]
DETECTION_FILES = [
    "premade_refusal_directions/gemma_refusal_layer_16.pt",
    #"premade_refusal_directions/llama3B_refusal_layer_16.pt",
    #"premade_refusal_directions/llama8B_refusal_layer_26.pt",
    #"premade_refusal_directions/phi_refusal_layer_16.pt",
    #"premade_refusal_directions/vicuna_refusal_layer_26.pt",
    #"premade_refusal_directions/granite_refusal_layer_16.pt"
]
LAYERS = [
        16,
        #16,
        #26,
        #16,
        #26,
        #16
]

# --- Log Parsing Constants ---
RE_MODEL = re.compile(r'gcg_experiment:   Model: (.*)$', re.MULTILINE)
RE_PROMPT = re.compile(r'Input prompt: (.*)$', re.MULTILINE)
COMPLETION_MARKER = "EXPERIMENT COMPLETED SUCCESSFULLY"

# --- Working File Constants ---
WORKING_FILE = "working_jobs.txt"
WORKING_FILE_LOCK = "working_jobs.txt.lock" # Lock file to prevent race conditions
JOB_DELIMITER = " | "

# Global variable to track current job for cleanup on exit
current_job = None

def cleanup_working_job(signum=None, frame=None):
    """Signal handler to remove current job from working file on exit"""
    global current_job
    if current_job:
        remove_from_working_file(current_job)
        print(f"\nSignal received. Removed {current_job} from working file.")
    if signum:
        sys.exit(1)

def read_prompts():
    """Read all prompts from the prompts file"""
    with open(PROMPTS_FILE, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def parse_log_file(log_path):
    """Parse a log file to extract model, prompt, and check for completion"""

    model = None
    prompt = None
    completed = False
    
    try:
        with open(log_path, 'rb') as f:
            lines = f.readlines()

            # Read first 100 lines to find model and prompt
            first_lines = b''.join(lines[:100]).decode('utf-8', errors='ignore')
            for line in first_lines.split('\n'):
                model_match = RE_MODEL.search(line)
                prompt_match = RE_PROMPT.search(line)
                if model_match:
                    model = model_match.group(1).strip()
                if prompt_match:
                    prompt = prompt_match.group(1).strip()

            # Read last 5 lines to check for completion marker
            last_lines = b''.join(lines[-5:]).decode('utf-8', errors='ignore')
            for line in last_lines.split('\n'):
                if COMPLETION_MARKER in line:
                    completed = True
                    break
    except FileNotFoundError:
        print(f"Warning: Log file not found, skipping: {log_path}")
    except Exception as e:
        print(f"Warning: Error parsing log file {log_path}: {e}")
            
    return model, prompt, completed

def get_completed_jobs(logs_dir):
    """Scan logs directory and return set of completed (model, prompt) tuples"""
    completed_jobs = set()
    
    if not os.path.exists(logs_dir):
        return completed_jobs
    
    for filename in os.listdir(logs_dir):
        if filename.endswith('.log'):
            log_path = os.path.join(logs_dir, filename)
            model, prompt, completed = parse_log_file(log_path)
            
            if model and prompt and completed:
                completed_jobs.add((model, prompt))
    
    return completed_jobs

def read_working_file():
    """
    Read current working jobs from the working file.
    This is used to prevent multiple runners from picking up the same job.
    """
    working_jobs = set()
    if os.path.exists(WORKING_FILE):
        with open(WORKING_FILE, 'r') as f:
            # Read all lines and parse jobs
            for line in f:
                line = line.strip()
                if JOB_DELIMITER in line:
                    parts = line.split(JOB_DELIMITER)
                    if len(parts) == 2:
                        working_jobs.add((parts[0].strip(), parts[1].strip()))
    return working_jobs

def add_to_working_file(model, prompt):
    """Add a job to the working file with file locking to prevent race conditions"""
    job_str = f"{model}{JOB_DELIMITER}{prompt}\n"
    
    # Use a lock file for atomic write operations
    with open(WORKING_FILE_LOCK, 'w') as lock_file:
        # Acquire an exclusive lock
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        
        # Read existing content
        existing = ""
        if os.path.exists(WORKING_FILE):
            with open(WORKING_FILE, 'r') as f:
                existing = f.read()
        
        # Append new job
        with open(WORKING_FILE, 'w') as f:
            f.write(existing)
            f.write(job_str)
        
        # Release the lock
        fcntl.flock(lock_file, fcntl.LOCK_UN)

def remove_from_working_file(job_tuple):
    """Remove a job from the working file with file locking"""
    model, prompt = job_tuple
    job_str = f"{model}{JOB_DELIMITER}{prompt}"
    
    # Use a lock file for atomic read/write operations
    with open(WORKING_FILE_LOCK, 'w') as lock_file:
        # Acquire an exclusive lock
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        
        if os.path.exists(WORKING_FILE):
            # Read all lines
            with open(WORKING_FILE, 'r') as f:
                lines = f.readlines()
            
            # Rewrite file without the completed job
            with open(WORKING_FILE, 'w') as f:
                for line in lines:
                    if line.strip() != job_str:
                        f.write(line)
        
        # Release the lock
        fcntl.flock(lock_file, fcntl.LOCK_UN)

def run_job(model, prompt, args):
    """Run a single job (one model, one prompt) using the main.py script"""
    global current_job
    current_job = (model, prompt)
    
    # Create temporary file for prompt
    # delete=False is needed so the file persists until os.unlink is called
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_f:
        temp_f.write(prompt)
        temp_f.flush()
        temp_file_path = temp_f.name
    
    # Get index to access corresponding direction files and layers
    try:
        model_index = MODELS.index(model)
    except ValueError:
        print(f"Error: Model {model} not found in configuration lists. Skipping.")
        os.unlink(temp_file_path) # Clean up temp file
        remove_from_working_file(current_job) # Remove from working file
        current_job = None
        return
    
    # Build command
    command = [
        sys.executable, SCRIPT_TO_RUN,
        "--pg_model", args.pg_model,
        "--model", model,
        "--target_direction_path", DIRECTION_FILES[model_index],
        "--detection_direction_path", DETECTION_FILES[model_index],
        "--target_layer", str(LAYERS[model_index]),
        "--logs-dir", args.logs_dir,
        "--prompt-file", temp_file_path,
    ]
    
    print(f"\nRunning job: {model} | {prompt[:50]}...")
    print(f"Command: {' '.join(command)}")
    
    try:
        # Run the command. check=True will raise an error if main.py fails.
        subprocess.run(command, check=True)
        print(f"Completed job: {model} | {prompt[:50]}...")
    except subprocess.CalledProcessError as e:
        print(f"Error running job: {model} | {prompt[:50]}...")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Stderr: {e.stderr}")
    finally:
        # Clean up temp file
        os.unlink(temp_file_path)
        
        # Remove from working file
        remove_from_working_file(current_job)
        current_job = None
    
def main():
    """
    Main script execution logic.
    
    1. Loads all prompts.
    2. Scans log directory for already completed jobs.
    3. Generates all possible (model, prompt) combinations.
    4. Iterates through combinations, skipping completed or in-progress jobs.
    5. Runs outstanding jobs one by one.
    """
    parser = argparse.ArgumentParser(description="Run main.py experiment script.")
    #parser.add_argument("--pg_model", type=str, default="meta-llama/Llama-Prompt-Guard-2-86M")
    parser.add_argument("--pg_model", type=str, default="ibm-granite/granite-guardian-hap-38m")
    parser.add_argument("--logs-dir", type=str, default="logs")
    parser.add_argument("--max-jobs", type=int, default=None, help="Maximum number of jobs to run")
    args = parser.parse_args()
    
    # Register signal handlers to clean up working file on interrupt
    signal.signal(signal.SIGINT, cleanup_working_job)
    signal.signal(signal.SIGTERM, cleanup_working_job)
    
    # Read all prompts
    prompts = read_prompts()
    print(f"Loaded {len(prompts)} prompts from {PROMPTS_FILE}")
    
    # Get set of jobs that have already finished successfully
    completed_jobs = get_completed_jobs(args.logs_dir)
    print(f"Found {len(completed_jobs)} completed jobs in logs")
    
    # Create a list of all (model, prompt) job tuples to be run
    all_jobs = []
    for model in MODELS:
        for prompt in prompts:
            for _ in range(REPEATS_PER_PROMPT):
                if (model, prompt) not in all_jobs:
                    all_jobs.append((model, prompt))
                else:
                    print(f"Warning: Duplicate job detected for {model} | {prompt[:50]}... Skipping duplicate.")

    print(f"Total possible jobs (with repeats): {len(all_jobs)}")
    
    # Iterate through all possible jobs and run the ones that are not completed or in progress
    jobs_run = 0
    for model, prompt in all_jobs:
        
        # Stop if max-jobs limit is reached
        if args.max_jobs and jobs_run >= args.max_jobs:
            print(f"\nReached maximum jobs limit ({args.max_jobs})")
            break
        
        # 1. Check if job is already done
        if (model, prompt) in completed_jobs:
            continue

        # 2. Check if job is currently running in another process
        working_jobs = read_working_file()
        if (model, prompt) in working_jobs:
            print(f"Skipping (already being worked on): {model} | {prompt[:50]}...")
            continue
        
        # 3. Add job to working file to mark it as in-progress
        add_to_working_file(model, prompt)
        
        # 4. Run the job
        run_job(model, prompt, args)

        # 5. Re-scan logs to update completed set (for next iterations)
        # This is useful if REPEATS_PER_PROMPT > 1
        completed_jobs = get_completed_jobs(args.logs_dir)
        
        jobs_run += 1
    
    if jobs_run == 0:
        print("\nNo new jobs to run - all jobs are either completed or being worked on")
    else:
        print(f"\nCompleted {jobs_run} new jobs")

if __name__ == "__main__":
    main()