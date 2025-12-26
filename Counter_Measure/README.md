# Countermeasure Experiments

## High-Level Overview

The Countermeasure works by tracing the cosine similarity to refusal over the token sequence. The result is a unique fingerprint trace we can use to identify if a model has been subjected to Super Suffix attacks.

## Quick start

After cloning the repository, ensure that all the packages in `install.sh` are properly installed, then run the countermeasure script with:

```bash
python countermeasure.py
```

This will generate a set of plots and statistics in the `plots/` directory, analyzing the logs from a previous Super Suffix generation experiment located in `PREVIOUS_RUNS/malicious_code/logs`.

## Practical Usage

This example is meant to be a simple PoC, and run complementary to the Super Suffix generation experiments.

The countermeasure experiments are designed to be a post-processing step to the Super Suffix generation experiments. During the Super Super generation, we log the cosine similarity to refusal at each token position. The countermeasure script analyzes these logs to determine if the model was successfully attacked. 

The Countermeasure script takes the parameters at the top of `countermeasure.py` to configure paths and settings. 

Below are the configuration variables in the script:

```python
# --- Configuration ---
LOG_DIR = 'PREVIOUS_RUNS/malicious_code/logs'
COST_PER_HOUR = 1.49
CACHE_DIR = "cache"  # New directory for cache files
PLOT_DIR = "plots"   # New base directory for plots

GOOD_PROMPTS_FILE = os.path.join("prompts", "bad_code", "good_code.txt")
BAD_PROMPTS_FILE = os.path.join("prompts", "bad_code", "bad_code.txt")

# Window configuration for analysis
TOKENS_BEFORE_OUTPUT = 4  # Number of tokens to analyze before output position
TOKENS_AFTER_OUTPUT = 4   # Number of tokens to analyze after output position
WINDOW_SIZE = TOKENS_BEFORE_OUTPUT + TOKENS_AFTER_OUTPUT + 1  # Total window size

# KNN Configuration
K_NEIGHBORS = 3  # Number of neighbors for KNN classifier
TEST_SIZE = 0.2  # 20% for testing, 80% for training
RANDOM_STATE = 42  # For reproducibility

# Model configurations
MODELS = ["google/gemma-2b-it", 
          "meta-llama/Llama-3.2-3B-instruct",
          "meta-llama/Llama-3.1-8B-instruct", 
          "microsoft/Phi-3-mini-128k-instruct",
          "lmsys/vicuna-7b-v1.5"]

MODEL = MODELS[0]
```

To select a model, change the `MODEL` variable to one of the options in the `MODELS` list. To process a particular set of logs, change the `LOG_DIR` variable to point to the desired log directory. To change the set of prompts to analyze, modify the `GOOD_PROMPTS_FILE` and `BAD_PROMPTS_FILE` variables.

To change the trace analysis window, adjust the `TOKENS_BEFORE_OUTPUT` and `TOKENS_AFTER_OUTPUT` variables. By default we use 4 as seen in the paper. 

## Output

In the plots folder, you will find a number of plots including a traces graph that looks like this:
![Example Traces Graph](examples/traces_example.pdf)

# Disclaimer

Malicious prompts from prompts/Harmbench are taken from the HarmBench repo you can find here: 
https://github.com/centerforaisafety/HarmBench

Additionally, DeltaGuard is *not* meant for production systems, and is meant to be a PoC experiment for research purposes only. 