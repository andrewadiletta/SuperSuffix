#!/usr/bin/env python3
"""
Saves a "concept direction" vector from a specific layer of a Llama model.

This script works by:
1. Loading pairs of positive/negative prompts (e.g., "be helpful" vs "be unhelpful").
2. Loading a pre-trained causal language model.
3. Feeding the prompts into the model and extracting the hidden states
   from the final token position at a specified layer.
4. Averaging the hidden states for all positive prompts.
5. Averaging the hidden states for all negative prompts.
6. Calculating the difference vector (positive_mean - negative_mean).
7. Normalizing and saving this vector to a .pt file.
"""

import os
import random
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse

# --- Configuration ---
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"  # Llama model for analysis
DEFAULT_DATA_SUBFOLDER = "hotel_recs"
N_INSTRUCTIONS = 300

# System prompt for Llama-3.2 (Note: Not used in get_hidden_states by default)
SYSTEM_PROMPT = "You are Llama, created by Meta. You are a helpful assistant."

# --- Model and Tokenizer Loading ---
def load_model_and_tokenizer(model_id):
    """
    Loads the language model and tokenizer from Hugging Face.
    
    Sets up dtype, attention implementation, and a default chat template if missing.
    """
    print(f"Loading model and tokenizer for '{model_id}'...")

    # Use bfloat16 and Flash Attention 2 on H100/A100+ GPUs
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
        attn_implementation = "flash_attention_2"
    else:
        torch_dtype = torch.float16
        attn_implementation = "eager"

    print(f"Using torch_dtype: {torch_dtype}, attn_implementation: {attn_implementation}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Check if a chat template is defined, and if not, set a default one
    if tokenizer.chat_template is None:
        print("Warning: No chat template found for tokenizer. Setting a default ChatML-style template.")
        # This is a common template format (ChatML) that works well.
        tokenizer.chat_template = (
            "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                    "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
                "{% elif message['role'] == 'user' %}"
                    "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
                "{% elif message['role'] == 'assistant' %}"
                    "{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )

    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

# --- Data Handling ---

def load_instructions_from_files(positive_path, negative_path, n_instructions):
    """
    Loads and samples a specified number of instructions from text files.
    
    Args:
        positive_path (str): Path to the file with positive prompts.
        negative_path (str): Path to the file with negative prompts.
        n_instructions (int): Number of prompts to sample from each file.
        
    Returns:
        tuple: (positive_sample, negative_sample)
    """
    
    def load_prompts(file_path):
        """Helper to load all non-empty lines from a file."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        prompts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
        return prompts

    print(f"Loading instructions from data files...")
    
    positive_instructions = load_prompts(positive_path)
    negative_instructions = load_prompts(negative_path)
    
    print(f"Loaded {len(positive_instructions)} positive and {len(negative_instructions)} negative prompts.")
    
    # Ensure we don't try to sample more prompts than available
    required_count = n_instructions
    if len(positive_instructions) < required_count:
        print(f"Warning: Positive prompts file has {len(positive_instructions)} instructions, using all available.")
        required_count = min(required_count, len(positive_instructions), len(negative_instructions))
    if len(negative_instructions) < required_count:
        print(f"Warning: Negative prompts file has {len(negative_instructions)} instructions, using all available.")
        required_count = min(required_count, len(positive_instructions), len(negative_instructions))

    # Shuffle and sample
    random.shuffle(positive_instructions)
    random.shuffle(negative_instructions)
    
    positive_sample = positive_instructions[:required_count]
    negative_sample = negative_instructions[:required_count]
    
    print(f"Sampled {len(positive_sample)} positive and {len(negative_sample)} negative instructions.")
    return positive_sample, negative_sample

# --- Concept Direction Extraction ---

def get_hidden_states(model, tokenizer, instructions, layer_idx):
    """
    Generates hidden states for a list of instructions at a specific layer.
    
    Args:
        model: The loaded AutoModelForCausalLM.
        tokenizer: The loaded AutoTokenizer.
        instructions (list): A list of prompt strings.
        layer_idx (int): The layer index to extract hidden states from.
        
    Returns:
        list: A list of hidden state tensors.
    """
    hidden_states = []
    for instruction in tqdm(instructions, desc=f"Generating hidden states at layer {layer_idx}"):
        
        # Note: System prompt is intentionally omitted to get the "raw"
        # user-prompt-based activation, rather than the system-conditioned one.
        conversation = [
            {"role": "user", "content": instruction},
        ]
        
        # Apply chat template to format the input
        inputs = tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            # Run generation for 1 token to get hidden states just before output
            outputs = model.generate(
                inputs, 
                use_cache=False, 
                max_new_tokens=1, 
                return_dict_in_generate=True,
                output_hidden_states=True, 
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Extract the hidden state
        # outputs.hidden_states[0]: States for the *first* (and only) generated token
        # [layer_idx]: Select the specific layer
        # [:, -1, :]: Select the state at the *last token position* of the input
        state = outputs.hidden_states[0][layer_idx][:, -1, :]
        hidden_states.append(state)
        
    return hidden_states

def calculate_concept_direction(positive_states, negative_states):
    """
    Calculates the concept direction vector from positive and negative hidden states.
    
    Args:
        positive_states (list): List of hidden state tensors for positive prompts.
        negative_states (list): List of hidden state tensors for negative prompts.
        
    Returns:
        torch.Tensor: The normalized concept direction vector.
    """
    print("Calculating concept direction...")
    # Calculate the mean activation for positive and negative prompts
    positive_mean = torch.stack(positive_states).mean(dim=0)
    negative_mean = torch.stack(negative_states).mean(dim=0)
    
    # Calculate the difference vector (this is the "concept direction")
    concept_dir = positive_mean - negative_mean
    
    # Normalize the vector
    concept_dir = concept_dir / concept_dir.norm()
    
    print("Concept direction calculated and normalized.")
    return concept_dir

# --- Main Execution ---

def main():
    """Main function to parse arguments, run extraction, and save the vector."""
    parser = argparse.ArgumentParser(description='Save concept direction from specific layer')
    parser.add_argument('--layer', type=int, required=True, 
                        help='Layer to analyze and save direction from')
    parser.add_argument('--n_instructions', type=int, default=N_INSTRUCTIONS, 
                        help='Number of instructions to use')
    parser.add_argument('--data_subfolder', type=str, default=DEFAULT_DATA_SUBFOLDER,
                       help=f'Data subfolder to use (default: {DEFAULT_DATA_SUBFOLDER})')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_ID,
                       help=f'Model to use (default: {DEFAULT_MODEL_ID})')
    parser.add_argument('--output_dir', type=str, default=".",
                       help='Directory to save output files (default: current directory)')
    
    args = parser.parse_args()
    
    layer_idx = args.layer
    n_instructions = args.n_instructions
    
    # Set up paths based on data subfolder
    data_dir = f"data/{args.data_subfolder}"
    positive_prompts_path = f"{data_dir}/direction1.txt"
    negative_prompts_path = f"{data_dir}/direction2.txt"
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define the output path for the tensor
    concept_direction_path = os.path.join(
        args.output_dir, 
        f"{args.data_subfolder}_concept_direction_layer_{layer_idx}.pt"
    )
    
    print(f"Starting concept direction extraction and save for layer {layer_idx}")
    print(f"Using data subfolder: {args.data_subfolder}")
    print(f"Using model: {args.model}")
    
    # Load prompts from files
    positive_instructions, negative_instructions = load_instructions_from_files(
        positive_prompts_path, negative_prompts_path, n_instructions
    )
    
    # Load the model
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Run the core extraction logic
    print(f"\nExtracting concept direction from layer {layer_idx}...")
    positive_hidden_states = get_hidden_states(model, tokenizer, positive_instructions, layer_idx)
    negative_hidden_states = get_hidden_states(model, tokenizer, negative_instructions, layer_idx)
    concept_direction = calculate_concept_direction(positive_hidden_states, negative_hidden_states)
    
    # Save the final tensor
    torch.save(concept_direction, concept_direction_path)
    print(f"Concept direction tensor saved to: {concept_direction_path}")

    print(f"\nConcept direction extraction complete!")

if __name__ == "__main__":
    main()