"""
Script that runs GCG with direction optimization instead of target string optimization
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import nanogcg
from nanogcg import DirectionGCGConfig, GCGConfig
import logging
import os
from datetime import datetime

def setup_logging(args) -> str:
    """Setup logging with timestamp-based filename and human-readable format"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logs directory if it doesn't exist
    log_dir = args.logs_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename
    log_filename = f"gcg_direction_opt_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()  # Also log to console
        ],
        force=True  # Force reconfiguration of the logging system
    )
    
    # Also configure the nanogcg logger specifically
    nanogcg_logger = logging.getLogger("nanogcg")
    nanogcg_logger.setLevel(logging.INFO)
    
    logger = logging.getLogger("gcg_experiment")
    
    # Log experiment configuration
    logger.info("=" * 80)
    logger.info("GCG DIRECTION OPTIMIZATION EXPERIMENT STARTED")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_path}")
    logger.info("")
    logger.info("EXPERIMENT CONFIGURATION:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  PG Model: {args.pg_model}")
    if args.prompt_file:
        logger.info(f"  Prompt file: {args.prompt_file}")
    else:
        logger.info(f"  Prompt: {args.prompt}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Data type: {args.dtype}")
    logger.info(f"  Probe sampling: {args.probe_sampling}")
    logger.info(f"  Number of optimization steps: {args.num_steps}")
    logger.info(f"  Target direction path: {args.target_direction_path}")
    logger.info(f"  Target layer: {args.target_layer}")
    logger.info("")
    
    return log_path

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-2b-it")
    parser.add_argument("--pg_model", type=str, default="meta-llama/Llama-Prompt-Guard-2-86M")
    parser.add_argument("--prompt", type=str, default=None)
    
    parser.add_argument("--use-last-token", action="store_true", default=True,
                       help="Use last token vs average over sequence")
    parser.add_argument("--prompt-file", type=str, default=None,
                       help="Path to file containing the prompt. If provided, overrides --prompt argument")
    
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--probe-sampling", action="store_true")
    parser.add_argument("--num-steps", type=int, default=300,
                       help="Number of optimization steps (default: 300)")
    parser.add_argument("--target-file", type=str, default=None)
    parser.add_argument("--target_direction_path", type=str, default=None)
    parser.add_argument("--target_layer", type=int, default=None)
    parser.add_argument("--detection_direction_path", type=str, default=None,
                       help="Path to detection direction tensor")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--window-size", type=int, default=5, help="Window size for oscillations")
    parser.add_argument("--primary-early-stop", type=int, default=5,
                       help="Early stopping steps for primary direction optimization")
    parser.add_argument("--super-early-stop", type=int, default=5, 
                       help="Early stopping steps for super suffix optimization")
    
    args = parser.parse_args()
    return args
    
def main():
    """Main function to run the GCG direction optimization experiment."""
    args = parse_args()
    
    # Setup logging
    log_path = setup_logging(args)
    logger = logging.getLogger("gcg_experiment")
    
    logger.info("Loading model and tokenizer...")
    pg_model = AutoModelForSequenceClassification.from_pretrained(
        args.pg_model, 
        torch_dtype=getattr(torch, args.dtype)
    ).to(args.device)

    text_gen_model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=getattr(torch, args.dtype)
    ).to(args.device)
    pg_tokenizer = AutoTokenizer.from_pretrained(args.pg_model)
    text_gen_tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Define the specific Llama-2-style guard template
    guard_template = (
        "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
                "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
            "{% endif %}"
        "{% endfor %}"
    )
    
    # Define a general-purpose default template (ChatML)
    default_chat_template = (
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

    # Check and set chat template for pg_tokenizer
    if pg_tokenizer.chat_template is None:
        logger.info("No chat template found for pg_tokenizer. Setting default guard template.")
        pg_tokenizer.chat_template = guard_template
    else:
        logger.info("Using existing chat template for pg_tokenizer.")

    # Check and set chat template for text_gen_tokenizer
    if text_gen_tokenizer.chat_template is None:
        logger.info("No chat template found for text_gen_tokenizer. Setting default ChatML-style template.")
        text_gen_tokenizer.chat_template = default_chat_template
    else:
        logger.info("Using existing chat template for text_gen_tokenizer.")

    # Setup padding tokens
    if text_gen_tokenizer.pad_token is None:
        text_gen_tokenizer.pad_token = text_gen_tokenizer.eos_token
        logger.info("Set text_gen_tokenizer pad_token to eos_token")
    
    if pg_tokenizer.pad_token is None:
        pg_tokenizer.pad_token = pg_tokenizer.eos_token
        logger.info("Set pg_tokenizer pad_token to eos_token")

    # Handle prompt input from file or command line
    if args.prompt_file:
        logger.info(f"Reading prompt from file: {args.prompt_file}")
        with open(args.prompt_file, 'r') as f:
            prompt_content = f.read().strip()
    else:
        prompt_content = args.prompt
    
    logger.info(f"Input prompt: {prompt_content}")
    
    messages = [{"role": "user", "content": prompt_content}]
    
    logger.info("Creating GCGConfig...")
    config = DirectionGCGConfig(
        verbosity="INFO",
        batch_size=70,
        use_prefix_cache=True,
        n_replace=3,
        optim_str_init="x x x x x x x x x x x x x x x x",
        num_steps=args.num_steps,
        #search_width=512,
        search_width=50,
        #topk=256,
        topk=100,
        #buffer_size=500,
        buffer_size=100,
        use_last_token=True,
        direction_weight=2.0,
        maximize_similarity=True,
        target_direction_path=args.target_direction_path,
        target_layer=args.target_layer,
        detection_direction_path=args.detection_direction_path,
        window_size=args.window_size,
        primary_early_stop=args.primary_early_stop,
        super_early_stop=args.super_early_stop
    )
    
    logger.info("Configuration created. Starting direction optimization...")
    logger.info("GCG Configuration:")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Search width: {config.search_width}")
    logger.info(f"  Number of steps: {config.num_steps}")
    logger.info(f"  Tokens to replace per step: {config.n_replace}")
    logger.info(f"  Top-k: {config.topk}")
    logger.info(f"  Buffer size: {config.buffer_size}")
    logger.info(f"  Direction weight: {config.direction_weight}")
    logger.info(f"  Use last token: {config.use_last_token}")
    logger.info(f"  Maximize similarity: {config.maximize_similarity}")
    logger.info(f"  Layer for direction: {config.target_layer}")
    logger.info(f"  Window size: {config.window_size}")
    logger.info(f"  Target direction path: {config.target_direction_path}")
    logger.info(f"  Detection direction path: {config.detection_direction_path}")
    logger.info(f"  Primary early stop: {args.primary_early_stop}")
    logger.info(f"  Super early stop: {args.super_early_stop}")
    logger.info("")

    # First run direction optimization to get the primary suffix
    primary_result = nanogcg.run_direction_optimization(
        pg_model,
        text_gen_model,
        pg_tokenizer,
        text_gen_tokenizer,
        messages,
        config=config
    )

    # get the last string and the target...
    primary_suffix=primary_result.last_string

    # only trucate to 30 chars if longer
    if len(primary_result.last_output) > 30:
        primary_result.last_output = primary_result.last_output[:30]
    target=primary_result.last_output[:30]

    logger.info("\n\n--------------------------------------------------\n\n")

    # append the best_string to the prompt
    for message in messages:
        if message["role"] == "user":
            message["content"] += " " + primary_suffix

    # Now run the full GCG with target optimization using the obtained target
    logger.info("Starting full GCG optimization with obtained target...")
    super_result = nanogcg.run_pg(
        pg_model,
        text_gen_model,
        pg_tokenizer,
        text_gen_tokenizer,
        messages,
        config=config,
        target=target,
    )
    
    # Print final results
    logger.info("Direction optimization completed!")
    logger.info(f"RESULTS:")
    logger.info(f"  Best loss: {super_result.best_loss:.6f}")
    logger.info(f"  Best optimized suffix: '{super_result.best_string}'")
    logger.info(f"  Total optimization steps: {len(super_result.losses)}")
    logger.info(f"  Final loss: {super_result.losses[-1]:.6f}")
    
    # Log loss progression (every 10 steps)
    logger.info("Loss progression (every 10 steps):")
    for i in range(0, len(super_result.losses), 10):
        logger.info(f"  Step {i:3d}: {super_result.losses[i]:.6f}")
 
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info(f"Log saved to: {log_path}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()