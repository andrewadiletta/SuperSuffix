# Nano GCG source Code
import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import transformers
from nanogcg.utils import logger

# import the main GCG classes
from nanogcg.directional_gcg import DirectionGCG
from nanogcg.pg_gcg import GCG_pg

# import the GCG config classes
from nanogcg.gcg_configs import GCGConfig, DirectionGCGConfig, GCGResult

@dataclass
class ProbeSamplingConfig:
    draft_model: transformers.PreTrainedModel
    draft_tokenizer: transformers.PreTrainedTokenizer
    r: int = 8
    sampling_factor: int = 16

# A wrapper around the GCG_pg class to run GCG optimization jointly
def run_pg(
    pg_model: transformers.PreTrainedModel,
    text_gen_model: transformers.PreTrainedModel,
    pg_tokenizer: transformers.PreTrainedTokenizer,
    text_gen_tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    target: str,
    config: Optional[GCGConfig] = None,
    force_eval: bool = False
) -> GCGResult:
    """Generates a single optimized string using GCG.

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        target: The target generation.
        config: The GCG configuration to use.

    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = GCGConfig()

    # Set the logger level based on config verbosity
    if hasattr(config, 'verbosity'):
        logger.setLevel(getattr(logging, config.verbosity, logging.INFO))

    gcg = GCG_pg(pg_model, text_gen_model, pg_tokenizer, text_gen_tokenizer, config, force_eval)
    result = gcg.run(messages, target)
    return result

# A wrapper around the DirectionGCG class to run direction optimization
def run_direction_optimization(
    pg_model: transformers.PreTrainedModel,
    text_gen_model: transformers.PreTrainedModel,
    pg_tokenizer: transformers.PreTrainedTokenizer,
    text_gen_tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    config: Optional[DirectionGCGConfig] = None
) -> GCGResult:
    """
    Optimizes input prompts to align model's hidden states with a target concept direction.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        messages: Input messages to optimize
        concept_direction_path: Path to saved concept direction tensor
        target_layer: Which layer to optimize alignment at
        config: Configuration for the optimization
    
    Returns:
        GCGResult with optimized prompt suffix
    """
    
    # Set the logger level based on config verbosity
    if hasattr(config, 'verbosity'):
        logger.setLevel(getattr(logging, config.verbosity, logging.INFO))

    gcg = DirectionGCG(pg_model, text_gen_model, pg_tokenizer, text_gen_tokenizer, config)
    result = gcg.run(messages)
    return result


