
import copy
import gc
import logging
import queue
import threading

from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from torch import Tensor
from transformers import set_seed
from scipy.stats import spearmanr
import os
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class GCGConfig:
    num_steps: int = 350
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: int = None
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    filter_ids: bool = False
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"
    target_loss_weight: float = 1.0
    benignity_threshold: float = 0.8 
    window_size: int = 5
    primary_early_stop: int = 5
    super_early_stop: int = 5



@dataclass
class DirectionGCGConfig(GCGConfig):
    target_direction_path: str = None  
    target_layer: int = None  
    direction_weight: float = 1.0  
    use_last_token: bool = True  
    maximize_similarity: bool = True  
    detection_direction_path: str = None 


@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]
    evals: List[float]
    last_output: str = ""
    last_string: str = ""