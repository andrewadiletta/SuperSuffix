import copy
import gc


from tqdm import tqdm
from typing import List, Union

import torch
import torch.nn.functional as F
import transformers
from torch import Tensor
import os


# import the GCG config classes
from nanogcg.gcg_configs import GCGConfig, GCGResult

from nanogcg.utils import (
    INIT_CHARS,
    _test_prompt_output,
    find_executable_batch_size,
    get_nonascii_toks,
    logger,
    filter_ids,
    sample_ids_from_grad,
    evaluate_response,
    _update_progress_plot,
    safety_model_override,
    safety_tokenizer_override,
    evaluate_with_prompt_guard,
)


class AttackBuffer:
    """
    A fixed-size buffer to store the 'best' (lowest loss) attack candidates.

    It maintains a sorted list of (loss, optim_ids) tuples,
    where the first element is always the one with the lowest loss.
    """

    def __init__(self, size: int):
        # The buffer stores tuples: (loss: float, optim_ids: Tensor)
        self.buffer: List[tuple[float, Tensor]] = []
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        """
        Adds a new candidate (loss and token IDs) to the buffer.

        If the buffer is full, the new candidate is only added if its loss
        is lower than the worst (highest) loss currently in the buffer,
        replacing it. The buffer is kept sorted.

        Args:
            loss (float): The candidate's loss.
            optim_ids (Tensor): The candidate's token IDs.
        """
        # If buffer size is 0, do nothing.
        if self.size == 0:
            return

        # If buffer isn't full, just add the new item.
        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        # If buffer is full, only add if new loss is better than the worst.
        elif loss < self.buffer[-1][0]:
            self.buffer[-1] = (loss, optim_ids)  # Replace the worst item
        else:
            # If the new loss is worse than the worst, do nothing.
            return

        # Re-sort the buffer by loss (ascending) after any change.
        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        """
        Returns the token ID tensor for the candidate with the lowest loss.

        Returns an empty tensor if the buffer is empty.
        """
        if not self.buffer:
            return torch.tensor([])  # Safety check
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        """
        Returns the lowest (best) loss value currently in the buffer.

        Returns infinity if the buffer is empty.
        """
        if not self.buffer:
            return float("inf")  # Safety check
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        """
        Returns the highest (worst) loss value currently in the buffer.

        Returns infinity if the buffer is empty.
        """
        if not self.buffer:
            return float("inf")  # Safety check
        return self.buffer[-1][0]

    def log_buffer(self, tokenizer):
        """
        Logs the current contents of the buffer (loss and decoded string).
        """
        message = "AttackBuffer Content:"
        if not self.buffer:
            message += "\n  (Buffer is empty)"
            # logger.info(message)
            return

        for i, (loss, ids) in enumerate(self.buffer):
            # Use decode() and squeeze() for robustness (handles [1, N] or [N])
            optim_str = tokenizer.decode(ids.squeeze(), skip_special_tokens=True)

            # Format newlines and backslashes for clean logging
            optim_str = optim_str.replace("\\", "\\\\").replace("\n", "\\n")
            message += f"\n  {i + 1}) Loss: {loss:.4f} | Suffix: '{optim_str}'"


class DirectionGCG:
    """
    Extended GCG class for optimizing prompts to align model's internal representations
    with target concept directions rather than generating specific outputs.
    """

    def __init__(
        self,
        pg_model: transformers.PreTrainedModel,
        text_gen: transformers.PreTrainedModel,
        pg_tokenizer: transformers.PreTrainedTokenizer,
        text_gen_tokenizer: transformers.PreTrainedTokenizer,
        config: GCGConfig,
    ):
        self.text_gen_model = text_gen
        self.pg_model = pg_model
        self.pg_tokenizer = pg_tokenizer
        self.text_gen_tokenizer = text_gen_tokenizer
        self.config = config

        self.pg_embedding_layer = pg_model.get_input_embeddings()
        self.text_gen_embedding_layer = text_gen.get_input_embeddings()
        self.not_allowed_ids = (
            None
            if config.allow_non_ascii
            else get_nonascii_toks(text_gen_tokenizer, device=pg_model.device)
        )
        self.prefix_cache = None
        self.draft_prefix_cache = None

        self.target_direction_path = config.target_direction_path
        self.target_layer = config.target_layer

        self.stop_flag = False
        self.detection_direction_path = config.detection_direction_path
        self.primary_early_stop = config.primary_early_stop
        self.super_early_stop = config.super_early_stop

    def run(self, messages: Union[str, List[dict]]) -> GCGResult:
        """
        Runs the direction-based GCG optimization attack.
        
        This method iteratively refines a suffix to maximize the model's
        hidden representation alignment with a target "direction vector"
        at a specific layer.

        Args:
            messages: A single user prompt string or a list of
                      conversation messages (e.g., [{"role": "user", "content": ...}]).

        Returns:
            GCGResult: An object containing the optimization results, including
                       the best suffix, losses, and model outputs.
        """

        # --- 1. Initialization and Setup ---

        target_layer = self.target_layer
        if target_layer is None:
            raise ValueError("target_layer must be specified for direction-based attack")

        # Load the target direction vector (what we want to align with)
        print(f"Loading direction tensor from: {self.target_direction_path}")
        direction_tensor = torch.load(
            self.target_direction_path, map_location=self.pg_model.device
        )
        if direction_tensor.dim() == 1:
            direction_tensor = direction_tensor.unsqueeze(0) # Ensure [1, D] shape

        # Load the detection vector (used for monitoring/testing, not optimization)
        if self.detection_direction_path:
            detection_direction_tensor = torch.load(
                self.detection_direction_path, map_location=self.pg_model.device
            )
            if detection_direction_tensor.dim() == 1:
                detection_direction_tensor = detection_direction_tensor.unsqueeze(0)
        else:
            detection_direction_tensor = None

        target_direction = direction_tensor.squeeze(0) # Use [D] shape for loss calcs

        # Set up model, tokenizer, and config for convenience
        model = self.text_gen_model
        tokenizer = self.text_gen_tokenizer
        config = self.config

        # --- 2. Prompt Templating ---
        
        # Ensure messages are in the standard list-of-dicts format
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)

        messages_orig = copy.deepcopy(messages) # Keep a copy without the suffix

        # Insert the {SUFFIX} placeholder if it's not already present
        if not any(["{SUFFIX}" in d["content"] for d in messages]):
            messages[-1]["content"] = messages[-1]["content"] + "{SUFFIX}"

        # Store the template for later testing
        self.original_messages = copy.deepcopy(messages)

        # Apply the model's chat template
        template = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Remove the BOS token if it's at the start, as tokenization will add it
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "")

        # Split the prompt into parts before and after the optimizable suffix
        before_str, after_str = template.split("{SUFFIX}")

        # --- 3. Tokenize and Embed Static Parts ---
        
        before_ids = tokenizer([before_str], padding=False, return_tensors="pt")[
            "input_ids"
        ].to(model.device, torch.int64)
        
        after_ids = tokenizer(
            [after_str], add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(model.device, torch.int64)

        # Get embeddings for the static parts
        embedding_layer = self.text_gen_embedding_layer
        before_embeds, after_embeds = [
            embedding_layer(ids) for ids in (before_ids, after_ids)
        ]
        
        self.before_embeds = before_embeds
        self.after_embeds = after_embeds

        # --- 4. Pre-calculate Prefix KV Cache ---
        
        # This is a major optimization: we compute the KV cache for the
        # `before_embeds` once and reuse it in every step of the loop.
        if config.use_prefix_cache:
            with torch.no_grad():
                output = model(inputs_embeds=before_embeds, use_cache=True)
                self.prefix_cache = output.past_key_values
        else:
            self.prefix_cache = None

        # --- 5. Initialize Attack Buffer ---
        
        # The buffer holds the best N suffixes found so far
        buffer = self.init_buffer_direction(target_direction, target_layer)
        optim_ids = buffer.get_best_ids() # Start with the best initial suffix

        # --- 6. Initialize Progress Trackers ---
        
        losses = []            # Loss from each step's best candidate
        steps = []             # Step numbers
        best_losses = []       # Best loss *so far* (from the buffer)
        Eval_scores = []       # Safety eval scores
        optim_strings = []     # Suffix string from each step
        pg_scores = []         # PromptGuard scores
        pg_loss = []           # Placeholder for plotting
        last_output = ""       # Last generated model output
        last_string = ""       # Last suffix string

        # --- 7. Run Baseline (No Suffix) ---
        
        # Evaluate the original prompt *without* any suffix for comparison.
        logger.info("Running baseline evaluation (no suffix)...")
        baseline_output, baseline_cosine_sims, baseline_output_start = (
            _test_prompt_output(
                model, tokenizer, messages_orig, "", detection_direction_tensor
            )
        )
        baseline_Eval_score = evaluate_response(
            safety_model_override, safety_tokenizer_override, baseline_output
        )
        baseline_pg_score_tensor = evaluate_with_prompt_guard(
            self.pg_model, self.pg_tokenizer, messages_orig, ""
        )
        baseline_pg_score = baseline_pg_score_tensor.squeeze().tolist()[0]
        
        # Log and print baseline results
        baseline_log = (
            f"Baseline (no suffix) | "
            f"PG Score: {baseline_pg_score:.6f} | "
            f"Eval Score: {baseline_Eval_score:.6f} | "
            f"Output: '{baseline_output}'"
        )
        logger.info(baseline_log)
        print(baseline_log)
        print("-" * 80)

        # --- 8. Start Optimization Loop ---
        
        logger.info("Starting optimization...")
        for step in tqdm(range(config.num_steps), desc="Optimizing Suffix"):
            
            # --- 8a. Get Gradient ---
            # Compute the gradient w.r.t. the one-hot token representations
            optim_ids_onehot_grad = self.compute_token_gradient_direction(
                optim_ids, target_direction, target_layer
            )
            
            with torch.no_grad():
                # --- 8b. Sample Candidates ---
                # Get a batch of new candidate token IDs by "swapping" tokens
                # based on the gradient.
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )

                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, tokenizer)

                new_search_width = sampled_ids.shape[0]
                batch_size = new_search_width or config.batch_size

                # --- 8c. Evaluate Candidates ---
                
                # Get embeddings for all sampled candidates
                sampled_embeds = embedding_layer(sampled_ids)
                
                # Construct the full input embeddings for the batch
                if self.prefix_cache:
                    # If using KV cache, we *only* need the suffix + after embeds
                    input_embeds = torch.cat(
                        [
                            sampled_embeds,
                            after_embeds.repeat(new_search_width, 1, 1),
                        ],
                        dim=1,
                    )
                else:
                    # Otherwise, we need the full sequence
                    input_embeds = torch.cat(
                        [
                            before_embeds.repeat(new_search_width, 1, 1),
                            sampled_embeds,
                            after_embeds.repeat(new_search_width, 1, 1),
                        ],
                        dim=1,
                    )

                # Calculate the direction-based loss for all candidates
                # `find_executable_batch_size` prevents OOM errors
                loss = find_executable_batch_size(
                    self._compute_candidates_loss_direction, batch_size
                )(input_embeds, target_direction, target_layer)

                # --- 8d. Select Best & Update Buffer ---
                
                # Find the best candidate *from this batch*
                current_loss = loss.min().item()
                best_batch_ids = sampled_ids[loss.argmin()].unsqueeze(0)
                losses.append(current_loss)
                
                # Add the best from this batch to the persistent buffer
                buffer.add(current_loss, best_batch_ids)

            # Get the *overall* best IDs from the buffer for the next iteration
            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)
            last_string = optim_str

            # --- 9. Log Step Info & Evaluate ---
            
            # Run a full generation test with the new best suffix
            test_output, cosine_sims, output_start = _test_prompt_output(
                model, tokenizer, messages, optim_str, detection_direction_tensor
            )
            last_output = test_output

            # Log step metrics
            step_num = len(losses)
            steps.append(step_num)
            best_losses.append(buffer.get_lowest_loss())

            # Get safety scores
            Eval_score = evaluate_response(
                safety_model_override, safety_tokenizer_override, test_output
            )
            Eval_scores.append(Eval_score)
            
            pg_score_tensor = evaluate_with_prompt_guard(
                self.pg_model, self.pg_tokenizer, messages, optim_str
            )
            pg_score = pg_score_tensor.squeeze().tolist()[0]
            pg_scores.append(pg_score)
            pg_loss.append(0) # NOTE: PG loss is not directly optimized in this loop

            # Log everything to file
            logger.info(f"OPTIMIZATION STEP {step_num}")
            logger.info(f"  Current batch loss: {current_loss:.6f}")
            logger.info(f"  Best buffer loss: {best_losses[-1]:.6f}")
            logger.info(f"  PG score: {pg_score:.6f}")
            logger.info(f"  Eval score: {Eval_score:.6f} (1=safe)")
            logger.info(f"  Optimized suffix: '{optim_str}'")
            logger.info(f"  Model output: '{test_output}'")
            logger.info(f"  Cosine similarities: {cosine_sims}")
            logger.info(f"  Output start position: {output_start}")
            logger.info("")

            # Print concise info to console
            print(
                f"Step {step_num}: Loss = {current_loss:.4f} | "
                f"Best = {best_losses[-1]:.4f} | "
                f"Eval = {Eval_score if Eval_score is not None else 'N/A'} | "
                f"PG = {pg_score:.4f}"
            )
            print(f"Suffix: '{optim_str}'")
            print(f"Output: '{test_output}'")
            print("-" * 80)

            # Update and save the progress plot
            _update_progress_plot(
                steps,
                pg_loss,     # The list of 0s
                losses,      # The target-generation loss
                pg_scores,
                Eval_scores,
                title_suffix=f"(Layer {target_layer})",
            )

            buffer.log_buffer(tokenizer) # Log the top N suffixes

            # --- 10. Check for Early Stopping ---
            
            # Stop if we get 5 (or more) low (unsafe) Eval scores
            low_eval_scores = [s for s in Eval_scores if s is not None and s < 0.1]
            if len(low_eval_scores) >= self.primary_early_stop and (Eval_scores[-1] is not None and Eval_scores[-1] < 0.1):
                logger.info(f"Early stopping due to {self.primary_early_stop}+ low Eval scores.")
                break
            else:
                logger.info(f"Number of low Eval scores (<0.1) so far: {len(low_eval_scores)}")

        # --- 11. Compile Final Result ---
        
        # Find the best string and loss from the entire run
        min_loss_index = losses.index(min(losses))
        best_loss = losses[min_loss_index]
        best_string = optim_strings[min_loss_index]

        logger.info(f"Optimization finished. Best loss: {best_loss:.6f}")
        logger.info(f"Best string: '{best_string}'")

        result = GCGResult(
            best_loss=best_loss,
            best_string=best_string,
            losses=losses,
            strings=optim_strings,
            evals=Eval_scores,
            last_output=last_output,
            last_string=last_string,
        )

        return result


    def init_buffer_direction(
        self, target_direction: Tensor, layer_idx: int
    ) -> AttackBuffer:
        """
        Initialize buffer for direction optimization.
        Simplified version that doesn't need target strings.
        """
        model = self.text_gen_model
        tokenizer = self.text_gen_tokenizer
        config = self.config

        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)

        if isinstance(config.optim_str_init, str):
            init_optim_ids = tokenizer(
                config.optim_str_init, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device)
            if config.buffer_size > 1:
                init_buffer_ids = (
                    tokenizer(
                        INIT_CHARS, add_special_tokens=False, return_tensors="pt"
                    )["input_ids"]
                    .squeeze()
                    .to(model.device)
                )
                init_indices = torch.randint(
                    0,
                    init_buffer_ids.shape[0],
                    (config.buffer_size - 1, init_optim_ids.shape[1]),
                )
                init_buffer_ids = torch.cat(
                    [init_optim_ids, init_buffer_ids[init_indices]], dim=0
                )
            else:
                init_buffer_ids = init_optim_ids

        else:  # assume list
            if len(config.optim_str_init) != config.buffer_size:
                logger.warning(
                    f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}"
                )
            try:
                init_buffer_ids = tokenizer(
                    config.optim_str_init, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(model.device)
            except ValueError:
                logger.error(
                    "Unable to create buffer. Ensure that all initializations tokenize to the same length."
                )

        true_buffer_size = max(1, config.buffer_size)

        # Compute the loss on the initial buffer entries
        if self.prefix_cache:
            init_buffer_embeds = torch.cat(
                [
                    self.text_gen_embedding_layer(init_buffer_ids),
                    self.after_embeds.repeat(true_buffer_size, 1, 1),
                ],
                dim=1,
            )
        else:
            init_buffer_embeds = torch.cat(
                [
                    self.before_embeds.repeat(true_buffer_size, 1, 1),
                    self.text_gen_embedding_layer(init_buffer_ids),
                    self.after_embeds.repeat(true_buffer_size, 1, 1),
                ],
                dim=1,
            )

        # Evaluate initial buffer entries with direction loss
        init_buffer_losses = find_executable_batch_size(
            self._compute_candidates_loss_direction, true_buffer_size
        )(init_buffer_embeds, target_direction, layer_idx)

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        buffer.log_buffer(tokenizer)

        logger.info("Initialized attack buffer.")

        return buffer

    def safe_cosine_similarity(self, v1, v2, dim=1, eps=1e-8):
        """
        Computes cosine similarity between two tensors with numerical stability.
        """
        v1_normalized = F.normalize(v1, p=2, dim=dim, eps=eps)
        v2_normalized = F.normalize(v2, p=2, dim=dim, eps=eps)
        return torch.sum(v1_normalized * v2_normalized, dim=dim)

    def compute_token_gradient_direction(
        self, optim_ids: Tensor, target_direction: Tensor, layer_idx: int
    ) -> Tensor:
        """
        Computes gradient of direction alignment loss w.r.t. one-hot token matrix.
        """
        model = self.text_gen_model
        embedding_layer = self.text_gen_embedding_layer

        # Create one-hot encoding of optimized tokens
        optim_ids_onehot = torch.nn.functional.one_hot(
            optim_ids, num_classes=embedding_layer.num_embeddings
        )
        optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
        optim_ids_onehot.requires_grad_()

        # Get embeddings
        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        # Construct full input embeds
        if self.prefix_cache:
            input_embeds = torch.cat([optim_embeds, self.after_embeds], dim=1)
            output = model(
                inputs_embeds=input_embeds,
                past_key_values=self.prefix_cache,
                output_hidden_states=True,
                use_cache=True,
            )
        else:
            input_embeds = torch.cat(
                [self.before_embeds, optim_embeds, self.after_embeds], dim=1
            )
            output = model(inputs_embeds=input_embeds, output_hidden_states=True)

        # Get hidden states at target layer
        hidden_states = output.hidden_states[layer_idx]

        # Use last token or average
        if self.config.use_last_token:
            hidden_to_align = hidden_states[:, -1, :]
        else:
            hidden_to_align = hidden_states.mean(dim=1)

        # Compute direction alignment loss
        cosine_sim = self.safe_cosine_similarity(
            hidden_to_align, target_direction.unsqueeze(0), dim=1
        )

        # Apply same maximize_similarity logic as in loss computation
        maximize_similarity = getattr(self.config, "maximize_similarity", False)
        if maximize_similarity:
            loss = -cosine_sim.mean() * self.config.direction_weight
        else:
            loss = cosine_sim.mean() * self.config.direction_weight

        # Get gradient
        optim_ids_onehot_grad = torch.autograd.grad(
            outputs=[loss], inputs=[optim_ids_onehot]
        )[0]

        return optim_ids_onehot_grad

    def _compute_candidates_loss_direction(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
        target_direction: Tensor,
        layer_idx: int,
    ) -> Tensor:
        """
        Computes direction alignment loss for candidate sequences.
        Instead of cross-entropy with target tokens, computes cosine similarity with target direction.
        """

        all_loss = []
        prefix_cache_batch = []

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i : i + search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                # Get hidden states at all layers
                print(
                    f"Debug batch {i}: input_embeds_batch.shape = {input_embeds_batch.shape}"
                )
                if self.prefix_cache:
                    print(
                        f"Debug batch {i}: Using prefix cache - only processing suffix + after tokens"
                    )
                    if (
                        not prefix_cache_batch
                        or current_batch_size != search_batch_size
                    ):
                        prefix_cache_batch = [
                            [
                                x.expand(current_batch_size, -1, -1, -1)
                                for x in self.prefix_cache[i]
                            ]
                            for i in range(len(self.prefix_cache))
                        ]

                    outputs = self.text_gen_model(
                        inputs_embeds=input_embeds_batch,
                        past_key_values=prefix_cache_batch,
                        output_hidden_states=True,
                        use_cache=True,
                    )
                else:
                    print(
                        f"Debug batch {i}: NOT using prefix cache - processing FULL sequence (slow!)"
                    )
                    outputs = self.text_gen_model(
                        inputs_embeds=input_embeds_batch, output_hidden_states=True
                    )

                # Extract hidden states at target layer
                hidden_states = outputs.hidden_states[layer_idx]

                # Use last token position (where assistant response starts)
                if self.config.use_last_token:
                    hidden_to_align = hidden_states[:, -1, :]
                else:
                    # Or average over sequence
                    hidden_to_align = hidden_states.mean(dim=1)

                # Compute cosine similarity with target direction
                # Flatten target_direction and broadcast to match batch size
                target_direction_flat = target_direction.view(-1)  # Flatten to 1D
                target_direction_batch = target_direction_flat.unsqueeze(0).expand(
                    current_batch_size, -1
                )
                cosine_sim = F.cosine_similarity(
                    hidden_to_align, target_direction_batch, dim=1
                )

                # Loss computation based on maximize_similarity flag
                direction_weight = getattr(self.config, "direction_weight", 1.0)
                maximize_similarity = getattr(self.config, "maximize_similarity", False)

                if maximize_similarity:
                    # Maximize similarity: loss = -cosine_sim (negative for minimization objective)
                    loss = -cosine_sim * direction_weight
                else:
                    # Minimize similarity: loss = cosine_sim (positive for minimization objective)
                    loss = cosine_sim * direction_weight

                print(
                    f"Debug batch {i}: current_batch_size = {current_batch_size}, hidden_to_align.shape = {hidden_to_align.shape}, cosine_sim.shape = {cosine_sim.shape}, loss.shape = {loss.shape}"
                )
                all_loss.append(loss)

                del outputs
                gc.collect()
                torch.cuda.empty_cache()

        final_loss = torch.cat(all_loss, dim=0)
        print(
            f"Debug _compute_candidates_loss_direction: final_loss.shape = {final_loss.shape}"
        )
        return final_loss
