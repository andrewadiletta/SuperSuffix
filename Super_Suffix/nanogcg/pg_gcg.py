# Nano GCG source Code

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
import time

# import the GCG config classes
from nanogcg.gcg_configs import GCGConfig, DirectionGCGConfig, GCGResult

from nanogcg.utils import (
    INIT_CHARS,
    _test_prompt_output,
    configure_pad_token,
    find_executable_batch_size,
    get_nonascii_toks,
    mellowmax,
    logger,
    filter_ids,
    sample_ids_from_grad,
    evaluate_response,
    _update_progress_plot,
    AttackBuffer,
    append_suffix_to_user_role,
    evaluate_with_prompt_guard,
    safety_model_override,
    safety_tokenizer_override,
    enfuse_direction_with_tokens,
)


class GCG_pg:
    def __init__(
        self,
        pg_model: transformers.PreTrainedModel,
        text_gen: transformers.PreTrainedModel,
        pg_tokenizer: transformers.PreTrainedTokenizer,
        text_gen_tokenizer: transformers.PreTrainedTokenizer,
        config: DirectionGCGConfig,
        force_text_gen: bool,
    ):
        self.pg_model = pg_model
        self.text_gen = text_gen
        self.pg_tokenizer = pg_tokenizer
        self.text_gen_tokenizer = text_gen_tokenizer
        self.config = config
        self.pg_embedding_layer = pg_model.get_input_embeddings()
        self.text_gen_embedding_layer = text_gen.get_input_embeddings()
        self.not_allowed_ids = (
            None
            if config.allow_non_ascii
            else get_nonascii_toks(pg_tokenizer, device=pg_model.device)
        )
        self.prefix_cache = None
        self.draft_prefix_cache = None
        self.stop_flag = False
        self.draft_model = None
        self.draft_tokenizer = None
        self.draft_embedding_layer = None
        self.force_text_gen = force_text_gen
        self.target_direction_path = config.target_direction_path
        self.target_layer = config.target_layer
        self.detection_direction_path = config.detection_direction_path
        self.window_size = config.window_size
        self.primary_early_stop = config.primary_early_stop
        self.super_early_stop = config.super_early_stop

    def run(self, messages: Union[str, List[dict]], target: str) -> GCGResult:
        # setup the direction tensor
        direction_tensor = torch.load(
            self.target_direction_path, map_location=self.pg_model.device
        )
        if direction_tensor.dim() == 1:
            direction_tensor = direction_tensor.unsqueeze(0)

        if self.detection_direction_path:
            detection_direction_tensor = torch.load(
                self.detection_direction_path, map_location=self.pg_model.device
            )
            if detection_direction_tensor.dim() == 1:
                detection_direction_tensor = detection_direction_tensor.unsqueeze(0)
        else:
            detection_direction_tensor = None

        target_direction = direction_tensor.squeeze(0)
        # --- 1. Initial Setup ---
        force_text_gen = self.force_text_gen

        messages = append_suffix_to_user_role(copy.deepcopy(messages))

        # Setup for Prompt Guard Model (self.pg_model, self.pg_tokenizer)
        template_pg = self.pg_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self.pg_tokenizer.bos_token and template_pg.startswith(
            self.pg_tokenizer.bos_token
        ):
            template_pg = template_pg.replace(self.pg_tokenizer.bos_token, "")
        before_str_pg, after_str_pg = template_pg.split("{SUFFIX}")
        before_ids_pg = self.pg_tokenizer(
            [before_str_pg], padding=False, return_tensors="pt"
        )["input_ids"].to(self.pg_model.device, torch.int64)
        after_ids_pg = self.pg_tokenizer(
            [after_str_pg], add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.pg_model.device, torch.int64)
        self.before_embeds_pg, self.after_embeds_pg = [
            self.pg_embedding_layer(ids) for ids in (before_ids_pg, after_ids_pg)
        ]

        # Setup for text_gen Model (self.text_gen_, self.text_gen__tokenizer)
        target_str = " " + target if self.config.add_space_before_target else target
        template_text_gen = self.text_gen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self.text_gen_tokenizer.bos_token and template_text_gen.startswith(
            self.text_gen_tokenizer.bos_token
        ):
            template_text_gen = template_text_gen.replace(
                self.text_gen_tokenizer.bos_token, ""
            )
        before_str_text_gen, after_str_text_gen = template_text_gen.split("{SUFFIX}")
        before_ids_text_gen = self.text_gen_tokenizer(
            [before_str_text_gen], padding=False, return_tensors="pt"
        )["input_ids"].to(self.text_gen.device, torch.int64)
        after_ids_text_gen = self.text_gen_tokenizer(
            [after_str_text_gen], add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.text_gen.device, torch.int64)
        self.target_ids_text_gen = self.text_gen_tokenizer(
            [target_str], add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.text_gen.device, torch.int64)
        (
            self.before_embeds_text_gen,
            self.after_embeds_text_gen,
            self.target_embeds_text_gen,
        ) = [
            self.text_gen_embedding_layer(ids)
            for ids in (
                before_ids_text_gen,
                after_ids_text_gen,
                self.target_ids_text_gen,
            )
        ]

        # --- 2. Initialize Buffer with ID Pairs ---
        buffer = self.init_buffer()

        # Initialize lists for tracking progress, including individual losses
        losses, optim_strings, best_losses, text_gen_scores, pg_scores, steps = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        pg_losses, text_gen_losses = [], []

        # --- 3. Main Optimization Loop ---
        for step_num in tqdm(
            range(1, self.config.num_steps + 1), desc="Optimizing Suffix"
        ):
            # print the target string
            logger.info(f"Target string: {target_str}")

            # Step A: Get the best pair of token IDs from the buffer
            optim_ids_pg, optim_ids_text_gen = buffer.get_best_ids()

            # Optional: Logic to stop optimizing for benignity if a threshold is met
            do_benign = True
            if (
                len(pg_scores) > 0 and pg_scores[-1] > self.config.benignity_threshold
            ):  # Assuming threshold in config
                logger.info(
                    f"High benignity score ({pg_scores[-1]:.4f}) achieved. Focusing on text_gen objective."
                )
                do_benign = False

            # Step B: Determine the objective for generating new candidates
            if (
                (step_num // self.window_size) % 2 == 0
                and do_benign
                and not force_text_gen
            ):
                objective = "pg"
                ids_for_grad = optim_ids_pg
                current_tokenizer = self.pg_tokenizer
            else:
                objective = "text_gen"
                ids_for_grad = optim_ids_text_gen
                current_tokenizer = self.text_gen_tokenizer

            logger.info(f"\n\n\n--- Step {step_num} ---")
            logger.info(
                f"Step {step_num}: Generating candidates with '{objective}' objective."
            )

            # Step C: Generate new candidate suffixes using the gradient from the selected objective
            gradient = self.compute_token_gradient(ids_for_grad, objective=objective)

            with torch.no_grad():
                sampled_ids = sample_ids_from_grad(
                    ids_for_grad.squeeze(0),
                    gradient.squeeze(0),
                    self.config.search_width,
                    self.config.topk,
                    self.config.n_replace,
                    not_allowed_ids=self.not_allowed_ids if objective == "pg" else None,
                )

                # Step D: text_genuate all candidates to make a selection

                if not force_text_gen:
                    loss_objective = "combined"
                else:
                    loss_objective = "text_gen"
                loss_fn_wrapper = lambda bs, s_ids: self._compute_candidates_loss_pg(
                    bs, s_ids, objective=loss_objective
                )

                # Ensure sampled_ids are in the PG tokenizer space before passing to the loss function
                if objective == "text_gen":
                    text_gen_strings = self.text_gen_tokenizer.batch_decode(
                        sampled_ids, skip_special_tokens=True
                    )

                    ids_for_loss = self.pg_tokenizer(
                        text_gen_strings,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    ).input_ids.to(self.pg_model.device)
                    # ids_for_loss = self.text_gen_tokenizer(text_gen_strings, return_tensors='pt', padding=True, add_special_tokens=False).input_ids.to(self.pg_model.device)
                else:
                    ids_for_loss = sampled_ids

                loss = find_executable_batch_size(
                    loss_fn_wrapper, self.config.batch_size
                )(ids_for_loss)

                # Step E: Sync the best new candidate and update the buffer
                best_new_ids_objective = sampled_ids[loss.argmin()]
                best_new_string = current_tokenizer.decode(
                    best_new_ids_objective, skip_special_tokens=True
                )

                # Re-tokenize with BOTH tokenizers to create the new synchronized pair
                new_optim_ids_pg = self.pg_tokenizer(
                    [best_new_string], return_tensors="pt", add_special_tokens=False
                ).input_ids.to(self.pg_model.device)
                new_optim_ids_text_gen = self.text_gen_tokenizer(
                    [best_new_string], return_tensors="pt", add_special_tokens=False
                ).input_ids.to(self.text_gen.device)

                current_loss = loss.min().item()
                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, (new_optim_ids_pg, new_optim_ids_text_gen))

                # --- 4. Logging and Plotting ---
                # Re-text_genuate the best new suffix to get its individual loss components for plotting
                ids_for_plotting = new_optim_ids_pg
                pg_loss_for_plot = self._compute_candidates_loss_pg(
                    1, ids_for_plotting, objective="pg"
                ).item()
                text_gen_loss_for_plot = self._compute_candidates_loss_pg(
                    1, ids_for_plotting, objective="text_gen"
                ).item()
                pg_losses.append(pg_loss_for_plot)
                text_gen_losses.append(text_gen_loss_for_plot)

                optim_strings.append(best_new_string)
                steps.append(step_num)
                best_losses.append(buffer.get_lowest_loss())

                # Test and text_genuate scores
                test_output, cosine_sims, output_start = _test_prompt_output(
                    self.text_gen,
                    self.text_gen_tokenizer,
                    copy.deepcopy(messages),
                    best_new_string,
                    detection_direction_tensor,
                )
                text_gen_score = evaluate_response(
                    safety_model_override, safety_tokenizer_override, test_output
                )
                text_gen_scores.append(text_gen_score)
                pg_score_tensor = evaluate_with_prompt_guard(
                    self.pg_model, self.pg_tokenizer, messages, best_new_string
                )
                pg_scores.append(pg_score_tensor.squeeze().tolist()[0])

                # Console prints
                print(
                    f"Step {step_num}: Selection Loss={current_loss:.4f} | PG Loss={pg_loss_for_plot:.4f} | text_gen Loss={text_gen_loss_for_plot:.4f}"
                )

                logger.info(f"  Selected Suffix: '{best_new_string}'")
                logger.info(
                    f"  Eval Score: {text_gen_score:.4f} | PG Score: {pg_score_tensor.squeeze().tolist()[0]:.4f}"
                )
                logger.info(
                    f"  text_gen Loss: {text_gen_losses[-1]:.4f} | PG Loss: {pg_losses[-1]:.4f}"
                )
                logger.info(f"  Current Loss: {current_loss:.4f}")
                logger.info(f"  Best Suffix: '{best_new_string}'")

                logger.info(f"  Model Output: '{test_output}'")
                print(f"Best Suffix: '{best_new_string}'")
                print(f"Model Output: '{test_output}'")
                print("-" * 80)

                # log the cosine similarities list
                logger.info(f"Cosine similarities: {cosine_sims}")
                logger.info(f"Output start position: {output_start}")

                # Pass all relevant data to the plotting function
                _update_progress_plot(
                    steps,
                    pg_losses,
                    text_gen_losses,
                    pg_scores,
                    text_gen_scores,
                    title_suffix="(Target)",
                )

                # Early stopping if we have 5 text_gen scores below 0.00
                super_suffix_count = 0
                for i in range(len(text_gen_scores)):
                    if text_gen_scores[i] <= 0.00 and pg_scores[i] > 0.85:
                        super_suffix_count += 1

                if super_suffix_count >= self.super_early_stop:
                    self.stop_flag = True

                if self.stop_flag:
                    logger.info("Early stopping due to finding a perfect match.")
                    break

        # --- 5. Return Final Result ---
        min_loss_index = losses.index(min(losses))
        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
            evals=text_gen_scores,
        )
        return result

    def init_buffer(self) -> AttackBuffer:
        """
        Initializes and populates the AttackBuffer with starting prompts.

        It evaluates one or more initial strings (from `self.config.optim_str_init`),
        calculates their 'combined' loss, and adds them to the buffer.
        
        Returns:
            AttackBuffer: The initialized buffer containing the best initial prompts.
            
        Raises:
            ValueError: If the configured buffer_size is less than 1.
        """
        # --- ADD THIS CHECK ---
        # Ensure the buffer has a valid, positive size.
        if self.config.buffer_size < 1:
            raise ValueError(
                f"AttackBuffer size must be at least 1, but got {self.config.buffer_size}. "
                "Please check your configuration."
            )
        # ----------------------

        logger.info(f"Initializing attack buffer of size {self.config.buffer_size}...")
        buffer = AttackBuffer(self.config.buffer_size)

        # Ensure `init_strings` is a list, even if only one is provided.
        init_strings = self.config.optim_str_init
        if isinstance(init_strings, str):
            init_strings = [init_strings]

        logger.info(f"Evaluating {len(init_strings)} initial suffix(es)...")

        # Process and evaluate each initial string.
        for s_idx, init_str in enumerate(init_strings):
            
            # 1. Tokenize the string for the PG (Prompt Guard) model.
            ids_pg = self.pg_tokenizer(
                [init_str], return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.pg_model.device)

            # 2. Compute the combined (PG + TextGen) loss for this string.
            # We use `_compute_candidates_loss_pg` to evaluate this single candidate.
            loss = self._compute_candidates_loss_pg(
                self.config.batch_size, # Use configured batch size for evaluation
                sampled_ids=ids_pg, 
                objective="combined"
            ).item() # .item() extracts the scalar loss value

            # 3. Tokenize the string for the Text Generation model.
            # We store both tokenizations in the buffer.
            ids_text_gen = self.text_gen_tokenizer(
                [init_str], return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.text_gen.device)

            logger.info(
                f"  Initial Suffix #{s_idx + 1}: '{init_str}' -> Loss: {loss:.4f}"
            )
            
            # 4. Add the (loss, token_tuple) pair to the buffer.
            # The buffer will automatically manage its size and keep the best entries.
            buffer.add(loss, (ids_pg, ids_text_gen))

        # Log the final state of the buffer after initialization.
        buffer.log_buffer(self.pg_tokenizer)
        logger.info("Initialized attack buffer.")
        return buffer

    def _compute_candidates_loss_pg(self, search_batch_size: int, sampled_ids: Tensor, objective: str = 'combined') -> Tensor:
        """
        Computes the loss for a batch of candidate prompts (`sampled_ids`)
        across two separate objectives:
        
        1.  'pg' (Prompt Guard): A classification loss to measure "benignity".
        2.  'text_gen' (Target Generation): An auto-regressive loss for a specific task, in this case, latent state
        
        It processes the total batch in smaller mini-batches (`search_batch_size`)
        to avoid out-of-memory (OOM) errors.

        Args:
            search_batch_size (int): The size of mini-batches to use for evaluation.
            sampled_ids (Tensor): The full batch of candidate token IDs to evaluate.
                                  These IDs are from the `pg_tokenizer`.
            objective (str): The loss to return ('pg', 'text_gen', or 'combined').

        Returns:
            Tensor: A 1D tensor of loss values (one per candidate) based on the 
                    specified objective.
        """
        
        start_time = time.time() # Start the timer

        # Lists to store per-batch losses
        all_loss_pg = []
        all_loss_target = []

        # --- Tokenization Mismatch Handling ---
        # The two models (`pg_model` and `text_gen`) use different tokenizers.
        # `sampled_ids` are from `pg_tokenizer`. To evaluate with `text_gen`,
        # we must decode the PG-tokens into strings and re-encode them
        # using the `text_gen_tokenizer`.
        
        # 1. Decode tokens from `pg_tokenizer` into text strings
        candidate_strings = self.pg_tokenizer.batch_decode(sampled_ids, skip_special_tokens=True)
        
        # 2. Re-encode the text strings using `text_gen_tokenizer`
        sampled_ids_text_gen = self.text_gen_tokenizer(
            candidate_strings, 
            return_tensors="pt", 
            padding=True, 
            add_special_tokens=False
        ).input_ids.to(self.text_gen.device)
        
        # --- Process in Mini-Batches ---
        # Iterate over all candidates in chunks of `search_batch_size`
        for i in range(0, sampled_ids.shape[0], search_batch_size):
            # We run this in `no_grad` mode because we are *evaluating* candidates,
            # not backpropagating through the models here.
            with torch.no_grad():
                # Get the current mini-batch
                current_batch_size = sampled_ids[i:i + search_batch_size].shape[0]
                sampled_ids_batch_pg = sampled_ids[i:i + current_batch_size]
                sampled_ids_batch_text_gen = sampled_ids_text_gen[i:i + current_batch_size]

                # --- 1. Prompt Guard Loss (Benignity) ---
                
                # Get embeddings for the PG model
                optim_embeds_pg = self.pg_embedding_layer(sampled_ids_batch_pg)
                
                # Construct full input: [before] + [candidate] + [after]
                # .repeat() broadcasts the static embeds to match the mini-batch size
                input_embeds_pg = torch.cat([
                    self.before_embeds_pg.repeat(current_batch_size, 1, 1), 
                    optim_embeds_pg, 
                    self.after_embeds_pg.repeat(current_batch_size, 1, 1)
                ], dim=1)
                
                # Forward pass through the PG model
                outputs_pg = self.pg_model(inputs_embeds=input_embeds_pg)
                logits_pg = outputs_pg.logits
                
                # The target for the 'pg' (benignity) classifier is always class 0
                target_pg = torch.zeros(current_batch_size, device=self.pg_model.device, dtype=torch.long)
                
                # Calculate loss, `reduction="none"` gives us a loss for *each* item
                loss_pg = F.cross_entropy(logits_pg.float(), target_pg, reduction="none")
                all_loss_pg.append(loss_pg)
                
                # --- 2. Target Generation Loss (Task-Specific) ---
                
                # Get embeddings for the Text Gen model
                optim_embeds_text_gen = self.text_gen_embedding_layer(sampled_ids_batch_text_gen)
                
                # Construct full input: [before] + [candidate] + [after] + [target_sequence]
                input_embeds_text_gen = torch.cat([
                    self.before_embeds_text_gen.repeat(current_batch_size, 1, 1), 
                    optim_embeds_text_gen, 
                    self.after_embeds_text_gen.repeat(current_batch_size, 1, 1), 
                    self.target_embeds_text_gen.repeat(current_batch_size, 1, 1)
                ], dim=1)
                
                # Forward pass through the Text Gen model
                outputs_text_gen = self.text_gen(inputs_embeds=input_embeds_text_gen)
                logits_text_gen = outputs_text_gen.logits
                
                # Standard auto-regressive loss: align logits and labels
                shift = input_embeds_text_gen.shape[1] - self.target_ids_text_gen.shape[1]
                shift_logits = logits_text_gen[..., shift - 1 : -1, :].contiguous()
                shift_labels = self.target_ids_text_gen.repeat(current_batch_size, 1)
                
                # Calculate cross-entropy loss for all tokens, `reduction="none"`
                loss_target = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)).float(), shift_labels.view(-1), reduction="none")
                
                # Reshape token-level losses to (batch_size, seq_len) and average
                # over the sequence length (dim=-1) to get one loss value per sample
                loss_target = loss_target.view(current_batch_size, -1).mean(dim=-1)
                all_loss_target.append(loss_target)

                # --- Memory Cleanup ---
                # Aggressively clear memory after each mini-batch
                del outputs_pg, outputs_text_gen, logits_pg, logits_text_gen
                del optim_embeds_pg, optim_embeds_text_gen
                del input_embeds_pg, input_embeds_text_gen
                gc.collect()
                torch.cuda.empty_cache()

        # --- Aggregate and Return Loss ---
        
        # Combine the lists of mini-batch losses into full tensors
        loss_pg_full = torch.cat(all_loss_pg, dim=0)
        loss_target_full = torch.cat(all_loss_target, dim=0)

        # Log the time taken
        end_time = time.time()
        logger.info(f"_compute_candidates_loss_pg took {end_time - start_time:.2f} seconds to execute.")

        # Return the correct loss based on the current objective for selection
        if objective == 'pg':
            logger.info("Using Prompt Guard loss for candidate selection.")
            return loss_pg_full
        elif objective == 'text_gen':
            logger.info("Using Target Generation loss for candidate selection.")
            return loss_target_full
        else: # 'combined' or any other default
            logger.info("Using combined loss for candidate selection.")
            target_weight = getattr(self.config, 'target_loss_weight', 1.0)
            
            # Calculate the weighted combined loss: loss_pg + (loss_target * weight)
            # Using explicit torch functions for clarity.
            term_target = torch.mul(loss_target_full, float(target_weight))
            combined_loss = torch.add(loss_pg_full, term_target)
            return combined_loss

    def compute_token_gradient(self, optim_ids: Tensor, objective: str) -> Tensor:
        """
        Computes the gradient for a given objective ('pg' or 'text_gen').

        This function calculates the gradient of the loss with respect to a one-hot
        representation of the input tokens (`optim_ids`). This is a common technique
        in prompt optimization (like GCG) where we need a "soft" proxy for
        discrete tokens to use gradient descent.

        Args:
            optim_ids (Tensor): The tensor of token IDs to be optimized.
            objective (str): The optimization objective. Must be 'pg' or 'text_gen'.

        Returns:
            Tensor: The gradient of the loss with respect to the one-hot encoded optim_ids.
        """

        # 1. --- Select Model and Tensors based on Objective ---
        # This block switches between two different optimization goals:
        # 'pg': Prompt Guard model loss (e.g., benignity classifier).
        # 'text_gen': Standard auto-regressive text generation loss.
        if objective == "pg":
            model = self.pg_model
            embedding_layer = self.pg_embedding_layer
            before_embeds = self.before_embeds_pg
            after_embeds = self.after_embeds_pg
            target_embeds = torch.tensor(
                [], device=model.device
            )  # No target sequence for 'pg'
        elif objective == "text_gen":
            model = self.text_gen
            embedding_layer = self.text_gen_embedding_layer
            before_embeds = self.before_embeds_text_gen
            after_embeds = self.after_embeds_text_gen
            target_embeds = (
                self.target_embeds_text_gen
            )  # Target sequence for auto-regressive loss
        else:
            raise ValueError("Objective must be 'pg' or 'text_gen'")

        # 2. --- Create Differentiable One-Hot Representation ---
        # We can't backpropagate through discrete token IDs (optim_ids) directly.
        # Instead, we create a one-hot vector for each ID and make *it* differentiable.
        # This allows us to get a gradient for *every* token in the vocabulary,
        # indicating which token would most decrease the loss.

        # F.one_hot requires a tensor of type torch.long
        one_hot = F.one_hot(
            optim_ids.long(), num_classes=embedding_layer.num_embeddings
        )
        one_hot = one_hot.to(model.device, model.dtype)

        # Tell PyTorch to track gradients for this one-hot tensor
        one_hot.requires_grad_()

        # Perform a *differentiable* embedding lookup using matrix multiplication.
        # (one_hot @ embedding_layer.weight) is mathematically equivalent to
        # embedding_layer(optim_ids) but allows us to get the gradient w.r.t. `one_hot`.
        optim_embeds = one_hot @ embedding_layer.weight

        # 3. --- Construct Full Input Embeddings ---
        # Assemble the complete input sequence from its parts:
        # [context before] + [optimizable tokens] + [context after] + [target tokens]
        if objective == "pg":
            # For 'pg', there are no target embeddings
            input_embeds = torch.cat([before_embeds, optim_embeds, after_embeds], dim=1)
        else:  # 'text_gen'
            # For 'text_gen', we append the target sequence embeddings
            input_embeds = torch.cat(
                [before_embeds, optim_embeds, after_embeds, target_embeds], dim=1
            )

        # 4. --- Forward Pass ---
        # Get the model's output (logits) using the constructed embeddings
        output = model(inputs_embeds=input_embeds)
        logits = output.logits

        # 5. --- Calculate Loss ---
        if objective == "pg":
            # 'pg' objective: Treat the output as a classifier.
            # The loss is calculated against a fixed target class [0].
            target = torch.tensor([0], device=model.device, dtype=torch.long)
            loss = F.cross_entropy(logits, target)
        else:  # 'text_gen' objective
            # 'text_gen' objective: Standard auto-regressive (next-token prediction) loss.
            # We need to align the logits with the labels (shift them by one).

            # Calculate the starting index of the target tokens' logits
            shift = input_embeds.shape[1] - self.target_ids_text_gen.shape[1]

            # Select the logits that correspond to predictions for the target tokens
            shift_logits = logits[..., shift - 1 : -1, :].contiguous()
            shift_labels = self.target_ids_text_gen

            # Calculate the cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        # 6. --- Backward Pass (Compute Gradient) ---
        # Manually compute the gradient of the loss with respect to the one-hot tensor.
        # This `gradient` tensor will have the shape:
        # (batch_size, optim_ids_length, num_embeddings)
        # It shows how much the loss would change for each possible token in the
        # vocabulary at each position in `optim_ids`.
        (gradient,) = torch.autograd.grad(outputs=loss, inputs=one_hot)

        return gradient