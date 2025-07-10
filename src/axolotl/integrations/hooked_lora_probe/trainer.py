"""
Custom trainer for hallucination probe training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from jaxtyping import Float, Int
from torch import Tensor
from typing import Dict, Optional, Any, Tuple, Union
from transformers import PreTrainedModel
from peft import PeftModelForCausalLM
from deepspeed import DeepSpeedEngine

from axolotl.core.trainers.base import AxolotlTrainer

from .losses import compute_vanilla_probe_loss, compute_max_span_aggregation_loss
from .model import HookedModel
from .eval_utils import create_probe_compute_metrics 


class HookedLoraProbeTrainer(AxolotlTrainer):
    """
    Custom trainer that combines language modeling loss with probe classification loss.
    
    This trainer computes a combined loss of:
    - Language modeling loss (next token prediction)
    - Probe classification loss (binary cross-entropy for hallucination detection)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Extract probe-specific configuration
        self.lambda_lm = self.args.lambda_lm
        self.anneal_max_aggr = self.args.anneal_max_aggr
        self.anneal_warmup = self.args.anneal_warmup
        self.span_weighting = self.args.span_weighting
        self.probe_threshold = self.args.probe_threshold
        # assert self.args.batch_eval_metrics == False, "batch_eval_metrics must be False for this trainer"
        # self.compute_metrics = create_probe_compute_metrics(probe_threshold=self.probe_threshold)

    def get_training_progress(self) -> float:
        """Get the current training progress as a float between 0 and 1."""
        if self.state.max_steps is None or self.state.max_steps == 0:
            return 1.0
        return min(1.0, self.state.global_step / self.state.max_steps)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute combined loss from language modeling and probe classification.
        
        Args:
            model: HallucinationProbeModel
            inputs: Dict containing input_ids, attention_mask, labels, probe_labels
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (unused)
            
        Returns:
            Combined loss tensor, optionally with outputs
        """
        input_ids: Int[Tensor, 'batch_size seq_len'] = inputs["input_ids"]
        attention_mask: Int[Tensor, 'batch_size seq_len'] = inputs["attention_mask"]
        lm_labels: Int[Tensor, 'batch_size seq_len'] = inputs["labels"]  # For language modeling loss
        probe_labels: Float[Tensor, 'batch_size seq_len'] = inputs["probe_labels"]  # For probe classification loss
        span_ids: Int[Tensor, 'batch_size seq_len'] = inputs["span_ids"]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
        )

        lm_logits: Float[Tensor, 'batch_size seq_len vocab_size'] = outputs["lm_logits"]
        probe_logits: Float[Tensor, 'batch_size seq_len'] = outputs["probe_logits"].squeeze(-1)
        lm_loss: Float[Tensor, ''] = outputs.get("lm_loss", torch.tensor(0.0, device=input_ids.device))
        
        if torch.isnan(lm_loss):
            lm_loss = torch.tensor(0.0, device=input_ids.device)

        vanilla_probe_loss = compute_vanilla_probe_loss(
            probe_logits,
            probe_labels,
            attention_mask,
            span_ids=span_ids,
            span_weighting=self.span_weighting,
        )
        
        if self.anneal_max_aggr:
            omega = min(1.0, self.get_training_progress() / self.anneal_warmup)
            max_aggr_loss = compute_max_span_aggregation_loss(
                probe_logits,
                probe_labels,
                span_ids,
            )
            probe_loss = vanilla_probe_loss * (1 - omega) + max_aggr_loss * omega
        else:
            omega = 0.0
            max_aggr_loss = torch.tensor(0.0)
            probe_loss = vanilla_probe_loss

        # Combine losses
        total_loss = self.lambda_lm * lm_loss + probe_loss
        
        self.log({
            "train_loss": float(total_loss.item()),
            "train_lm_loss": float(lm_loss.item()),
            "train_probe_loss": float(probe_loss.item()),
            "train_vanilla_probe_loss": float(vanilla_probe_loss.item()),
            "train_max_aggr_loss": float(max_aggr_loss.item()),
            "train_omega": omega,
        })
        
        if return_outputs:
            return total_loss, outputs
        return total_loss
    
    def prediction_step(
        self, 
        model, 
        inputs, 
        prediction_loss_only, 
        ignore_keys=None
    ) -> Tuple[Optional[Float[Tensor, '']], Optional[Float[Tensor, 'batch_size seq_len 1']], Optional[Tuple[Float[Tensor, 'batch_size seq_len'], Int[Tensor, 'batch_size seq_len'], Int[Tensor, 'batch_size seq_len']]]]:
        """
        Perform a prediction step.
        
        Args:
            model: HallucinationProbeModel
            inputs: Dict containing input_ids, attention_mask, labels, probe_labels
            prediction_loss_only: Whether to return only the loss
            ignore_keys: Keys to ignore in outputs
            
        Returns:
            Tuple of (loss, logits, labels) where labels includes probe_labels, attention_mask, and span_ids for evaluation
        """
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
            
            # Compute loss
            loss = self.compute_loss(model, inputs)
            
            if prediction_loss_only:
                return (loss, None, None)
            
            # Return probe logits and structured labels for evaluation
            probe_logits: Float[Tensor, 'batch_size seq_len 1'] = outputs["probe_logits"]
            probe_labels: Float[Tensor, 'batch_size seq_len'] = inputs["probe_labels"]
            attention_mask: Int[Tensor, 'batch_size seq_len'] = inputs["attention_mask"]
            span_ids: Int[Tensor, 'batch_size seq_len'] = inputs["span_ids"]
            
            # Package labels as tuple for eval_utils
            # TODO: Make sure we don't break the callback if we change this
            structured_labels = (probe_labels, attention_mask, span_ids)
            
            return (loss, probe_logits, structured_labels)
    
    def _get_probe_head_from_model(self, model: Union[PreTrainedModel, PeftModelForCausalLM, HookedModel, DeepSpeedEngine]):
        """Extract probe head from model, handling different model types."""
        if isinstance(model, DeepSpeedEngine):
            # DeepSpeed wraps the model
            base_model = model.module
        else:
            base_model = model
            
        # Handle PeftModel wrapping
        if hasattr(base_model, 'base_model'):
            # PeftModel structure: model.base_model.model.probe_head
            return base_model.base_model.model.probe_head
        else:
            # Direct model structure: model.probe_head
            return base_model.probe_head
    
    def _save_probe_head(self, output_dir: str, model: Union[PreTrainedModel, PeftModelForCausalLM, HookedModel, DeepSpeedEngine]):
        """Save probe head weights to output directory."""
        try:
            probe_head = self._get_probe_head_from_model(model)
            probe_head_path = os.path.join(output_dir, "probe_head.bin")
            
            # Save probe head state dict
            torch.save(probe_head.state_dict(), probe_head_path)
            print(f"Saved probe head weights to {probe_head_path}")
            
        except Exception as e:
            print(f"Failed to save probe head weights: {e}")
    
    def _save_checkpoint(self, model, trial, **kwargs):
        """Override to save probe head with each checkpoint."""
        # Call parent method to save standard checkpoint
        checkpoint_folder = super()._save_checkpoint(model, trial, **kwargs)
        
        # Save probe head weights
        if checkpoint_folder:
            self._save_probe_head(checkpoint_folder, model)
            
        return checkpoint_folder
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Override to save probe head with final model."""
        # Call parent method to save standard model
        super()._save(output_dir, state_dict)
        
        # Save probe head weights
        if output_dir:
            self._save_probe_head(output_dir, self.model)
    
    def _debug_model_weights(self, model: Union[PreTrainedModel, PeftModelForCausalLM, HookedModel, DeepSpeedEngine]):
        print(f"--------------------------------")
        print(f"src.axolotl.integrations.hooked_lora_probe.trainer.HookedLoraProbeTrainer._debug_model_weights:253")
        print(f"type(model): {type(model)}")
        print(f"model: {repr(model)[:300]}")
        if not isinstance(model, DeepSpeedEngine):
            print(f"linear_head weights: {model.base_model.model.probe_head.linear.weight}")
            print(f"model.model.layers.13.mlp.down_proj.lora_B.default.weight: {model.base_model.model.model.layers[13].mlp.down_proj.lora_B.default.weight}")
        else:
            print(f"(DeepSpeedEngine detected)")
            print(f"linear_head weights: {model.module.base_model.model.probe_head.linear.weight}")
            print(f"model.model.layers.13.mlp.down_proj.lora_B.default.weight: {model.module.base_model.model.model.layers[13].mlp.down_proj.lora_B.default.weight}")
        print(f"--------------------------------")