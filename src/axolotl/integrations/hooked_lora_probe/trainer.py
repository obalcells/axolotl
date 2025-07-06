"""
Custom trainer for hallucination probe training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from typing import Dict, Optional, Any, Tuple, Union
from transformers import PreTrainedModel
from peft import PeftModelForCausalLM
from deepspeed import DeepSpeedEngine

from axolotl.core.trainers.base import AxolotlTrainer

from .model import HookedModel, add_probe_head

def get_token_labels(input_ids: Int[Tensor, 'batch_size seq_len']) -> Float[Tensor, 'batch_size seq_len']:
    """
    Generate token labels for hallucination detection.
    
    Labels are 1 for frequent tokens that are likely to be hallucinated,
    0 for all other tokens.
    
    Args:
        input_ids: [batch_size, seq_len] token IDs
        
    Returns:
        token_labels: [batch_size, seq_len] binary labels
    """
    # Define frequent tokens that are likely to be hallucinated
    frequent_token_ids = {
        # 1820,  # 'the'
        # 438,   # 'and' 
        # 285,   # 'is'
        # 258,   # 'in'
        # 998,   # 'to'
        # 1073,  # 'of'
        # 64,    # 'a'
        # 9210,  # 'that'
        # 275,   # 'it'
        # 4291,  # 'with'
    }
    
    # Create labels tensor with same shape as input_ids
    token_labels = torch.zeros_like(input_ids, dtype=torch.float)
    
    # Set labels to 1 for frequent tokens
    for token_id in frequent_token_ids:
        token_labels[input_ids == token_id] = 1.0
    
    return token_labels

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
        self.lambda_lm = 0.0
        self.ignore_index = -100.0
        self.max_clipped_logits = 100.0
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute combined loss from language modeling and probe classification.
        
        Args:
            model: HallucinationProbeModel
            inputs: Dict containing input_ids, attention_mask, labels, token_labels
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (unused)
            
        Returns:
            Combined loss tensor, optionally with outputs
        """

        # print("-----------")
        # print(f"src.axolotl.integrations.hooked_lora_probe.trainer.HookedLoraProbeTrainer.compute_loss:99")
        # for i, group in enumerate(self.optimizer.param_groups):
        #     print(f"Parameter group {i}:")
        #     for j, param in enumerate(group['params']):
        #         print(f"  Parameter {j}: shape {param.shape}, requires_grad={param.requires_grad}")
        # print("-----------")

        # Extract inputs
        input_ids: Int[Tensor, 'batch_size seq_len'] = inputs["input_ids"]
        attention_mask: Int[Tensor, 'batch_size seq_len'] = inputs["attention_mask"]
        lm_labels: Int[Tensor, 'batch_size seq_len'] = inputs["labels"]  # For language modeling loss
        # token_labels: Float[Tensor, 'batch_size seq_len'] = inputs["token_labels"]  # For probe classification loss
        token_labels: Float[Tensor, 'batch_size seq_len'] = get_token_labels(input_ids)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
        )

        # TODO: Remove this
        self._debug_model_weights(model)

        # Get outputs
        lm_logits: Float[Tensor, 'batch_size seq_len vocab_size'] = outputs["lm_logits"]
        probe_logits: Float[Tensor, 'batch_size seq_len'] = outputs["probe_logits"].squeeze(-1)
        lm_loss: Float[Tensor, ''] = outputs.get("lm_loss", torch.tensor(0.0, device=input_ids.device))
        
        # Handle NaN in LM loss
        if torch.isnan(lm_loss):
            lm_loss = torch.tensor(0.0, device=input_ids.device)
        
        # Compute probe loss
        probe_loss = self._compute_probe_loss(probe_logits, token_labels, attention_mask)
        
        # Combine losses
        if self.lambda_lm > 0:
            total_loss = self.lambda_lm * lm_loss + (1 - self.lambda_lm) * probe_loss
        else:
            total_loss = probe_loss
        
        # Log metrics
        self.log({
            "train_loss": total_loss.item(),
            "train_lm_loss": lm_loss.item(),
            "train_probe_loss": probe_loss.item(),
        })
        
        if return_outputs:
            return total_loss, outputs
        return total_loss
    
    def _compute_probe_loss(
        self, 
        probe_logits: Float[Tensor, 'batch_size seq_len'], 
        token_labels: Float[Tensor, 'batch_size seq_len'], 
        attention_mask: Int[Tensor, 'batch_size seq_len']
    ) -> Float[Tensor, '']:
        """
        Compute binary cross-entropy loss for probe classification.
        
        Args:
            probe_logits: Raw logits from probe head
            token_labels: Target labels (0.0, 1.0, -100.0)
            attention_mask: Attention mask
            
        Returns:
            Probe classification loss
        """
        # Clip logits to prevent extreme values
        probe_logits_clipped = torch.clamp(
            probe_logits,
            min=-self.max_clipped_logits,
            max=self.max_clipped_logits
        )
        
        # Create mask for valid positions (not padding, not ignore_index)
        valid_mask = (token_labels != self.ignore_index) & (attention_mask == 1)
        
        if not valid_mask.any():
            # No valid positions, return zero loss
            return torch.tensor(0.0, device=probe_logits.device, requires_grad=True)
        
        try:
            # Compute BCE loss using the same pattern as probe_loss.py
            bce_loss = F.binary_cross_entropy_with_logits(
                probe_logits_clipped,
                token_labels,
                reduction='none'
            )
            
            # Apply mask and compute mean
            bce_loss = bce_loss[valid_mask].mean()
            
            # Check for NaN
            if torch.isnan(bce_loss):
                print(f"WARNING: NaN detected in probe BCE loss")
                bce_loss = torch.tensor(0.0, device=probe_logits.device)
            
            return bce_loss
            
        except Exception as e:
            print(f"Error in probe BCE loss calculation: {e}\nFallback to setting loss to 0.0")
            return torch.tensor(0.0, device=probe_logits.device)
    
    def prediction_step(
        self, 
        model, 
        inputs, 
        prediction_loss_only, 
        ignore_keys=None
    ) -> Tuple[Optional[Float[Tensor, '']], Optional[Float[Tensor, 'batch_size seq_len 1']], Optional[Float[Tensor, 'batch_size seq_len']]]:
        """
        Perform a prediction step.
        
        Args:
            model: HallucinationProbeModel
            inputs: Dict containing input_ids, attention_mask, labels, token_labels
            prediction_loss_only: Whether to return only the loss
            ignore_keys: Keys to ignore in outputs
            
        Returns:
            Tuple of (loss, logits, labels)
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
            
            # Return probe logits and token labels for evaluation
            probe_logits: Float[Tensor, 'batch_size seq_len 1'] = outputs["probe_logits"]
            # token_labels: Float[Tensor, 'batch_size seq_len'] = inputs["token_labels"]
            token_labels: Float[Tensor, 'batch_size seq_len'] = get_token_labels(inputs["input_ids"])
            
            return (loss, probe_logits, token_labels)
    
    def _wrap_model(self, model, training=True, dataloader=None):
        wrapped_model = super()._wrap_model(model, training=training, dataloader=dataloader)
        print("------------")
        print(f"HookedLoraProbeTrainer._wrap_model:272")
        print(f"type(model): {type(model)}")
        print(f"_wrap_model:276")
        print(f"wrapped_model: {type(wrapped_model)}")
        print("------------")
        return wrapped_model

    def _debug_model_weights(self, model: Union[PreTrainedModel, PeftModelForCausalLM, HookedModel, DeepSpeedEngine]):
        print(f"--------------------------------")
        print(f"src.axolotl.integrations.hooked_lora_probe.trainer.HookedLoraProbeTrainer._debug_model_weights:253")
        print(f"type(model): {type(model)}")
        print(f"model: {repr(model)[:300]}")

        if not isinstance(model, DeepSpeedEngine):
            print(f"linear_head weights: {model.probe_head.linear.weight}")
            print(f"model.model.layers.13.mlp.down_proj.lora_B.default.weight: {model._model.model.layers[13].mlp.down_proj.lora_B.default.weight}")
        else:
            print(f"(DeepSpeedEngine detected)")
            print(f"linear_head weights: {model.module.probe_head.linear.weight}")
            print(f"model.model.layers.13.mlp.down_proj.lora_B.default.weight: {model.module._model.model.layers[13].mlp.down_proj.lora_B.default.weight}")

        print(f"--------------------------------")