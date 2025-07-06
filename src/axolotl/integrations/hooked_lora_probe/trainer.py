"""
Custom trainer for hallucination probe training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from typing import Dict, Optional, Any, Tuple

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
        1820,  # 'the'
        438,   # 'and' 
        285,   # 'is'
        258,   # 'in'
        998,   # 'to'
        1073,  # 'of'
        64,    # 'a'
        9210,  # 'that'
        275,   # 'it'
        4291,  # 'with'
    }
    
    # Create labels tensor with same shape as input_ids
    token_labels = torch.zeros_like(input_ids, dtype=torch.float)
    
    # Set labels to 1 for frequent tokens
    for token_id in frequent_token_ids:
        token_labels[input_ids == token_id] = 1.0
    
    return token_labels.float()

class HookedLoraProbeTrainer(AxolotlTrainer):
    """
    Custom trainer that combines language modeling loss with probe classification loss.
    
    This trainer computes a combined loss of:
    - Language modeling loss (next token prediction)
    - Probe classification loss (binary cross-entropy for hallucination detection)
    """
    
    def __init__(self, *args, **kwargs):
        print("HookedLoraProbeTrainer.__init__")
        print(f"kwargs: {kwargs.keys()}")
        print(f"args: {args}")
        print("-" * 100 + "\n")

        model = kwargs.pop("model")
        hooked_model = add_probe_head(model, {"probe_layer_idx": 30})
        kwargs["model"] = hooked_model

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
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log training metrics.
        
        Args:
            logs: Dict of metrics to log
        """
        # Add to trainer's log history
        if self.state.log_history:
            self.state.log_history[-1].update(logs)
        else:
            self.state.log_history.append(logs)
        
        # Print metrics every logging_steps
        if self.state.global_step % self.args.logging_steps == 0:
            print(f"Step {self.state.global_step}: {logs}")
        
        # Call parent log method
        super().log(logs)

    def _wrap_model(self, model, training=True, dataloader=None):
        print(f"HookedLoraProbeTrainer._wrap_model")
        return super()._wrap_model(model._model, training=training, dataloader=dataloader)