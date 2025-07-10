"""
Loss functions for hallucination probe training with span-based aggregation.
"""

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from typing import Optional
from transformers import AutoModelForCausalLM
from peft import PeftModel


def compute_vanilla_probe_loss(
    probe_logits: Float[Tensor, 'batch_size seq_len'],
    probe_labels: Float[Tensor, 'batch_size seq_len'],
    attention_mask: Int[Tensor, 'batch_size seq_len'],
    span_ids: Optional[Int[Tensor, 'batch_size seq_len']] = None,
    span_weighting: float = 1.0,
    ignore_label: float = -100.0,
) -> torch.Tensor:
    """
    Compute vanilla (token-level) probe loss.
    
    Args:
        probe_logits: Probe predictions [batch_size, seq_len]
        probe_labels: Token-level labels [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        span_ids: Span IDs for each token [batch_size, seq_len] (optional)
        span_weighting: Weighting scheme (optional)
        ignore_label: Label value to ignore
        
    Returns:
        Binary cross-entropy loss
    """
    # Create mask for valid tokens (attended and not ignored)
    mask = (attention_mask == 1) & (probe_labels != ignore_label)
    
    if mask.sum() == 0:
        return torch.tensor(0.0, device=probe_logits.device, dtype=probe_logits.dtype)

    weight = torch.ones_like(probe_labels)

    if span_weighting != 1.0:
        annotated_span_mask = mask & (span_ids != -100)
        weight[annotated_span_mask] = span_weighting
    
    # Get valid logits and labels
    valid_logits = probe_logits[mask]
    valid_labels = probe_labels[mask]
    weight = weight[mask]
    
    # Compute binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels, weight=weight, reduction='mean')
    
    # Check for NaN
    if torch.isnan(loss):
        print(f"WARNING: NaN detected in vanilla probe loss. Returning 0.0")
        loss = torch.tensor(0.0, device=probe_logits.device, dtype=probe_logits.dtype)
    
    return loss


def compute_max_span_aggregation_loss(
    probe_logits: Float[Tensor, 'batch_size seq_len'],
    probe_labels: Float[Tensor, 'batch_size seq_len'],
    span_ids: Int[Tensor, 'batch_size seq_len'],
    max_clipped_logits: float = 100.0,
    sparsity_penalty_weight: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute span-level max-aggregation loss using span_ids.
    
    For each span:
    - Take the maximum logit within the span
    - Compute BCE loss with the span's label (1.0 for hallucination, 0.0 for supported)
    
    Args:
        probe_logits: Probe predictions [batch_size, seq_len]
        probe_labels: Token-level labels [batch_size, seq_len]
        span_ids: Span IDs for each token [batch_size, seq_len]
                 -100 means ignore, 0+ means span ID
        max_clipped_logits: Maximum absolute value for logits
        sparsity_penalty_weight: Weight for sparsity penalty (optional)
        
    Returns:
        Aggregated loss tensor
    """
    device = probe_logits.device
    dtype = probe_logits.dtype
    
    # Clip logits to prevent extreme values
    probe_logits_clipped = torch.clamp(
        probe_logits,
        min=-max_clipped_logits,
        max=max_clipped_logits
    )
    
    span_losses = []
    
    batch_size, seq_len = probe_logits.shape[:2]
    
    for batch_idx in range(batch_size):
        batch_span_ids = span_ids[batch_idx]
        batch_labels = probe_labels[batch_idx]
        batch_logits = probe_logits_clipped[batch_idx]
        
        # Get unique span IDs (excluding ignore label)
        unique_span_ids = torch.unique(batch_span_ids)
        unique_span_ids = unique_span_ids[unique_span_ids != -100]
        
        for span_id in unique_span_ids:
            # Get positions for this span
            span_mask = (batch_span_ids == span_id)
            span_positions = torch.where(span_mask)[0]
            
            if len(span_positions) == 0:
                continue
                
            # Get labels for this span - they should all be the same
            span_labels = batch_labels[span_positions]
            
            # Skip if any token in span is ignored
            if (span_labels == -100.0).any():
                continue
                
            # Check that all labels in span are the same
            unique_labels = torch.unique(span_labels)
            if len(unique_labels) != 1:
                continue
                
            span_label = unique_labels[0].item()
            
            # Skip if span label is ignore
            if span_label == -100.0:
                continue
                
            # Get logits for this span and take maximum
            span_logits = batch_logits[span_positions]
            max_logit = torch.max(span_logits)
            
            # Compute BCE loss
            target = torch.tensor(span_label, device=device, dtype=dtype)
            loss = F.binary_cross_entropy_with_logits(max_logit, target, reduction='none')
            span_losses.append(loss)
    
    if not span_losses:
        # No valid spans found in the batch
        return torch.tensor(0.0, device=device, dtype=dtype)
    
    # Compute mean loss over all spans
    final_loss = torch.mean(torch.stack(span_losses))
    
    # Add sparsity penalty if specified
    if sparsity_penalty_weight is not None:
        sparsity_loss = compute_sparsity_loss(
            probe_logits=probe_logits,
            probe_labels=probe_labels,
        )
        final_loss = final_loss + sparsity_penalty_weight * sparsity_loss
    
    # Check for NaN
    if torch.isnan(final_loss):
        print(f"WARNING: NaN detected in span aggregation loss. Returning 0.0")
        final_loss = torch.tensor(0.0, device=device, dtype=dtype)
    
    return final_loss


def compute_sparsity_loss(
    probe_logits: Float[Tensor, 'batch_size seq_len'],
    probe_labels: Float[Tensor, 'batch_size seq_len'],
) -> torch.Tensor:
    """
    Penalize probe firing on tokens that are not hallucinations.
    
    Args:
        probe_logits: Probe predictions [batch_size, seq_len]
        probe_labels: Token-level labels [batch_size, seq_len]
        
    Returns:
        Sparsity penalty loss
    """
    # Find positions that are supported (label 0.0)
    mask = (probe_labels == 0.0)
    
    if mask.sum() == 0:
        return torch.tensor(0.0, device=probe_logits.device)
    
    # Apply sigmoid to get probabilities
    probe_probs = torch.sigmoid(probe_logits)
    
    # Penalize positive predictions on supported positions
    sparsity_probs = probe_probs[mask]
    sparsity_loss = torch.mean(sparsity_probs ** 2)
    
    return sparsity_loss


def compute_kl_penalty(
    model: AutoModelForCausalLM,
    input_ids: Float[Tensor, 'batch_size seq_len'],
    attention_mask: Float[Tensor, 'batch_size seq_len'],
    temperature: float = 1.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute KL divergence penalty between base model and LoRA-adapted model.
    
    Args:
        lm_model: The language model (potentially with LoRA adapters)
        input_ids: Input token IDs
        attention_mask: Attention mask
        temperature: Temperature for softmax (default: 1.0)
        reduction: How to reduce the KL divergence ('mean', 'sum', 'none')
        
    Returns:
        KL divergence penalty as a scalar tensor
    """
    device = input_ids.device
    
    # Get logits from base model (without adapters)
    try:
        with model.disable_adapter():
            base_logits: Float[Tensor, 'batch_size seq_len vocab_size'] = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            ).logits
    except:
        print(f"WARNING: Failed to get logits from base model. Returning 0.0")
        return torch.tensor(0.0, device=device)
    
    # Get logits from adapted model (with adapters)
    adapted_logits: Float[Tensor, 'batch_size seq_len vocab_size'] = lm_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
    ).logits
    
    # Apply temperature scaling
    base_logits = base_logits / temperature
    adapted_logits = adapted_logits / temperature
    
    # Convert to log probabilities
    base_log_probs = F.log_softmax(base_logits, dim=-1)
    adapted_log_probs = F.log_softmax(adapted_logits, dim=-1)
    
    # Compute KL divergence: KL(adapted || base)
    # KL(P||Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
    adapted_probs = torch.exp(adapted_log_probs)
    kl_div = adapted_probs * (adapted_log_probs - base_log_probs)
    
    # Sum over vocabulary dimension
    kl_div = kl_div.sum(dim=-1)  # Shape: [batch_size, seq_len]
    
    # Apply attention mask to ignore padded tokens
    if attention_mask is not None:
        kl_div = kl_div * attention_mask
        
    # Reduce according to specified reduction method
    if reduction == 'mean':
        if attention_mask is not None:
            # Mean over non-padded tokens
            kl_penalty = kl_div.sum() / attention_mask.sum()
        else:
            kl_penalty = kl_div.mean()
    elif reduction == 'sum':
        kl_penalty = kl_div.sum()
    elif reduction == 'none':
        kl_penalty = kl_div
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")
    
    # Check for NaN
    if torch.isnan(kl_penalty):
        print(f"WARNING: NaN detected in KL penalty. Returning 0.0")
        kl_penalty = torch.tensor(0.0, device=device)
    
    return kl_penalty