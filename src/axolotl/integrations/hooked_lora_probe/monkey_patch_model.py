"""
Model transformation approach for hallucination probe training.
This approach transforms an existing model in-place rather than wrapping it.
"""

import torch
import torch.nn as nn
import os
from jaxtyping import Float, Int
from torch import Tensor
from typing import List, Optional, Dict, Any
from transformers import PreTrainedModel


class LinearHead(nn.Module):
    """Linear probe head."""
    
    def __init__(self, hidden_size: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, device=device, dtype=dtype)
        
        # Initialize with small weights
        with torch.no_grad():
            self.linear.weight.data.normal_(mean=0.0, std=0.01)
            if self.linear.bias is not None:
                self.linear.bias.data.zero_()
    
    def forward(self, hidden_states: Float[Tensor, 'batch_size seq_len hidden_size']) -> Float[Tensor, 'batch_size seq_len 1']:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            logits: [batch_size, seq_len, 1]
        """
        return self.linear(hidden_states)


def _get_target_layer(module: nn.Module, probe_layer_idx: int) -> nn.Module:
    """Get the target layer for hooking."""
    if hasattr(module, 'layers'):
        # LLaMA-style models
        return module.layers[probe_layer_idx]
    elif hasattr(module, 'transformer') and hasattr(module.transformer, 'h'):
        # GPT-style models
        return module.transformer.h[probe_layer_idx]
    elif hasattr(module, 'base_model'):
        # PeftModel or a model with adapters
        return _get_target_layer(module.base_model, probe_layer_idx)
    elif hasattr(module, 'model'):
        return _get_target_layer(module.model, probe_layer_idx)
    else:
        raise ValueError(f"Unsupported model architecture: {type(module)}\nModule:\n{module}")


def _register_probe_hook(model):
    """Register forward hook to capture activations."""
    target_layer = _get_target_layer(model, model.probe_layer_idx)
    
    def hook_fn(module, input, output):
        # For transformer layers, output is typically (hidden_states, attention_weights)
        if isinstance(output, tuple):
            model._hooked_activations = output[0]  # Take hidden states
        else:
            model._hooked_activations = output
    
    model._probe_hook_handle = target_layer.register_forward_hook(hook_fn)


def _hooked_forward(self, input_ids: Int[Tensor, 'batch_size seq_len'], attention_mask: Optional[Int[Tensor, 'batch_size seq_len']] = None, 
            labels: Optional[Int[Tensor, 'batch_size seq_len']] = None, **kwargs) -> Dict[str, Float[Tensor, '...']]:
    """
    Forward pass that returns both LM and probe outputs.
    
    Args:
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        labels: [batch_size, seq_len] - LM labels for next token prediction
        **kwargs: Additional arguments
        
    Returns:
        Dict containing:
            - lm_logits: [batch_size, seq_len, vocab_size]
            - probe_logits: [batch_size, seq_len, 1]
            - lm_loss: Language modeling loss (if labels provided)
    """
    # Reset hooked activations
    self._hooked_activations = None

    # Forward pass through the model using the original forward method
    outputs = self._original_forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        **kwargs
    )
    
    # Get probe logits from hooked activations
    if self._hooked_activations is None:
        raise RuntimeError("No activations were hooked. Check model architecture.")
    
    probe_logits: Float[Tensor, 'batch_size seq_len 1'] = self.probe_head(self._hooked_activations)
    
    result = {
        "lm_logits": outputs.logits,
        "probe_logits": probe_logits,
    }
    
    if hasattr(outputs, 'loss') and outputs.loss is not None:
        result["lm_loss"] = outputs.loss
    
    return result


def _cleanup_hook(self):
    """Clean up hook on destruction."""
    if hasattr(self, '_probe_hook_handle') and self._probe_hook_handle is not None:
        self._probe_hook_handle.remove()
        self._probe_hook_handle = None


def _load_legacy_probe_head_weights(probe_head: nn.Module, probe_head_path: str) -> bool:
    """
    Load probe head weights from legacy format (weight, bias keys).
    TODO: Remove this function once all probes are migrated to new format.
    
    Args:
        probe_head: The probe head module to load weights into
        probe_head_path: Path to the probe_head.bin file
        
    Returns:
        True if weights were loaded successfully, False otherwise
    """
    if not os.path.exists(probe_head_path):
        return False
    
    try:
        probe_state_dict = torch.load(probe_head_path, map_location='cpu')
        
        # Convert legacy format (weight, bias) to new format (linear.weight, linear.bias)
        if "weight" in probe_state_dict and "bias" in probe_state_dict:
            converted_state_dict = {
                "linear.weight": probe_state_dict["weight"],
                "linear.bias": probe_state_dict["bias"]
            }
            probe_head.load_state_dict(converted_state_dict)
            print(f"Loaded probe head weights from {probe_head_path} (legacy format)")
            return True
        
        return False
    except Exception as e:
        print(f"Failed to load legacy probe head weights from {probe_head_path}: {e}")
        return False


def _load_probe_head_weights(probe_head: nn.Module, probe_head_path: str) -> bool:
    """
    Load probe head weights from a .bin file.
    
    Args:
        probe_head: The probe head module to load weights into
        probe_head_path: Path to the probe_head.bin file
        
    Returns:
        True if weights were loaded successfully, False otherwise
    """
    if not os.path.exists(probe_head_path):
        return False
    
    try:
        probe_state_dict = torch.load(probe_head_path, map_location='cpu')
        probe_head.load_state_dict(probe_state_dict)
        print(f"Loaded probe head weights from {probe_head_path}")
        return True
    except Exception as e:
        print(f"Failed to load probe head weights from {probe_head_path}: {e}")
        # Try legacy format as fallback
        return _load_legacy_probe_head_weights(probe_head, probe_head_path)


def convert_to_hooked_model(model: PreTrainedModel, probe_layer_idx: int, hidden_size: Optional[int] = None, 
                           cfg: Optional[dict] = None) -> PreTrainedModel:
    """
    Convert an existing model to a hooked model by transforming it in-place.
    
    Args:
        model: Pre-trained model (potentially with LoRA adapters)
        probe_layer_idx: Layer index to hook for probe
        hidden_size: Hidden size for probe head (auto-detected if None)
        cfg: Configuration dict that may contain lora_model_dir or resume_from_checkpoint
        
    Returns:
        The same model object, but transformed to include probe functionality
    """
    # Auto-detect hidden size if not provided
    if hidden_size is None:
        hidden_size = model.config.hidden_size
        
        # Handle special cases for different model architectures
        if hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'hidden_size'):
            # Some models like Gemma have nested config
            hidden_size = model.config.text_config.hidden_size
    
    # Store original forward method
    model._original_forward = model.forward
    
    # Add probe-specific attributes
    model.probe_layer_idx = probe_layer_idx
    model._hooked_activations = None
    model._probe_hook_handle = None
    model.probe_head = LinearHead(hidden_size, device=model.device, dtype=model.dtype)
    
    # Try to load probe head weights from various locations
    probe_head_loaded = False
    if cfg:
        # Check lora_model_dir first
        if cfg.get("lora_model_dir"):
            probe_head_path = os.path.join(cfg["lora_model_dir"], "probe_head.bin")
            probe_head_loaded = _load_probe_head_weights(model.probe_head, probe_head_path)
        
        # If not found in lora_model_dir, check resume_from_checkpoint
        if not probe_head_loaded and cfg.get("resume_from_checkpoint"):
            probe_head_path = os.path.join(cfg["resume_from_checkpoint"], "probe_head.bin")
            probe_head_loaded = _load_probe_head_weights(model.probe_head, probe_head_path)
    
    if not probe_head_loaded:
        print("No probe head weights found, using random initialization")
    
    # Replace forward method with hooked version
    model.forward = _hooked_forward.__get__(model, model.__class__)
    
    # Add cleanup method
    model._cleanup_hook = _cleanup_hook.__get__(model, model.__class__)
    
    # Register hook
    _register_probe_hook(model)
    
    # Override __del__ to clean up hook
    original_del = getattr(model.__class__, '__del__', None)
    
    def new_del(self):
        self._cleanup_hook()
        if original_del is not None:
            original_del(self)
    
    model.__class__.__del__ = new_del
    
    return model


def restore_original_model(model: PreTrainedModel) -> PreTrainedModel:
    """
    Restore a hooked model back to its original state.
    
    Args:
        model: Hooked model to restore
        
    Returns:
        The same model object, but restored to original state
    """
    if not hasattr(model, '_original_forward'):
        raise ValueError("Model does not appear to be a hooked model")
    
    # Clean up hook
    if hasattr(model, '_cleanup_hook'):
        model._cleanup_hook()
    
    # Restore original forward method
    model.forward = model._original_forward
    
    # Remove probe-specific attributes
    if hasattr(model, 'probe_head'):
        delattr(model, 'probe_head')
    if hasattr(model, 'probe_layer_idx'):
        delattr(model, 'probe_layer_idx')
    if hasattr(model, '_hooked_activations'):
        delattr(model, '_hooked_activations')
    if hasattr(model, '_probe_hook_handle'):
        delattr(model, '_probe_hook_handle')
    if hasattr(model, '_original_forward'):
        delattr(model, '_original_forward')
    if hasattr(model, '_cleanup_hook'):
        delattr(model, '_cleanup_hook')
    
    return model


def add_probe_head_transform(model: PreTrainedModel, cfg: dict) -> PreTrainedModel:
    """
    Add probe head to a pre-trained model using transformation approach.
    
    Args:
        model: Pre-trained model (potentially with LoRA adapters)
        cfg: Configuration object
        
    Returns:
        The same model object, but transformed to include probe functionality
    """
    # Determine probe layer
    probe_layer_idx = cfg.get("probe_layer_idx")
    if probe_layer_idx is None:
        # Default to last layer - 1 (since we want intermediate representations)
        if hasattr(model, 'config'):
            if hasattr(model.config, 'num_hidden_layers'):
                probe_layer_idx = model.config.num_hidden_layers - 1
            elif hasattr(model.config, 'n_layer'):
                probe_layer_idx = model.config.n_layer - 1
            else:
                raise ValueError("Cannot determine number of layers in model")
        else:
            raise ValueError("Model does not have config attribute")
    
    # Get hidden size
    hidden_size = model.config.hidden_size
    
    # Handle special cases for different model architectures
    if hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'hidden_size'):
        # Some models like Gemma have nested config
        hidden_size = model.config.text_config.hidden_size
    
    # Transform the model
    return convert_to_hooked_model(
        model=model,
        probe_layer_idx=probe_layer_idx,
        hidden_size=hidden_size,
        cfg=cfg
    )