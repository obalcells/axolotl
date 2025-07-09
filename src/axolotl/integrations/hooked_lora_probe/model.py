"""
Model modifications for hallucination probe training
"""

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from typing import List, Optional, Dict, Any
from transformers import PreTrainedModel
from abc import ABC
from peft import PeftModel, PeftModelForCausalLM
from collections.abc import Callable

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


class HookedModel(nn.Module):
    """Wrapper that hooks a linear head to a language model and delegates all calls to the underlying model."""
    
    def __init__(self, model: PreTrainedModel, probe_head_layer: int, hidden_size: int):
        super().__init__()

        self.model = model
        self.probe_head_layer = probe_head_layer
        self._hooked_activations = None
        self._probe_hook_handle = None

        self.probe_head = LinearHead(hidden_size, device=model.device, dtype=model.dtype)
        
        self._register_probe_hook()
    
    def _register_probe_hook(self):
        """Register forward hook to capture activations."""
        target_layer = self._get_target_layer(self._model)
        
        def hook_fn(module, input, output):
            # For transformer layers, output is typically (hidden_states, attention_weights)
            if isinstance(output, tuple):
                self._hooked_activations = output[0]  # Take hidden states
            else:
                self._hooked_activations = output
        
        self._probe_hook_handle = target_layer.register_forward_hook(hook_fn)
    
    def _get_target_layer(self, module: nn.Module) -> nn.Module:
        """Get the target layer for hooking."""
        if hasattr(module, 'layers'):
            # LLaMA-style models
            return module.layers[self.probe_head_layer]
        elif hasattr(module, 'transformer') and hasattr(module.transformer, 'h'):
            # GPT-style models
            return module.transformer.h[self.probe_head_layer]
        elif hasattr(module, 'base_model'):
            # PeftModel or a model with adapters
            return self._get_target_layer(module.base_model)
        elif hasattr(module, 'model'):
            return self._get_target_layer(module.model)
        else:
            raise ValueError(f"Unsupported model architecture: {type(module)}\nModule:\n{module}")
    
    def forward(self, input_ids: Int[Tensor, 'batch_size seq_len'], attention_mask: Optional[Int[Tensor, 'batch_size seq_len']] = None, 
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

        # Forward pass through the model
        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        # Get probe logits from hooked activations
        if self._hooked_activations is None:
            raise RuntimeError("No activations were hooked. Check model architecture.")
        
        probe_logits = self.probe_head(self._hooked_activations)
        
        result = {
            "lm_logits": outputs.logits,
            "probe_logits": probe_logits,
        }
        
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            result["lm_loss"] = outputs.loss
        
        return result
    
    def __setattr__(self, name, value):
        """Delegate attribute setting to the underlying model when appropriate."""
        # Always set our core attributes on self first
        our_attributes = {'_model', 'probe_head', 'probe_head_layer', '_hooked_activations', '_probe_hook_handle'}
        
        if name in our_attributes:
            # Always set our own core attributes on self
            super().__setattr__(name, value)
        elif not hasattr(self, 'model'):
            # During initialization, before model is set, set everything on self
            super().__setattr__(name, value)
        else:
            # After initialization, try to set on the underlying model if it has this attribute
            if hasattr(self._model, name):
                setattr(self._model, name, value)
            else:
                # Otherwise set on self
                super().__setattr__(name, value)

    def __getattr__(self, name):
        """Delegate ALL attribute access to the underlying model."""
        # This method is only called when the attribute is not found on self
        # By this point, self.model should always be set (after __init__)
        our_attributes = {'_model', 'probe_head', 'probe_head_layer', '_hooked_activations', '_probe_hook_handle'}

        print(f"HookedModel.__getattr__: {name}")

        if name in our_attributes:
            return super().__getattr__(name)
        
        try:
            return getattr(self._model, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __call__(self, *args, **kwargs):
        """Ensure calling the object uses our forward method."""
        return self.forward(*args, **kwargs)
    
    def __repr__(self):
        """Provide a helpful representation."""
        return f"HookedModel(\n  model={repr(self._model)},\n  probe_head_layer={self.probe_head_layer}\n)"
    
    def __str__(self):
        """Provide a helpful string representation."""
        return f"HookedModel wrapping {type(self._model).__name__} with probe at layer {self.probe_head_layer}"
    
    @property
    def device(self):
        """Delegate device property to the underlying model."""
        return self._model.device
    
    @property
    def dtype(self):
        """Delegate dtype property to the underlying model."""
        return self._model.dtype
    
    def to(self, *args, **kwargs):
        """Override to method to move both model and probe head."""
        self._model = self._model.to(*args, **kwargs)
        self.probe_head = self.probe_head.to(*args, **kwargs)
        return self
    
    def train(self, mode=True):
        """Override train method to set both model and probe head."""
        self._model.train(mode)
        self.probe_head.train(mode)
        return self
    
    def eval(self):
        """Override eval method to set both model and probe head."""
        self._model.eval()
        self.probe_head.eval()
        return self
    
    def parameters(self, recurse=True):
        """Return parameters from both model and probe head."""
        for param in self._model.parameters(recurse=recurse):
            yield param
        for param in self.probe_head.parameters(recurse=recurse):
            yield param
    
    def named_parameters(self, prefix='', recurse=True):
        """Return named parameters from both model and probe head."""
        probe_prefix = f"{prefix}probe_head." if prefix else "probe_head."
        
        # Return model parameters without additional prefix
        print(f"HookedModel.named_parameters: prefix={prefix}")
        for name, param in self._model.named_parameters(prefix=prefix, recurse=recurse):
            # print(f"HookedModel.named_parameters: yielding {repr(name)}")
            yield name, param
        # Return probe head parameters with probe_head prefix
        for name, param in self.probe_head.named_parameters(prefix=probe_prefix[:-1], recurse=recurse):
            yield name, param
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Return state dict including both model and probe head."""
        if destination is None:
            destination = {}

        # print(f"HookedModel.state_dict: prefix={prefix}")
        # print(f"HookedModel.state_dict: keep_vars={keep_vars}")

        # Get model state dict without additional prefix
        model_state = self._model.state_dict(prefix=prefix, keep_vars=keep_vars)
        destination.update(model_state)
        
        # Get probe head state dict with probe_head prefix
        probe_state = self.probe_head.state_dict(prefix=f"{prefix}probe_head.", keep_vars=keep_vars)
        destination.update(probe_state)

        # print(f"destination.keys(): {destination.keys()}")
        
        return destination
    
    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load state dict for both model and probe head."""
        # Separate model and probe head state dicts
        model_state = {}
        probe_state = {}

        print(f"--------------------------------")
        print(f"HookedModel.load_state_dict: state_dict.keys(): {repr(state_dict.keys())[:1_000]}")
        print(f"--------------------------------")
        
        for key, value in state_dict.items():
            if key.startswith('probe_head.'):
                probe_state[key[11:]] = value  # Remove 'probe_head.' prefix
            else:
                # All other keys are assumed to be model parameters
                model_state[key] = value
        
        # Load state dicts
        if model_state:
            self._model.load_state_dict(model_state, strict=strict)
        if probe_state:
            self.probe_head.load_state_dict(probe_state, strict=strict)

    def __del__(self):
        """Clean up hook on destruction."""
        if hasattr(self, '_probe_hook_handle') and self._probe_hook_handle is not None:
            self._probe_hook_handle.remove()

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self._model.gradient_checkpointing_enable(*args, **kwargs)


def add_probe_head(model: PreTrainedModel, cfg: dict) -> HookedModel:
    """
    Add probe head to a pre-trained model.
    
    Args:
        model: Pre-trained model (potentially with LoRA adapters)
        cfg: Configuration object
        
    Returns:
        HallucinationProbeModel with probe head attached
    """
    # Determine probe layer
    probe_head_layer = cfg.get("probe_head_layer", None)
    if probe_head_layer is None:
        # Default to last layer - 1 (since we want intermediate representations)
        if hasattr(model, 'config'):
            if hasattr(model.config, 'num_hidden_layers'):
                probe_head_layer = model.config.num_hidden_layers - 1
            elif hasattr(model.config, 'n_layer'):
                probe_head_layer = model.config.n_layer - 1
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
    
    # Create probe model
    probe_model = HookedModel(
        model=model,
        probe_head_layer=probe_head_layer,
        hidden_size=hidden_size
    )

    return probe_model

# Register HookedModel as a virtual subclass of PeftModelForCausalLM
# This allows isinstance(hooked_model, PeftModelForCausalLM) to return True
# without requiring multiple inheritance
# PeftModelForCausalLM.register(HookedModel)