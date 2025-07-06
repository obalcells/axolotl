"""
Configuration arguments for Hallucination Probe Plugin
"""

from typing import Optional, List, Union
from pydantic import BaseModel, field_validator


class HookedLoraProbeArgs(BaseModel):
    """Configuration arguments for hallucination probe training."""
    
    # Enable/disable probe training
    hooked_lora_probe_enabled: Optional[bool] = False
    
    # Probe configuration
    probe_layer_idx: Optional[int] = None  # Which layer to attach probe to (default: last layer)
    probe_threshold: float = 0.5  # Classification threshold for evaluation
    
    # Loss configuration
    lambda_lm: float = 0.0  # Weight for language modeling loss (0.0 = only probe loss)
    
    # # Dataset configuration - expects 'token_labels' field with values {0.0, 1.0, -100.0}
    # token_labels_field: str = "token_labels"
    
    @field_validator("probe_threshold")
    @classmethod
    def validate_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("probe_threshold must be between 0.0 and 1.0")
        return v
    
    @field_validator("lambda_lm")
    @classmethod
    def validate_lambda_lm(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("lambda_lm must be between 0.0 and 1.0")
        return v