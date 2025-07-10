"""
Configuration arguments for Hallucination Probe Plugin
"""

from typing import Optional
from dataclasses import dataclass
from pydantic import BaseModel

class ProbeArgs(BaseModel):
    """Configuration arguments for Hallucination Probe Plugin."""

    hooked_lora_probe_enabled: bool = False
    probe_head_layer: int = -1  # Which layer to attach probe to (default: last layer)
    lambda_lm: float = 0.0  # Weight for language modeling loss (0.0 = only probe loss)
    anneal_max_aggr: bool = True  # Whether to anneal max_aggr loss
    anneal_warmup: float = 1.0  # Warmup for annealing max_aggr loss
    probe_threshold: float = 0.5  # Classification threshold for evaluation
    span_weighting: float = 10.0  # Weight for the tokens within annotated spans


@dataclass
class ProbeTrainingArgsMixin:
    """Configuration arguments for hallucination probe training."""
    
    # Probe configuration
    probe_head_layer: int = -1  # Which layer to attach probe to (default: last layer)

    # Loss configuration
    lambda_lm: float = 0.0  # Weight for language modeling loss (0.0 = only probe loss)
    anneal_max_aggr: bool = True  # Whether to anneal max_aggr loss
    anneal_warmup: float = 1.0  # Warmup for annealing max_aggr loss
    span_weighting: float = 10.0  # Weight for the tokens within annotated spans
    probe_threshold: float = 0.5  # Classification threshold for evaluation
