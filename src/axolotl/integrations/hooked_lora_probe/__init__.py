"""
Hallucination Probe Plugin for Axolotl

This plugin enables training hallucination detection probes using the Axolotl framework.
It implements a custom trainer that combines standard language modeling loss with
a binary classification loss from a linear probe head attached to intermediate layers.
"""

from typing import Optional

from axolotl.integrations.base import BasePlugin

from .model import HookedModel
from .args import HookedLoraProbeArgs # pylint: disable=unused-import. # noqa: F401

class HookedLoraProbePlugin(BasePlugin):
    """
    Plugin for training hallucination detection probes.
    
    This plugin:
    1. Adds a linear probe head to an intermediate layer
    2. Implements custom trainer with combined LM + probe loss
    3. Supports LoRA adapters for parameter-efficient training
    4. Handles datasets with token-level hallucination labels
    """
    
    def get_input_args(self):
        return "axolotl.integrations.hooked_lora_probe.HookedLoraProbeArgs"
    
    def get_trainer_cls(self, cfg):
        """Return custom trainer class for hallucination probe training."""
        if cfg.hooked_lora_probe_enabled:
            from .trainer import HookedLoraProbeTrainer
            return HookedLoraProbeTrainer
        return None

    def get_training_args(self, cfg):
        return {
            "label_names": ["probe_labels", "labels"],
        }
    
    def get_collator_cls_and_kwargs(self, cfg, is_eval=False):
        """Return custom data collator for probe training."""
        from .collator import DataCollatorForProbe
        return DataCollatorForProbe, {}
    
    def pre_lora_load(self, cfg, model) -> Optional[HookedModel]:
        """Add probe head to the model after loading."""
        if cfg.hooked_lora_probe_enabled:
            from .model import add_probe_head
            hooked_model = add_probe_head(model, cfg)
            return hooked_model
        return None

    def post_lora_load(self, cfg, model):
        if cfg.hooked_lora_probe_enabled:
            model.probe_head.requires_grad_(True)

    def add_callbacks_post_trainer(self, cfg, trainer):
        """Add probe-specific callbacks."""
        if cfg.hooked_lora_probe_enabled:
            from .callbacks import HookedLoraProbeEvaluationCallback
            callback = HookedLoraProbeEvaluationCallback(
                probe_threshold=cfg.probe_threshold,
                eval_steps=cfg.probe_eval_steps,
            )
            return [callback]
        return []