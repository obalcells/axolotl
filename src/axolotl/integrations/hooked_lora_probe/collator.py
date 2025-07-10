"""
Data collator for hooked LoRA probe training.
Simplified version based on axolotl KD collator.
"""

from dataclasses import dataclass
from typing import Any, Optional, Union, List, Dict

import torch
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from axolotl.utils.collators.batching import DataCollatorForSeq2Seq


@dataclass
class DataCollatorForProbe(DataCollatorForSeq2Seq):
    """
    Data collator for probe training with probe_labels.
    Handles standard padding for all fields including probe_labels.
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        assert self.tokenizer.padding_side == "right"
        if self.tokenizer.pad_token is None:
            print(f"tokenizer.pad_token is None, setting to eos_token {repr(self.tokenizer.eos_token)}")
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, torch.Tensor]:
        if return_tensors is None:
            return_tensors = self.return_tensors

        # Pad labels, probe_labels, and position_ids first (before tokenizer.pad)
        for feature_name, pad_token_id in [
            ("labels", self.label_pad_token_id),
            ("probe_labels", self.label_pad_token_id),
            ("span_ids", self.label_pad_token_id),
            ("position_ids", self.position_pad_token_id),
        ]:
            feat = (
                [feature[feature_name] for feature in features]
                if feature_name in features[0].keys()
                else None
            )
            
            if feat is not None:
                max_feature_length = max(len(l) for l in feat)
                if self.pad_to_multiple_of is not None:
                    max_feature_length = (
                        (max_feature_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                for feature in features:
                    remainder_len = max_feature_length - len(feature[feature_name])
                    if feature_name == "position_ids":
                        remainder = list(range(remainder_len))
                    else:
                        remainder = [pad_token_id] * remainder_len
                    
                    if isinstance(feature[feature_name], list):
                        feature[feature_name] = (
                            feature[feature_name] + remainder
                            if padding_side == "right"
                            else remainder + feature[feature_name]
                        )
                    elif padding_side == "right":
                        feature[feature_name] = np.concatenate(
                            [feature[feature_name], remainder]
                        ).astype(np.int64 if feature_name == "position_ids" else np.float32)
                    else:
                        feature[feature_name] = np.concatenate(
                            [remainder, feature[feature_name]]
                        ).astype(np.int64 if feature_name == "position_ids" else np.float32)

        # Use tokenizer padding for remaining fields
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        return features