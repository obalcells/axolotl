"""
Callbacks for hallucination probe training
"""

import torch
import numpy as np
from jaxtyping import Float, Int
from torch import Tensor
from typing import Dict, Optional, Any
from transformers import TrainerCallback, TrainerState, TrainerControl
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score



class HookedLoraProbeEvaluationCallback(TrainerCallback):
    """
    Callback for evaluating probe performance during training.
    
    Computes classification metrics like accuracy, precision, recall, F1, and AUC
    for the hallucination detection probe.
    """
    
    def __init__(self, probe_threshold: float = 0.5, eval_steps: Optional[int] = None):
        """
        Args:
            probe_threshold: Classification threshold for binary predictions
            eval_steps: How often to run evaluation (if None, uses trainer's eval_steps)
        """
        self.probe_threshold = probe_threshold
        self.eval_steps = eval_steps
        
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, 
                   model=None, eval_dataloader=None, **kwargs) -> None:
        """
        Called during evaluation to compute probe-specific metrics.
        
        Args:
            args: Training arguments
            state: Current trainer state
            control: Trainer control object
            model: The model being trained
            eval_dataloader: Evaluation dataloader
            **kwargs: Additional arguments
        """
        if eval_dataloader is None:
            return
            
        # Set model to evaluation mode
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                
                # Get probe outputs
                probe_logits: Float[Tensor, 'batch_size seq_len 1'] = outputs["probe_logits"]
                probe_logits = probe_logits.squeeze(-1)  # [batch_size, seq_len]
                
                # Get labels
                token_labels: Float[Tensor, 'batch_size seq_len'] = batch["probe_labels"]
                attention_mask: Int[Tensor, 'batch_size seq_len'] = batch["attention_mask"]
                
                # Create mask for valid positions
                valid_mask = (token_labels != -100.0) & (attention_mask == 1)
                
                if valid_mask.any():
                    # Get valid predictions and labels
                    valid_logits = probe_logits[valid_mask]
                    valid_labels = token_labels[valid_mask]
                    
                    # Convert to probabilities
                    valid_probs = torch.sigmoid(valid_logits)
                    
                    # Convert to binary predictions
                    valid_preds = (valid_probs >= self.probe_threshold)
                    
                    # Collect for metrics computation
                    all_predictions.extend(valid_preds.float().cpu().numpy())
                    all_labels.extend(valid_labels.float().cpu().numpy())
                    all_probs.extend(valid_probs.float().cpu().numpy())
        
        # Compute metrics if we have predictions
        if len(all_predictions) > 0:
            metrics = self._compute_metrics(
                predictions=np.array(all_predictions),
                labels=np.array(all_labels),
                probs=np.array(all_probs)
            )
            
            # Log metrics
            for key, value in metrics.items():
                print(f"Probe {key}: {value:.4f}")
                
            # Add to trainer's log history
            # if state.log_history:
            #     state.log_history[-1].update(metrics)
            # else:
            #     state.log_history.append(metrics)
    
    def _compute_metrics(self, predictions: np.ndarray, labels: np.ndarray, 
                        probs: np.ndarray) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            predictions: Binary predictions
            labels: True labels
            probs: Prediction probabilities
            
        Returns:
            Dict of computed metrics
        """
        metrics = {}
        
        # Basic accuracy
        metrics["probe_accuracy"] = float(accuracy_score(labels, predictions))
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        metrics["probe_precision"] = float(precision)
        metrics["probe_recall"] = float(recall)
        metrics["probe_f1"] = float(f1)
        
        # AUC if we have both classes
        if len(np.unique(labels)) > 1:
            try:
                metrics["probe_auc"] = float(roc_auc_score(labels, probs))
            except ValueError:
                # Handle case where AUC can't be computed
                metrics["probe_auc"] = 0.0
        else:
            metrics["probe_auc"] = 0.0
        
        # Class distribution
        positive_rate = float(np.mean(labels))
        metrics["probe_positive_rate"] = positive_rate
        
        # Prediction distribution
        pred_positive_rate = float(np.mean(predictions))
        metrics["probe_pred_positive_rate"] = pred_positive_rate
        
        return metrics
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, 
               model=None, **kwargs) -> None:
        """
        Called when logging metrics.
        
        Args:
            args: Training arguments
            state: Current trainer state
            control: Trainer control object
            model: The model being trained
            **kwargs: Additional arguments
        """
        # Add probe-specific logging if needed
        pass