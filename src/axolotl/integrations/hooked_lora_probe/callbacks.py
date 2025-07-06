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
                # token_labels: Float[Tensor, 'batch_size seq_len'] = batch["token_labels"]
                token_labels: Float[Tensor, 'batch_size seq_len'] = get_token_labels(batch["input_ids"])
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
            if state.log_history:
                state.log_history[-1].update(metrics)
            else:
                state.log_history.append(metrics)
    
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
        metrics["probe_accuracy"] = accuracy_score(labels, predictions)
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        metrics["probe_precision"] = precision
        metrics["probe_recall"] = recall
        metrics["probe_f1"] = f1
        
        # AUC if we have both classes
        if len(np.unique(labels)) > 1:
            try:
                metrics["probe_auc"] = roc_auc_score(labels, probs)
            except ValueError:
                # Handle case where AUC can't be computed
                metrics["probe_auc"] = 0.0
        else:
            metrics["probe_auc"] = 0.0
        
        # Class distribution
        positive_rate = np.mean(labels)
        metrics["probe_positive_rate"] = positive_rate
        
        # Prediction distribution
        pred_positive_rate = np.mean(predictions)
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