"""
Callbacks for hallucination probe training
"""

import torch
import numpy as np
from jaxtyping import Float, Int
from torch import Tensor
from typing import Dict, Optional, Any, List
from transformers import TrainerCallback, TrainerState, TrainerControl
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, precision_score, recall_score, f1_score, roc_curve


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
        Called during evaluation to compute probe-specific metrics at three aggregation levels:
        1. 'all': all tokens (exclude only padding tokens)
        2. 'span': only tokens that belong to spans (based on span_ids)
        3. 'span_max': span-level predictions using max aggregation
        
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
        
        # Initialize metrics collectors for different aggregation levels - keep as tensors
        all_probs = {'all': [], 'span': [], 'span_max': []}
        all_preds = {'all': [], 'span': [], 'span_max': []}
        all_labels = {'all': [], 'span': [], 'span_max': []}
        
        with torch.no_grad():
            for batch in eval_dataloader:

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch.get("labels"),
                )
                
                probe_logits: Float[Tensor, 'batch_size seq_len'] = outputs["probe_logits"].squeeze(-1)
                probe_probs: Float[Tensor, 'batch_size seq_len'] = torch.sigmoid(probe_logits)
                probe_preds: Float[Tensor, 'batch_size seq_len'] = (probe_probs > self.probe_threshold).float()
                
                attention_mask: Int[Tensor, 'batch_size seq_len'] = batch["attention_mask"]
                probe_labels: Float[Tensor, 'batch_size seq_len'] = batch["probe_labels"]
                span_ids: Int[Tensor, 'batch_size seq_len'] = batch["span_ids"]
                
                # 1. All tokens metrics (exclude padding and ignore labels)
                all_tokens_mask = (attention_mask == 1) & (probe_labels != -100.0)
                if all_tokens_mask.any():
                    all_probs['all'].append(probe_probs[all_tokens_mask])
                    all_preds['all'].append(probe_preds[all_tokens_mask])
                    all_labels['all'].append(probe_labels[all_tokens_mask])
                
                # 2. Span tokens metrics (only tokens that belong to annotated spans)
                span_tokens_mask = (span_ids != -100) & (probe_labels != -100.0)
                if span_tokens_mask.any():
                    all_probs['span'].append(probe_probs[span_tokens_mask])
                    all_preds['span'].append(probe_preds[span_tokens_mask])
                    all_labels['span'].append(probe_labels[span_tokens_mask])
                
                # 3. Span max metrics (max aggregation over spans)
                batch_size = probe_logits.shape[0]
                span_max_probs = []
                span_max_preds = []
                span_max_labels = []
                
                for batch_idx in range(batch_size):
                    batch_span_ids = span_ids[batch_idx]
                    batch_labels = probe_labels[batch_idx]
                    batch_probs = probe_probs[batch_idx]
                    batch_preds = probe_preds[batch_idx]
                    
                    # Get unique span IDs (excluding ignore label)
                    unique_span_ids = torch.unique(batch_span_ids)
                    unique_span_ids = unique_span_ids[unique_span_ids != -100]
                    
                    for span_id in unique_span_ids:
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
                        
                        span_label = unique_labels[0]
                        
                        # Get max probability and prediction for this span
                        span_probs = batch_probs[span_positions]
                        span_preds = batch_preds[span_positions]
                        
                        max_prob = torch.max(span_probs)
                        max_pred = torch.max(span_preds)
                        
                        span_max_probs.append(max_prob)
                        span_max_preds.append(max_pred)
                        span_max_labels.append(span_label)
                
                # Add batch span results if any
                if span_max_probs:
                    all_probs['span_max'].append(torch.stack(span_max_probs))
                    all_preds['span_max'].append(torch.stack(span_max_preds))
                    all_labels['span_max'].append(torch.stack(span_max_labels))
        
        # Compute aggregated metrics
        aggregated_metrics = self._compute_aggregated_metrics(all_probs, all_preds, all_labels)
        
        # Display metrics in a pretty table format
        print(f"DISPLAYING METRICS")
        self._display_metrics_table(aggregated_metrics)

        # Convert to float so that we can serialize it to JSON later
        aggregated_metrics = {k: float(v) for k, v in aggregated_metrics.items()}
        
        # Add to trainer's log history
        if state.log_history:
            state.log_history[-1].update(aggregated_metrics)
        else:
            state.log_history.append(aggregated_metrics)

        return aggregated_metrics
    
    def _compute_aggregated_metrics(self, all_probs: Dict[str, List], all_preds: Dict[str, List], all_labels: Dict[str, List]) -> Dict[str, float]:
        """
        Compute classification metrics for different aggregation levels.
        
        Args:
            all_probs: Dictionary with lists of tensors for each aggregation level
            all_preds: Dictionary with lists of tensors for each aggregation level  
            all_labels: Dictionary with lists of tensors for each aggregation level
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Compute classification metrics for each aggregation level
        for agg_level in ['all', 'span', 'span_max']:
            if len(all_preds[agg_level]) > 0:
                try:
                    # Concatenate all tensors for this aggregation level and convert to numpy
                    preds_tensor = torch.cat(all_preds[agg_level], dim=0)
                    labels_tensor = torch.cat(all_labels[agg_level], dim=0)
                    probs_tensor = torch.cat(all_probs[agg_level], dim=0)
                    
                    # Convert to numpy arrays on CPU
                    preds = preds_tensor.cpu().numpy().flatten()
                    labels = labels_tensor.cpu().numpy().flatten()
                    probs = probs_tensor.cpu().numpy().flatten()
                    
                    # Compute metrics
                    accuracy = accuracy_score(labels, preds)
                    precision = precision_score(labels, preds, average='binary', zero_division=0)
                    recall = recall_score(labels, preds, average='binary', zero_division=0)
                    f1 = f1_score(labels, preds, average='binary', zero_division=0)
                    
                    # AUC and TPR at specific FPR values
                    if len(np.unique(labels)) > 1:
                        auc = roc_auc_score(labels, probs)
                        
                        # Compute TPR at specific FPR values
                        fpr, tpr, _ = roc_curve(labels, probs)
                        tpr_at_fpr_01 = self._get_tpr_at_fpr(fpr, tpr, target_fpr=0.1)
                        tpr_at_fpr_001 = self._get_tpr_at_fpr(fpr, tpr, target_fpr=0.01)
                    else:
                        auc = 0.0
                        tpr_at_fpr_01 = 0.0
                        tpr_at_fpr_001 = 0.0
                    
                    # Add metrics with agg_level prefix
                    metrics[f"{agg_level}_accuracy"] = accuracy
                    metrics[f"{agg_level}_precision"] = precision
                    metrics[f"{agg_level}_recall"] = recall
                    metrics[f"{agg_level}_f1"] = f1
                    metrics[f"{agg_level}_auc"] = auc
                    metrics[f"{agg_level}_tpr_at_fpr_01"] = tpr_at_fpr_01
                    metrics[f"{agg_level}_tpr_at_fpr_001"] = tpr_at_fpr_001
                    metrics[f"{agg_level}_count"] = len(preds)
                    
                    # Class distribution
                    positive_rate = np.mean(labels)
                    pred_positive_rate = np.mean(preds)
                    metrics[f"{agg_level}_positive_rate"] = positive_rate
                    metrics[f"{agg_level}_pred_positive_rate"] = pred_positive_rate
                    
                except Exception as e:
                    print(f"Error computing {agg_level} metrics: {e}")
                    metrics[f"{agg_level}_accuracy"] = 0.0
                    metrics[f"{agg_level}_precision"] = 0.0
                    metrics[f"{agg_level}_recall"] = 0.0
                    metrics[f"{agg_level}_f1"] = 0.0
                    metrics[f"{agg_level}_auc"] = 0.0
                    metrics[f"{agg_level}_tpr_at_fpr_01"] = 0.0
                    metrics[f"{agg_level}_tpr_at_fpr_001"] = 0.0
                    metrics[f"{agg_level}_count"] = 0
                    metrics[f"{agg_level}_positive_rate"] = 0.0
                    metrics[f"{agg_level}_pred_positive_rate"] = 0.0
        
        return metrics
    
    def _get_tpr_at_fpr(self, fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
        """
        Get the True Positive Rate (TPR) at a specific False Positive Rate (FPR).
        
        Args:
            fpr: False positive rates from ROC curve
            tpr: True positive rates from ROC curve
            target_fpr: Target FPR value
            
        Returns:
            TPR at the target FPR (interpolated if necessary)
        """
        if len(fpr) == 0 or len(tpr) == 0:
            return 0.0
        
        # Find the index where FPR is closest to target_fpr
        idx = np.searchsorted(fpr, target_fpr)
        
        if idx == 0:
            return tpr[0]
        elif idx >= len(fpr):
            return tpr[-1]
        else:
            # Linear interpolation between adjacent points
            fpr_low, fpr_high = fpr[idx-1], fpr[idx]
            tpr_low, tpr_high = tpr[idx-1], tpr[idx]
            
            if fpr_high == fpr_low:
                return tpr_high
            
            # Interpolate
            alpha = (target_fpr - fpr_low) / (fpr_high - fpr_low)
            return tpr_low + alpha * (tpr_high - tpr_low)
    
    def _display_metrics_table(self, metrics: Dict[str, float]) -> None:
        """
        Display metrics in a nicely formatted table.
        
        Args:
            metrics: Dictionary of computed metrics
        """
        # Define the metrics to display and their display names
        metric_names = {
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1': 'F1',
            'auc': 'AUC',
            'tpr_at_fpr_01': 'TPR@FPR=0.1',
            'tpr_at_fpr_001': 'TPR@FPR=0.01',
            'count': 'Count',
            'positive_rate': 'Pos Rate',
            'pred_positive_rate': 'Pred Pos Rate'
        }
        
        # Aggregation levels
        agg_levels = ['all', 'span', 'span_max']
        agg_display_names = {
            'all': 'All Tokens',
            'span': 'Span Tokens', 
            'span_max': 'Span Max'
        }
        
        # Calculate column widths
        metric_col_width = max(len(name) for name in metric_names.values()) + 2
        agg_col_width = 12  # Fixed width for aggregation columns
        
        # Print header
        print("\n" + "="*80)
        print("PROBE EVALUATION METRICS")
        print("="*80)
        
        # Print column headers
        header = f"{'Metric':<{metric_col_width}}"
        for agg_level in agg_levels:
            header += f"{agg_display_names[agg_level]:^{agg_col_width}}"
        print(header)
        print("-" * len(header))
        
        # Print each metric row
        for metric_key, metric_display_name in metric_names.items():
            row = f"{metric_display_name:<{metric_col_width}}"
            
            for agg_level in agg_levels:
                metric_full_key = f"{agg_level}_{metric_key}"
                value = metrics.get(metric_full_key, 0.0)
                
                # Format the value based on metric type
                if metric_key == 'count':
                    formatted_value = f"{int(value)}"
                elif metric_key in ['positive_rate', 'pred_positive_rate']:
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = f"{value:.4f}"
                
                row += f"{formatted_value:^{agg_col_width}}"
            
            print(row)
        
        print("="*80 + "\n")
    
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
    
    # def on_log(self, args, state: TrainerState, control: TrainerControl, 
    #            model=None, **kwargs) -> None:
    #     """
    #     Called when logging metrics.
        
    #     Args:
    #         args: Training arguments
    #         state: Current trainer state
    #         control: Trainer control object
    #         model: The model being trained
    #         **kwargs: Additional arguments
    #     """
    #     # Add probe-specific logging if needed
    #     pass