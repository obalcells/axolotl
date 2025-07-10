"""
Evaluation utilities for hallucination probe training.

This module provides clean evaluation functions that can be used directly 
with the trainer's compute_metrics functionality instead of callbacks.
"""

import torch
import numpy as np
from jaxtyping import Float, Int
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from transformers import EvalPrediction


class ProbeEvaluator:
    """
    Evaluator for hallucination probe metrics at multiple aggregation levels.
    
    Computes classification metrics like accuracy, precision, recall, F1, and AUC
    for the hallucination detection probe at three aggregation levels:
    1. 'all': all tokens (exclude only padding tokens)
    2. 'span': only tokens that belong to spans (based on span_ids)
    3. 'span_max': span-level predictions using max aggregation
    """
    
    def __init__(self, probe_threshold: float = 0.5):
        """
        Args:
            probe_threshold: Classification threshold for binary predictions
        """
        self.probe_threshold = probe_threshold
    
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute probe metrics for use with trainer's compute_metrics.
        
        Args:
            eval_pred: EvalPrediction containing predictions and labels
            
        Returns:
            Dictionary of computed metrics
        """
        # Extract predictions and labels
        predictions = eval_pred.predictions
        label_ids = eval_pred.label_ids
        
        # Handle different prediction formats
        if isinstance(predictions, tuple):
            # If predictions is tuple (logits, other_outputs), take the first element
            probe_logits = predictions[0]
        else:
            probe_logits = predictions
        
        # Extract relevant tensors from label_ids
        # Assuming label_ids contains: [probe_labels, attention_mask, span_ids]
        if isinstance(label_ids, tuple) or isinstance(label_ids, list):
            probe_labels = label_ids[0]
            attention_mask = label_ids[1] if len(label_ids) > 1 else None
            span_ids = label_ids[2] if len(label_ids) > 2 else None
        else:
            probe_labels = label_ids
            attention_mask = None
            span_ids = None
        
        # Convert to tensors if needed
        if isinstance(probe_logits, np.ndarray):
            probe_logits = torch.from_numpy(probe_logits)
        if isinstance(probe_labels, np.ndarray):
            probe_labels = torch.from_numpy(probe_labels)
        if attention_mask is not None and isinstance(attention_mask, np.ndarray):
            attention_mask = torch.from_numpy(attention_mask)
        if span_ids is not None and isinstance(span_ids, np.ndarray):
            span_ids = torch.from_numpy(span_ids)
        
        # Compute probabilities and predictions
        probe_probs = torch.sigmoid(probe_logits.squeeze(-1))
        probe_preds = (probe_probs > self.probe_threshold).float()
        
        # Compute metrics at different aggregation levels
        metrics = self._compute_aggregated_metrics(
            probe_probs, probe_preds, probe_labels, attention_mask, span_ids
        )
        
        return metrics
    
    def _compute_aggregated_metrics(
        self, 
        probe_probs: Float[Tensor, 'batch_size seq_len'],
        probe_preds: Float[Tensor, 'batch_size seq_len'], 
        probe_labels: Float[Tensor, 'batch_size seq_len'],
        attention_mask: Optional[Int[Tensor, 'batch_size seq_len']] = None,
        span_ids: Optional[Int[Tensor, 'batch_size seq_len']] = None
    ) -> Dict[str, float]:
        """
        Compute classification metrics for different aggregation levels using vectorized operations.
        
        Args:
            probe_probs: Probe prediction probabilities
            probe_preds: Probe binary predictions
            probe_labels: True labels
            attention_mask: Attention mask to exclude padding tokens
            span_ids: Span IDs for span-level aggregation
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # 1. All tokens metrics (exclude padding and ignore labels)
        if attention_mask is not None:
            all_tokens_mask = (attention_mask == 1) & (probe_labels != -100.0)
        else:
            all_tokens_mask = (probe_labels != -100.0)
        
        if all_tokens_mask.any():
            all_probs = probe_probs[all_tokens_mask]
            all_preds = probe_preds[all_tokens_mask]
            all_labels = probe_labels[all_tokens_mask]
            
            all_metrics = self._compute_level_metrics(all_probs, all_preds, all_labels, 'all')
            metrics.update(all_metrics)
        
        # 2. Span tokens metrics (only tokens that belong to annotated spans)
        if span_ids is not None:
            span_tokens_mask = (span_ids != -100) & (probe_labels != -100.0)
            
            if span_tokens_mask.any():
                span_probs = probe_probs[span_tokens_mask]
                span_preds = probe_preds[span_tokens_mask]
                span_labels = probe_labels[span_tokens_mask]
                
                span_metrics = self._compute_level_metrics(span_probs, span_preds, span_labels, 'span')
                metrics.update(span_metrics)
            
            # 3. Span max metrics (max aggregation over spans)
            span_max_metrics = self._compute_span_max_metrics_vectorized(
                probe_probs, probe_preds, probe_labels, span_ids
            )
            metrics.update(span_max_metrics)
        
        return metrics
    
    def _compute_span_max_metrics_vectorized(
        self,
        probe_probs: Float[Tensor, 'batch_size seq_len'],
        probe_preds: Float[Tensor, 'batch_size seq_len'],
        probe_labels: Float[Tensor, 'batch_size seq_len'],
        span_ids: Int[Tensor, 'batch_size seq_len']
    ) -> Dict[str, float]:
        """
        Compute span-level max aggregation metrics.
        
        Args:
            probe_probs: Probe prediction probabilities
            probe_preds: Probe binary predictions
            probe_labels: True labels
            span_ids: Span IDs
            
        Returns:
            Dictionary of span_max metrics
        """
        batch_size = probe_probs.shape[0]
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
        
        if len(span_max_probs) == 0:
            return {
                'span_max_accuracy': 0.0,
                'span_max_precision': 0.0,
                'span_max_recall': 0.0,
                'span_max_f1': 0.0,
                'span_max_auc': 0.0,
                'span_max_tpr_at_fpr_01': 0.0,
                'span_max_tpr_at_fpr_001': 0.0,
                'span_max_count': 0,
                'span_max_positive_rate': 0.0,
                'span_max_pred_positive_rate': 0.0
            }
        
        # Stack the results
        span_max_probs = torch.stack(span_max_probs)
        span_max_preds = torch.stack(span_max_preds)
        span_max_labels = torch.stack(span_max_labels)
        
        # Compute metrics
        span_max_metrics = self._compute_level_metrics(
            span_max_probs, span_max_preds, span_max_labels, 'span_max'
        )
        
        return span_max_metrics
    
    def _compute_level_metrics(
        self,
        probs: Tensor,
        preds: Tensor,
        labels: Tensor,
        agg_level: str
    ) -> Dict[str, float]:
        """
        Compute classification metrics for a specific aggregation level.
        
        Args:
            probs: Probability tensor
            preds: Prediction tensor
            labels: Label tensor
            agg_level: Aggregation level name
            
        Returns:
            Dictionary of computed metrics for this level
        """
        metrics = {}
        
        try:
            # Convert to numpy arrays on CPU
            preds_np = preds.cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()
            probs_np = probs.cpu().numpy().flatten()
            
            # Compute basic metrics
            accuracy = accuracy_score(labels_np, preds_np)
            precision = precision_score(labels_np, preds_np, average='binary', zero_division=0)
            recall = recall_score(labels_np, preds_np, average='binary', zero_division=0)
            f1 = f1_score(labels_np, preds_np, average='binary', zero_division=0)
            
            # AUC and TPR at specific FPR values
            if len(np.unique(labels_np)) > 1:
                auc = roc_auc_score(labels_np, probs_np)
                
                # Compute TPR at specific FPR values
                fpr, tpr, _ = roc_curve(labels_np, probs_np)
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
            metrics[f"{agg_level}_count"] = len(preds_np)
            
            # Class distribution
            positive_rate = np.mean(labels_np)
            pred_positive_rate = np.mean(preds_np)
            metrics[f"{agg_level}_positive_rate"] = positive_rate
            metrics[f"{agg_level}_pred_positive_rate"] = pred_positive_rate
            
        except Exception as e:
            print(f"Error computing {agg_level} metrics: {e}")
            # Return zero metrics on error
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


def create_probe_compute_metrics(probe_threshold: float = 0.5):
    """
    Create a compute_metrics function for use with Trainer.
    
    Args:
        probe_threshold: Classification threshold for binary predictions
        
    Returns:
        Callable that can be used as compute_metrics in Trainer
    """
    evaluator = ProbeEvaluator(probe_threshold=probe_threshold)
    return evaluator.compute_metrics


def display_metrics_table(metrics: Dict[str, float]) -> None:
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