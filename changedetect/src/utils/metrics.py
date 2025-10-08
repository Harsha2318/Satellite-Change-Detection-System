"""
Evaluation metrics for change detection
"""
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def calculate_iou(pred, target):
    """
    Calculate Intersection over Union (IoU).
    
    Args:
        pred: Predicted binary mask (torch.Tensor or np.ndarray)
        target: Ground truth binary mask (torch.Tensor or np.ndarray)
        
    Returns:
        IoU score
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Ensure binary masks
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    # Calculate intersection and union
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    
    # Avoid division by zero
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def calculate_dice(pred, target):
    """
    Calculate Dice coefficient (F1 score).
    
    Args:
        pred: Predicted binary mask (torch.Tensor or np.ndarray)
        target: Ground truth binary mask (torch.Tensor or np.ndarray)
        
    Returns:
        Dice coefficient
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Ensure binary masks
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    # Calculate true positives
    tp = np.logical_and(pred, target).sum()
    
    # Calculate Dice coefficient
    dice = 2 * tp / (pred.sum() + target.sum())
    
    # Handle edge cases
    if pred.sum() + target.sum() == 0:
        return 1.0  # Both pred and target are empty
    
    return dice


def calculate_precision_recall_f1(pred, target):
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        pred: Predicted binary mask (torch.Tensor or np.ndarray)
        target: Ground truth binary mask (torch.Tensor or np.ndarray)
        
    Returns:
        Tuple of (precision, recall, F1 score)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Flatten the arrays
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Calculate metrics
    precision = precision_score(target_flat, pred_flat, zero_division=1)
    recall = recall_score(target_flat, pred_flat, zero_division=1)
    f1 = f1_score(target_flat, pred_flat, zero_division=1)
    
    return precision, recall, f1


def calculate_confusion_matrix(pred, target):
    """
    Calculate confusion matrix.
    
    Args:
        pred: Predicted binary mask (torch.Tensor or np.ndarray)
        target: Ground truth binary mask (torch.Tensor or np.ndarray)
        
    Returns:
        Confusion matrix (TN, FP, FN, TP)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Flatten the arrays
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(target_flat, pred_flat, labels=[0, 1]).ravel()
    
    return tn, fp, fn, tp


def calculate_accuracy(pred, target):
    """
    Calculate accuracy.
    
    Args:
        pred: Predicted binary mask (torch.Tensor or np.ndarray)
        target: Ground truth binary mask (torch.Tensor or np.ndarray)
        
    Returns:
        Accuracy
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Flatten the arrays
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Calculate accuracy
    accuracy = (pred_flat == target_flat).mean()
    
    return accuracy


def calculate_all_metrics(pred, target):
    """
    Calculate all evaluation metrics.
    
    Args:
        pred: Predicted binary mask (torch.Tensor or np.ndarray)
        target: Ground truth binary mask (torch.Tensor or np.ndarray)
        
    Returns:
        Dictionary of metrics
    """
    # Calculate confusion matrix
    tn, fp, fn, tp = calculate_confusion_matrix(pred, target)
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    
    # Return all metrics
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'dice': f1,  # Dice coefficient is equivalent to F1 score for binary classification
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def calculate_per_class_metrics(pred, target, num_classes=2):
    """
    Calculate metrics for each class in a multi-class segmentation.
    
    Args:
        pred: Predicted class indices (torch.Tensor or np.ndarray)
        target: Ground truth class indices (torch.Tensor or np.ndarray)
        num_classes: Number of classes
        
    Returns:
        Dictionary of per-class metrics
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Calculate metrics for each class
    for i in range(num_classes):
        # Create binary masks for this class
        pred_class = (pred == i).astype(bool)
        target_class = (target == i).astype(bool)
        
        # Calculate metrics for this class
        class_metrics = calculate_all_metrics(pred_class, target_class)
        
        # Add to metrics dictionary
        metrics[f'class_{i}'] = class_metrics
    
    # Calculate average metrics
    avg_metrics = {
        'avg_accuracy': np.mean([metrics[f'class_{i}']['accuracy'] for i in range(num_classes)]),
        'avg_precision': np.mean([metrics[f'class_{i}']['precision'] for i in range(num_classes)]),
        'avg_recall': np.mean([metrics[f'class_{i}']['recall'] for i in range(num_classes)]),
        'avg_f1': np.mean([metrics[f'class_{i}']['f1'] for i in range(num_classes)]),
        'avg_iou': np.mean([metrics[f'class_{i}']['iou'] for i in range(num_classes)]),
    }
    
    metrics['avg'] = avg_metrics
    
    return metrics