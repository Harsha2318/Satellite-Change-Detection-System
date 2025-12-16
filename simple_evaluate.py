#!/usr/bin/env python
"""
Evaluation script for satellite change detection
"""
import numpy as np
from pathlib import Path
from PIL import Image
import json

def calculate_metrics(pred_binary, gt_binary):
    """
    Calculate evaluation metrics
    
    Args:
        pred_binary: Predicted binary mask (0 or 1)
        gt_binary: Ground truth binary mask (0 or 1)
    
    Returns:
        Dictionary of metrics
    """
    
    pred = pred_binary.flatten().astype(bool)
    gt = gt_binary.flatten().astype(bool)
    
    # Calculate TP, TN, FP, FN
    tp = np.sum((pred == 1) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    return {
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'iou': float(iou),
        'dice': float(dice),
    }

def evaluate_single(pred_binary_path, gt_path):
    """Evaluate single prediction against ground truth"""
    
    pred = np.array(Image.open(pred_binary_path).convert('L')) > 127
    gt = np.array(Image.open(gt_path).convert('L')) > 127
    
    metrics = calculate_metrics(pred, gt)
    return metrics

def evaluate_batch(pred_dir, gt_dir):
    """Evaluate all predictions in directory"""
    
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    
    pred_files = sorted(pred_dir.glob('*_binary.png'))
    
    print("[EVALUATION START]")
    print(f"  Predictions: {pred_dir}")
    print(f"  Ground truth: {gt_dir}")
    print(f"  Found {len(pred_files)} predictions")
    
    all_metrics = []
    
    for i, pred_file in enumerate(pred_files):
        basename = pred_file.stem.replace('_binary', '')
        gt_file = gt_dir / f'{basename}.png'
        
        if not gt_file.exists():
            print(f"[WARNING] Ground truth not found for {basename}")
            continue
        
        print(f"\n[{i+1}/{len(pred_files)}] {basename}")
        
        try:
            metrics = evaluate_single(pred_file, gt_file)
            metrics['image'] = basename
            all_metrics.append(metrics)
            
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
            print(f"  IoU:       {metrics['iou']:.4f}")
            print(f"  Dice:      {metrics['dice']:.4f}")
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue
    
    # Calculate average metrics
    if all_metrics:
        avg_metrics = {}
        for key in ['accuracy', 'precision', 'recall', 'f1', 'iou', 'dice']:
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        print(f"\n[AVERAGE METRICS]")
        print(f"  Accuracy:  {avg_metrics['accuracy']:.4f}")
        print(f"  Precision: {avg_metrics['precision']:.4f}")
        print(f"  Recall:    {avg_metrics['recall']:.4f}")
        print(f"  F1 Score:  {avg_metrics['f1']:.4f}")
        print(f"  IoU:       {avg_metrics['iou']:.4f}")
        print(f"  Dice:      {avg_metrics['dice']:.4f}")
        
        # Save results
        results = {
            'total_evaluated': len(all_metrics),
            'individual_results': all_metrics,
            'average_metrics': avg_metrics,
        }
        
        eval_path = Path('evaluation') / 'evaluation_results.json'
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(eval_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n[OK] Results saved: {eval_path}")
    
    return all_metrics

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        pred_dir = sys.argv[1]
        gt_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/test/labels'
    else:
        pred_dir = 'predictions'
        gt_dir = 'data/test/labels'
    
    print(f"Evaluating predictions against ground truth...")
    metrics = evaluate_batch(pred_dir, gt_dir)
    
    print("\n[NEXT STEPS]")
    print("  1. Review evaluation results: evaluation/evaluation_results.json")
    print("  2. Adjust threshold if needed: python simple_inference.py [threshold]")
    print("  3. Retrain with more epochs for better accuracy")
