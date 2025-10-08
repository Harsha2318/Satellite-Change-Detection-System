"""
Unit tests for metrics module
"""
import unittest
import numpy as np
import torch
import sys
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from changedetect.src.utils.metrics import (
    calculate_iou, calculate_dice, calculate_precision_recall_f1,
    calculate_confusion_matrix, calculate_all_metrics, calculate_accuracy,
    calculate_per_class_metrics
)

class TestMetrics(unittest.TestCase):
    """Test cases for metrics module"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test arrays
        self.pred_perfect = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ])
        
        self.target_perfect = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ])
        
        self.pred_partial = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0]
        ])
        
        self.target_partial = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ])
        
        self.pred_empty = np.zeros((4, 4))
        self.target_empty = np.zeros((4, 4))
        
        # Create torch tensors
        self.pred_torch = torch.from_numpy(self.pred_partial).float()
        self.target_torch = torch.from_numpy(self.target_partial).float()
    
    def test_iou_perfect_match(self):
        """Test IoU calculation with perfect prediction"""
        iou = calculate_iou(self.pred_perfect, self.target_perfect)
        self.assertEqual(iou, 1.0)
    
    def test_iou_partial_match(self):
        """Test IoU calculation with partial prediction"""
        iou = calculate_iou(self.pred_partial, self.target_partial)
        # IoU = 3/(3+0+1) = 3/4 = 0.75
        self.assertAlmostEqual(iou, 0.75, places=2)
    
    def test_iou_empty(self):
        """Test IoU calculation with empty arrays"""
        iou = calculate_iou(self.pred_empty, self.target_empty)
        self.assertEqual(iou, 1.0)  # Both empty is 1.0
    
    def test_iou_torch_tensor(self):
        """Test IoU calculation with torch tensors"""
        iou = calculate_iou(self.pred_torch, self.target_torch)
        self.assertAlmostEqual(iou, 0.75, places=2)
    
    def test_dice_perfect_match(self):
        """Test Dice calculation with perfect prediction"""
        dice = calculate_dice(self.pred_perfect, self.target_perfect)
        self.assertEqual(dice, 1.0)
    
    def test_dice_partial_match(self):
        """Test Dice calculation with partial prediction"""
        dice = calculate_dice(self.pred_partial, self.target_partial)
        # Dice = 2*3/(3+4) = 6/7 ≈ 0.857
        self.assertAlmostEqual(dice, 0.857, places=3)
    
    def test_dice_torch_tensor(self):
        """Test Dice calculation with torch tensors"""
        dice = calculate_dice(self.pred_torch, self.target_torch)
        self.assertAlmostEqual(dice, 0.857, places=3)
    
    def test_precision_recall_f1(self):
        """Test precision, recall, and F1 calculation"""
        precision, recall, f1 = calculate_precision_recall_f1(
            self.pred_partial, self.target_partial
        )
        self.assertEqual(precision, 1.0)  # All predicted positives are true
        self.assertAlmostEqual(recall, 0.75, places=2)  # 3/4 true positives found
        self.assertAlmostEqual(f1, 0.857, places=3)  # F1 = 2*1*0.75/(1+0.75) = 2*0.75/1.75 ≈ 0.857
    
    def test_confusion_matrix(self):
        """Test confusion matrix calculation"""
        tn, fp, fn, tp = calculate_confusion_matrix(
            self.pred_partial, self.target_partial
        )
        self.assertEqual(tn, 12)  # True negatives
        self.assertEqual(fp, 0)   # False positives
        self.assertEqual(fn, 1)   # False negatives
        self.assertEqual(tp, 3)   # True positives
    
    def test_accuracy(self):
        """Test accuracy calculation"""
        accuracy = calculate_accuracy(self.pred_partial, self.target_partial)
        # Accuracy = (12+3)/(12+0+1+3) = 15/16 = 0.9375
        self.assertAlmostEqual(accuracy, 0.9375, places=4)
    
    def test_all_metrics(self):
        """Test all metrics calculation"""
        metrics = calculate_all_metrics(
            self.pred_partial, self.target_partial
        )
        
        self.assertAlmostEqual(metrics['iou'], 0.75, places=2)
        self.assertAlmostEqual(metrics['dice'], 0.857, places=3)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertAlmostEqual(metrics['recall'], 0.75, places=2)
        self.assertAlmostEqual(metrics['f1'], 0.857, places=3)
        self.assertAlmostEqual(metrics['accuracy'], 0.9375, places=4)
    
    def test_per_class_metrics(self):
        """Test per-class metrics calculation"""
        # Create multi-class predictions and targets
        multi_pred = np.zeros((4, 4), dtype=int)
        multi_pred[1:3, 1:3] = 1
        
        multi_target = np.zeros((4, 4), dtype=int)
        multi_target[1:3, 1:4] = 1
        
        # Calculate per-class metrics
        metrics = calculate_per_class_metrics(multi_pred, multi_target, num_classes=2)
        
        # Check class 0 metrics (background)
        self.assertAlmostEqual(metrics['class_0']['iou'], 0.923, places=3)
        
        # Check class 1 metrics (foreground)
        self.assertAlmostEqual(metrics['class_1']['iou'], 0.8, places=3)
        
        # Check average metrics
        self.assertAlmostEqual(metrics['avg']['avg_iou'], 0.8615, places=3)


if __name__ == '__main__':
    unittest.main()