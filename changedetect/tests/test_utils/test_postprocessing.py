"""
Unit tests for postprocessing module
"""
import unittest
import numpy as np
import sys
import os
from pathlib import Path
import cv2
from skimage import morphology

# Add project root to path
project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from changedetect.src.utils.postprocessing import (
    apply_morphological_operations, remove_small_objects, smooth_boundaries,
    apply_threshold, remove_small_holes
)

class TestPostprocessing(unittest.TestCase):
    """Test cases for postprocessing module"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test prediction with noise
        self.noisy_pred = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 0],  # Single noise pixel
            [0, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],  # Small noise object
            [0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        # Create a mask with holes
        self.holey_mask = np.ones((10, 10), dtype=np.uint8)
        self.holey_mask[3, 3] = 0  # Small hole
        self.holey_mask[4:6, 4:6] = 0  # Larger hole
    
    def test_morphological_operations(self):
        """Test morphological operations (opening/closing)"""
        # Apply opening to remove noise
        cleaned = apply_morphological_operations(self.noisy_pred, operation='opening', kernel_size=3)
        
        # The isolated noise pixels should be removed
        self.assertEqual(cleaned[2, 6], 0)  # Single noise pixel removed
        self.assertEqual(cleaned[6, 2], 0)  # Small noise object removed
        
        # The main object should be preserved
        self.assertTrue(np.sum(cleaned[1:5, 1:5] > 0) >= 10)  # Main blob preserved
        
        # Test closing
        binary_mask = np.zeros((8, 8), dtype=bool)
        binary_mask[2:5, 2:5] = True
        binary_mask[3, 3] = False  # Create a hole
        
        closed = apply_morphological_operations(binary_mask, operation='closing', kernel_size=3)
        self.assertTrue(closed[3, 3])  # Hole should be filled
    
    def test_remove_small_objects(self):
        """Test removal of small objects"""
        # Remove objects smaller than 4 pixels
        cleaned = remove_small_objects(self.noisy_pred, min_size=4)
        
        # The isolated noise pixels should be removed
        self.assertEqual(cleaned[2, 6], 0)  # Single noise pixel removed
        self.assertEqual(cleaned[6, 2], 0)  # Small noise object removed
        
        # The main object should be preserved
        self.assertTrue(np.array_equal(cleaned[1:5, 1:5], self.noisy_pred[1:5, 1:5]))
    
    def test_smooth_boundaries(self):
        """Test boundary smoothing"""
        # Create a prediction with jagged edges
        jagged = np.zeros((10, 10))
        jagged[2:8, 2:8] = 1
        jagged[4, 2] = 0  # Create a jagged edge
        jagged[5, 2] = 0
        jagged[2, 5] = 0
        
        # Smooth the boundaries
        smoothed = smooth_boundaries(jagged, sigma=1.0)
        
        # Convert back to binary (threshold at 0.5)
        binary_smoothed = (smoothed > 0.5).astype(np.uint8)
        
        # Check that the smoothed version has fewer edge pixels
        edges_before = np.sum((jagged - np.roll(jagged, 1, axis=0)) != 0) + \
                      np.sum((jagged - np.roll(jagged, 1, axis=1)) != 0)
                      
        edges_after = np.sum((binary_smoothed - np.roll(binary_smoothed, 1, axis=0)) != 0) + \
                     np.sum((binary_smoothed - np.roll(binary_smoothed, 1, axis=1)) != 0)
        
        # The smoothed image should have fewer or equal edge pixels
        self.assertLessEqual(edges_after, edges_before)
    
    def test_combined_operations(self):
        """Test a combination of postprocessing operations"""
        # Create a mask with noise and holes
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 1
        mask[4, 4] = 0  # Small hole
        mask[0, 0] = 1  # Isolated pixel
        
        # Apply a series of operations
        # 1. Remove small objects
        cleaned = remove_small_objects(mask, min_size=10)
        
        # 2. Fill small holes
        filled = remove_small_holes(cleaned, min_size=5)
        
        # 3. Apply morphological closing
        closed = apply_morphological_operations(filled, operation='closing', kernel_size=3)
        
        # Check that isolated pixel is removed
        self.assertEqual(closed[0, 0], 0)
        
        # Check that small hole is filled
        self.assertEqual(closed[4, 4], 1)
        
        # Check that main object is preserved
        self.assertTrue(np.sum(closed[2:8, 2:8] > 0) >= 30)


if __name__ == '__main__':
    unittest.main()