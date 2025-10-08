"""
Unit tests for Siamese U-Net model
"""
import unittest
import torch
import numpy as np
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from changedetect.src.models.siamese_unet import get_change_detection_model


class TestSiameseUNet(unittest.TestCase):
    """Test cases for Siamese U-Net model"""
    
    def setUp(self):
        """Set up test environment"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.channels = 3
        self.height = 256
        self.width = 256
    
    def test_siamese_unet_model(self):
        """Test Siamese U-Net model forward pass"""
        # Create random input data
        x1 = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        x2 = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        
        # Create model
        model = get_change_detection_model('siamese_unet', in_channels=self.channels).to(self.device)
        
        # Run forward pass
        output = model(x1, x2)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1, self.height, self.width))
        
        # Check model parameters
        num_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(num_params, 1000)  # Make sure model has reasonable size
    
    def test_siamese_diff_model(self):
        """Test Siamese difference model forward pass"""
        # Create random input data
        x1 = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        x2 = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        
        # Create model
        model = get_change_detection_model('siamese_diff', in_channels=self.channels).to(self.device)
        
        # Run forward pass
        output = model(x1, x2)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1, self.height, self.width))
    
    def test_fcn_diff_model(self):
        """Test FCN difference model forward pass"""
        # Create random input data
        x1 = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        x2 = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        
        # Create model
        model = get_change_detection_model('fcn_diff', in_channels=self.channels).to(self.device)
        
        # Run forward pass
        output = model(x1, x2)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1, self.height, self.width))
    
    def test_invalid_model_type(self):
        """Test invalid model type raises ValueError"""
        with self.assertRaises(ValueError):
            get_change_detection_model('invalid_model_type')
    
    def test_dropout(self):
        """Test model with dropout"""
        # Create random input data
        x1 = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        x2 = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        
        # Create model with dropout
        model = get_change_detection_model('siamese_unet', in_channels=self.channels, dropout=0.5).to(self.device)
        
        # Set model to train mode (to enable dropout)
        model.train()
        
        # Run forward pass multiple times
        output1 = model(x1, x2)
        output2 = model(x1, x2)
        
        # Check that outputs are different due to dropout
        self.assertFalse(torch.allclose(output1, output2))
        
        # Set model to eval mode (to disable dropout)
        model.eval()
        
        # Run forward pass multiple times
        output1 = model(x1, x2)
        output2 = model(x1, x2)
        
        # Check that outputs are the same in eval mode
        self.assertTrue(torch.allclose(output1, output2))


if __name__ == '__main__':
    unittest.main()