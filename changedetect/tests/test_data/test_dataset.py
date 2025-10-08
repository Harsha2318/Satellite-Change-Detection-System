"""
Unit tests for dataset module
"""
import unittest
import os
import sys
import shutil
import numpy as np
import torch
import rasterio
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from changedetect.src.data.dataset import (
    ChangeDetectionDataset,
    create_dataloaders,
    create_test_dataloader,
    get_data_transforms
)


class TestChangeDetectionDataset(unittest.TestCase):
    """Test cases for ChangeDetectionDataset class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with sample data"""
        cls.test_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'temp_test_data'
        cls.images_dir = cls.test_dir / 'images'
        cls.masks_dir = cls.test_dir / 'masks'
        
        # Create directories
        os.makedirs(cls.images_dir, exist_ok=True)
        os.makedirs(cls.masks_dir, exist_ok=True)
        
        # Create sample images and masks
        cls.image_shape = (3, 256, 256)  # (C, H, W)
        cls.mask_shape = (1, 256, 256)   # (C, H, W)
        
        # Create sample data for two image pairs
        for i in range(2):
            # Create t1 image
            t1_data = np.random.randint(0, 255, size=cls.image_shape).astype(np.uint8)
            t1_path = cls.images_dir / f"sample_{i}_t1.tif"
            
            # Create t2 image
            t2_data = np.random.randint(0, 255, size=cls.image_shape).astype(np.uint8)
            t2_path = cls.images_dir / f"sample_{i}_t2.tif"
            
            # Create mask
            mask_data = np.random.randint(0, 2, size=cls.mask_shape[1:]).astype(np.uint8)
            mask_path = cls.masks_dir / f"sample_{i}_mask.tif"
            
            # Save as GeoTIFF files
            profile = {
                'driver': 'GTiff',
                'dtype': 'uint8',
                'count': cls.image_shape[0],
                'height': cls.image_shape[1],
                'width': cls.image_shape[2],
                'crs': '+proj=latlong',
                'transform': rasterio.transform.from_bounds(0, 0, 1, 1, cls.image_shape[2], cls.image_shape[1])
            }
            
            with rasterio.open(t1_path, 'w', **profile) as dst:
                for i in range(cls.image_shape[0]):
                    dst.write(t1_data[i], i+1)
            
            with rasterio.open(t2_path, 'w', **profile) as dst:
                for i in range(cls.image_shape[0]):
                    dst.write(t2_data[i], i+1)
            
            mask_profile = profile.copy()
            mask_profile['count'] = 1
            
            with rasterio.open(mask_path, 'w', **mask_profile) as dst:
                dst.write(mask_data, 1)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.test_dir)
    
    def test_dataset_initialization(self):
        """Test dataset initialization"""
        dataset = ChangeDetectionDataset(
            image_pairs_dir=str(self.images_dir),
            mask_dir=str(self.masks_dir),
            tile_size=128,
            phase="train"
        )
        
        # Check that image pairs were found
        self.assertEqual(len(dataset.image_pairs), 2)
        
        # Check that tiles were created
        self.assertGreater(len(dataset.tiles), 0)
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ method"""
        dataset = ChangeDetectionDataset(
            image_pairs_dir=str(self.images_dir),
            mask_dir=str(self.masks_dir),
            tile_size=128,
            phase="train"
        )
        
        # Get first item
        item = dataset[0]
        
        # Check that item contains expected keys
        self.assertIn('t1', item)
        self.assertIn('t2', item)
        self.assertIn('mask', item)
        self.assertIn('name', item)
        self.assertIn('window', item)
        self.assertIn('transform', item)
        self.assertIn('profile', item)
        
        # Check data types and shapes
        self.assertIsInstance(item['t1'], torch.Tensor)
        self.assertIsInstance(item['t2'], torch.Tensor)
        self.assertIsInstance(item['mask'], torch.Tensor)
        
        # Check shapes
        self.assertEqual(item['t1'].shape[0], 3)  # 3 channels
        self.assertEqual(item['t2'].shape[0], 3)  # 3 channels
        self.assertEqual(item['mask'].shape[0], 1)  # 1 channel
    
    def test_data_transforms(self):
        """Test data transformations"""
        train_transform = get_data_transforms("train")
        val_transform = get_data_transforms("val")
        
        # Create sample data
        img1 = np.random.randint(0, 255, size=(128, 128, 3)).astype(np.uint8)
        img2 = np.random.randint(0, 255, size=(128, 128, 3)).astype(np.uint8)
        mask = np.random.randint(0, 2, size=(128, 128)).astype(np.float32)
        
        # Apply transformations
        train_augmented = train_transform(image=img1, image2=img2, mask=mask)
        val_augmented = val_transform(image=img1, image2=img2, mask=mask)
        
        # Check that transformations returned expected keys
        self.assertIn('image', train_augmented)
        self.assertIn('image2', train_augmented)
        self.assertIn('mask', train_augmented)
        
        # Check that validation transformation doesn't modify shapes
        self.assertEqual(val_augmented['image'].shape, img1.shape)
        self.assertEqual(val_augmented['image2'].shape, img2.shape)
        self.assertEqual(val_augmented['mask'].shape, mask.shape)
    
    def test_create_dataloaders(self):
        """Test create_dataloaders function"""
        train_loader, val_loader = create_dataloaders(
            image_pairs_dir=str(self.images_dir),
            mask_dir=str(self.masks_dir),
            tile_size=128,
            batch_size=2,
            val_split=0.5,
            num_workers=0  # Use 0 workers for testing
        )
        
        # Check that dataloaders were created
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        
        # Get a batch from each loader
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        # Check batch structure
        self.assertIn('t1', train_batch)
        self.assertIn('t2', train_batch)
        self.assertIn('mask', train_batch)
        
        # Check batch shapes
        self.assertEqual(train_batch['t1'].shape[0], 2)  # Batch size
        self.assertEqual(train_batch['t1'].shape[1], 3)  # Channels
    
    def test_create_test_dataloader(self):
        """Test create_test_dataloader function"""
        test_loader = create_test_dataloader(
            image_pairs_dir=str(self.images_dir),
            tile_size=128,
            batch_size=2,
            num_workers=0  # Use 0 workers for testing
        )
        
        # Check that test dataloader was created
        self.assertIsNotNone(test_loader)
        
        # Get a batch from the loader
        test_batch = next(iter(test_loader))
        
        # Check batch structure
        self.assertIn('t1', test_batch)
        self.assertIn('t2', test_batch)
        self.assertIn('mask', test_batch)
        
        # Check batch shapes
        self.assertEqual(test_batch['t1'].shape[0], 2)  # Batch size
        self.assertEqual(test_batch['t1'].shape[1], 3)  # Channels


if __name__ == '__main__':
    unittest.main()