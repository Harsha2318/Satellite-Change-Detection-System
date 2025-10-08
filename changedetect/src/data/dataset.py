"""
Dataset classes for satellite image change detection
"""
import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from pathlib import Path
import random
from typing import Tuple, List, Dict, Any, Optional

from changedetect.src.data.tile import create_tiles, generate_tile_windows
from changedetect.src.data.preprocess import preprocess_pair, normalize_image


class ChangeDetectionDataset(Dataset):
    """Dataset for satellite image change detection"""
    
    def __init__(
        self,
        image_pairs_dir: str,
        mask_dir: Optional[str] = None,
        tile_size: int = 256,
        transform=None,
        phase: str = "train",
        overlap: int = 0,
    ):
        """
        Initialize the change detection dataset.
        
        Args:
            image_pairs_dir: Directory containing image pairs (t1 and t2)
            mask_dir: Directory containing mask images (optional for test phase)
            tile_size: Size of image tiles to extract
            transform: Image transformations to apply
            phase: 'train', 'val', or 'test'
            overlap: Overlap between tiles in pixels
        """
        self.image_pairs_dir = Path(image_pairs_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.tile_size = tile_size
        self.transform = transform
        self.phase = phase
        self.overlap = overlap
        
        # Get list of image pair files
        self.image_pairs = self._get_image_pairs()
        
        # Create tiles for all images if needed
        self.tiles = self._create_all_tiles()
        
    def _get_image_pairs(self) -> List[Dict[str, str]]:
        """
        Get list of image pair files.
        
        Returns:
            List of dictionaries containing t1, t2, and mask file paths
        """
        image_pairs = []
        
        # List all t1 images
        t1_images = sorted(list(self.image_pairs_dir.glob("*_t1.tif")))
        
        for t1_path in t1_images:
            # Get corresponding t2 image
            name = t1_path.stem.replace("_t1", "")
            t2_path = self.image_pairs_dir / f"{name}_t2.tif"
            
            if not t2_path.exists():
                continue
            
            # Get corresponding mask if in train/val phase
            mask_path = None
            if self.mask_dir is not None:
                mask_path = self.mask_dir / f"{name}_mask.tif"
                if not mask_path.exists() and self.phase != "test":
                    continue
            
            image_pairs.append({
                "name": name,
                "t1": str(t1_path),
                "t2": str(t2_path),
                "mask": str(mask_path) if mask_path is not None else None
            })
        
        return image_pairs
    
    def _create_all_tiles(self) -> List[Dict[str, Any]]:
        """
        Create tiles for all image pairs.
        
        Returns:
            List of dictionaries containing tile information
        """
        all_tiles = []
        
        for pair in self.image_pairs:
            # Get image and mask dimensions
            with rasterio.open(pair["t1"]) as src:
                height, width = src.height, src.width
                t1_profile = src.profile.copy()
            
            # Create tiles for this image pair (in-memory windows)
            tiles = generate_tile_windows(
                (height, width),
                tile_size=self.tile_size,
                overlap=self.overlap,
                src_transform=t1_profile.get('transform') if 'transform' in t1_profile else None
            )
            
            for tile in tiles:
                tile_info = {
                    "name": f"{pair['name']}_{tile['id']}",
                    "t1": pair["t1"],
                    "t2": pair["t2"],
                    "mask": pair["mask"],
                    "window": tile["window"],
                    "transform": tile["transform"],
                    "profile": t1_profile,
                }
                all_tiles.append(tile_info)
        
        return all_tiles
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.tiles)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing t1, t2, and mask tensors
        """
        tile_info = self.tiles[idx]
        
        # Read image tiles
        with rasterio.open(tile_info["t1"]) as src:
            t1_img = src.read(window=tile_info["window"])
            
        with rasterio.open(tile_info["t2"]) as src:
            t2_img = src.read(window=tile_info["window"])
        
        # Preprocess images
        t1_img, t2_img = preprocess_pair(t1_img, t2_img)
        
        # Transpose from (C,H,W) to (H,W,C) for transformations
        t1_img = np.transpose(t1_img, (1, 2, 0))
        t2_img = np.transpose(t2_img, (1, 2, 0))
        
        # Get mask if available
        if tile_info["mask"] is not None:
            with rasterio.open(tile_info["mask"]) as src:
                mask = src.read(1, window=tile_info["window"])
                mask = (mask > 0).astype(np.float32)  # Binarize mask
        else:
            # Create dummy mask for test phase
            mask = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        
        # Apply transforms if specified
        if self.transform is not None:
            augmented = self.transform(image=t1_img, image2=t2_img, mask=mask)
            t1_img = augmented["image"]
            t2_img = augmented["image2"]
            mask = augmented["mask"]
        
        # Convert to PyTorch tensors
        t1_tensor = torch.from_numpy(t1_img.transpose(2, 0, 1))
        t2_tensor = torch.from_numpy(t2_img.transpose(2, 0, 1))
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        # Convert window and transform to serializable/primitive types so DataLoader can collate
        win = tile_info["window"]
        window_tuple = (int(win.col_off), int(win.row_off), int(win.width), int(win.height))

        tr = tile_info.get("transform")
        # Ensure transform is a numeric tuple (no None) so DataLoader can collate
        if tr is not None:
            transform_tuple = tuple(float(x) for x in tr)
        else:
            # Affine transform has 6 params; use zeros as a safe default
            transform_tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return {
            "name": tile_info["name"],
            "t1": t1_tensor,
            "t2": t2_tensor,
            "mask": mask_tensor,
            "window": window_tuple,
            "transform": transform_tuple,
        }


def get_data_transforms(phase: str) -> Dict[str, Any]:
    """
    Get data transformations based on dataset phase.
    
    Args:
        phase: 'train', 'val', or 'test'
        
    Returns:
        Dictionary containing transformations for each phase
    """
    if phase == "train":
        # Use a minimal, safer augmentation pipeline to avoid native crashes on some systems
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ], additional_targets={'image2': 'image'})
    elif phase == "val" or phase == "test":
        return A.Compose([], additional_targets={'image2': 'image'})
    else:
        raise ValueError(f"Invalid phase: {phase}")


def create_dataloaders(
    image_pairs_dir: str,
    mask_dir: Optional[str] = None,
    tile_size: int = 256,
    batch_size: int = 16,
    val_split: float = 0.2,
    num_workers: int = 4,
    overlap: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        image_pairs_dir: Directory containing image pairs
        mask_dir: Directory containing mask images
        tile_size: Size of image tiles to extract
        batch_size: Batch size for dataloaders
        val_split: Fraction of data to use for validation
        num_workers: Number of worker threads for data loading
        overlap: Overlap between tiles in pixels
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Get all image pairs
    dataset = ChangeDetectionDataset(
        image_pairs_dir=image_pairs_dir,
        mask_dir=mask_dir,
        tile_size=tile_size,
        transform=None,
        phase="train",
        overlap=overlap
    )
    
    # Split into train and validation sets
    total_samples = len(dataset.tiles)
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    split_idx = int(total_samples * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create train dataset
    train_tiles = [dataset.tiles[i] for i in train_indices]
    train_dataset = ChangeDetectionDataset(
        image_pairs_dir=image_pairs_dir,
        mask_dir=mask_dir,
        tile_size=tile_size,
        transform=get_data_transforms("train"),
        phase="train",
        overlap=overlap
    )
    train_dataset.tiles = train_tiles
    
    # Create validation dataset
    val_tiles = [dataset.tiles[i] for i in val_indices]
    val_dataset = ChangeDetectionDataset(
        image_pairs_dir=image_pairs_dir,
        mask_dir=mask_dir,
        tile_size=tile_size,
        transform=get_data_transforms("val"),
        phase="val",
        overlap=overlap
    )
    val_dataset.tiles = val_tiles
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    return train_dataloader, val_dataloader


def create_test_dataloader(
    image_pairs_dir: str,
    tile_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 4,
    overlap: int = 64
) -> DataLoader:
    """
    Create test dataloader.
    
    Args:
        image_pairs_dir: Directory containing image pairs
        tile_size: Size of image tiles to extract
        batch_size: Batch size for dataloader
        num_workers: Number of worker threads for data loading
        overlap: Overlap between tiles in pixels
        
    Returns:
        Test dataloader
    """
    test_dataset = ChangeDetectionDataset(
        image_pairs_dir=image_pairs_dir,
        mask_dir=None,
        tile_size=tile_size,
        transform=get_data_transforms("test"),
        phase="test",
        overlap=overlap
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return test_dataloader