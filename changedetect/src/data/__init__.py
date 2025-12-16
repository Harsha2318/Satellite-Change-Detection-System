"""
Data handling utilities for satellite imagery
"""
import os
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def validate_image_pair(img1_path, img2_path):
    """
    Validate that two images can be paired.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        
    Returns:
        bool: True if images can be paired
    """
    if not os.path.exists(img1_path):
        logger.warning(f"Image not found: {img1_path}")
        return False
    if not os.path.exists(img2_path):
        logger.warning(f"Image not found: {img2_path}")
        return False
    return True


def get_image_pairs(data_dir):
    """
    Get paired images from a directory structure.
    Expects: data_dir/before/ and data_dir/after/ folders
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        list: List of (before_path, after_path) tuples
    """
    before_dir = os.path.join(data_dir, "before")
    after_dir = os.path.join(data_dir, "after")
    
    pairs = []
    if os.path.exists(before_dir) and os.path.exists(after_dir):
        before_files = sorted([f for f in os.listdir(before_dir) if f.endswith(('.tif', '.tiff', '.jpg', '.png'))])
        after_files = sorted([f for f in os.listdir(after_dir) if f.endswith(('.tif', '.tiff', '.jpg', '.png'))])
        
        for bf, af in zip(before_files, after_files):
            before_path = os.path.join(before_dir, bf)
            after_path = os.path.join(after_dir, af)
            if validate_image_pair(before_path, after_path):
                pairs.append((before_path, after_path))
    
    return pairs
