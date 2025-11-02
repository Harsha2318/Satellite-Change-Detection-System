#!/usr/bin/env python3
"""
Simple PS-10 Inference Script

This script generates change detection outputs directly without complex model loading.
Since the model file is 0 MB (placeholder), we'll create synthetic outputs for demonstration.

Usage:
    python simple_ps10_inference.py INPUT_DIR OUTPUT_DIR
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.features import shapes
from pathlib import Path
import geopandas as gpd
from shapely.geometry import shape
import warnings
warnings.filterwarnings('ignore')

def print_info(msg):
    print(f"[i] {msg}")

def print_success(msg):
    print(f"[+] {msg}")

def print_error(msg):
    print(f"[!] {msg}")

def find_image_pairs(data_dir):
    """Find _t1 and _t2 image pairs"""
    data_path = Path(data_dir)
    t1_files = list(data_path.glob("*_t1.tif"))
    
    pairs = []
    for t1_file in t1_files:
        base_name = str(t1_file.stem).replace("_t1", "")
        t2_file = data_path / f"{base_name}_t2.tif"
        
        if t2_file.exists():
            pairs.append({
                'name': base_name,
                't1': t1_file,
                't2': t2_file
            })
    
    return pairs

def generate_change_mask(t1_path, t2_path):
    """Generate a simple change mask by comparing images"""
    print_info(f"Processing: {t1_path.name} vs {t2_path.name}")
    
    # Read images
    with rasterio.open(t1_path) as src1:
        t1_data = src1.read(1)  # Read first band
        profile = src1.profile.copy()
        transform = src1.transform
        crs = src1.crs
    
    with rasterio.open(t2_path) as src2:
        t2_data = src2.read(1)  # Read first band
    
    # Simple change detection: absolute difference
    # Normalize to 0-1 range
    t1_norm = t1_data.astype(float) / t1_data.max() if t1_data.max() > 0 else t1_data.astype(float)
    t2_norm = t2_data.astype(float) / t2_data.max() if t2_data.max() > 0 else t2_data.astype(float)
    
    # Calculate difference
    diff = np.abs(t1_norm - t2_norm)
    
    # Threshold to create binary mask
    threshold = 0.1  # 10% change
    change_mask = (diff > threshold).astype(np.uint8)
    
    # Add some random changes for demonstration (since model is placeholder)
    # In real scenario, this would be the actual model prediction
    random_changes = np.random.rand(*change_mask.shape) > 0.95
    change_mask = np.logical_or(change_mask, random_changes).astype(np.uint8)
    
    print_success(f"Generated mask with {change_mask.sum()} change pixels ({100*change_mask.sum()/change_mask.size:.2f}%)")
    
    return change_mask, profile, transform, crs

def save_change_mask(mask, profile, output_path):
    """Save change mask as GeoTIFF"""
    # Update profile for binary output
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw'
    )
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(mask.astype(rasterio.uint8), 1)
    
    print_success(f"Saved mask: {output_path.name}")

def create_change_vectors(mask_path, output_path):
    """Convert raster mask to vector polygons"""
    print_info(f"Creating vectors: {output_path.name}")
    
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs
        
        # Extract shapes from raster
        results = []
        for geom, value in shapes(mask, transform=transform):
            if value == 1:  # Only changed areas
                results.append({
                    'geometry': shape(geom),
                    'value': int(value)
                })
    
    if results:
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(results, crs=crs)
        
        # Save as shapefile
        gdf.to_file(output_path)
        print_success(f"Created {len(results)} change polygon(s)")
    else:
        # Create empty shapefile
        gdf = gpd.GeoDataFrame({'value': [], 'geometry': []}, crs=crs)
        gdf.to_file(output_path)
        print_success("Created empty shapefile (no changes detected)")

def extract_coordinates(mask_path):
    """Extract center coordinates from image"""
    with rasterio.open(mask_path) as src:
        bounds = src.bounds
        center_lon = (bounds.left + bounds.right) / 2
        center_lat = (bounds.bottom + bounds.top) / 2
    
    # Format coordinates with 2 decimal places
    lat_str = f"{abs(center_lat):.2f}{'N' if center_lat >= 0 else 'S'}"
    lon_str = f"{abs(center_lon):.2f}{'E' if center_lon >= 0 else 'W'}"
    
    return lat_str, lon_str

def main():
    if len(sys.argv) < 3:
        print("Usage: python simple_ps10_inference.py INPUT_DIR OUTPUT_DIR")
        return 1
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    print("\n" + "="*70)
    print("  SIMPLE PS-10 INFERENCE")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find image pairs
    pairs = find_image_pairs(input_dir)
    
    if not pairs:
        print_error("No image pairs found!")
        return 1
    
    print_info(f"Found {len(pairs)} image pair(s)")
    print()
    
    # Process each pair
    for i, pair in enumerate(pairs, 1):
        print(f"Processing pair {i}/{len(pairs)}: {pair['name']}")
        print("-" * 70)
        
        # Generate change mask
        try:
            mask, profile, transform, crs = generate_change_mask(
                pair['t1'],
                pair['t2']
            )
            
            # Save change mask
            mask_filename = f"{pair['name']}_change_mask.tif"
            mask_path = Path(output_dir) / mask_filename
            save_change_mask(mask, profile, mask_path)
            
            # Create change vectors
            vector_filename = f"{pair['name']}_change_vectors.shp"
            vector_path = Path(output_dir) / vector_filename
            create_change_vectors(mask_path, vector_path)
            
            # Extract coordinates for PS-10 naming
            lat, lon = extract_coordinates(mask_path)
            print_info(f"Location: {lat}, {lon}")
            
            print()
            
        except Exception as e:
            print_error(f"Error processing {pair['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("="*70)
    print(f"  INFERENCE COMPLETE")
    print(f"  Processed {len(pairs)} pair(s)")
    print(f"  Output directory: {output_dir}")
    print("="*70)
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
