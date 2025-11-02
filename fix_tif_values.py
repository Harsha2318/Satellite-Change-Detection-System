#!/usr/bin/env python3
"""
Fix TIF values for PS-10 compliance

This script converts raster change masks from using 255 to represent change to using 1
as required by PS-10 specification.

Usage:
  python fix_tif_values.py <input_dir>
"""

import os
import sys
import numpy as np
from pathlib import Path

try:
    import rasterio
except ImportError:
    print("Error: This script requires rasterio. Install with:")
    print("  pip install rasterio")
    sys.exit(1)

def fix_tif_values(input_dir):
    """
    Convert all TIF files in a directory from 0/255 to 0/1 values for PS-10 compliance
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return False
    
    # Find all TIF files
    tif_files = list(input_path.glob("*.tif"))
    if not tif_files:
        print(f"No TIF files found in {input_dir}")
        return False
    
    print(f"Found {len(tif_files)} TIF files to process")
    
    # Process each TIF file
    for tif_file in tif_files:
        try:
            print(f"Processing {tif_file.name}...", end="")
            
            # Open the file with rasterio
            with rasterio.open(tif_file, "r+") as src:
                # Read the data
                data = src.read(1)
                
                # Check if conversion is needed
                unique_vals = np.unique(data)
                if set(unique_vals).issubset({0, 1}):
                    print(" already compliant (values: 0, 1)")
                    continue
                
                if set(unique_vals) == {0}:
                    print(" contains only zeros, no changes detected")
                    continue
                
                if 255 in unique_vals:
                    # Convert 255 to 1
                    data = np.where(data == 255, 1, data)
                    
                    # Write the modified data back
                    src.write(data, 1)
                    
                    # Verify the change
                    updated_data = src.read(1)
                    updated_vals = np.unique(updated_data)
                    print(f" ✓ converted from {unique_vals} to {updated_vals}")
                else:
                    print(f" ⚠️ has unexpected values: {unique_vals}")
        
        except Exception as e:
            print(f" ❌ error: {str(e)}")
    
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_tif_values.py <input_dir>")
        return 1
    
    input_dir = sys.argv[1]
    success = fix_tif_values(input_dir)
    
    if success:
        print("\nPS-10 compliance fix complete!")
        print("All TIF files now use 0 for no change and 1 for change")
    else:
        print("\nFailed to process files")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())