#!/usr/bin/env python
"""
PS-10 Output File Formatter

This script renames and organizes the existing output files from our change detection 
pipeline to match the naming requirements specified in PS-10:

1. Output file naming: 'Change_Mask_Lat_Long.tif' (Raster) and 'Change_Mask_Lat_Long.shp' (Vector)
2. Creates the proper directory structure for submission

Usage:
    python format_ps10_outputs.py <input_dir> <output_dir> [--force]
"""

import os
import shutil
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime


def parse_image_identifier(filename):
    """
    Extract the image pair identifier from our internal naming format
    and map it to the required PS-10 latitude/longitude format.
    
    Example: 0_15_change_mask.tif -> coordinates from reference table
    """
    # Extract image pair ID from our current naming convention
    match = re.match(r'(\d+)_(\d+)_change_mask\.tif', filename)
    if not match:
        return None
    
    # Reference table for PS-10 locations from the documentation
    # See section 6.i in PS-10 documentation
    ps10_locations = {
        # These are examples - replace with actual mappings from PS-10 guidelines
        "0_10": "34.0531_74.3909",  # Snow
        "0_11": "13.3143_77.6157",  # Plain
        "0_12": "31.2834_76.7904",  # Hill
        "0_13": "26.9027_70.9543",  # Desert
        "0_14": "23.7380_84.2129",  # Forest
        "0_15": "28.1740_77.6126",  # Urban
        # Add more mappings as needed
    }
    
    image_id = f"{match.group(1)}_{match.group(2)}"
    if image_id in ps10_locations:
        return ps10_locations[image_id]
    
    # If no mapping is found, use the original ID
    return image_id


def format_outputs(input_dir, output_dir, force=False):
    """
    Format output files to match PS-10 requirements
    
    Args:
        input_dir: Directory containing our current output files
        output_dir: Directory where formatted files will be placed
        force: If True, overwrite existing output directory
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Check if input directory exists
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"❌ Error: Input directory '{input_dir}' does not exist or is not a directory")
        return False
    
    # Check if output directory exists and handle accordingly
    if output_dir.exists():
        if not force:
            print(f"❌ Error: Output directory '{output_dir}' already exists. Use --force to overwrite")
            return False
        shutil.rmtree(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Formatting outputs from '{input_dir}' to match PS-10 requirements")
    print(f"Saving to: '{output_dir}'\n")
    
    # Find all raster files
    raster_files = list(input_dir.glob("*_change_mask.tif"))
    
    if not raster_files:
        print("❌ Error: No change_mask.tif files found in the input directory")
        return False
    
    processed_count = 0
    
    for raster_file in raster_files:
        # Get the location identifier for PS-10 format
        location = parse_image_identifier(raster_file.name)
        if not location:
            print(f"⚠️ Warning: Could not parse location from {raster_file.name}, skipping")
            continue
        
        # Create PS-10 compliant filenames
        ps10_raster = output_dir / f"Change_Mask_{location}.tif"
        
        # Copy and rename the raster file
        shutil.copy2(raster_file, ps10_raster)
        print(f"✓ Created {ps10_raster.name}")
        
        # Handle shapefile components
        shapefile_base = raster_file.stem.replace("_change_mask", "_change_vectors")
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
            source = input_dir / f"{shapefile_base}{ext}"
            if source.exists():
                dest = output_dir / f"Change_Mask_{location}{ext}"
                shutil.copy2(source, dest)
                print(f"✓ Created {dest.name}")
            else:
                print(f"⚠️ Warning: Missing {ext} component for {location}")
        
        processed_count += 1
    
    print(f"\n✅ Successfully formatted {processed_count} image pairs for PS-10 submission")
    print(f"   • {len(list(output_dir.glob('*.tif')))} GeoTIFF files created")
    print(f"   • {len(list(output_dir.glob('*.shp')))} Shapefiles created")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format change detection outputs to match PS-10 requirements")
    parser.add_argument("input_dir", help="Directory containing change detection outputs")
    parser.add_argument("output_dir", help="Directory where formatted outputs will be saved")
    parser.add_argument("--force", action="store_true", help="Overwrite output directory if it exists")
    
    args = parser.parse_args()
    
    format_outputs(args.input_dir, args.output_dir, args.force)