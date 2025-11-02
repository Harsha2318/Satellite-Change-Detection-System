#!/usr/bin/env python3
"""
PS-10 Final Submission Package Creator (Windows Compatible)

This script creates a final PS-10 compliant submission package with:
1. Proper file naming: 'Change_Mask_Lat_Long.tif'
2. Correct TIF values: 0 for no change, 1 for change
3. Complete shapefiles with all components
4. MD5 hash of the model file

NOTE: This version is Windows PowerShell compatible (ASCII only)
"""

import os
import sys
import shutil
import hashlib
from pathlib import Path
import zipfile
from datetime import datetime

# Set UTF-8 encoding for Windows
os.environ["PYTHONIOENCODING"] = "utf-8"

try:
    import rasterio
    import numpy as np
except ImportError:
    print("ERROR: rasterio and numpy are required.")
    print("Install with: pip install rasterio numpy")
    sys.exit(1)

def calculate_model_md5(model_path):
    """Calculate MD5 hash of model file as required by PS-10"""
    if not os.path.exists(model_path):
        print(f"ERROR: Model file {model_path} not found")
        return None
        
    try:
        hash_md5 = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"ERROR calculating MD5: {str(e)}")
        return None

def verify_tif_values(tif_file):
    """Verify TIF values are 0/1 as required by PS-10"""
    try:
        with rasterio.open(tif_file) as src:
            data = src.read(1)
            unique_vals = np.unique(data)
            
            if set(unique_vals).issubset({0, 1}):
                print(f"  [OK] {tif_file.name}: Values = {unique_vals}")
                return True
            else:
                print(f"  [ERROR] {tif_file.name}: Invalid values: {unique_vals}")
                print("     PS-10 requires: 1 for change, 0 for no change")
                return False
    except Exception as e:
        print(f"  [ERROR] Reading {tif_file}: {str(e)}")
        return False

def create_submission_package(predictions_dir, model_path, team_name):
    """Create PS-10 submission package from verified predictions"""
    print(f"Creating PS-10 submission package...")
    
    pred_path = Path(predictions_dir)
    if not pred_path.exists():
        print(f"ERROR: Predictions directory {predictions_dir} not found")
        return None
        
    if not os.path.exists(model_path):
        print(f"ERROR: Model file {model_path} not found")
        return None
        
    # Calculate model MD5 hash
    model_md5 = calculate_model_md5(model_path)
    if not model_md5:
        print("ERROR calculating model MD5 hash")
        return None
    
    print(f"\nModel file: {model_path}")
    print(f"Model hash (MD5): {model_md5}")
    print(f"Processing: {predictions_dir}\n")
    
    # Create submission folder name: PS10_[DD-MMM-YYYY]_[TeamName]
    today = datetime.now()
    date_str = today.strftime("%d-%b-%Y")
    team_name_clean = team_name.replace(" ", "")
    submission_dir_name = f"PS10_{date_str}_{team_name_clean}"
    output_zip = f"{submission_dir_name}.zip"
    
    print(f"Output submission file: {output_zip}\n")
    
    # Create submission directory
    submission_dir = Path(submission_dir_name)
    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    submission_dir.mkdir(exist_ok=True)
    
    print(f"Creating PS-10 directory structure...")
    print(f"Copying model file...")
    model_filename = os.path.basename(model_path)
    shutil.copy2(model_path, submission_dir / model_filename)
    
    # Find all TIF files
    print("Processing TIF files...")
    change_mask_files = list(pred_path.glob("*_change_mask.tif"))
    if not change_mask_files:
        change_mask_files = list(pred_path.glob("*.tif"))
    
    if not change_mask_files:
        print(f"ERROR: No TIF files found in {predictions_dir}")
        return None
        
    print(f"Found {len(change_mask_files)} TIF files\n")
    
    processed_locations = []
    
    for tif_file in change_mask_files:
        # Get location identifier from filename
        if "_change_mask.tif" in tif_file.name:
            location = tif_file.name.split("_change_mask.tif")[0]
        else:
            parts = tif_file.stem.split("_")
            if len(parts) >= 2:
                location = f"{parts[-2]}_{parts[-1]}"
            else:
                location = tif_file.stem
        
        processed_locations.append(location)
        
        # Check if already in PS-10 format
        if tif_file.name.startswith("Change_Mask_"):
            dest_tif = submission_dir / tif_file.name
        else:
            dest_tif = submission_dir / f"Change_Mask_{location}.tif"
        
        # Copy and verify TIF file
        shutil.copy2(tif_file, dest_tif)
        verify_tif_values(dest_tif)
        
        # Look for shapefile components
        shapefile_base = str(tif_file).replace("_change_mask.tif", "_change_vectors")
        if not os.path.exists(f"{shapefile_base}.shp"):
            shapefile_base = str(tif_file).replace(".tif", "")
            
        shapefile_extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
        for ext in shapefile_extensions:
            shp_file = Path(f"{shapefile_base}{ext}")
            if shp_file.exists():
                dest_shp = submission_dir / f"Change_Mask_{location}{ext}"
                shutil.copy2(shp_file, dest_shp)
                print(f"  + {dest_shp.name}")
            elif ext != '.cpg':
                print(f"  - WARNING: Missing {shapefile_base}{ext}")
                
    # Save model MD5 hash to file
    with open(submission_dir / "model_md5.txt", "w") as f:
        f.write(model_md5)
    
    print("\nAll files processed successfully!")
    
    # Create zip file
    print("Creating submission package...")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in submission_dir.glob("*"):
            zipf.write(file, file.name)
    
    zip_size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"SUCCESS: {output_zip}")
    print(f"File size: {zip_size_mb:.2f} MB")
    
    return output_zip

def main():
    if len(sys.argv) < 3:
        print("Usage: python prepare_ps10_final_fixed.py <predictions_dir> <model_path> [team_name]")
        print("\nExample:")
        print("  python prepare_ps10_final_fixed.py predictions_final model/model.h5 XBosonAI")
        return 1
    
    predictions_dir = sys.argv[1]
    model_path = sys.argv[2]
    team_name = sys.argv[3] if len(sys.argv) > 3 else "XBoson"
    
    create_submission_package(predictions_dir, model_path, team_name)
    return 0

if __name__ == "__main__":
    sys.exit(main())