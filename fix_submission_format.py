#!/usr/bin/env python3
"""
PS-10 CRITICAL FIX: Correct Submission Filename Format

THE ISSUE: Previous submissions failed because filenames were wrong!
- ❌ Wrong: Change_Mask_0_0.tif, Change_Mask_22_28.tif (integer lat/long)
- ✅ Right: Change_Mask_LAT_LONG.tif (decimal coordinates from image metadata!)

This script:
1. Extracts lat/long from GeoTIFF metadata
2. Renames files to: Change_Mask_LAT_LONG.{tif,shp,shx,dbf,prj,cpg}
3. Creates compliant ZIP for submission
"""

import os
import sys
import shutil
import hashlib
import zipfile
from pathlib import Path
from datetime import datetime

try:
    import rasterio
    import geopandas as gpd
    import numpy as np
except ImportError:
    print("ERROR: Required packages missing!")
    print("Install with: pip install rasterio geopandas shapely")
    sys.exit(1)


def extract_lat_long_from_tif(tif_path):
    """Extract latitude and longitude from GeoTIFF bounds"""
    try:
        with rasterio.open(tif_path) as src:
            # Get bounds (left, bottom, right, top)
            bounds = src.bounds
            
            # Get center coordinates
            center_long = (bounds.left + bounds.right) / 2
            center_lat = (bounds.bottom + bounds.top) / 2
            
            return round(center_lat, 4), round(center_long, 4)
    except Exception as e:
        print(f"Error reading {tif_path}: {e}")
        return None, None


def get_shapefile_base_name(shp_path):
    """Get the base name without extension for shapefile components"""
    return str(shp_path).rsplit('.', 1)[0]


def rename_predictions_with_correct_format(input_dir, output_dir):
    """Rename all prediction files to PS-10 correct format"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return False
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("PS-10 FILENAME FORMAT CORRECTION")
    print("="*70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Find all TIF files
    tif_files = sorted(input_path.glob("*change_mask.tif")) + sorted(input_path.glob("*.tif"))
    tif_files = list(set(tif_files))  # Remove duplicates
    
    if not tif_files:
        print("❌ No TIF files found in input directory!")
        return False
    
    print(f"Found {len(tif_files)} TIF files\n")
    
    processed_files = set()
    renamed_count = 0
    
    for tif_file in tif_files:
        # Skip if already processed (avoid duplicates)
        if str(tif_file) in processed_files:
            continue
        
        # Extract lat/long from metadata
        lat, long = extract_lat_long_from_tif(str(tif_file))
        
        if lat is None or long is None:
            print(f"⚠️  Skipped {tif_file.name}: Could not extract coordinates")
            continue
        
        # Create new filename: Change_Mask_LAT_LONG.tif
        new_tif_name = f"Change_Mask_{lat}_{long}.tif"
        new_tif_path = output_path / new_tif_name
        
        # Copy TIF file
        try:
            shutil.copy2(tif_file, new_tif_path)
            print(f"✅ {tif_file.name} → {new_tif_name}")
            processed_files.add(str(tif_file))
            renamed_count += 1
        except Exception as e:
            print(f"❌ Failed to copy {tif_file.name}: {e}")
            continue
        
        # Now handle associated shapefiles
        # Look for pattern: find *_change_vectors.shp or similar
        base_name_patterns = [
            str(tif_file.stem).replace("_change_mask", ""),
            str(tif_file.stem).replace(".tif", "").replace("_mask", ""),
        ]
        
        # Find all shapefile components
        shp_base = None
        for shp_candidate in input_path.glob("*change_vectors.shp"):
            # Check if this shapefile corresponds to our TIF
            candidate_stem = str(shp_candidate.stem).replace("_change_vectors", "")
            tif_stem = str(tif_file.stem).replace("_change_mask", "")
            
            if candidate_stem == tif_stem:
                shp_base = shp_candidate
                break
        
        # If we found a shapefile, copy all its components with new name
        if shp_base:
            shp_extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
            new_shp_base = f"Change_Mask_{lat}_{long}"
            
            for ext in shp_extensions:
                old_file = input_path / f"{shp_base.stem}{ext}"
                if old_file.exists():
                    new_file = output_path / f"{new_shp_base}{ext}"
                    try:
                        shutil.copy2(old_file, new_file)
                    except Exception as e:
                        print(f"  ⚠️  Could not copy {ext} component: {e}")
    
    print(f"\n✅ Renamed {renamed_count} TIF files with correct PS-10 format")
    return True


def create_submission_zip(formatted_dir, model_path, team_name):
    """Create final submission ZIP with model hash"""
    
    formatted_path = Path(formatted_dir)
    
    if not formatted_path.exists():
        print(f"❌ Formatted directory not found: {formatted_dir}")
        return None
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    # Calculate model MD5
    print(f"\nCalculating model MD5 hash...")
    try:
        hash_md5 = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        model_hash = hash_md5.hexdigest()
        print(f"✅ Model hash: {model_hash}")
    except Exception as e:
        print(f"❌ Error calculating hash: {e}")
        return None
    
    # Create ZIP filename
    today = datetime.now()
    date_str = today.strftime("%d-%b-%Y")  # DD-MMM-YYYY
    team_name_clean = team_name.replace(" ", "")
    zip_name = f"PS10_{date_str}_{team_name_clean}.zip"
    
    print(f"\nCreating submission ZIP: {zip_name}")
    
    try:
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add all formatted files
            for file_path in sorted(formatted_path.glob("*")):
                if file_path.is_file():
                    arcname = file_path.name
                    zf.write(file_path, arcname)
                    print(f"  Added: {arcname}")
            
            # Add model hash file
            hash_file = "model_md5.txt"
            with open(hash_file, 'w') as f:
                f.write(f"{model_hash}\n")
            zf.write(hash_file, hash_file)
            print(f"  Added: {hash_file}")
    
    except Exception as e:
        print(f"❌ Error creating ZIP: {e}")
        return None
    
    print(f"\n✅ Submission package created: {zip_name}")
    
    # Print instructions
    print("\n" + "="*70)
    print("SUBMISSION CHECKLIST")
    print("="*70)
    print(f"✅ ZIP file created: {zip_name}")
    print(f"✅ Model hash saved: {hash_file}")
    print(f"✅ Filenames corrected to PS-10 format")
    print()
    print("ON OCTOBER 31 AT SUBMISSION TIME:")
    print(f"1. Upload: {zip_name}")
    print(f"2. Upload hash from: {hash_file}")
    print(f"3. Submit before 16:00 (4 PM) IST")
    print("="*70)
    
    return zip_name


def main():
    """Main execution"""
    
    if len(sys.argv) < 3:
        print("Usage: python fix_submission_format.py <input_predictions_dir> <model_path> [team_name]")
        print()
        print("Example:")
        print("  python fix_submission_format.py predictions_final models/xboson_change_detector.pt 'XBoson AI'")
        print()
        print("This will:")
        print("1. Extract lat/long from GeoTIFF metadata")
        print("2. Rename files to: Change_Mask_LAT_LONG.{tif,shp,...}")
        print("3. Create PS10_[DATE]_[TEAM].zip submission package")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    model_path = sys.argv[2]
    team_name = sys.argv[3] if len(sys.argv) > 3 else "XBoson"
    
    # Step 1: Rename with correct format
    formatted_dir = f"{input_dir}_formatted"
    if not rename_predictions_with_correct_format(input_dir, formatted_dir):
        print("❌ Format correction failed!")
        sys.exit(1)
    
    # Step 2: Create submission ZIP
    zip_file = create_submission_zip(formatted_dir, model_path, team_name)
    if not zip_file:
        print("❌ ZIP creation failed!")
        sys.exit(1)
    
    print("\n✅ ALL DONE! Ready for submission on October 31")


if __name__ == "__main__":
    main()
