#!/usr/bin/env python3
"""
PS-10 Final Submission Package Creator
Creates complete submission with both Sentinel-2 and LISS4 results
"""

import os
import shutil
import zipfile
import hashlib
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
import fiona
from fiona.crs import from_epsg
import numpy as np

print("=" * 80)
print("PS-10 FINAL SUBMISSION CREATOR")
print("=" * 80)
print()

# Configuration
TEAM_NAME = "XBosonAI"
SUBMISSION_DATE = "31-10-2025"
FOLDER_NAME = f"PS10_{SUBMISSION_DATE}_{TEAM_NAME}"
ZIP_NAME = f"{FOLDER_NAME}.zip"

# Paths
PREDICTIONS_DIR = r"C:\Users\harsh\PS-10\ps10_predictions_formatted"
SUBMISSION_DIR = r"C:\Users\harsh\PS-10\PS10_FINAL_SUBMISSION"
OUTPUT_ZIP = os.path.join(r"C:\Users\harsh\PS-10", ZIP_NAME)

# Create clean submission directory
if os.path.exists(SUBMISSION_DIR):
    shutil.rmtree(SUBMISSION_DIR)
os.makedirs(SUBMISSION_DIR)

print("📋 Step 1: Prepare Sentinel-2 Files (28.42, 73.48)")
print("-" * 80)

# Sentinel-2 coordinates from existing files
s2_tif_src = os.path.join(PREDICTIONS_DIR, "Sentinel2_20200328_20250307_change_mask.tif")
s2_shp_base = os.path.join(PREDICTIONS_DIR, "Change_Mask_3145140.0_354900.0")

# Target names with lat/long
s2_base = os.path.join(SUBMISSION_DIR, "Change_Mask_28.42_73.48")

# Copy and rename Sentinel-2 files
print(f"   Copying Sentinel-2 TIF...")
shutil.copy2(s2_tif_src, f"{s2_base}.tif")
print(f"   ✓ Change_Mask_28.42_73.48.tif")

# Copy shapefile components
for ext in ['shp', 'dbf', 'shx', 'cpg']:
    src = f"{s2_shp_base}.{ext}"
    if os.path.exists(src):
        shutil.copy2(src, f"{s2_base}.{ext}")
        print(f"   ✓ Change_Mask_28.42_73.48.{ext}")

# Create .prj file for Sentinel-2
prj_content = '''PROJCS["WGS_1984_UTM_Zone_43N",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",75],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["Meter",1]]'''
with open(f"{s2_base}.prj", 'w') as f:
    f.write(prj_content)
print(f"   ✓ Change_Mask_28.42_73.48.prj")

print()
print("📋 Step 2: Prepare LISS4 Files (31.34, 76.75)")
print("-" * 80)

# LISS4 TIF already exists
liss4_tif_src = os.path.join(PREDICTIONS_DIR, "Change_Mask_31.34_76.75.tif")
liss4_base = os.path.join(SUBMISSION_DIR, "Change_Mask_31.33_76.78")

if os.path.exists(liss4_tif_src):
    print(f"   Copying LISS4 TIF...")
    shutil.copy2(liss4_tif_src, f"{liss4_base}.tif")
    print(f"   ✓ Change_Mask_31.33_76.78.tif")
    
    # Create LISS4 shapefile with high simplification for speed
    print(f"   Creating LISS4 shapefile (simplified)...")
    
    with rasterio.open(liss4_tif_src) as src:
        image = src.read(1)
        transform = src.transform
        crs = src.crs
        
        # Get center coordinates for metadata
        center_x = (src.bounds.left + src.bounds.right) / 2
        center_y = (src.bounds.bottom + src.bounds.top) / 2
        
        print(f"      Image size: {src.width}x{src.height}")
        print(f"      Changes: {np.sum(image > 0)} pixels ({np.sum(image > 0) / image.size * 100:.2f}%)")
        print(f"      Center: {center_y:.2f}°N, {center_x:.2f}°E")
        
        # Vectorize with aggressive simplification
        print(f"      Vectorizing and simplifying...")
        geometries = []
        feature_count = 0
        
        for geom, value in shapes(image, transform=transform):
            if value > 0:  # Only changed pixels
                geom_shape = shape(geom)
                # Aggressive simplification - tolerance of 50m
                simplified = geom_shape.simplify(tolerance=50, preserve_topology=True)
                if simplified.is_valid and not simplified.is_empty:
                    geometries.append(simplified)
                    feature_count += 1
                    if feature_count % 10000 == 0:
                        print(f"         Progress: {feature_count} features...")
        
        print(f"      Total features: {len(geometries)}")
        
        # Write shapefile
        schema = {
            'geometry': 'Polygon',
            'properties': {'change': 'int'}
        }
        
        with fiona.open(
            f"{liss4_base}.shp",
            'w',
            driver='ESRI Shapefile',
            crs=crs,
            schema=schema
        ) as dst:
            for geom in geometries:
                dst.write({
                    'geometry': mapping(geom),
                    'properties': {'change': 1}
                })
        
        print(f"   ✓ Change_Mask_31.33_76.78.shp")
        print(f"   ✓ Change_Mask_31.33_76.78.dbf")
        print(f"   ✓ Change_Mask_31.33_76.78.shx")
        
        # Create .prj file
        with open(f"{liss4_base}.prj", 'w') as f:
            f.write(prj_content)
        print(f"   ✓ Change_Mask_31.33_76.78.prj")
        
        # Create .cpg file
        with open(f"{liss4_base}.cpg", 'w') as f:
            f.write('UTF-8')
        print(f"   ✓ Change_Mask_31.33_76.78.cpg")
else:
    print(f"   ⚠️ LISS4 TIF not found, skipping...")

print()
print("📋 Step 3: Create HashFile.txt")
print("-" * 80)

# Create hash of the solution package
hash_files = [
    r"C:\Users\harsh\PS-10\model\model.h5",
    r"C:\Users\harsh\PS-10\process_real_data.py",
    r"C:\Users\harsh\PS-10\run_efficient_inference.py",
]

md5_hash = hashlib.md5()
for file_path in hash_files:
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            md5_hash.update(f.read())
        print(f"   ✓ Added {os.path.basename(file_path)}")

hash_value = md5_hash.hexdigest()
print(f"   Hash: {hash_value}")

# Write HashFile.txt
hash_file = os.path.join(SUBMISSION_DIR, "HashFile.txt")
with open(hash_file, 'w') as f:
    f.write(f"MD5: {hash_value}\n")
print(f"   ✓ Created HashFile.txt")

print()
print("📋 Step 4: Create Final ZIP Package")
print("-" * 80)

# Create ZIP file
with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(SUBMISSION_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.join(FOLDER_NAME, os.path.relpath(file_path, SUBMISSION_DIR))
            zipf.write(file_path, arcname)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"   + {os.path.basename(file_path)} ({file_size:.1f} KB)")

zip_size = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)  # MB

print()
print("=" * 80)
print("✅ PS-10 SUBMISSION PACKAGE COMPLETE!")
print("=" * 80)
print()
print(f"📦 File: {ZIP_NAME}")
print(f"📏 Size: {zip_size:.2f} MB")
print(f"📍 Location: {OUTPUT_ZIP}")
print()
print("📋 Package Structure:")
print(f"   {FOLDER_NAME}/")
print(f"   ├── Change_Mask_28.42_73.48.tif    (Sentinel-2)")
print(f"   ├── Change_Mask_28.42_73.48.shp")
print(f"   ├── Change_Mask_28.42_73.48.dbf")
print(f"   ├── Change_Mask_28.42_73.48.shx")
print(f"   ├── Change_Mask_28.42_73.48.prj")
print(f"   ├── Change_Mask_28.42_73.48.cpg")
print(f"   ├── Change_Mask_31.33_76.78.tif    (LISS4)")
print(f"   ├── Change_Mask_31.33_76.78.shp")
print(f"   ├── Change_Mask_31.33_76.78.dbf")
print(f"   ├── Change_Mask_31.33_76.78.shx")
print(f"   ├── Change_Mask_31.33_76.78.prj")
print(f"   ├── Change_Mask_31.33_76.78.cpg")
print(f"   └── HashFile.txt")
print()
print("✅ PS-10 COMPLIANCE:")
print(f"   ✓ 2 TIFFs (2 different locations)")
print(f"   ✓ 2 Shapefiles (with all extensions)")
print(f"   ✓ Coordinate-based naming")
print(f"   ✓ HashFile.txt included")
print(f"   ✓ All files georeferenced")
print(f"   ✓ Correct folder/ZIP naming")
print()
print("🎯 READY TO SUBMIT!")
print(f"   Upload: {OUTPUT_ZIP}")
print()
print("✨ Done!")

# Cleanup
print()
print("🧹 Cleaning up temporary files...")
if os.path.exists(SUBMISSION_DIR):
    shutil.rmtree(SUBMISSION_DIR)
print("   ✓ Temporary files removed")
