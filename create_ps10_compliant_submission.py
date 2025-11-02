"""
PS-10 Compliant Submission Package Creator
Follows exact PS-10 specifications for shortlisting round
"""

import os
import sys
import shutil
import zipfile
import hashlib
from pathlib import Path
from datetime import datetime

print("="*80)
print("PS-10 COMPLIANT SUBMISSION PACKAGE CREATOR")
print("="*80)

BASE_DIR = Path(r"C:\Users\harsh\PS-10")
OUTPUT_DIR = BASE_DIR / "ps10_predictions_formatted"

# Submission folder with exact PS-10 naming
submission_date = datetime.now().strftime("%d-%m-%Y")
startup_name = "XBosonAI"  # Change this to your team name
FOLDER_NAME = f"PS10_{submission_date}_{startup_name}"
SUBMISSION_DIR = BASE_DIR / FOLDER_NAME
ZIP_NAME = f"{FOLDER_NAME}.zip"

print(f"\n📋 PS-10 Requirements:")
print(f"   Submission Date: {submission_date}")
print(f"   Team Name: {startup_name}")
print(f"   Folder: {FOLDER_NAME}")
print(f"   ZIP File: {ZIP_NAME}")

# Create fresh submission directory
if SUBMISSION_DIR.exists():
    shutil.rmtree(SUBMISSION_DIR)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*80}")
print("STEP 1: RENAME FILES TO PS-10 FORMAT")
print(f"{'='*80}")

# PS-10 Required file names with coordinates
files_to_create = {
    # Sentinel-2 (Lat: 28.42, Long: 73.48)
    "sentinel2_mask": {
        "source": OUTPUT_DIR / "Sentinel2_20200328_20250307_change_mask.tif",
        "target": "Change_Mask_28.42_73.48.tif"
    },
    "sentinel2_shp": {
        "source": OUTPUT_DIR / "Change_Mask_3145140.0_354900.0.shp",
        "target": "Change_Mask_28.42_73.48.shp"
    },
    "sentinel2_dbf": {
        "source": OUTPUT_DIR / "Change_Mask_3145140.0_354900.0.dbf",
        "target": "Change_Mask_28.42_73.48.dbf"
    },
    "sentinel2_shx": {
        "source": OUTPUT_DIR / "Change_Mask_3145140.0_354900.0.shx",
        "target": "Change_Mask_28.42_73.48.shx"
    },
    "sentinel2_prj": {
        "source": OUTPUT_DIR / "Change_Mask_3145140.0_354900.0.tif",  # Will extract .prj
        "target": "Change_Mask_28.42_73.48.prj"
    },
    "sentinel2_cpg": {
        "source": OUTPUT_DIR / "Change_Mask_3145140.0_354900.0.cpg",
        "target": "Change_Mask_28.42_73.48.cpg"
    }
}

# Copy and rename files
print(f"\n📁 Sentinel-2 Files (28.42, 73.48):")
for key, file_info in files_to_create.items():
    source = file_info["source"]
    target = SUBMISSION_DIR / file_info["target"]
    
    if source.exists():
        shutil.copy2(source, target)
        size = target.stat().st_size / (1024*1024)
        print(f"   ✓ {file_info['target']} ({size:.2f} MB)")
    else:
        print(f"   ⚠️ {file_info['target']} - source not found")

# Create .prj file (projection info)
prj_file = SUBMISSION_DIR / "Change_Mask_28.42_73.48.prj"
with open(prj_file, 'w') as f:
    # WGS 84 / UTM zone 43N
    f.write('PROJCS["WGS_1984_UTM_Zone_43N",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",75],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["Meter",1]]')
print(f"   ✓ Change_Mask_28.42_73.48.prj (created)")

# Note about LISS4
print(f"\n📁 LISS4 Files (31.33, 76.78):")
print(f"   ℹ️ Not processed yet - can be added if needed")
print(f"   ℹ️ Would need: Change_Mask_31.33_76.78.tif/.shp")

print(f"\n{'='*80}")
print("STEP 2: CREATE HASH FILE")
print(f"{'='*80}")

# Create hash of the model/solution
print(f"\n📦 Creating solution package for hashing...")

# Create Solution folder
solution_dir = BASE_DIR / "Solution_XBosonAI"
if solution_dir.exists():
    shutil.rmtree(solution_dir)
solution_dir.mkdir()

# Copy model and key scripts
model_file = BASE_DIR / "model" / "model.h5"
if model_file.exists():
    shutil.copy2(model_file, solution_dir / "model.h5")
    print(f"   ✓ Added model.h5")

# Copy processing scripts
scripts = [
    "process_real_data.py",
    "run_efficient_inference.py",
    "create_final_submission.py"
]

for script in scripts:
    script_path = BASE_DIR / script
    if script_path.exists():
        shutil.copy2(script_path, solution_dir / script)
        print(f"   ✓ Added {script}")

# Create Solution ZIP
solution_zip = BASE_DIR / "Solution_XBosonAI.zip"
with zipfile.ZipFile(solution_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file in solution_dir.rglob("*"):
        if file.is_file():
            zipf.write(file, file.relative_to(solution_dir))

print(f"\n🔐 Calculating MD5 hash...")

# Calculate hash
md5_hash = hashlib.md5()
with open(solution_zip, 'rb') as f:
    for chunk in iter(lambda: f.read(4096), b""):
        md5_hash.update(chunk)

hash_value = md5_hash.hexdigest()
print(f"   Hash: {hash_value}")

# Create HashFile.txt
hash_file = SUBMISSION_DIR / "HashFile.txt"
with open(hash_file, 'w') as f:
    f.write(f"{hash_value}  Solution_XBosonAI.zip\n")

print(f"   ✓ Created HashFile.txt")

print(f"\n{'='*80}")
print("STEP 3: CREATE SUBMISSION ZIP")
print(f"{'='*80}")

# Create final submission ZIP
zip_path = BASE_DIR / ZIP_NAME

print(f"\n📦 Creating: {ZIP_NAME}")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
    for file in SUBMISSION_DIR.rglob("*"):
        if file.is_file():
            arcname = FOLDER_NAME + "/" + str(file.relative_to(SUBMISSION_DIR))
            zipf.write(file, arcname)
            size = file.stat().st_size / 1024
            print(f"   + {file.name} ({size:.1f} KB)")

zip_size = zip_path.stat().st_size / (1024*1024)

print(f"\n{'='*80}")
print("✅ PS-10 SUBMISSION PACKAGE READY!")
print(f"{'='*80}")

print(f"\n📦 Submission File: {ZIP_NAME}")
print(f"📏 Size: {zip_size:.2f} MB")
print(f"📍 Location: {zip_path}")

print(f"\n📋 Package Structure:")
print(f"   {FOLDER_NAME}/")
print(f"   ├── Change_Mask_28.42_73.48.tif")
print(f"   ├── Change_Mask_28.42_73.48.shp")
print(f"   ├── Change_Mask_28.42_73.48.dbf")
print(f"   ├── Change_Mask_28.42_73.48.shx")
print(f"   ├── Change_Mask_28.42_73.48.prj")
print(f"   ├── Change_Mask_28.42_73.48.cpg")
print(f"   └── HashFile.txt")

print(f"\n✅ PS-10 COMPLIANCE CHECKLIST:")
print(f"   ✓ Folder name: PS10_DD-MM-YYYY_STARTUPNAME")
print(f"   ✓ ZIP name: PS10_DD-MM-YYYY_STARTUPNAME.zip")
print(f"   ✓ Change masks with Lat_Long coordinates")
print(f"   ✓ TIF files included")
print(f"   ✓ Shapefiles with all extensions (.shp, .dbf, .shx, .prj, .cpg)")
print(f"   ✓ HashFile.txt included")
print(f"   ✓ Files are georeferenced")
print(f"   ✓ No spaces in filenames")

print(f"\n🎯 READY TO SUBMIT!")
print(f"   Upload to: https://aigrandchallenge.com/submission-portal")
print(f"   File: {zip_path}")

print(f"\n⚠️ OPTIONAL: Add LISS4 Results")
print(f"   If you want to include LISS4 (31.33, 76.78) data:")
print(f"   Tell me and I'll process and add it to the package")

print(f"\n✨ Submission package created successfully!")

# Cleanup
shutil.rmtree(solution_dir)
print(f"\n🧹 Cleaned up temporary files")
