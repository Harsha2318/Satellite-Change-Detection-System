"""
Quick Submission Package Creator
Uses existing results and completes missing parts
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

print("="*80)
print("PS-10 QUICK SUBMISSION PACKAGE")
print("="*80)

BASE_DIR = Path(r"C:\Users\harsh\PS-10")
OUTPUT_DIR = BASE_DIR / "ps10_predictions_formatted"
SUBMISSION_DIR = BASE_DIR / "PS10_Final_Submission"
ZIP_NAME = f"PS10_{datetime.now().strftime('%d-%b-%Y')}_ChangeDetection.zip"

print(f"\n📁 Source: {OUTPUT_DIR}")
print(f"📦 Target: {ZIP_NAME}")

# Check existing files
existing = list(OUTPUT_DIR.glob("*"))
print(f"\n📊 Found {len(existing)} files:")
for f in existing:
    if f.is_file():
        size = f.stat().st_size / (1024*1024)
        print(f"   - {f.name} ({size:.2f} MB)")

# Create submission directory
if SUBMISSION_DIR.exists():
    shutil.rmtree(SUBMISSION_DIR)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Copy files
print(f"\n📋 Copying files...")
for file in existing:
    if file.is_file():
        dest = SUBMISSION_DIR / file.name
        shutil.copy2(file, dest)
        print(f"   ✓ {file.name}")

# Create ZIP
print(f"\n📦 Creating ZIP package...")
zip_path = BASE_DIR / ZIP_NAME

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
    for file in SUBMISSION_DIR.rglob("*"):
        if file.is_file():
            arcname = file.relative_to(SUBMISSION_DIR)
            zipf.write(file, arcname)
            print(f"   + {arcname}")

zip_size = zip_path.stat().st_size / (1024*1024)

print(f"\n{'='*80}")
print(f"✅ SUBMISSION PACKAGE CREATED!")
print(f"{'='*80}")
print(f"\n📦 File: {zip_path.name}")
print(f"📏 Size: {zip_size:.2f} MB")
print(f"📍 Location: {zip_path}")

print(f"\n📋 Package contains:")
with zipfile.ZipFile(zip_path, 'r') as zipf:
    for info in zipf.filelist:
        print(f"   - {info.filename} ({info.file_size / 1024:.1f} KB)")

print(f"\n🎯 READY FOR SUBMISSION!")
print(f"   Upload: {zip_path}")

print(f"\n✨ Done!")
