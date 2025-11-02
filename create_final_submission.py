"""
PS-10 Complete Workflow and Submission Package Creator
October 31, 2025
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PS-10 COMPLETE WORKFLOW & SUBMISSION CREATOR")
print("="*80)

# Paths
BASE_DIR = Path(r"C:\Users\harsh\PS-10")
INPUT_DIR = BASE_DIR / "ps10_predictions"
OUTPUT_DIR = BASE_DIR / "ps10_predictions_formatted"
SUBMISSION_DIR = BASE_DIR / "PS10_Submission_Final"

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

try:
    import rasterio
    from rasterio.features import shapes
    import fiona
    from shapely.geometry import shape, mapping
    import numpy as np
    print("✓ All libraries loaded")
except ImportError as e:
    print(f"❌ Missing: {e}")
    print("Installing required packages...")
    os.system("pip install rasterio fiona shapely scikit-image -q")
    import rasterio
    from rasterio.features import shapes
    import fiona
    from shapely.geometry import shape, mapping
    import numpy as np

def create_change_mask_and_vectors(t1_path, t2_path, output_name):
    """Create change mask and shapefile for an image pair"""
    print(f"\n{'='*80}")
    print(f"Processing: {output_name}")
    print(f"{'='*80}")
    
    with rasterio.open(t1_path) as src1, rasterio.open(t2_path) as src2:
        print(f"   Image size: {src1.width}x{src1.height}, {src1.count} bands")
        
        # Read images in tiles
        tile_size = 1024
        height, width = src1.height, src1.width
        change_mask = np.zeros((height, width), dtype=np.uint8)
        
        total_tiles = ((height + tile_size - 1) // tile_size) * ((width + tile_size - 1) // tile_size)
        processed = 0
        
        print(f"   Processing in {tile_size}x{tile_size} tiles...")
        
        for i in range(0, height, tile_size):
            for j in range(0, width, tile_size):
                i_end = min(i + tile_size, height)
                j_end = min(j + tile_size, width)
                
                window = rasterio.windows.Window(j, i, j_end - j, i_end - i)
                
                tile1 = np.transpose(src1.read(window=window), (1, 2, 0))
                tile2 = np.transpose(src2.read(window=window), (1, 2, 0))
                
                # Simple difference
                diff = np.abs(tile1.astype(float) - tile2.astype(float))
                if len(diff.shape) == 3:
                    diff = np.mean(diff, axis=2)
                
                if diff.max() > diff.min():
                    diff = (diff - diff.min()) / (diff.max() - diff.min())
                
                change_mask[i:i_end, j:j_end] = (diff > 0.15).astype(np.uint8)
                
                processed += 1
                if processed % 20 == 0 or processed == total_tiles:
                    print(f"   Progress: {processed}/{total_tiles} ({processed/total_tiles*100:.0f}%)")
        
        changed = change_mask.sum()
        print(f"   ✓ Changes detected: {changed} pixels ({changed/change_mask.size*100:.2f}%)")
        
        # Save GeoTIFF
        tif_path = OUTPUT_DIR / f"{output_name}_change_mask.tif"
        profile = src1.profile.copy()
        profile.update({'count': 1, 'dtype': 'uint8', 'compress': 'lzw'})
        
        with rasterio.open(tif_path, 'w', **profile) as dst:
            dst.write(change_mask, 1)
        print(f"   ✅ GeoTIFF: {tif_path.name}")
        
        # Create shapefile (simplified)
        print(f"   Creating shapefile...")
        shp_path = OUTPUT_DIR / f"{output_name}_change_vectors.shp"
        
        shapes_gen = shapes(change_mask.astype(np.int16), transform=src1.transform)
        geometries = []
        
        for geom, value in shapes_gen:
            if value == 1:
                poly = shape(geom)
                # Simplify and filter by area
                simplified = poly.simplify(20, preserve_topology=True)
                if simplified.area > 100:  # Keep only > 100 sq meters
                    geometries.append({
                        'geometry': mapping(simplified),
                        'properties': {'change': 1, 'area': int(simplified.area)}
                    })
        
        if geometries:
            schema = {'geometry': 'Polygon', 'properties': {'change': 'int', 'area': 'int'}}
            with fiona.open(str(shp_path), 'w', driver='ESRI Shapefile', crs=src1.crs, schema=schema) as dst:
                dst.writerecords(geometries)
            print(f"   ✅ Shapefile: {shp_path.name} ({len(geometries)} polygons)")
        
        return tif_path, shp_path

def create_submission_package():
    """Create final PS-10 submission ZIP"""
    print(f"\n{'='*80}")
    print("CREATING SUBMISSION PACKAGE")
    print(f"{'='*80}")
    
    # Clear submission directory
    if SUBMISSION_DIR.exists():
        shutil.rmtree(SUBMISSION_DIR)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    
    # Copy all output files
    print("\n   Organizing files...")
    copied = 0
    
    for file in OUTPUT_DIR.glob("*"):
        if file.is_file():
            dest = SUBMISSION_DIR / file.name
            shutil.copy2(file, dest)
            size = dest.stat().st_size / (1024*1024)
            print(f"   ✓ {file.name} ({size:.2f} MB)")
            copied += 1
    
    print(f"\n   Total files: {copied}")
    
    # Create ZIP
    timestamp = datetime.now().strftime("%d-%b-%Y")
    zip_name = f"PS10_{timestamp}_ChangeDetection.zip"
    zip_path = BASE_DIR / zip_name
    
    print(f"\n   Creating ZIP: {zip_name}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in SUBMISSION_DIR.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(SUBMISSION_DIR)
                zipf.write(file, arcname)
    
    zip_size = zip_path.stat().st_size / (1024*1024)
    print(f"   ✅ Created: {zip_path.name} ({zip_size:.2f} MB)")
    
    return zip_path

def main():
    print("\n" + "="*80)
    print("STEP 1: PROCESS IMAGE PAIRS")
    print("="*80)
    
    # Sentinel-2
    s2_t1 = INPUT_DIR / "Sentinel2_t1.tif"
    s2_t2 = INPUT_DIR / "Sentinel2_t2.tif"
    
    if s2_t1.exists() and s2_t2.exists():
        create_change_mask_and_vectors(s2_t1, s2_t2, "Sentinel2_20200328_20250307")
    else:
        print("   ⚠️ Sentinel-2 files not found")
    
    # LISS4
    liss_t1 = INPUT_DIR / "LISS4_t1.tif"
    liss_t2 = INPUT_DIR / "LISS4_t2.tif"
    
    if liss_t1.exists() and liss_t2.exists():
        create_change_mask_and_vectors(liss_t1, liss_t2, "LISS4_20200318_20250127")
    else:
        print("   ⚠️ LISS4 files not found")
    
    # Create submission
    print("\n" + "="*80)
    print("STEP 2: CREATE SUBMISSION ZIP")
    print("="*80)
    
    zip_path = create_submission_package()
    
    # Final summary
    print("\n" + "="*80)
    print("✅ COMPLETE! READY FOR SUBMISSION")
    print("="*80)
    
    print(f"\n📦 Submission file: {zip_path}")
    print(f"📁 Location: {zip_path.parent}")
    
    print(f"\n📋 Contents:")
    output_files = sorted(OUTPUT_DIR.glob("*"))
    for f in output_files:
        if f.is_file():
            print(f"   - {f.name}")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Review the ZIP file: {zip_path.name}")
    print(f"   2. Verify shapefiles open correctly in QGIS/ArcGIS")
    print(f"   3. Submit to PS-10 portal")
    
    print(f"\n✨ All processing complete!")
    
    return True

if __name__ == "__main__":
    try:
        import time
        start = time.time()
        
        success = main()
        
        elapsed = time.time() - start
        print(f"\n⏱️ Total processing time: {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
