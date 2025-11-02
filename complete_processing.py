"""
Complete PS-10 Processing - Final Version
Optimized vectorization for large change areas
"""

import os
import sys
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PS-10 FINAL PROCESSING - OPTIMIZED")
print("="*80)

# Paths
BASE_DIR = Path(r"C:\Users\harsh\PS-10")
INPUT_DIR = BASE_DIR / "ps10_predictions"
OUTPUT_DIR = BASE_DIR / "ps10_predictions_formatted"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    import rasterio
    from rasterio.features import shapes
    import fiona
    from shapely.geometry import shape, mapping, box
    from shapely.ops import unary_union
    print("✓ Libraries loaded")
except ImportError as e:
    print(f"❌ Missing library: {e}")
    sys.exit(1)

def create_simplified_shapefile(change_mask, transform, crs, output_path, simplify_tolerance=10):
    """Create shapefile with simplified geometries"""
    print(f"   📐 Creating simplified shapefile...")
    
    # Vectorize
    shapes_gen = shapes(change_mask.astype(np.int16), transform=transform)
    
    geometries = []
    count = 0
    
    for geom, value in shapes_gen:
        if value == 1:
            # Create shapely geometry and simplify
            poly = shape(geom)
            simplified = poly.simplify(simplify_tolerance, preserve_topology=True)
            
            # Only keep if area is significant (> 100 sq meters)
            if simplified.area > 100:
                geometries.append({
                    'geometry': mapping(simplified),
                    'properties': {
                        'change': 1,
                        'area_sqm': int(simplified.area)
                    }
                })
                count += 1
                
                if count % 1000 == 0:
                    print(f"   Processed {count} polygons...")
    
    print(f"   ✓ Created {len(geometries)} polygons")
    
    if geometries:
        schema = {
            'geometry': 'Polygon',
            'properties': {
                'change': 'int',
                'area_sqm': 'int'
            }
        }
        
        with fiona.open(str(output_path), 'w', 
                      driver='ESRI Shapefile',
                      crs=crs,
                      schema=schema) as dst:
            dst.writerecords(geometries)
        
        return True
    return False

def process_existing_mask(mask_path, output_prefix):
    """Process an existing change mask"""
    print(f"\n🔄 Processing: {output_prefix}")
    print(f"   Input: {mask_path.name}")
    
    with rasterio.open(mask_path) as src:
        change_mask = src.read(1)
        
        changed_pixels = change_mask.sum()
        total_pixels = change_mask.size
        pct = (changed_pixels / total_pixels) * 100
        
        print(f"   Size: {src.width}x{src.height}")
        print(f"   Changes: {changed_pixels} pixels ({pct:.2f}%)")
        
        # Create shapefile
        shp_output = OUTPUT_DIR / f"{output_prefix}_change_vectors.shp"
        
        success = create_simplified_shapefile(
            change_mask, 
            src.transform, 
            src.crs,
            shp_output,
            simplify_tolerance=20  # More aggressive simplification
        )
        
        if success:
            print(f"   ✅ Saved: {shp_output.name}")
            
            # Check file sizes
            shp_size = shp_output.stat().st_size / (1024*1024)
            print(f"   Size: {shp_size:.2f} MB")
        
        return success

def process_liss4_pair():
    """Process LISS4 pair with tiled approach"""
    print(f"\n{'='*80}")
    print("PROCESSING LISS4 PAIR")
    print(f"{'='*80}")
    
    t1 = INPUT_DIR / "LISS4_t1.tif"
    t2 = INPUT_DIR / "LISS4_t2.tif"
    
    if not (t1.exists() and t2.exists()):
        print("   ⚠️ LISS4 files not found")
        return False
    
    print(f"\n🔄 Processing: LISS4_20200318_20250127")
    print(f"   T1: {t1.name}")
    print(f"   T2: {t2.name}")
    
    with rasterio.open(t1) as src1, rasterio.open(t2) as src2:
        print(f"   Size: {src1.width}x{src1.height}, {src1.count} bands")
        print(f"   Processing in 512x512 tiles...")
        
        height = src1.height
        width = src1.width
        tile_size = 512
        
        change_mask = np.zeros((height, width), dtype=np.uint8)
        
        total_tiles = ((height + tile_size - 1) // tile_size) * ((width + tile_size - 1) // tile_size)
        processed = 0
        
        for i in range(0, height, tile_size):
            for j in range(0, width, tile_size):
                i_end = min(i + tile_size, height)
                j_end = min(j + tile_size, width)
                
                window = rasterio.windows.Window(j, i, j_end - j, i_end - i)
                
                tile1 = src1.read(window=window)
                tile2 = src2.read(window=window)
                
                tile1 = np.transpose(tile1, (1, 2, 0))
                tile2 = np.transpose(tile2, (1, 2, 0))
                
                diff = np.abs(tile1.astype(float) - tile2.astype(float))
                
                if len(diff.shape) == 3:
                    diff = np.mean(diff, axis=2)
                
                if diff.max() > diff.min():
                    diff = (diff - diff.min()) / (diff.max() - diff.min())
                
                tile_mask = (diff > 0.15).astype(np.uint8)
                change_mask[i:i_end, j:j_end] = tile_mask
                
                processed += 1
                if processed % 10 == 0:
                    progress = (processed / total_tiles) * 100
                    print(f"   Progress: {progress:.1f}% ({processed}/{total_tiles})")
        
        changed = change_mask.sum()
        total = change_mask.size
        print(f"   ✓ Changes: {changed} pixels ({changed/total*100:.2f}%)")
        
        # Save mask
        tif_output = OUTPUT_DIR / "LISS4_20200318_20250127_change_mask.tif"
        profile = src1.profile.copy()
        profile.update({'count': 1, 'dtype': 'uint8', 'compress': 'lzw'})
        
        with rasterio.open(tif_output, 'w', **profile) as dst:
            dst.write(change_mask, 1)
        
        print(f"   ✅ Saved: {tif_output.name}")
        
        # Create shapefile
        shp_output = OUTPUT_DIR / "LISS4_20200318_20250127_change_vectors.shp"
        success = create_simplified_shapefile(
            change_mask,
            src1.transform,
            src1.crs,
            shp_output,
            simplify_tolerance=15
        )
        
        if success:
            print(f"   ✅ Saved: {shp_output.name}")
        
        return True

def main():
    print("\n" + "="*80)
    print("STEP 1: PROCESS SENTINEL-2 MASK")
    print("="*80)
    
    # Check if Sentinel-2 mask exists
    s2_mask = OUTPUT_DIR / "Sentinel2_20200328_20250307_change_mask.tif"
    
    if s2_mask.exists():
        print(f"\n✓ Found existing Sentinel-2 mask")
        process_existing_mask(s2_mask, "Sentinel2_20200328_20250307")
    else:
        print(f"\n⚠️ Sentinel-2 mask not found, skipping...")
    
    print("\n" + "="*80)
    print("STEP 2: PROCESS LISS4 PAIR")
    print("="*80)
    
    process_liss4_pair()
    
    print("\n" + "="*80)
    print("✅ PROCESSING COMPLETE!")
    print("="*80)
    
    # List outputs
    output_files = sorted(OUTPUT_DIR.glob("*"))
    print(f"\n📄 Output files:")
    for f in output_files:
        size = f.stat().st_size / (1024*1024)
        print(f"   - {f.name} ({size:.2f} MB)")
    
    print(f"\n🎯 READY FOR SUBMISSION!")
    print(f"   All change masks and shapefiles created")
    print(f"   Location: {OUTPUT_DIR}")
    
    return True

if __name__ == "__main__":
    try:
        import time
        start = time.time()
        
        success = main()
        
        elapsed = time.time() - start
        print(f"\n⏱️ Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
