"""
Efficient Real Data Inference for PS-10
Processes large images in tiles to avoid memory issues
"""

import os
import sys
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PS-10 CHANGE DETECTION - EFFICIENT INFERENCE")
print("="*80)

# Paths
BASE_DIR = Path(r"C:\Users\harsh\PS-10")
INPUT_DIR = BASE_DIR / "ps10_predictions"
OUTPUT_DIR = BASE_DIR / "ps10_predictions_formatted"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Configuration:")
print(f"   Input: {INPUT_DIR}")
print(f"   Output: {OUTPUT_DIR}")

try:
    import rasterio
    from rasterio.features import shapes
    import fiona
    from shapely.geometry import shape, mapping
    print("✓ Geospatial libraries loaded")
except ImportError as e:
    print(f"❌ Missing library: {e}")
    sys.exit(1)

def simple_change_detection_tiled(src1, src2, tile_size=512):
    """Process large images in tiles to avoid memory issues"""
    print(f"   Processing in {tile_size}x{tile_size} tiles...")
    
    height = src1.height
    width = src1.width
    
    # Create output array
    change_mask = np.zeros((height, width), dtype=np.uint8)
    
    total_tiles = ((height + tile_size - 1) // tile_size) * ((width + tile_size - 1) // tile_size)
    processed = 0
    
    # Process in tiles
    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            # Calculate tile bounds
            i_end = min(i + tile_size, height)
            j_end = min(j + tile_size, width)
            
            # Read tile from both images
            window = rasterio.windows.Window(j, i, j_end - j, i_end - i)
            
            tile1 = src1.read(window=window)  # Shape: (bands, h, w)
            tile2 = src2.read(window=window)
            
            # Transpose to (h, w, bands)
            tile1 = np.transpose(tile1, (1, 2, 0))
            tile2 = np.transpose(tile2, (1, 2, 0))
            
            # Calculate difference
            diff = np.abs(tile1.astype(float) - tile2.astype(float))
            
            # Average across bands
            if len(diff.shape) == 3:
                diff = np.mean(diff, axis=2)
            
            # Normalize
            if diff.max() > diff.min():
                diff = (diff - diff.min()) / (diff.max() - diff.min())
            
            # Threshold
            threshold = 0.15
            tile_mask = (diff > threshold).astype(np.uint8)
            
            # Store in output
            change_mask[i:i_end, j:j_end] = tile_mask
            
            processed += 1
            if processed % 10 == 0:
                progress = (processed / total_tiles) * 100
                print(f"   Progress: {progress:.1f}% ({processed}/{total_tiles} tiles)")
    
    changed_pixels = change_mask.sum()
    total_pixels = change_mask.size
    print(f"   ✓ Detected changes: {changed_pixels} pixels ({changed_pixels/total_pixels*100:.2f}%)")
    
    return change_mask

def process_image_pair(t1_path, t2_path, output_prefix):
    """Process a single image pair efficiently"""
    print(f"\n🔄 Processing: {output_prefix}")
    print(f"   T1: {t1_path.name}")
    print(f"   T2: {t2_path.name}")
    
    # Open files
    with rasterio.open(t1_path) as src1, rasterio.open(t2_path) as src2:
        print(f"   Image size: {src1.width}x{src1.height}, {src1.count} bands")
        
        # Run tiled change detection
        change_mask = simple_change_detection_tiled(src1, src2, tile_size=512)
        
        # Save GeoTIFF
        tif_output = OUTPUT_DIR / f"{output_prefix}_change_mask.tif"
        profile = src1.profile.copy()
        profile.update({
            'count': 1,
            'dtype': 'uint8',
            'compress': 'lzw'
        })
        
        with rasterio.open(tif_output, 'w', **profile) as dst:
            dst.write(change_mask, 1)
        
        print(f"   ✅ Saved: {tif_output.name}")
        
        # Create shapefile
        print(f"   📐 Creating shapefile...")
        shp_output = OUTPUT_DIR / f"{output_prefix}_change_vectors.shp"
        
        # Vectorize (only changed pixels)
        shapes_gen = shapes(change_mask.astype(np.int16), transform=src1.transform)
        geometries = []
        
        count = 0
        for geom, value in shapes_gen:
            if value == 1:  # Only changed areas
                geometries.append({
                    'geometry': geom,
                    'properties': {'change': 1}
                })
                count += 1
                if count % 100 == 0:
                    print(f"   Vectorizing... {count} polygons")
        
        if geometries:
            # Save shapefile
            schema = {
                'geometry': 'Polygon',
                'properties': {'change': 'int'}
            }
            
            with fiona.open(str(shp_output), 'w', 
                          driver='ESRI Shapefile',
                          crs=src1.crs,
                          schema=schema) as dst:
                dst.writerecords(geometries)
            
            print(f"   ✅ Saved: {shp_output.name} ({len(geometries)} polygons)")
        else:
            print(f"   ℹ️ No changes detected (or too small to vectorize)")
        
        return True

def main():
    print("\n" + "="*80)
    print("STARTING INFERENCE")
    print("="*80)
    
    # Find image pairs
    pairs = []
    
    # Sentinel-2
    s2_t1 = INPUT_DIR / "Sentinel2_t1.tif"
    s2_t2 = INPUT_DIR / "Sentinel2_t2.tif"
    if s2_t1.exists() and s2_t2.exists():
        pairs.append((s2_t1, s2_t2, "Sentinel2_20200328_20250307"))
    
    # LISS4
    liss_t1 = INPUT_DIR / "LISS4_t1.tif"
    liss_t2 = INPUT_DIR / "LISS4_t2.tif"
    if liss_t1.exists() and liss_t2.exists():
        pairs.append((liss_t1, liss_t2, "LISS4_20200318_20250127"))
    
    print(f"\n📊 Found {len(pairs)} image pair(s) to process")
    
    if not pairs:
        print("❌ No image pairs found!")
        return False
    
    # Process each pair
    success_count = 0
    for idx, (t1, t2, prefix) in enumerate(pairs, 1):
        print(f"\n{'='*80}")
        print(f"PAIR {idx}/{len(pairs)}: {prefix}")
        print(f"{'='*80}")
        
        try:
            if process_image_pair(t1, t2, prefix):
                success_count += 1
        except Exception as e:
            print(f"   ❌ Error processing {prefix}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print(f"✅ INFERENCE COMPLETE: {success_count}/{len(pairs)} pairs processed")
    print("="*80)
    print(f"\n📁 Results saved to: {OUTPUT_DIR}")
    
    # List output files
    output_files = list(OUTPUT_DIR.glob("*"))
    if output_files:
        print(f"\n📄 Output files:")
        for f in sorted(output_files):
            size_mb = f.stat().st_size / (1024*1024)
            print(f"   - {f.name} ({size_mb:.2f} MB)")
    
    print(f"\n🎯 Next step: Create PS-10 submission package")
    print(f"   All change masks and shapefiles are ready!")
    
    return success_count == len(pairs)

if __name__ == "__main__":
    try:
        import time
        start_time = time.time()
        
        success = main()
        
        elapsed = time.time() - start_time
        print(f"\n⏱️ Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
