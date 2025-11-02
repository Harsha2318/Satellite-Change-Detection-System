"""
PS-10: Process LISS4 Image Pair Only
Quick processing for second dataset
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.coords import BoundingBox
import fiona
from fiona.crs import from_epsg
from shapely.geometry import shape, mapping, box
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PS-10: LISS4 CHANGE DETECTION")
print("="*80)

# Paths
input_dir = r"C:\Users\harsh\PS-10\ps10_predictions"
output_dir = r"C:\Users\harsh\PS-10\ps10_predictions_formatted"

t1_path = os.path.join(input_dir, "LISS4_t1.tif")
t2_path = os.path.join(input_dir, "LISS4_t2.tif")

print(f"\n📁 Input files:")
print(f"   T1: {os.path.basename(t1_path)}")
print(f"   T2: {os.path.basename(t2_path)}")

# Read images
print(f"\n🔄 Reading images...")
with rasterio.open(t1_path) as src:
    img1 = src.read()
    profile = src.profile
    transform = src.transform
    crs = src.crs
    bounds = src.bounds
    
with rasterio.open(t2_path) as src:
    img2 = src.read()

print(f"   Size: {img1.shape[1]}x{img1.shape[2]}, {img1.shape[0]} bands")

# Calculate center coordinates
center_lon = (bounds.left + bounds.right) / 2
center_lat = (bounds.bottom + bounds.top) / 2

# Transform to lat/lon if needed
if crs and crs.is_projected:
    from pyproj import Transformer
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    center_lon, center_lat = transformer.transform(center_lon, center_lat)

print(f"   Center coordinates: {center_lat:.2f}°N, {center_lon:.2f}°E")

# Simple change detection
print(f"\n🔍 Detecting changes...")
img1_norm = img1.astype(np.float32) / 255.0
img2_norm = img2.astype(np.float32) / 255.0

# Calculate difference
diff = np.abs(img1_norm - img2_norm)
change_magnitude = np.mean(diff, axis=0)

# Threshold
threshold = 0.15
change_mask = (change_magnitude > threshold).astype(np.uint8)
change_pixels = np.sum(change_mask)
total_pixels = change_mask.shape[0] * change_mask.shape[1]
change_percent = (change_pixels / total_pixels) * 100

print(f"   Changes detected: {change_pixels} pixels ({change_percent:.2f}%)")

# Save TIF
output_name = f"Change_Mask_{center_lat:.2f}_{center_lon:.2f}"
tif_path = os.path.join(output_dir, f"{output_name}.tif")

profile.update({
    'count': 1,
    'dtype': 'uint8',
    'compress': 'lzw',
    'nodata': 0
})

print(f"\n💾 Saving TIF...")
with rasterio.open(tif_path, 'w', **profile) as dst:
    dst.write(change_mask, 1)
print(f"   ✓ {os.path.basename(tif_path)}")

# Create shapefile
print(f"\n📐 Creating shapefile...")
shp_path = os.path.join(output_dir, f"{output_name}.shp")

# Get changed pixel coordinates and create polygons
from rasterio.features import shapes

print(f"   Vectorizing change mask...")
mask = change_mask == 1
polygons = []
count = 0

for geom, val in shapes(change_mask, mask=mask, transform=transform):
    if val == 1:
        polygons.append(shape(geom))
        count += 1
        if count % 10000 == 0:
            print(f"   Progress: {count} features...")

print(f"   Total features: {count}")

# Simplify if too many
if count > 5000:
    print(f"   Simplifying geometries...")
    # Merge nearby polygons
    simplified = unary_union(polygons)
    if simplified.geom_type == 'Polygon':
        polygons = [simplified]
    else:
        polygons = list(simplified.geoms)
    print(f"   Reduced to: {len(polygons)} features")

# Write shapefile
schema = {
    'geometry': 'Polygon',
    'properties': {'change': 'int'}
}

print(f"   Writing shapefile...")
with fiona.open(shp_path, 'w', driver='ESRI Shapefile', 
                crs=crs, schema=schema) as dst:
    for poly in polygons:
        dst.write({
            'geometry': mapping(poly),
            'properties': {'change': 1}
        })

# Get file sizes
tif_size = os.path.getsize(tif_path) / (1024*1024)
shp_size = os.path.getsize(shp_path) / (1024*1024)

print(f"\n✅ LISS4 Processing Complete!")
print(f"   📄 TIF: {output_name}.tif ({tif_size:.2f} MB)")
print(f"   📄 SHP: {output_name}.shp ({shp_size:.2f} MB)")
print(f"   📍 Location: {center_lat:.2f}°N, {center_lon:.2f}°E")
print(f"   🔄 Change: {change_percent:.2f}%")

print("\n" + "="*80)
