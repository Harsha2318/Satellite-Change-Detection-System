"""Quick check of CRS in processed files"""
import rasterio
from pathlib import Path

processed_dir = Path(r"C:\Users\harsh\PS-10\changedetect\data\processed")

for tif_file in processed_dir.glob("*.tif"):
    with rasterio.open(tif_file) as src:
        print(f"\n{tif_file.name}:")
        print(f"  CRS: {src.crs}")
        print(f"  Size: {src.width}x{src.height}")
        print(f"  Bands: {src.count}")
        print(f"  Bounds: {src.bounds}")
