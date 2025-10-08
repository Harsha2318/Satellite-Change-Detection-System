"""Generate blank mask TIFFs for each tile listed in a manifest CSV.

This is useful when you don't have ground-truth masks but need a mask_dir
argument for the training CLI. The generated masks are all zeros (no-change)
and preserve the spatial profile of the corresponding tile (so geotransform
and CRS match).
"""
import csv
from pathlib import Path
import numpy as np
import rasterio


def generate_blank_masks(manifest_csv: str, out_dir: str, overwrite: bool = False) -> int:
    manifest_csv = Path(manifest_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with manifest_csv.open('r', newline='') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"Manifest {manifest_csv} contains no rows")

    written = 0
    for r in rows:
        tile_id = r.get('tile_id') or r.get('id') or None
        pre = Path(r['pre_tile'])

        if tile_id is None:
            tile_id = pre.stem

        mask_path = out_dir / f"{tile_id}_mask.tif"
        if mask_path.exists() and not overwrite:
            continue

        # Read profile from pre tile and write a single-band uint8 mask of zeros
        with rasterio.open(str(pre)) as src:
            profile = src.profile.copy()
            width = src.width
            height = src.height
            transform = src.transform
            crs = src.crs

        mask_profile = profile
        mask_profile.update({
            'count': 1,
            'dtype': 'uint8'
        })

        zeros = np.zeros((height, width), dtype=np.uint8)

        with rasterio.open(str(mask_path), 'w', **mask_profile) as dst:
            dst.write(zeros, 1)

        written += 1

    return written


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Generate blank mask TIFFs from manifest')
    p.add_argument('manifest', help='Path to manifest.csv')
    p.add_argument('out_dir', help='Output directory for mask files')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    n = generate_blank_masks(args.manifest, args.out_dir, overwrite=args.overwrite)
    print(f"Wrote {n} mask files to {args.out_dir}")
