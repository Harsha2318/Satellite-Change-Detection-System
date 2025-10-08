"""
Utilities to stack single-band Sentinel-2 TIFFs from extracted folders
into multi-band GeoTIFFs (and optional NumPy arrays).

Expected folder layout (each folder is a product):

  <input_dir>/R2...GTDA/BAND2.tif
  <input_dir>/R2...GTDA/BAND3.tif
  <input_dir>/R2...GTDA/BAND4.tif

This module reads requested band files in each product folder, stacks
them in the requested order and writes a multi-band GeoTIFF to the
output directory.
"""
import os
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio

from ..utils.geoutils import write_geotiff

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _find_band_file(folder: Path, band_name: str) -> Optional[Path]:
    """Look for a file matching the band name (case-insensitive)."""
    for p in folder.iterdir():
        if p.is_file() and p.name.upper().startswith(band_name.upper()):
            return p
    return None


def stack_bands_from_folder(input_dir: str,
                            output_dir: str,
                            bands: Optional[List[str]] = None,
                            out_format: str = 'tif',
                            save_npy: bool = False,
                            overwrite: bool = False) -> List[str]:
    """
    Stack bands from each product folder in `input_dir` and write multi-band files

    Args:
        input_dir: directory containing extracted product folders
        output_dir: directory to save stacked outputs
        bands: list of band filenames (e.g. ['BAND4','BAND3','BAND2']) in desired order
               If None, defaults to ['BAND4','BAND3','BAND2'] (R,G,B)
        out_format: 'tif' or 'npy' for the primary output (GeoTIFF recommended)
        save_npy: also save a .npy copy alongside the GeoTIFF
        overwrite: overwrite existing outputs

    Returns:
        List of output file paths written
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if bands is None:
        # Default to Red, Green, Blue order from Sentinel-2 band names
        bands = ['BAND4', 'BAND3', 'BAND2']

    written_files = []

    # Iterate over product folders (directories) in input_dir
    for entry in sorted(input_path.iterdir()):
        if not entry.is_dir():
            continue

        product_name = entry.name
        logger.info(f"Processing product folder: {product_name}")

        band_arrays = []
        meta = None
        missing = []

        for band in bands:
            bf = _find_band_file(entry, band)
            if bf is None:
                missing.append(band)
                continue

            with rasterio.open(str(bf)) as src:
                arr = src.read(1)
                band_meta = src.meta.copy()

            if meta is None:
                # We'll use the first band's metadata as base
                meta = band_meta
            else:
                # Basic sanity check: shapes must match
                if arr.shape != (meta['height'], meta['width']):
                    logger.error(
                        f"Shape mismatch for {bf.name}: {arr.shape} != {(meta['height'], meta['width'])}. Skipping product."
                    )
                    band_arrays = []
                    break

            band_arrays.append(arr)

        if missing:
            logger.warning(f"Missing bands {missing} in product {product_name}; skipping this product.")
            continue

        if not band_arrays:
            continue

        # Stack bands into a (bands, H, W) array
        stacked = np.stack(band_arrays, axis=0)

        # Prepare output filenames
        out_basename = f"{product_name}_stacked"
        tif_out = output_path / f"{out_basename}.tif"
        npy_out = output_path / f"{out_basename}.npy"

        if tif_out.exists() and not overwrite:
            logger.info(f"Output already exists and overwrite=False: {tif_out}; skipping")
            written_files.append(str(tif_out))
        else:
            # Update metadata for multi-band
            write_meta = meta.copy()
            write_meta.update({
                'count': stacked.shape[0],
                'height': stacked.shape[1],
                'width': stacked.shape[2],
                'dtype': stacked.dtype
            })

            write_geotiff(str(tif_out), stacked, write_meta)
            written_files.append(str(tif_out))

        if save_npy or out_format == 'npy':
            if npy_out.exists() and not overwrite:
                logger.info(f"NumPy output exists and overwrite=False: {npy_out}")
            else:
                np.save(str(npy_out), stacked)
                written_files.append(str(npy_out))

    return written_files


if __name__ == '__main__':
    # Simple CLI for quick local use
    import argparse

    parser = argparse.ArgumentParser(description='Stack single-band TIFFs into multi-band GeoTIFFs')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--bands', default='BAND4,BAND3,BAND2', help='Comma-separated band names in desired order')
    parser.add_argument('--out-format', choices=['tif', 'npy'], default='tif')
    parser.add_argument('--save-npy', action='store_true', help='Also save NumPy arrays')
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()
    bands = [b.strip() for b in args.bands.split(',') if b.strip()]
    outs = stack_bands_from_folder(args.input_dir, args.output_dir, bands=bands, out_format=args.out_format, save_npy=args.save_npy, overwrite=args.overwrite)
    print('\n'.join(outs))
