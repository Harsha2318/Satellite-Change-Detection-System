"""Convert paired tile TIFFs (from manifest CSV) to normalized .npz arrays ready for training.

The script supports normalization modes: minmax (0-1) or zscore (per-band mean/std).
Output files are saved next to the CSV as <tile_id>.npz unless output_dir is provided.
"""
from pathlib import Path
import csv
import numpy as np
import logging
import rasterio

logger = logging.getLogger(__name__)


def _normalize_minmax(arr: np.ndarray):
    # arr shape: (bands, H, W)
    arr = arr.astype('float32')
    minv = arr.min(axis=(1,2), keepdims=True)
    maxv = arr.max(axis=(1,2), keepdims=True)
    denom = (maxv - minv)
    denom[denom==0] = 1.0
    return (arr - minv) / denom, {'min': minv.squeeze().tolist(), 'max': maxv.squeeze().tolist()}


def _normalize_zscore(arr: np.ndarray):
    arr = arr.astype('float32')
    mean = arr.mean(axis=(1,2), keepdims=True)
    std = arr.std(axis=(1,2), keepdims=True)
    std[std==0] = 1.0
    return (arr - mean) / std, {'mean': mean.squeeze().tolist(), 'std': std.squeeze().tolist()}


def convert_manifest_to_npz(manifest_csv: str, output_dir: str = None, norm: str = 'minmax') -> int:
    """Read manifest CSV and produce .npz files for each row.

    Returns the number of files written.
    """
    mpath = Path(manifest_csv)
    if not mpath.exists():
        raise FileNotFoundError(manifest_csv)

    outroot = Path(output_dir) if output_dir else mpath.parent
    outroot.mkdir(parents=True, exist_ok=True)

    written = 0
    with mpath.open('r', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tile_id = row['tile_id']
            pre = Path(row['pre_tile'])
            post = Path(row['post_tile'])

            # read data
            with rasterio.open(str(pre)) as src:
                pre_arr = src.read().astype('float32')
            with rasterio.open(str(post)) as src:
                post_arr = src.read().astype('float32')

            # Stack into single array with shape (2, bands, H, W)
            # Ensure both have same shape
            if pre_arr.shape != post_arr.shape:
                logger.warning(f"Shape mismatch for {tile_id}: {pre_arr.shape} vs {post_arr.shape}; skipping")
                continue

            pair_arr = np.stack([pre_arr, post_arr], axis=0)

            # Normalize per-image (apply same norm separately to pre and post)
            meta = {'tile_id': tile_id}
            if norm == 'minmax':
                pre_norm, pre_stats = _normalize_minmax(pair_arr[0])
                post_norm, post_stats = _normalize_minmax(pair_arr[1])
            else:
                pre_norm, pre_stats = _normalize_zscore(pair_arr[0])
                post_norm, post_stats = _normalize_zscore(pair_arr[1])

            # Save into .npz
            outp = outroot / f"{tile_id}.npz"
            np.savez_compressed(str(outp), pre=pre_norm, post=post_norm, pre_stats=pre_stats, post_stats=post_stats, meta=meta)
            written += 1

    logger.info(f"Wrote {written} npz files to {outroot}")
    return written


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert manifest CSV to normalized .npz arrays')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out-dir')
    parser.add_argument('--norm', choices=['minmax','zscore'], default='minmax')
    args = parser.parse_args()
    print(convert_manifest_to_npz(args.manifest, args.out_dir, args.norm))
