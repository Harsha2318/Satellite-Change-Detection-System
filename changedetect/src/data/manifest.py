"""Create a CSV manifest pairing pre/post tile files for training.

The manifest contains rows: tile_id, pre_tile, post_tile, width, height, bands, dtype

It expects tile folders named like <product>_stacked and files inside named
<product>_stacked_tile_i_j.tif so matching is done on the trailing `_tile_i_j` id.
"""
from pathlib import Path
import csv
import logging
from typing import Optional
import rasterio

logger = logging.getLogger(__name__)


def _tile_id_from_name(fname: str) -> Optional[str]:
    """Return tile id after the last '_tile_' occurrence, or None if not found."""
    if '_tile_' not in fname:
        return None
    return fname.split('_tile_', 1)[1].rsplit('.', 1)[0]


def make_manifest(tiles_parent_dir: str, output_csv: str, pre_folder: Optional[str] = None, post_folder: Optional[str] = None) -> str:
    """Scan tiles and write a manifest CSV pairing tiles by tile id.

    If pre_folder and post_folder are None, the function will pick the two subfolders
    found inside tiles_parent_dir (sorted) and treat the first as pre and second as post.
    """
    parent = Path(tiles_parent_dir)
    if not parent.exists():
        raise FileNotFoundError(f"Tiles parent directory not found: {parent}")

    subdirs = [p for p in sorted(parent.iterdir()) if p.is_dir()]
    if pre_folder and post_folder:
        pre_dir = parent / pre_folder
        post_dir = parent / post_folder
    else:
        if len(subdirs) < 2:
            raise ValueError(f"Need at least two product folders under {parent} to make pairs; found {len(subdirs)}")
        pre_dir, post_dir = subdirs[0], subdirs[1]

    logger.info(f"Pairing pre folder: {pre_dir.name} with post folder: {post_dir.name}")

    pre_files = { _tile_id_from_name(f.name): f for f in pre_dir.glob('*.tif') if _tile_id_from_name(f.name) }
    post_files = { _tile_id_from_name(f.name): f for f in post_dir.glob('*.tif') if _tile_id_from_name(f.name) }

    common_ids = sorted(set(pre_files.keys()).intersection(post_files.keys()))
    if not common_ids:
        raise ValueError('No matching tile ids found between the two folders')

    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['tile_id', 'pre_tile', 'post_tile', 'width', 'height', 'bands', 'dtype'])

        for tid in common_ids:
            pre = pre_files[tid]
            post = post_files[tid]

            # read basic metadata from pre (assume same for post)
            with rasterio.open(str(pre)) as src:
                width = src.width
                height = src.height
                bands = src.count
                dtype = src.dtypes[0] if src.dtypes else 'uint8'

            writer.writerow([tid, str(pre), str(post), width, height, bands, dtype])

    logger.info(f"Wrote manifest with {len(common_ids)} rows to {out_csv}")
    return str(out_csv)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Make manifest CSV pairing tiles')
    parser.add_argument('--tiles-dir', required=True)
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--pre-folder')
    parser.add_argument('--post-folder')
    args = parser.parse_args()
    print(make_manifest(args.tiles_dir, args.out_csv, args.pre_folder, args.post_folder))
