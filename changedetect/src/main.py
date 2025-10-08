"""
Main entry point for satellite image change detection
"""
import os
import sys
import logging
import argparse
import datetime
from pathlib import Path
import json

# Prefer EPSG parameters from the registry over GeoTIFF keys when possible.
# This helps avoid PROJ/proj.db conflicts on Windows systems with multiple PROJ installations.
os.environ.setdefault('GTIFF_SRS_SOURCE', 'EPSG')

# When running this file directly (python main.py) ensure the project root is on sys.path
# so imports like 'changedetect.src.*' work. main.py is at changedetect/src/main.py, so the
# project root is two parents up.
try:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
except Exception:
    pass

# Defer heavy/optional imports (download, train, inference, evaluate) to handlers
# to avoid importing optional dependencies (sentinelsat, torch, etc.) at module import time.

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Satellite Image Change Detection for PS-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Download command
    download_parser = subparsers.add_parser(
        'download', 
        help='Download satellite imagery'
    )
    download_parser.add_argument(
        "--source", 
        choices=["sentinel2", "resourcesat2"],
        required=True,
        help="Source of satellite imagery"
    )
    download_parser.add_argument(
        "--output", 
        required=True,
        help="Output directory"
    )
    download_parser.add_argument(
        "--lat", 
        type=float,
        help="Latitude of point of interest"
    )
    download_parser.add_argument(
        "--lon", 
        type=float,
        help="Longitude of point of interest"
    )
    download_parser.add_argument(
        "--start-date", 
        help="Start date (YYYY-MM-DD)"
    )
    download_parser.add_argument(
        "--end-date", 
        help="End date (YYYY-MM-DD)"
    )
    download_parser.add_argument(
        "--list-locations",
        action="store_true",
        help="List sample locations from the problem statement"
    )
    download_parser.add_argument(
        "--cloud-cover", 
        type=int,
        default=10,
        help="Maximum cloud cover percentage (for Sentinel-2)"
    )
    download_parser.add_argument(
        "--user", 
        type=str,
        required=True,
        help="Username for Sentinel Hub (if using Sentinel-2)"
    )
    download_parser.add_argument(
        "--password", 
        type=str,
        required=True,
        help="Password for Sentinel Hub (if using Sentinel-2)"
    )
    
    # Train command
    train_parser = subparsers.add_parser(
        'train', 
        help='Train a change detection model'
    )
    train_parser.add_argument(
        "--image_dir", 
        required=True,
        help="Directory containing image pairs (t1 and t2)"
    )
    train_parser.add_argument(
        "--mask_dir", 
        required=True,
        help="Directory containing mask images"
    )
    train_parser.add_argument(
        "--output_dir", 
        default="./outputs",
        help="Directory to save model checkpoints and logs"
    )
    train_parser.add_argument(
        "--model_type", 
        choices=["siamese_unet", "siamese_diff", "fcn_diff"],
        default="siamese_unet",
        help="Type of change detection model to use"
    )
    train_parser.add_argument(
        "--in_channels", 
        type=int,
        default=3,
        help="Number of input channels per image"
    )
    train_parser.add_argument(
        "--batch_size", 
        type=int,
        default=16,
        help="Batch size for training"
    )
    train_parser.add_argument(
        "--num_epochs", 
        type=int,
        default=100,
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--resume", 
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    # Inference command
    inference_parser = subparsers.add_parser(
        'inference', 
        help='Run inference for change detection'
    )
    inference_parser.add_argument(
        "--image_dir", 
        required=True,
        help="Directory containing image pairs (t1 and t2)"
    )
    inference_parser.add_argument(
        "--model_path", 
        required=True,
        help="Path to trained model checkpoint"
    )
    inference_parser.add_argument(
        "--output_dir", 
        default="./predictions",
        help="Directory to save predictions"
    )
    inference_parser.add_argument(
        "--model_type", 
        default="siamese_unet",
        choices=["siamese_unet", "siamese_diff", "fcn_diff"],
        help="Type of change detection model"
    )
    inference_parser.add_argument(
        "--in_channels", 
        type=int,
        default=3,
        help="Number of input channels per image"
    )
    inference_parser.add_argument(
        "--no_vector", 
        action="store_true",
        help="Skip vector output generation"
    )
    inference_parser.add_argument(
        "--min_area", 
        type=int,
        default=10,
        help="Minimum area in pixels for vector features"
    )
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        'evaluate', 
        help='Evaluate change detection results'
    )
    evaluate_parser.add_argument(
        "--pred_dir", 
        required=True,
        help="Directory containing prediction masks"
    )
    evaluate_parser.add_argument(
        "--gt_dir", 
        required=True,
        help="Directory containing ground truth masks"
    )
    evaluate_parser.add_argument(
        "--output_dir", 
        default="./evaluation",
        help="Directory to save evaluation results"
    )
    evaluate_parser.add_argument(
        "--vector", 
        action="store_true",
        help="Evaluate vector results"
    )

    # Stack command - create multi-band images from extracted band files
    stack_parser = subparsers.add_parser(
        'stack',
        help='Stack single-band TIFFs in extracted product folders into multi-band GeoTIFFs'
    )
    stack_parser.add_argument(
        '--input-dir',
        required=True,
        help='Directory containing extracted product folders (e.g., data/raw)'
    )
    stack_parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to save stacked outputs'
    )
    stack_parser.add_argument(
        '--bands',
        default='BAND4,BAND3,BAND2',
        help='Comma-separated band names in desired order (default: BAND4,BAND3,BAND2)'
    )
    stack_parser.add_argument(
        '--save-npy',
        action='store_true',
        help='Also save NumPy .npy copies alongside GeoTIFFs'
    )
    stack_parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing outputs'
    )

    # Tile command - break large stacked images into tiles
    tile_parser = subparsers.add_parser(
        'tile',
        help='Tile multi-band GeoTIFFs into smaller patches for model input'
    )
    tile_parser.add_argument(
        '--input-dir',
        required=True,
        help='Directory containing stacked GeoTIFFs'
    )
    tile_parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to save tiles'
    )
    tile_parser.add_argument(
        '--tile-size',
        type=int,
        default=512,
        help='Tile size in pixels (default: 512)'
    )
    tile_parser.add_argument(
        '--overlap',
        type=int,
        default=64,
        help='Overlap between tiles in pixels (default: 64)'
    )
    tile_parser.add_argument(
        '--bands',
        default=None,
        help='Comma-separated 1-based band indices to include (e.g. 1,2,3). Default: all bands'
    )
    tile_parser.add_argument(
        '--prefix',
        default=None,
        help='Optional prefix for tile filenames'
    )
    tile_parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker processes to use for tiling (default: 1)'
    )
    tile_parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing tiles'
    )

    # Manifest command - create CSV pairing pre/post tiles
    manifest_parser = subparsers.add_parser(
        'manifest',
        help='Create a CSV manifest pairing pre and post tiles'
    )
    manifest_parser.add_argument('--tiles-dir', required=True, help='Parent directory containing product tile subfolders')
    manifest_parser.add_argument('--out-csv', required=True, help='CSV output path')
    manifest_parser.add_argument('--pre-folder', help='Optional pre product folder name')
    manifest_parser.add_argument('--post-folder', help='Optional post product folder name')

    # Convert command - convert manifest rows to normalized .npz arrays
    convert_parser = subparsers.add_parser(
        'convert',
        help='Convert manifest CSV entries to normalized .npz arrays'
    )
    convert_parser.add_argument('--manifest', required=True, help='Manifest CSV path')
    convert_parser.add_argument('--out-dir', help='Output directory for .npz files')
    convert_parser.add_argument('--norm', choices=['minmax','zscore'], default='minmax', help='Normalization mode')
    
    # Prepare-pairs command - create *_t1.tif and *_t2.tif files from manifest
    prep_parser = subparsers.add_parser(
        'prepare-pairs',
        help='Create paired *_t1.tif and *_t2.tif files from manifest CSV for training'
    )
    prep_parser.add_argument('--manifest', required=True, help='Path to manifest CSV')
    prep_parser.add_argument('--out-dir', required=True, help='Output directory for paired files')
    prep_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')

    # Generate-masks command - create blank mask TIFFs from manifest
    mask_parser = subparsers.add_parser(
        'generate-masks',
        help='Generate blank mask TIFFs for each manifest entry'
    )
    mask_parser.add_argument('--manifest', required=True, help='Path to manifest CSV')
    mask_parser.add_argument('--out-dir', required=True, help='Output directory for masks')
    mask_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing masks')
    
    return parser


def handle_download(args):
    """Handle the download command"""
    # Import download utilities here to avoid optional dependency import at module load
    from changedetect.src.data.download import (
        download_sentinel2_copernicus,
        download_resourcesat2_bhoonidhi,
        list_sample_locations,
        download_sentinel2
    )

    if args.list_locations:
        list_sample_locations()
        return 0

    if args.lat is None or args.lon is None:
        logger.error("Latitude and longitude are required")
        return 1
    
    if args.start_date is None or args.end_date is None:
        logger.error("Start date and end date are required")
        return 1
    
    date_range = (args.start_date, args.end_date)
    
    if args.source == "sentinel2":
        download_sentinel2_copernicus(
            args.output, 
            args.lat, 
            args.lon, 
            date_range, 
            cloud_cover_max=args.cloud_cover
        )
    elif args.source == "resourcesat2":
        download_resourcesat2_bhoonidhi(
            args.output, 
            args.lat, 
            args.lon, 
            date_range
        )
    
    return 0


def handle_stack(args):
    """Handle stack command"""
    from changedetect.src.data.stack import stack_bands_from_folder

    bands = [b.strip() for b in args.bands.split(',') if b.strip()]
    written = stack_bands_from_folder(
        args.input_dir,
        args.output_dir,
        bands=bands,
        save_npy=args.save_npy,
        overwrite=args.overwrite
    )

    for w in written:
        logger.info(f"Wrote: {w}")

    return 0


def handle_tile(args):
    """Handle the tile command"""
    from changedetect.src.data.tile import create_tiles
    import glob

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse bands argument
    bands = None
    if args.bands:
        bands = [int(b.strip()) for b in args.bands.split(',') if b.strip()]

    # Find stacked GeoTIFFs in input_dir
    tifs = sorted(input_path.glob('*_stacked.tif'))
    written = []

    for tif in tifs:
        prefix = args.prefix or tif.stem
        tile_out_dir = output_path / tif.stem
        tile_out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Tiling {tif} -> {tile_out_dir}")
        tiles = create_tiles(str(tif), str(tile_out_dir), tile_size=args.tile_size, overlap=args.overlap, bands=bands, prefix=prefix)
        written.extend(tiles)

    for w in written:
        logger.info(f"Wrote tile: {w}")

    return 0


def handle_manifest(args):
    from changedetect.src.data.manifest import make_manifest
    out = make_manifest(args.tiles_dir, args.out_csv, args.pre_folder, args.post_folder)
    logger.info(f"Manifest created at: {out}")
    return 0


def handle_convert(args):
    from changedetect.src.data.convert_tiles import convert_manifest_to_npz
    n = convert_manifest_to_npz(args.manifest, args.out_dir, args.norm)
    logger.info(f"Converted {n} items from manifest")
    return 0


def handle_train(args):
    """Handle the train command"""
    # Import train main here to avoid importing torch at module import time
    from changedetect.src.train import main as train_main

    # Convert args to list and set sys.argv for train_main
    train_args = []
    for key, value in vars(args).items():
        if key != 'command' and value is not None:
            if isinstance(value, bool):
                if value:
                    train_args.append(f"--{key}")
            else:
                train_args.append(f"--{key}")
                train_args.append(str(value))

    # Set sys.argv so the train.main() parser reads the correct arguments
    sys_argv_backup = sys.argv
    try:
        sys.argv = ['train'] + train_args
        return train_main()
    finally:
        sys.argv = sys_argv_backup


def handle_inference(args):
    """Handle the inference command"""
    # Import inference main here to avoid importing torch at module import time
    from changedetect.src.inference import main as inference_main

    # Convert args to list and set sys.argv for inference_main
    inference_args = []
    for key, value in vars(args).items():
        if key != 'command' and value is not None:
            if isinstance(value, bool):
                if value:
                    inference_args.append(f"--{key}")
            else:
                inference_args.append(f"--{key}")
                inference_args.append(str(value))

    sys_argv_backup = sys.argv
    try:
        sys.argv = ['inference'] + inference_args
        return inference_main()
    finally:
        sys.argv = sys_argv_backup


def handle_evaluate(args):
    """Handle the evaluate command"""
    # Import evaluate main here to avoid importing heavy deps at module import time
    from changedetect.src.evaluate import main as evaluate_main

    # Convert args to list and set sys.argv for evaluate_main
    evaluate_args = []
    for key, value in vars(args).items():
        if key != 'command' and value is not None:
            if isinstance(value, bool):
                if value:
                    evaluate_args.append(f"--{key}")
            else:
                evaluate_args.append(f"--{key}")
                evaluate_args.append(str(value))

    sys_argv_backup = sys.argv
    try:
        sys.argv = ['evaluate'] + evaluate_args
        return evaluate_main()
    finally:
        sys.argv = sys_argv_backup


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute the appropriate command
    if args.command == 'download':
        return handle_download(args)
    elif args.command == 'stack':
        return handle_stack(args)
    elif args.command == 'tile':
        return handle_tile(args)
    elif args.command == 'manifest':
        return handle_manifest(args)
    elif args.command == 'convert':
        return handle_convert(args)
    elif args.command == 'prepare-pairs':
        # lazy import
        from changedetect.src.data.prepare_pairs import make_pair_folder
        make_pair_folder(args.manifest, args.out_dir, overwrite=args.overwrite)
        logger.info(f"Created paired files in: {args.out_dir}")
        return 0
    elif args.command == 'generate-masks':
        from changedetect.src.data.generate_masks import generate_blank_masks
        n = generate_blank_masks(args.manifest, args.out_dir, overwrite=args.overwrite)
        logger.info(f"Wrote {n} mask files to: {args.out_dir}")
        return 0
    elif args.command == 'train':
        return handle_train(args)
    elif args.command == 'inference':
        return handle_inference(args)
    elif args.command == 'evaluate':
        return handle_evaluate(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())