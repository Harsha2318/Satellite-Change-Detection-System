"""
Evaluation script for satellite image change detection
"""
import os
import sys
import time
import logging
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import geopandas as gpd
import torch
from tqdm import tqdm

from changedetect.src.utils.metrics import (
    calculate_all_metrics,
    calculate_iou,
    calculate_dice,
    calculate_precision_recall_f1,
    calculate_confusion_matrix
)
from changedetect.src.models.siamese_unet import get_change_detection_model
from changedetect.src.inference import load_model, predict_large_image
from changedetect.src.data.dataset import create_test_dataloader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_predictions(pred_dir, gt_dir):
    """
    Evaluate model predictions against ground truth.
    
    Args:
        pred_dir: Directory containing prediction masks
        gt_dir: Directory containing ground truth masks
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating predictions in {pred_dir} against ground truth in {gt_dir}")
    
    # Find all prediction files
    pred_files = list(Path(pred_dir).glob("*_change_mask.tif"))
    
    if not pred_files:
        logger.error(f"No prediction files found in {pred_dir}")
        return None
    
    # Initialize metrics
    all_metrics = {
        'iou': [],
        'dice': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'accuracy': [],
        'tp': [],
        'fp': [],
        'fn': [],
        'tn': []
    }
    
    # Evaluate each prediction
    for pred_file in tqdm(pred_files, desc="Evaluating"):
        # Get corresponding ground truth file
        name = pred_file.stem.replace("_change_mask", "")
        gt_file = Path(gt_dir) / f"{name}_mask.tif"
        
        if not gt_file.exists():
            logger.warning(f"Ground truth not found for {name}")
            continue
        
        # Read prediction and ground truth
        with rasterio.open(pred_file) as src:
            pred = src.read(1)
            pred_binary = (pred > 0).astype(bool)
        
        with rasterio.open(gt_file) as src:
            gt = src.read(1)
            gt_binary = (gt > 0).astype(bool)
        
        # Calculate metrics
        metrics = calculate_all_metrics(pred_binary, gt_binary)
        
        # Store metrics
        for key, value in metrics.items():
            if key in all_metrics:
                all_metrics[key].append(value)
        
        logger.info(f"Metrics for {name}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}, F1={metrics['f1']:.4f}")
    
    # Calculate average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    std_metrics = {f"{key}_std": np.std(values) for key, values in all_metrics.items()}
    
    # Combine average and standard deviation metrics
    result_metrics = {**avg_metrics, **std_metrics}
    result_metrics['num_samples'] = len(pred_files)
    
    logger.info(f"Average metrics: IoU={avg_metrics['iou']:.4f}, Dice={avg_metrics['dice']:.4f}, F1={avg_metrics['f1']:.4f}")
    
    return result_metrics


def evaluate_vector_results(pred_dir, gt_dir):
    """
    Evaluate vector results against ground truth.
    
    Args:
        pred_dir: Directory containing prediction vectors
        gt_dir: Directory containing ground truth vectors
        
    Returns:
        Dictionary of vector evaluation metrics
    """
    logger.info(f"Evaluating vector predictions in {pred_dir} against ground truth in {gt_dir}")
    
    # Find all prediction files
    pred_files = list(Path(pred_dir).glob("*_change_vectors.shp"))
    
    if not pred_files:
        logger.error(f"No vector prediction files found in {pred_dir}")
        return None
    
    # Initialize metrics
    vector_metrics = {
        'iou': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'num_pred_polygons': [],
        'num_gt_polygons': [],
        'area_ratio': []
    }
    
    # Evaluate each prediction
    for pred_file in tqdm(pred_files, desc="Evaluating vectors"):
        # Get corresponding ground truth file
        name = pred_file.stem.replace("_change_vectors", "")
        gt_file = Path(gt_dir) / f"{name}_vector.shp"
        
        if not gt_file.exists():
            logger.warning(f"Ground truth vector not found for {name}")
            continue
        
        try:
            # Read prediction and ground truth
            pred_gdf = gpd.read_file(pred_file)
            gt_gdf = gpd.read_file(gt_file)
            
            # Make sure they have the same CRS
            if pred_gdf.crs != gt_gdf.crs:
                gt_gdf = gt_gdf.to_crs(pred_gdf.crs)
            
            # Create unions for overall comparison
            pred_union = pred_gdf.unary_union
            gt_union = gt_gdf.unary_union
            
            # Calculate intersection and union
            intersection = pred_union.intersection(gt_union).area
            union = pred_union.union(gt_union).area
            
            # Calculate metrics
            iou = intersection / union if union > 0 else 0
            precision = intersection / pred_union.area if pred_union.area > 0 else 0
            recall = intersection / gt_union.area if gt_union.area > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store metrics
            vector_metrics['iou'].append(iou)
            vector_metrics['precision'].append(precision)
            vector_metrics['recall'].append(recall)
            vector_metrics['f1'].append(f1)
            vector_metrics['num_pred_polygons'].append(len(pred_gdf))
            vector_metrics['num_gt_polygons'].append(len(gt_gdf))
            vector_metrics['area_ratio'].append(pred_union.area / gt_union.area if gt_union.area > 0 else float('inf'))
            
            logger.info(f"Vector metrics for {name}: IoU={iou:.4f}, F1={f1:.4f}")
        
        except Exception as e:
            logger.error(f"Error evaluating vector {name}: {e}")
    
    # Calculate average metrics
    avg_metrics = {key: np.mean(values) for key, values in vector_metrics.items()}
    std_metrics = {f"{key}_std": np.std(values) for key, values in vector_metrics.items()}
    
    # Combine average and standard deviation metrics
    result_metrics = {**avg_metrics, **std_metrics}
    result_metrics['num_samples'] = len(vector_metrics['iou'])
    
    logger.info(f"Average vector metrics: IoU={avg_metrics['iou']:.4f}, F1={avg_metrics['f1']:.4f}")
    
    return result_metrics


def plot_metrics(metrics_list, output_dir):
    """
    Plot evaluation metrics.
    
    Args:
        metrics_list: List of metrics dictionaries
        output_dir: Directory to save plots
        
    Returns:
        List of saved plot paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = []
    
    # Extract metrics
    metrics_names = ['iou', 'dice', 'precision', 'recall', 'f1']
    
    for metric_name in metrics_names:
        plt.figure(figsize=(10, 6))
        
        for i, metrics in enumerate(metrics_list):
            if metric_name in metrics:
                plt.bar(i, metrics[metric_name], yerr=metrics.get(f"{metric_name}_std", 0))
        
        plt.xlabel('Models')
        plt.ylabel(metric_name.upper())
        plt.title(f'{metric_name.upper()} Comparison')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save plot
        plot_path = os.path.join(output_dir, f'{metric_name}_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        saved_plots.append(plot_path)
    
    return saved_plots


def evaluate_model_on_dataset(model, dataloader, device):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_metrics = {
        'iou': [],
        'dice': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'loss': []
    }
    
    criterion = torch.nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get data
            t1_images = batch['t1'].to(device).float()
            t2_images = batch['t2'].to(device).float()
            masks = batch['mask'].to(device).float()
            
            # Forward pass
            outputs = model(t1_images, t2_images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            preds = torch.sigmoid(outputs) > 0.5
            batch_iou = calculate_iou(preds, masks > 0.5)
            batch_dice = calculate_dice(preds, masks > 0.5)
            batch_precision, batch_recall, batch_f1 = calculate_precision_recall_f1(preds, masks > 0.5)
            
            # Store metrics
            all_metrics['iou'].append(batch_iou)
            all_metrics['dice'].append(batch_dice)
            all_metrics['precision'].append(batch_precision)
            all_metrics['recall'].append(batch_recall)
            all_metrics['f1'].append(batch_f1)
            all_metrics['loss'].append(loss.item())
    
    # Calculate average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    std_metrics = {f"{key}_std": np.std(values) for key, values in all_metrics.items()}
    
    # Combine average and standard deviation metrics
    result_metrics = {**avg_metrics, **std_metrics}
    
    return result_metrics


def compare_models(model_paths, test_data_dir, output_dir):
    """
    Compare multiple models on a test dataset.
    
    Args:
        model_paths: List of model paths
        test_data_dir: Directory containing test data
        output_dir: Directory to save results
        
    Returns:
        Dictionary of comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test dataloader
    test_loader = create_test_dataloader(
        test_data_dir,
        tile_size=256,
        batch_size=16,
        num_workers=4,
        overlap=32
    )
    
    # Evaluate each model
    all_results = []
    
    for i, model_path in enumerate(model_paths):
        logger.info(f"Evaluating model {i+1}/{len(model_paths)}: {Path(model_path).name}")
        
        # Load model
        model_type = "siamese_unet"  # Default model type
        in_channels = 3  # Default input channels
        
        if Path(model_path).with_suffix('.json').exists():
            with open(Path(model_path).with_suffix('.json'), 'r') as f:
                model_info = json.load(f)
                model_type = model_info.get('model_type', model_type)
                in_channels = model_info.get('in_channels', in_channels)
        
        model = load_model(model_path, model_type, in_channels, device)
        
        # Evaluate model
        metrics = evaluate_model_on_dataset(model, test_loader, device)
        metrics['model_path'] = model_path
        metrics['model_name'] = Path(model_path).stem
        
        all_results.append(metrics)
        
        logger.info(f"Model {i+1} metrics: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}, F1={metrics['f1']:.4f}")
    
    # Save results
    results_path = os.path.join(output_dir, 'model_comparison.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Plot comparison
    plot_metrics(all_results, output_dir)
    
    # Return best model based on IoU
    best_model_idx = np.argmax([m['iou'] for m in all_results])
    best_model = all_results[best_model_idx]
    
    logger.info(f"Best model: {best_model['model_name']} with IoU={best_model['iou']:.4f}")
    
    return {
        'all_results': all_results,
        'best_model': best_model,
        'best_model_path': model_paths[best_model_idx]
    }


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate satellite image change detection results")
    
    # Main command
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Evaluate predictions command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate predictions')
    eval_parser.add_argument("--pred_dir", type=str, required=True,
                           help="Directory containing prediction masks")
    eval_parser.add_argument("--gt_dir", type=str, required=True,
                           help="Directory containing ground truth masks")
    eval_parser.add_argument("--output_dir", type=str, default="./evaluation",
                           help="Directory to save evaluation results")
    eval_parser.add_argument("--vector", action="store_true",
                           help="Evaluate vector results")
    
    # Compare models command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument("--model_paths", type=str, nargs='+', required=True,
                              help="Paths to model checkpoints")
    compare_parser.add_argument("--test_data_dir", type=str, required=True,
                              help="Directory containing test data")
    compare_parser.add_argument("--output_dir", type=str, default="./comparison",
                              help="Directory to save comparison results")
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    if args.command == 'evaluate':
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Evaluate raster predictions
        raster_metrics = evaluate_predictions(args.pred_dir, args.gt_dir)
        
        if raster_metrics:
            raster_metrics_path = os.path.join(args.output_dir, 'raster_metrics.json')
            with open(raster_metrics_path, 'w') as f:
                json.dump(raster_metrics, f, indent=4)
            
            logger.info(f"Raster evaluation metrics saved to {raster_metrics_path}")
        
        # Evaluate vector predictions if requested
        if args.vector:
            vector_metrics = evaluate_vector_results(args.pred_dir, args.gt_dir)
            
            if vector_metrics:
                vector_metrics_path = os.path.join(args.output_dir, 'vector_metrics.json')
                with open(vector_metrics_path, 'w') as f:
                    json.dump(vector_metrics, f, indent=4)
                
                logger.info(f"Vector evaluation metrics saved to {vector_metrics_path}")
    
    elif args.command == 'compare':
        # Compare models
        comparison = compare_models(args.model_paths, args.test_data_dir, args.output_dir)
        
        # Log best model
        logger.info(f"Best model: {comparison['best_model']['model_name']}")
        logger.info(f"IoU: {comparison['best_model']['iou']:.4f}")
        logger.info(f"Dice: {comparison['best_model']['dice']:.4f}")
        logger.info(f"F1: {comparison['best_model']['f1']:.4f}")
    
    else:
        logger.error("No command specified")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())