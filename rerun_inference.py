"""
Re-run inference with a lower threshold to detect changes
"""
import os
import sys
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import model-related code
from changedetect.src.models.siamese_unet import get_change_detection_model
from changedetect.src.data.preprocess import preprocess_pair, normalize_image
from changedetect.src.utils.geoutils import read_geotiff, write_geotiff, raster_to_vector

def load_model(model_path, model_type='siamese_unet', in_channels=3, device='cpu'):
    """Load the trained model."""
    print(f"Loading model from {model_path}")
    
    # Create model architecture
    model = get_change_detection_model(
        model_type=model_type,
        in_channels=in_channels,
        out_channels=1,
        features=64,
        bilinear=False
    )
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded with IoU: {checkpoint.get('val_iou', 'N/A')}")
    
    return model

def predict_tile(model, t1_tile, t2_tile, device, threshold=0.1):
    """
    Make a prediction for a single tile with a lower threshold.
    """
    # Preprocess tiles
    t1_tile, t2_tile = preprocess_pair(t1_tile, t2_tile)
    
    # Convert to PyTorch tensors
    t1_tensor = torch.from_numpy(t1_tile).unsqueeze(0).to(device).float()
    t2_tensor = torch.from_numpy(t2_tile).unsqueeze(0).to(device).float()
    
    # Make prediction
    with torch.no_grad():
        output = model(t1_tensor, t2_tensor)
        # Apply custom threshold
        prediction = torch.sigmoid(output) > threshold
        
    # Convert to numpy array
    prediction = prediction.squeeze().cpu().numpy().astype(np.uint8) * 255
    
    return prediction

def run_inference_with_lower_threshold(model, image_pairs, output_dir, threshold=0.1, device='cpu'):
    """
    Run inference on multiple image pairs with a lower threshold.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_paths = []
    
    for pair in tqdm(image_pairs, desc="Processing image pairs"):
        t1_path = pair['t1']
        t2_path = pair['t2']
        name = pair['name']
        
        print(f"\nProcessing image pair: {name}")
        
        # Create output paths
        raster_output = os.path.join(output_dir, f"{name}_change_mask.tif")
        vector_output = os.path.join(output_dir, f"{name}_change_vectors.shp")
        visualization_output = os.path.join(output_dir, f"{name}_visualization.png")
        
        # Load images
        try:
            with rasterio.open(t1_path) as src:
                t1_profile = src.profile.copy()
                t1_img = src.read()
                
            with rasterio.open(t2_path) as src:
                t2_img = src.read()
                
            # Skip if images are empty
            if t1_img.min() == t1_img.max() == 0 or t2_img.min() == t2_img.max() == 0:
                print(f"Skipping {name}: Empty images")
                continue
                
            # Make prediction
            prediction = predict_tile(model, t1_img, t2_img, device, threshold)
            
            # Set output profile
            out_profile = t1_profile.copy()
            out_profile.update({
                'count': 1,
                'dtype': 'uint8',
                'compress': 'lzw',
                'nodata': 0
            })
            
            # Write prediction to file
            with rasterio.open(raster_output, 'w', **out_profile) as dst:
                dst.write(prediction, 1)
                
            print(f"Saved prediction to {raster_output}")
            
            # Create vector output
            try:
                raster_to_vector(raster_output, vector_output, value=1, min_area=10)
                print(f"Saved vector output to {vector_output}")
            except Exception as e:
                print(f"Failed to create vector output: {e}")
            
            # Create visualization
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            
            # Normalize for better visualization
            def normalize_for_display(img):
                result = np.zeros_like(img, dtype=np.float32)
                for b in range(img.shape[0]):
                    band = img[b].astype(np.float32)
                    p2 = np.percentile(band, 2)
                    p98 = np.percentile(band, 98)
                    if p98 > p2:
                        result[b] = np.clip((band - p2) / (p98 - p2), 0, 1)
                    else:
                        result[b] = band / band.max() if band.max() > 0 else band
                return np.transpose(result, (1, 2, 0))
            
            t1_vis = normalize_for_display(t1_img)
            t2_vis = normalize_for_display(t2_img)
            
            axs[0, 0].imshow(t1_vis)
            axs[0, 0].set_title("Before (T1)")
            axs[0, 0].axis('off')
            
            axs[0, 1].imshow(t2_vis)
            axs[0, 1].set_title("After (T2)")
            axs[0, 1].axis('off')
            
            axs[1, 0].imshow(prediction, cmap='hot')
            axs[1, 0].set_title(f"Change Mask (threshold={threshold})")
            axs[1, 0].axis('off')
            
            # Create overlay of prediction on T2 image
            overlay = t2_vis.copy()
            mask = prediction > 0
            overlay_mask = np.zeros_like(overlay)
            overlay_mask[..., 0] = mask  # Red channel
            overlay = np.clip(overlay * 0.7 + overlay_mask * 0.3, 0, 1)
            
            axs[1, 1].imshow(overlay)
            axs[1, 1].set_title("Changes Overlay on T2")
            axs[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(visualization_output, dpi=150)
            plt.close(fig)
            
            print(f"Saved visualization to {visualization_output}")
            
            # Count changes
            change_pixels = (prediction > 0).sum()
            change_percent = (change_pixels / (prediction.shape[0] * prediction.shape[1])) * 100
            print(f"Detected changes: {change_pixels} pixels ({change_percent:.2f}%)")
            
            output_paths.append({
                'name': name,
                'raster': raster_output,
                'vector': vector_output,
                'visualization': visualization_output,
                'change_pixels': int(change_pixels),
                'change_percent': float(change_percent)
            })
            
        except Exception as e:
            print(f"Error processing {name}: {e}")
    
    # Save summary
    summary_path = os.path.join(output_dir, "inference_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Inference with threshold={threshold}\n")
        f.write(f"Total image pairs processed: {len(output_paths)}\n")
        f.write("\nSummary of detected changes:\n")
        
        # Sort by amount of changes
        output_paths.sort(key=lambda x: x['change_percent'], reverse=True)
        
        for output in output_paths:
            f.write(f"{output['name']}: {output['change_pixels']} pixels ({output['change_percent']:.2f}%)\n")
    
    print(f"\nInference completed for {len(output_paths)} image pairs")
    print(f"Results saved to {output_dir}")
    
    return output_paths

def get_image_pairs(image_dir):
    """
    Get list of image pair files.
    """
    image_dir = Path(image_dir)
    image_pairs = []
    
    # List all t1 images
    t1_images = sorted(list(image_dir.glob("*_t1.tif")))
    
    for t1_path in t1_images:
        # Get corresponding t2 image
        name = t1_path.stem.replace("_t1", "")
        t2_path = image_dir / f"{name}_t2.tif"
        
        if not t2_path.exists():
            print(f"Missing t2 image for {name}")
            continue
        
        image_pairs.append({
            "name": name,
            "t1": str(t1_path),
            "t2": str(t2_path),
        })
    
    return image_pairs

if __name__ == "__main__":
    # Set parameters
    model_path = "changedetect/training_runs/run1_small/best_model.pth"
    image_dir = "changedetect/data/processed/train_pairs"
    output_dir = "predictions_threshold_0.1"
    threshold = 0.1
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path, device=device)
    
    # Get image pairs
    image_pairs = get_image_pairs(image_dir)
    print(f"Found {len(image_pairs)} image pairs")
    
    # Run inference
    output_paths = run_inference_with_lower_threshold(
        model,
        image_pairs,
        output_dir,
        threshold=threshold,
        device=device
    )