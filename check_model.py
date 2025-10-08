"""
Check the model file and try a simple inference without thresholding
"""
import os
import sys
import torch
import numpy as np
import rasterio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import model-related code
try:
    from changedetect.src.models.siamese_unet import get_change_detection_model
    from changedetect.src.data.preprocess import preprocess_pair, normalize_image
except ImportError:
    print("ERROR: Cannot import required modules. Please run from project root.")
    sys.exit(1)

def check_model_file(model_path):
    """Check if the model file exists and is valid."""
    if not os.path.exists(model_path):
        print(f"ERROR: Model file {model_path} does not exist.")
        return False
    
    try:
        # Try to load the model file
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Check if it's a valid checkpoint
        if 'model_state_dict' not in checkpoint:
            print(f"ERROR: Model file {model_path} does not contain model_state_dict.")
            print(f"Contents: {list(checkpoint.keys())}")
            return False
        
        # Check for training metrics
        if 'val_iou' in checkpoint:
            print(f"Model validation IoU: {checkpoint['val_iou']}")
        
        if 'epoch' in checkpoint:
            print(f"Model trained for {checkpoint['epoch']} epochs")
        
        return True
    except Exception as e:
        print(f"ERROR: Failed to load model file {model_path}: {str(e)}")
        return False

def test_inference_without_threshold(model_path, t1_path, t2_path, model_type='siamese_unet', in_channels=3):
    """Try inference with the model but don't apply thresholding."""
    if not os.path.exists(t1_path) or not os.path.exists(t2_path):
        print(f"ERROR: Image files do not exist: {t1_path} or {t2_path}")
        return
    
    print(f"Loading model from {model_path}")
    device = torch.device('cpu')
    
    # Create model architecture
    model = get_change_detection_model(
        model_type=model_type,
        in_channels=in_channels,
        out_channels=1,
        features=64,
        bilinear=False
    )
    
    # Load model weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        return
    
    # Load images
    print(f"Loading images: {t1_path} and {t2_path}")
    try:
        with rasterio.open(t1_path) as src:
            t1_img = src.read()
        
        with rasterio.open(t2_path) as src:
            t2_img = src.read()
        
        print(f"Loaded images with shapes: {t1_img.shape} and {t2_img.shape}")
    except Exception as e:
        print(f"ERROR loading images: {str(e)}")
        return
    
    # Check for valid pixel values
    print(f"T1 stats: min={t1_img.min()}, max={t1_img.max()}, mean={t1_img.mean():.4f}")
    print(f"T2 stats: min={t2_img.min()}, max={t2_img.max()}, mean={t2_img.mean():.4f}")
    
    if t1_img.min() == t1_img.max() or t2_img.min() == t2_img.max():
        print("WARNING: One of the images has uniform values (no variation)")
    
    # Preprocess images
    t1_img, t2_img = preprocess_pair(t1_img, t2_img)
    
    # Convert to PyTorch tensors
    t1_tensor = torch.from_numpy(t1_img).unsqueeze(0).to(device).float()
    t2_tensor = torch.from_numpy(t2_img).unsqueeze(0).to(device).float()
    
    # Make prediction
    print("Running inference...")
    with torch.no_grad():
        output = model(t1_tensor, t2_tensor)
        # Raw logits output
        raw_output = output.squeeze().cpu().numpy()
        
        # Apply sigmoid to get probabilities
        prob_output = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Apply different thresholds
        thresh_outputs = {}
        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
            thresholded = (prob_output > threshold).astype(np.uint8)
            thresh_outputs[threshold] = thresholded
    
    # Print statistics about the output
    print("\nRaw output statistics:")
    print(f"Shape: {raw_output.shape}")
    print(f"Min: {raw_output.min():.4f}, Max: {raw_output.max():.4f}")
    print(f"Mean: {raw_output.mean():.4f}, Std: {raw_output.std():.4f}")
    
    print("\nProbability output statistics:")
    print(f"Min: {prob_output.min():.4f}, Max: {prob_output.max():.4f}")
    print(f"Mean: {prob_output.mean():.4f}, Std: {prob_output.std():.4f}")
    
    for threshold, thresh_out in thresh_outputs.items():
        pixel_count = thresh_out.sum()
        percentage = (pixel_count / thresh_out.size) * 100
        print(f"\nThreshold {threshold}: {pixel_count} pixels ({percentage:.2f}%) classified as change")

if __name__ == "__main__":
    # Use the model files we found
    model_path = "changedetect/training_runs/run1_small/best_model.pth"
    
    print(f"Using model file: {model_path}")
    
    if check_model_file(model_path):
        # Find an image pair with actual data
        print("\nLooking for image pairs...")
        pairs_dir = Path("changedetect/data/processed/train_pairs")
        
        # Try to find a valid image pair
        test_samples = ["0_10", "0_11", "0_12"]
        
        for sample in test_samples:
            t1_path = pairs_dir / f"{sample}_t1.tif"
            t2_path = pairs_dir / f"{sample}_t2.tif"
            
            if t1_path.exists() and t2_path.exists():
                print(f"\nUsing sample {sample} for testing")
                test_inference_without_threshold(model_path, str(t1_path), str(t2_path))
                break
        else:
            print("ERROR: Could not find valid image pairs for testing")