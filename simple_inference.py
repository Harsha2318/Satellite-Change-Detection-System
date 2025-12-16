#!/usr/bin/env python
"""
Simple inference script for satellite change detection
"""
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np
import json
import os

# Simple U-Net model (same as training)
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.upconv = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Output
        self.final = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool(enc1)
        enc2 = self.enc2(x)
        x = self.pool(enc2)
        
        # Decoder
        x = self.upconv(enc2)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Output
        x = self.final(x)
        x = self.sigmoid(x)
        
        return x

def inference(model_path, before_image_path, after_image_path, output_dir, threshold=0.5):
    """
    Run inference on image pair
    
    Args:
        model_path: Path to trained model
        before_image_path: Path to before image
        after_image_path: Path to after image
        output_dir: Directory to save predictions
        threshold: Threshold for binary classification
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFERENCE START]")
    print(f"  Model: {model_path}")
    print(f"  Before: {before_image_path}")
    print(f"  After: {after_image_path}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"  Device: {device}")
    
    # Load images
    before = np.array(Image.open(before_image_path).convert('RGB'), dtype=np.float32) / 255.0
    after = np.array(Image.open(after_image_path).convert('RGB'), dtype=np.float32) / 255.0
    
    print(f"  Image size: {before.shape}")
    
    # Combine channels (6 channels total)
    combined = np.concatenate([before, after], axis=2)
    combined = combined.transpose(2, 0, 1)  # CHW
    x = torch.from_numpy(combined).unsqueeze(0).to(device)  # Add batch dimension
    
    # Inference
    with torch.no_grad():
        pred_proba = model(x)
    
    # Convert to numpy
    pred_proba = pred_proba.squeeze().cpu().numpy()
    pred_binary = (pred_proba > threshold).astype(np.uint8) * 255
    
    print(f"  Prediction shape: {pred_proba.shape}")
    print(f"  Changed pixels (>{threshold}): {(pred_proba > threshold).sum()} / {pred_proba.size}")
    print(f"  Change %: {(pred_proba > threshold).sum() / pred_proba.size * 100:.2f}%")
    
    # Save results
    basename = Path(before_image_path).stem
    
    # Save probability map
    prob_path = output_dir / f'{basename}_probability.png'
    prob_img = Image.fromarray((pred_proba * 255).astype(np.uint8))
    prob_img.save(prob_path)
    print(f"[OK] Probability map saved: {prob_path}")
    
    # Save binary map
    binary_path = output_dir / f'{basename}_binary.png'
    binary_img = Image.fromarray(pred_binary)
    binary_img.save(binary_path)
    print(f"[OK] Binary map saved: {binary_path}")
    
    # Save statistics
    stats = {
        'image': basename,
        'model': str(model_path),
        'threshold': threshold,
        'total_pixels': int(pred_proba.size),
        'changed_pixels': int((pred_proba > threshold).sum()),
        'change_percentage': float((pred_proba > threshold).sum() / pred_proba.size * 100),
        'mean_probability': float(pred_proba.mean()),
        'max_probability': float(pred_proba.max()),
        'min_probability': float(pred_proba.min()),
    }
    
    stats_path = output_dir / f'{basename}_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"[OK] Statistics saved: {stats_path}")
    
    return pred_proba, pred_binary, stats

def batch_inference(model_path, before_dir, after_dir, output_dir, threshold=0.5):
    """
    Run inference on all image pairs in directories
    """
    
    before_dir = Path(before_dir)
    after_dir = Path(after_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    before_images = sorted(list(before_dir.glob('*.png')) + list(before_dir.glob('*.jpg')))
    after_images = sorted(list(after_dir.glob('*.png')) + list(after_dir.glob('*.jpg')))
    
    print(f"\n[BATCH INFERENCE START]")
    print(f"  Found {len(before_images)} image pairs")
    
    all_stats = []
    
    for i, (before_path, after_path) in enumerate(zip(before_images, after_images)):
        print(f"\n[{i+1}/{len(before_images)}] Processing {before_path.name}")
        
        try:
            pred_proba, pred_binary, stats = inference(
                model_path, before_path, after_path, output_dir, threshold
            )
            all_stats.append(stats)
        except Exception as e:
            print(f"[ERROR] Failed to process {before_path.name}: {e}")
            continue
    
    # Save summary
    summary_path = output_dir / 'batch_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'total_processed': len(all_stats),
            'threshold': threshold,
            'results': all_stats,
            'average_change_percentage': np.mean([s['change_percentage'] for s in all_stats]) if all_stats else 0,
        }, f, indent=4)
    
    print(f"\n[OK] Batch inference complete!")
    print(f"[OK] Summary saved: {summary_path}")

if __name__ == '__main__':
    import sys
    
    # Single inference example
    model_path = 'changedetect/models/best_model.pth'
    before_img = 'data/test/before/img_000.png'
    after_img = 'data/test/after/img_000.png'
    output_dir = 'predictions'
    
    if len(sys.argv) > 1 and sys.argv[1] == 'batch':
        # Batch inference
        batch_inference(
            model_path,
            'data/test/before',
            'data/test/after',
            output_dir
        )
    else:
        # Single inference
        print("Running single image inference...")
        inference(model_path, before_img, after_img, output_dir)
        
        print("\n[NEXT STEPS]")
        print("  1. View probability map: predictions/img_000_probability.png")
        print("  2. View binary map: predictions/img_000_binary.png")
        print("  3. Check statistics: predictions/img_000_stats.json")
        print("  4. For batch inference: python simple_inference.py batch")
