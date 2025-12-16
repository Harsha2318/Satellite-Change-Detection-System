#!/usr/bin/env python
"""
Simplified training script without problematic imports
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import time
from datetime import datetime

# Simple dataset loader
class SimpleChangeDetectionDataset(Dataset):
    def __init__(self, before_dir, after_dir, mask_dir):
        from PIL import Image
        import numpy as np
        
        self.before_dir = Path(before_dir)
        self.after_dir = Path(after_dir)
        self.mask_dir = Path(mask_dir)
        
        self.before_images = sorted(list(self.before_dir.glob('*.png')) + list(self.before_dir.glob('*.jpg')))
        self.after_images = sorted(list(self.after_dir.glob('*.png')) + list(self.after_dir.glob('*.jpg')))
        self.masks = sorted(list(self.mask_dir.glob('*.png')) + list(self.mask_dir.glob('*.jpg')))
        
        assert len(self.before_images) == len(self.after_images) == len(self.masks), \
            f"Mismatched lengths: before={len(self.before_images)}, after={len(self.after_images)}, masks={len(self.masks)}"
        
    def __len__(self):
        return len(self.before_images)
    
    def __getitem__(self, idx):
        from PIL import Image
        import numpy as np
        
        # Load before and after images
        before = Image.open(self.before_images[idx]).convert('RGB')
        after = Image.open(self.after_images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')
        
        # Stack before and after (6 channels total)
        img_array = np.concatenate([np.array(before, dtype=np.float32) / 255.0,
                                    np.array(after, dtype=np.float32) / 255.0], axis=2)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0).unsqueeze(0)
        
        return img_tensor, mask_tensor

# Simple U-Net model
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

def train():
    # Paths
    before_dir = Path('data/train/before')
    after_dir = Path('data/train/after')
    mask_dir = Path('data/train/labels')
    output_dir = Path('changedetect/models')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Hyperparameters
    batch_size = 4
    num_epochs = 3
    learning_rate = 0.001
    
    print("[TRAINING START]")
    print(f"  Before dir: {before_dir}")
    print(f"  After dir: {after_dir}")
    print(f"  Mask dir: {mask_dir}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    
    # Create dataset
    dataset = SimpleChangeDetectionDataset(before_dir, after_dir, mask_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"  Dataset size: {len(dataset)} image pairs")
    
    # Model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    model = SimpleUNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    start_time = time.time()
    history = {'loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % max(1, len(dataloader)//2) == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        history['loss'].append(float(avg_loss))
        print(f"[OK] Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
    
    # Save model
    model_path = output_dir / 'best_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"[OK] Model saved to: {model_path}")
    
    # Save history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"[OK] History saved to: {history_path}")
    
    elapsed = time.time() - start_time
    print(f"[OK] Training completed in {elapsed:.2f} seconds")
    
    return model, history

if __name__ == '__main__':
    train()
