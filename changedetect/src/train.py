"""
Training script for satellite image change detection
"""
import os
import sys
import time
import datetime
import logging
import argparse
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter  # Disabled due to compatibility issues

from changedetect.src.data.dataset import (
    ChangeDetectionDataset,
    create_dataloaders,
    get_data_transforms
)
from changedetect.src.models.siamese_unet import get_change_detection_model
from changedetect.src.utils.md5_utils import compute_md5, save_md5_hash
from changedetect.src.utils.metrics import (
    calculate_iou,
    calculate_dice,
    calculate_precision_recall_f1,
    calculate_confusion_matrix
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """Dice loss for image segmentation"""
    
    def __init__(self, smooth=1.0):
        """
        Initialize the Dice loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        """
        Calculate the Dice loss.
        
        Args:
            logits: Predicted logits
            targets: Ground truth labels
            
        Returns:
            Dice loss
        """
        batch_size = logits.size(0)
        
        # Apply sigmoid activation to logits
        probs = torch.sigmoid(logits)
        
        # Flatten the tensors
        probs_flat = probs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)
        
        # Calculate intersection and union
        intersection = (probs_flat * targets_flat).sum(1)
        union = probs_flat.sum(1) + targets_flat.sum(1)
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss (1 - Dice coefficient)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice loss for image segmentation"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        """
        Initialize the combined loss.
        
        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
            smooth: Smoothing factor for Dice loss
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)
        
    def forward(self, logits, targets):
        """
        Calculate the combined loss.
        
        Args:
            logits: Predicted logits
            targets: Ground truth labels
            
        Returns:
            Combined BCE and Dice loss
        """
        bce_loss = self.bce_loss(logits, targets)
        dice_loss = self.dice_loss(logits, targets)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    output_dir,
    save_interval=5,
    early_stopping_patience=10
):
    """
    Train the change detection model.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use for training (cuda or cpu)
        num_epochs: Number of training epochs
        output_dir: Directory to save model checkpoints
        save_interval: Interval for saving model checkpoints
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        
    Returns:
        Trained model and training history
    """
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # TensorBoard logging disabled due to compatibility issues
    writer = None
    
    # Initialize variables for tracking progress
    best_val_loss = float('inf')
    best_val_iou = 0.0
    epochs_no_improve = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': [],
        'val_f1': []
    }
    
    # Move model to device
    model.to(device)
    
    # Start training
    start_time = time.time()
    logger.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Get data from batch
            t1_images = batch['t1'].to(device).float()
            t2_images = batch['t2'].to(device).float()
            masks = batch['mask'].to(device).float()
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(t1_images, t2_images)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}")
        
        # Calculate average training loss for the epoch
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        val_precision, val_recall, val_f1 = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Get data from batch
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
                
                # Update statistics
                val_loss += loss.item()
                val_iou += batch_iou
                val_dice += batch_dice
                val_precision += batch_precision
                val_recall += batch_recall
                val_f1 += batch_f1
        
        # Calculate average validation metrics for the epoch
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)
        val_precision /= len(val_loader)
        val_recall /= len(val_loader)
        val_f1 /= len(val_loader)
        
        # Update history
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)
        history['val_f1'].append(val_f1)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Log validation results
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {time.time() - epoch_start_time:.2f} seconds")
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}")
        logger.info(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        
        # TensorBoard logging disabled
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/val_iou', val_iou, epoch)
            writer.add_scalar('Metrics/val_dice', val_dice, epoch)
            writer.add_scalar('Metrics/val_f1', val_f1, epoch)
        
        # Save checkpoint if it's the best model so far
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'val_dice': val_dice,
                'val_f1': val_f1
            }, best_model_path)
            
            # Compute and save MD5 hash of the best model
            model_hash = compute_md5(best_model_path)
            save_md5_hash(best_model_path, model_hash)
            
            logger.info(f"Saved best model with IoU: {val_iou:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        # Save checkpoint at regular intervals
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou
            }, checkpoint_path)
            logger.info(f"Saved checkpoint at epoch {epoch+1}")
        
        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_iou': val_iou
    }, final_model_path)
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in history.items()}, f, indent=4)
    
    # Close TensorBoard writer if enabled
    if writer:
        writer.close()
    
    # Log total training time
    total_time = time.time() - start_time
    logger.info(f"Training completed in {datetime.timedelta(seconds=int(total_time))}")
    
    return model, history


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Train a satellite image change detection model")
    
    # Data parameters
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Directory containing image pairs (t1 and t2)")
    parser.add_argument("--mask_dir", type=str, required=True,
                       help="Directory containing mask images")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Directory to save model checkpoints and logs")
    parser.add_argument("--tile_size", type=int, default=256,
                       help="Size of image tiles to extract")
    parser.add_argument("--overlap", type=int, default=32,
                       help="Overlap between tiles in pixels")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="siamese_unet",
                       choices=["siamese_unet", "siamese_diff", "fcn_diff"],
                       help="Type of change detection model to use")
    parser.add_argument("--in_channels", type=int, default=3,
                       help="Number of input channels per image")
    parser.add_argument("--features", type=int, default=64,
                       help="Number of features in the first layer")
    parser.add_argument("--bilinear", action="store_true",
                       help="Use bilinear interpolation instead of transposed convolutions")
    parser.add_argument("--dropout", type=float, default=0.2,
                       help="Dropout probability")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay (L2 penalty)")
    parser.add_argument("--val_split", type=float, default=0.2,
                       help="Fraction of data to use for validation")
    parser.add_argument("--save_interval", type=int, default=5,
                       help="Interval for saving model checkpoints")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of worker threads for data loading")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    
    return parser.parse_args()


def main():
    """Main entry point"""
    # Parse command-line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        args.image_dir,
        args.mask_dir,
        tile_size=args.tile_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        overlap=args.overlap
    )
    logger.info(f"Created data loaders with {len(train_loader)} training batches and {len(val_loader)} validation batches")
    
    # Create model
    logger.info(f"Creating {args.model_type} model...")
    model = get_change_detection_model(
        args.model_type,
        in_channels=args.in_channels,
        out_channels=1,
        features=args.features,
        bilinear=args.bilinear,
        dropout=args.dropout
    )
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Define loss function and optimizer
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # 'verbose' kwarg removed for compatibility with some torch versions
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume is not None and os.path.isfile(args.resume):
        logger.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info(f"Resuming from epoch {start_epoch}")
    
    # Train model
    logger.info("Starting training...")
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        args.num_epochs,
        args.output_dir,
        save_interval=args.save_interval,
        early_stopping_patience=args.patience
    )
    
    logger.info("Training completed")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())