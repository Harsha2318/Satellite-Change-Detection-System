"""
Siamese U-Net architecture for satellite image change detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from changedetect.src.models.unet import UNet, ConvBlock, DownBlock, UpBlock, OutConv


class SiameseUNet(nn.Module):
    """Siamese U-Net architecture for change detection"""
    
    def __init__(self, in_channels, out_channels=1, features=64, bilinear=True, dropout=0.0):
        """
        Initialize the Siamese U-Net model.
        
        Args:
            in_channels: Number of input channels per image (typically 3 for RGB or more for multispectral)
            out_channels: Number of output channels (typically 1 for binary change detection)
            features: Number of features in the first layer
            bilinear: Whether to use bilinear interpolation or transposed convolutions
            dropout: Dropout probability
        """
        super(SiameseUNet, self).__init__()
        
        # Shared encoder
        self.encoder = UNet(in_channels, out_channels, features, bilinear, dropout)
        
        # Fusion module
        self.fusion = ConvBlock(features * 16 // (2 if bilinear else 1) * 2, 
                               features * 16 // (2 if bilinear else 1), dropout=dropout)
        
        # Decoder path
        factor = 2 if bilinear else 1
        self.up1 = UpBlock(features * 16, features * 8 // factor, bilinear, dropout=dropout)
        self.up2 = UpBlock(features * 8, features * 4 // factor, bilinear, dropout=dropout)
        self.up3 = UpBlock(features * 4, features * 2 // factor, bilinear, dropout=dropout)
        self.up4 = UpBlock(features * 2, features, bilinear, dropout=dropout)
        
        # Output layer
        self.outc = OutConv(features, out_channels)
    
    def forward(self, x1, x2):
        """
        Forward pass through the Siamese U-Net.
        
        Args:
            x1: Input image at time 1
            x2: Input image at time 2
            
        Returns:
            Change detection mask
        """
        # Encode both images using the shared encoder
        features1 = self.encoder.encode(x1)
        features2 = self.encoder.encode(x2)
        
        # Extract the bottleneck features
        f1_bottleneck = features1[-1]
        f2_bottleneck = features2[-1]
        
        # Fusion at the bottleneck
        fused = torch.cat([f1_bottleneck, f2_bottleneck], dim=1)
        fused = self.fusion(fused)
        
        # Decoder path with skip connections from both encoders
        x = self.up1(fused, torch.abs(features1[-2] - features2[-2]))
        x = self.up2(x, torch.abs(features1[-3] - features2[-3]))
        x = self.up3(x, torch.abs(features1[-4] - features2[-4]))
        x = self.up4(x, torch.abs(features1[-5] - features2[-5]))
        
        # Output
        logits = self.outc(x)
        
        return logits


class SiameseUNetDifference(nn.Module):
    """Siamese U-Net with early fusion by difference"""
    
    def __init__(self, in_channels, out_channels=1, features=64, bilinear=True, dropout=0.0):
        """
        Initialize the Siamese U-Net Difference model.
        
        Args:
            in_channels: Number of input channels per image
            out_channels: Number of output channels
            features: Number of features in the first layer
            bilinear: Whether to use bilinear interpolation or transposed convolutions
            dropout: Dropout probability
        """
        super(SiameseUNetDifference, self).__init__()
        
        # Use a standard UNet with double the input channels
        self.unet = UNet(in_channels * 2, out_channels, features, bilinear, dropout)
        
    def forward(self, x1, x2):
        """
        Forward pass through the model.
        
        Args:
            x1: Input image at time 1
            x2: Input image at time 2
            
        Returns:
            Change detection mask
        """
        # Concatenate the two images along the channel dimension
        x = torch.cat([x1, x2], dim=1)
        
        # Pass through the U-Net
        return self.unet(x)


class SiameseFCNDifference(nn.Module):
    """Simple FCN-based Siamese network with difference operation"""
    
    def __init__(self, in_channels, out_channels=1, features=64):
        """
        Initialize the Siamese FCN Difference model.
        
        Args:
            in_channels: Number of input channels per image
            out_channels: Number of output channels
            features: Number of features in the network
        """
        super(SiameseFCNDifference, self).__init__()
        
        # Define the shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(features, features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(features * 2, features * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(inplace=True)
        )
        
        # Define the decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(features * 4, features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(features * 2, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(features, out_channels, kernel_size=1)
        )
    
    def forward(self, x1, x2):
        """
        Forward pass through the Siamese FCN.
        
        Args:
            x1: Input image at time 1
            x2: Input image at time 2
            
        Returns:
            Change detection mask
        """
        # Encode both images
        features1 = self.encoder(x1)
        features2 = self.encoder(x2)
        
        # Take the absolute difference
        diff = torch.abs(features1 - features2)
        
        # Decode the difference
        out = self.decoder(diff)
        
        return out


def get_change_detection_model(model_type='siamese_unet', in_channels=3, out_channels=1, 
                              features=64, bilinear=True, dropout=0.2):
    """
    Factory function to create a change detection model.
    
    Args:
        model_type: Type of model to create (siamese_unet, siamese_diff, fcn_diff)
        in_channels: Number of input channels per image
        out_channels: Number of output channels
        features: Number of features in the first layer
        bilinear: Whether to use bilinear interpolation or transposed convolutions
        dropout: Dropout probability
        
    Returns:
        Initialized model
    """
    if model_type == 'siamese_unet':
        return SiameseUNet(in_channels, out_channels, features, bilinear, dropout)
    elif model_type == 'siamese_diff':
        return SiameseUNetDifference(in_channels, out_channels, features, bilinear, dropout)
    elif model_type == 'fcn_diff':
        return SiameseFCNDifference(in_channels, out_channels, features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create some random input data
    batch_size = 2
    channels = 4  # Sentinel-2 has 4 bands (RGB + NIR)
    height, width = 256, 256
    
    x1 = torch.randn(batch_size, channels, height, width).to(device)
    x2 = torch.randn(batch_size, channels, height, width).to(device)
    
    # Test each model
    for model_type in ['siamese_unet', 'siamese_diff', 'fcn_diff']:
        print(f"Testing {model_type}...")
        
        model = get_change_detection_model(model_type, in_channels=channels).to(device)
        output = model(x1, x2)
        
        print(f"Input shapes: {x1.shape}, {x2.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        print("-" * 50)