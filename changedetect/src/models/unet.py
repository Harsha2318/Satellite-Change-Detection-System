"""
U-Net architecture for image segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block for U-Net"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.0):
        """
        Initialize the double convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            mid_channels: Number of channels after the first convolution
            dropout: Dropout probability
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        
    def forward(self, x):
        """Forward pass"""
        x = self.double_conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class DownBlock(nn.Module):
    """Downsampling block for U-Net"""
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        """
        Initialize the downsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dropout: Dropout probability
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout=dropout)
        )
    
    def forward(self, x):
        """Forward pass"""
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upsampling block for U-Net"""
    
    def __init__(self, in_channels, out_channels, bilinear=True, dropout=0.0):
        """
        Initialize the upsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bilinear: Whether to use bilinear interpolation or transposed convolutions
            dropout: Dropout probability
        """
        super().__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            )
            self.conv = ConvBlock(in_channels, out_channels, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels, dropout=dropout)
    
    def forward(self, x1, x2):
        """
        Forward pass with skip connections
        
        Args:
            x1: Features from the encoder path
            x2: Features from the decoder path
            
        Returns:
            Output tensor
        """
        x1 = self.up(x1)
        
        # Adjust the sizes
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                         diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution block"""
    
    def __init__(self, in_channels, out_channels):
        """
        Initialize the output convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        """Forward pass"""
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture for image segmentation"""
    
    def __init__(self, in_channels, out_channels=1, features=64, bilinear=True, dropout=0.0):
        """
        Initialize the U-Net model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            features: Number of features in the first layer (will be doubled in each down step)
            bilinear: Whether to use bilinear interpolation or transposed convolutions for upsampling
            dropout: Dropout probability
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.bilinear = bilinear
        
        # Encoder path
        self.inc = ConvBlock(in_channels, features, dropout=dropout)
        self.down1 = DownBlock(features, features * 2, dropout=dropout)
        self.down2 = DownBlock(features * 2, features * 4, dropout=dropout)
        self.down3 = DownBlock(features * 4, features * 8, dropout=dropout)
        
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(features * 8, features * 16 // factor, dropout=dropout)
        
        # Decoder path
        self.up1 = UpBlock(features * 16, features * 8 // factor, bilinear, dropout=dropout)
        self.up2 = UpBlock(features * 8, features * 4 // factor, bilinear, dropout=dropout)
        self.up3 = UpBlock(features * 4, features * 2 // factor, bilinear, dropout=dropout)
        self.up4 = UpBlock(features * 2, features, bilinear, dropout=dropout)
        
        # Output layer
        self.outc = OutConv(features, out_channels)
    
    def forward(self, x):
        """
        Forward pass through the U-Net.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        
        return logits
    
    def encode(self, x):
        """
        Encode input through the encoder path.
        
        Args:
            x: Input tensor
            
        Returns:
            List of feature maps
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        return [x1, x2, x3, x4, x5]