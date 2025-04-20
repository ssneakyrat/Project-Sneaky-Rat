#!/usr/bin/env python3
"""
layers.py - Custom layers for the student HiFi-GAN model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable 2D convolution with adaptive channel handling."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        # Handle channel dimension mismatch
        if x.size(1) != self.in_channels:
            print(f"WARNING: Channel mismatch in DepthwiseSeparableConv2d - input has {x.size(1)} channels, expected {self.in_channels}")
            if x.size(1) > self.in_channels:
                # Truncate channels
                x = x[:, :self.in_channels]
            else:
                # Pad with zeros
                padding = torch.zeros(x.size(0), self.in_channels - x.size(1), x.size(2), x.size(3), device=x.device)
                x = torch.cat([x, padding], dim=1)
        
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ResBlock2d(nn.Module):
    """Residual block with multiple dilated 2D convolutions."""
    
    def __init__(self, channels, kernel_size, dilations, use_depthwise=False, activation='LeakyRelu'):
        super().__init__()
        self.convs = nn.ModuleList()
        self.use_depthwise = use_depthwise
        
        # Select activation function
        if activation == 'LeakyRelu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'Relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.LeakyReLU(0.1)  # Default
        
        for d in dilations:
            # Calculate padding for 'same' output - FIXED VERSION
            if isinstance(kernel_size, (list, tuple)) and isinstance(d, (list, tuple)):
                # Both kernel_size and d are tuples/lists
                padding = ((kernel_size[0] * d[0] - d[0]) // 2, 
                          (kernel_size[1] * d[1] - d[1]) // 2)
            elif isinstance(kernel_size, (list, tuple)):
                # kernel_size is a tuple/list but d is a scalar
                padding = ((kernel_size[0] * d - d) // 2, 
                          (kernel_size[1] * d - d) // 2)
            elif isinstance(d, (list, tuple)):
                # kernel_size is a scalar but d is a tuple/list
                padding = ((kernel_size * d[0] - d[0]) // 2, 
                          (kernel_size * d[1] - d[1]) // 2)
            else:
                # Both are scalars
                padding = (kernel_size * d - d) // 2
            
            if use_depthwise:
                self.convs.append(
                    DepthwiseSeparableConv2d(
                        channels, channels, kernel_size,
                        dilation=d, padding=padding
                    )
                )
            else:
                self.convs.append(
                    nn.Conv2d(
                        channels, channels, kernel_size,
                        dilation=d, padding=padding
                    )
                )
    
    def forward(self, x):
        for conv in self.convs:
            residual = x
            x = self.activation(x)
            x = conv(x)
            
            # Handle channel dimension mismatch between residual and processed output
            if x.size(1) != residual.size(1):
                print(f"Handling channel mismatch in ResBlock2d: output has {x.size(1)} channels, residual has {residual.size(1)}")
                if x.size(1) < residual.size(1):
                    # Truncate channels from residual to match x
                    residual = residual[:, :x.size(1)]
                else:
                    # Pad residual with zeros to match x
                    padding = torch.zeros(
                        residual.size(0), x.size(1) - residual.size(1),
                        residual.size(2), residual.size(3), device=residual.device
                    )
                    residual = torch.cat([residual, padding], dim=1)
            
            x = x + residual
        return x


class SimplifiedMRF2d(nn.Module):
    """Simplified Multi-Receptive Field Fusion module with 2D convolutions."""
    
    def __init__(self, channels, kernel_sizes, dilations, use_depthwise=False, activation='LeakyRelu'):
        super().__init__()
        self.resblocks = nn.ModuleList()
        
        for k, d in zip(kernel_sizes, dilations):
            self.resblocks.append(ResBlock2d(channels, k, d, use_depthwise, activation))
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        for resblock in self.resblocks:
            for conv in resblock.convs:
                if isinstance(conv, nn.Conv2d):
                    nn.init.kaiming_normal_(conv.weight)
                    if conv.bias is not None:
                        nn.init.zeros_(conv.bias)
                elif hasattr(conv, 'depthwise'):
                    nn.init.kaiming_normal_(conv.depthwise.weight)
                    if conv.depthwise.bias is not None:
                        nn.init.zeros_(conv.depthwise.bias)
                    nn.init.kaiming_normal_(conv.pointwise.weight)
                    if conv.pointwise.bias is not None:
                        nn.init.zeros_(conv.pointwise.bias)
    
    def forward(self, x):
        outputs = []
        for resblock in self.resblocks:
            outputs.append(resblock(x))
        
        # Average the outputs from different receptive fields
        return torch.stack(outputs).mean(dim=0)