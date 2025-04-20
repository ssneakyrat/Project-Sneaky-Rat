#!/usr/bin/env python3
"""
model.py - Student HiFi-GAN model implementation with fixed channel handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import StudentConfig
from .layers import DepthwiseSeparableConv2d, ResBlock2d, SimplifiedMRF2d


class AdaptiveChannelLayer(nn.Module):
    """
    Adapts channel dimensions between layers to ensure compatibility.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.needs_adaptation = in_channels != out_channels
        
        if self.needs_adaptation:
            self.adaptation = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        if not self.needs_adaptation:
            return x
            
        # Check input channels and adapt if needed
        if x.size(1) != self.in_channels:
            print(f"AdaptiveChannelLayer: Input has {x.size(1)} channels, expected {self.in_channels}")
            if x.size(1) > self.in_channels:
                # Truncate channels
                x = x[:, :self.in_channels]
            else:
                # Pad with zeros
                padding = torch.zeros(x.size(0), self.in_channels - x.size(1), 
                                     x.size(2), x.size(3), device=x.device)
                x = torch.cat([x, padding], dim=1)
                
        return self.adaptation(x)


class HiFiGANStudent2D(nn.Module):
    """
    Student HiFi-GAN model with 2D convolutions.
    Efficiently generates audio waveforms from mel-spectrograms.
    """
    
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = StudentConfig()
        
        self.config = config
        self.input_channels = getattr(config, 'input_channels', 1)
        
        # Handle the case where upsample_rates is a single integer (0)
        self.upsample_rates = config.upsample_rates
        if isinstance(self.upsample_rates, int):
            # Convert to list with default values if it's 0
            if self.upsample_rates == 0:
                self.upsample_rates = [8, 8, 4, 4]
            else:
                self.upsample_rates = [self.upsample_rates] * len(config.upsample_kernel_sizes)
        
        self.upsample_kernel_sizes = config.upsample_kernel_sizes
        self.upsample_initial_channel = config.upsample_initial_channel
        self.resblock_kernel_sizes = config.resblock_kernel_sizes
        self.resblock_dilation_sizes = config.resblock_dilation_sizes
        self.use_depthwise_separable = getattr(config, 'use_depthwise_separable', True)
        self.activation_function = getattr(config, 'activation_function', 'LeakyRelu')
        
        # Select activation function
        if self.activation_function == 'LeakyRelu':
            self.activation = nn.LeakyReLU(getattr(config, 'leaky_relu_alpha', 0.1))
        elif self.activation_function == 'Relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.LeakyReLU(0.1)  # Default
            
        # Initial convolution
        self.conv_pre = nn.Conv2d(self.input_channels, self.upsample_initial_channel, (7, 7), 1, padding=(3, 3))
        
        # Upsampling layers with channel adaptations
        self.upsamples = nn.ModuleList()
        self.channel_adaptations = nn.ModuleList()
        self.post_mrf_adaptations = nn.ModuleList()  # NEW: Add post-MRF adaptations
        
        # Define channel progressions explicitly
        # Start with initial channel count
        upsample_channels = [self.upsample_initial_channel]
        
        # Calculate each subsequent layer's channel count (halve each time)
        for i in range(len(self.upsample_rates)):
            upsample_channels.append(upsample_channels[-1] // 2)
            
        # MRF block channel counts (fixed to 16 for all blocks)
        mrf_channels = 16
        
        # Now create the layers with proper channel handling
        for i, (u, k) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes)):
            in_channels = upsample_channels[i]
            out_channels = upsample_channels[i+1]
            
            # Calculate appropriate padding
            padding = ((k[0]-1)//2, (k[1]-1)//2)
            
            # Create upsampling block: upsample + conv
            self.upsamples.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=u, mode='nearest'),
                    nn.Conv2d(in_channels, out_channels, k, 1, padding=padding)
                )
            )
            
            # Add channel adaptation layer between upsample output and MRF block
            self.channel_adaptations.append(
                AdaptiveChannelLayer(out_channels, mrf_channels)
            )
            
            # NEW: Add post-MRF adaptation to prepare for next upsampling layer
            # This adapts from the MRF output back to the expected channel count
            if i < len(self.upsample_rates) - 1:  # Don't need an adapter after the last MRF
                next_in_channels = upsample_channels[i+1]  # Channel count expected by next upsampling
                self.post_mrf_adaptations.append(
                    AdaptiveChannelLayer(mrf_channels, next_in_channels)
                )
            else:
                # For the last layer, still create an adapter but it's identity (for consistent indexing)
                self.post_mrf_adaptations.append(
                    AdaptiveChannelLayer(mrf_channels, mrf_channels)
                )
        
        # MRF blocks (all use the same mrf_channels value)
        self.mrfs = nn.ModuleList()
        for i in range(len(self.upsample_rates)):
            self.mrfs.append(
                SimplifiedMRF2d(
                    mrf_channels,
                    self.resblock_kernel_sizes,
                    self.resblock_dilation_sizes,
                    self.use_depthwise_separable,
                    self.activation_function
                )
            )
            
        # Add final channel adaptation before output conv
        self.final_adaptation = AdaptiveChannelLayer(mrf_channels, mrf_channels)
            
        # Output convolution
        self.conv_post = nn.Conv2d(mrf_channels, 1, (7, 7), 1, padding=(3, 3))
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        # Custom weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass of the student HiFi-GAN model.
        
        Args:
            x: Input mel-spectrogram [batch_size, channels, height, width]
            
        Returns:
            Generated audio waveform [batch_size, 1, height_out, width_out]
        """
        # Initial convolution
        x = self.conv_pre(x)
        
        # Upsampling stages with MRF blocks
        for i, (up, adapt, mrf, post_adapt) in enumerate(zip(
            self.upsamples, self.channel_adaptations, self.mrfs, self.post_mrf_adaptations
        )):
            if i > 0:
                # From the second iteration onward, we need to use the post-adaptation
                # from the previous MRF to prepare for this upsampling
                x = post_adapt(x)
                
            # Apply upsampling
            x = up(x)
            
            # Adapt channels for MRF block
            x = adapt(x)
            
            # Apply MRF block
            x = mrf(x)
        
        # Final channel adaptation
        x = self.final_adaptation(x)
        
        # Final convolution and activation
        x = self.activation(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x
        
    @classmethod
    def from_config(cls, config_path):
        """Create a model instance from a configuration file."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.StudentConfig()
        return cls(config)
        
    def get_mapping_info(self):
        """Return information for weight mapping from teacher model."""
        return {
            'conv_pre': {
                'type': 'input', 
                'shape': self.conv_pre.weight.shape
            },
            'upsamples': [
                {
                    'type': 'upsample',
                    'index': i,
                    'shape': up[1].weight.shape  # The conv after upsampling
                } for i, up in enumerate(self.upsamples)
            ],
            'channel_adaptations': [
                {
                    'type': 'adaptation',
                    'index': i,
                    'shape': adapt.adaptation.weight.shape if hasattr(adapt, 'adaptation') else None
                } for i, adapt in enumerate(self.channel_adaptations)
            ],
            'post_mrf_adaptations': [  # Add mapping info for new adaptations
                {
                    'type': 'post_mrf_adaptation',
                    'index': i,
                    'shape': adapt.adaptation.weight.shape if hasattr(adapt, 'adaptation') else None
                } for i, adapt in enumerate(self.post_mrf_adaptations)
            ],
            'mrfs': [
                {
                    'type': 'mrf',
                    'index': i,
                    'resblocks': [
                        {
                            'kernel_size': k,
                            'convs': [
                                {
                                    'dilation': d,
                                    'shape': resblock.convs[j].weight.shape if isinstance(resblock.convs[j], nn.Conv2d)
                                             else (resblock.convs[j].depthwise.weight.shape, resblock.convs[j].pointwise.weight.shape)
                                } for j, d in enumerate(self.resblock_dilation_sizes[k_idx])
                            ]
                        } for k_idx, (k, resblock) in enumerate(zip(self.resblock_kernel_sizes, mrf.resblocks))
                    ]
                } for i, mrf in enumerate(self.mrfs)
            ],
            'final_adaptation': {
                'type': 'adaptation',
                'shape': self.final_adaptation.adaptation.weight.shape if hasattr(self.final_adaptation, 'adaptation') else None
            },
            'conv_post': {
                'type': 'output',
                'shape': self.conv_post.weight.shape
            }
        }