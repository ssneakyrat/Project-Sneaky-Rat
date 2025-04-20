#!/usr/bin/env python3
"""
model.py - Student HiFi-GAN model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import StudentConfig
from .layers import DepthwiseSeparableConv2d, ResBlock2d, SimplifiedMRF2d


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
        self.upsample_rates = config.upsample_rates
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
        
        # Upsampling layers
        self.upsamples = nn.ModuleList()
        in_channels = self.upsample_initial_channel
        for i, (u, k) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes)):
            # Calculate output channels for this stage
            out_channels = in_channels // 2
            
            # Calculate appropriate padding
            padding = ((k[0]-1)//2, (k[1]-1)//2)
            
            # Create upsampling block: upsample + conv
            self.upsamples.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=u, mode='nearest'),
                    nn.Conv2d(in_channels, out_channels, k, 1, padding=padding)
                )
            )
            in_channels = out_channels
        
        # MRF blocks
        self.mrfs = nn.ModuleList()
        for i in range(len(self.upsample_rates)):
            channels = in_channels
            self.mrfs.append(
                SimplifiedMRF2d(
                    channels,
                    self.resblock_kernel_sizes,
                    self.resblock_dilation_sizes,
                    self.use_depthwise_separable,
                    self.activation_function
                )
            )
            
        # Output convolution
        self.conv_post = nn.Conv2d(in_channels, 1, (7, 7), 1, padding=(3, 3))
        
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
        for i, (up, mrf) in enumerate(zip(self.upsamples, self.mrfs)):
            x = up(x)
            x = mrf(x)
        
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
            'conv_post': {
                'type': 'output',
                'shape': self.conv_post.weight.shape
            }
        }
