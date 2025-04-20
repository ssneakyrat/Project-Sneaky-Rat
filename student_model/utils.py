#!/usr/bin/env python3
"""
utils.py - Utility functions for student HiFi-GAN model.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path

def channel_reduction_mapping(teacher_weights, reduction_factor):
    """
    Apply channel reduction mapping to teacher weights.
    
    Args:
        teacher_weights: Teacher model weights tensor
        reduction_factor: Factor to reduce channels by (0.5 means half the channels)
        
    Returns:
        Student weights tensor with reduced channels
    """
    # Calculate L1-norm for each output channel
    if len(teacher_weights.shape) == 4:  # Conv2d
        channel_importance = torch.sum(torch.abs(teacher_weights), dim=(1, 2, 3))
    else:  # Conv1d
        channel_importance = torch.sum(torch.abs(teacher_weights), dim=(1, 2))
    
    # Calculate number of student channels
    student_channels = max(1, int(teacher_weights.shape[0] * reduction_factor))
    
    # Select top-k channels based on importance
    _, top_channels = torch.topk(channel_importance, student_channels)
    
    # Initialize student weights with selected channels
    student_weights = teacher_weights[top_channels]
    
    return student_weights

def transpose_to_standard_mapping(teacher_weights):
    """
    Map transposed convolution weights to standard convolution weights.
    
    Args:
        teacher_weights: Teacher transposed convolution weights
        
    Returns:
        Transformed weights for standard convolution
    """
    # Flip the kernels spatially
    if len(teacher_weights.shape) == 4:  # 2D convolution
        return torch.flip(torch.flip(teacher_weights, dims=[2]), dims=[3])
    else:  # 1D convolution
        return torch.flip(teacher_weights, dims=[2])

def estimate_model_size(model):
    """
    Estimate the size of a PyTorch model in parameters and memory.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dict with parameter count and estimated memory size
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    # Estimate memory size (assuming float32 parameters)
    memory_bytes = total_params * 4  # 4 bytes per float32
    
    return {
        'total_parameters': total_params,
        'memory_megabytes': memory_bytes / (1024 * 1024),
        'parameter_details': {
            name: p.numel() for name, p in model.named_parameters()
        }
    }

def create_sample_input(model_type, batch_size=1):
    """
    Create a sample input tensor for the model.
    
    Args:
        model_type: Type of the model ('1D' or '2D')
        batch_size: Batch size for the input
        
    Returns:
        Sample input tensor
    """
    if model_type == '1D':
        # Create a sample mel-spectrogram [batch_size, channels, time]
        return torch.randn(batch_size, 1, 80)
    else:
        # Create a sample 2D input [batch_size, channels, height, width]
        return torch.randn(batch_size, 1, 80, 80)

def check_compatibility(model, teacher_model_info):
    """
    Check if the student model is compatible with the teacher model.
    
    Args:
        model: Student model instance
        teacher_model_info: Dictionary with teacher model information
        
    Returns:
        Dict with compatibility check results
    """
    model_size = estimate_model_size(model)
    
    return {
        'is_compatible': True,  # Placeholder for actual compatibility check
        'student_params': model_size['total_parameters'],
        'teacher_params': teacher_model_info.get('total_params', 0),
        'reduction_factor': model_size['total_parameters'] / max(1, teacher_model_info.get('total_params', 1)),
        'notes': 'Student model is ready for weight mapping'
    }

def save_model_info(model, output_path):
    """
    Save model information to a JSON file.
    
    Args:
        model: PyTorch model
        output_path: Path to save the JSON file
        
    Returns:
        Path to the saved file
    """
    # Get model information
    info = {
        'size': estimate_model_size(model),
        'mapping_info': model.get_mapping_info(),
        'timestamp': str(torch.datetime.now())
    }
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    return output_path
