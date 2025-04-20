#!/usr/bin/env python3
"""
weight_mapping.py - Implement weight mapping from teacher to student HiFi-GAN model.
"""

import argparse
import onnx
import numpy as np
import torch
import os
import json
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
import warnings

# Suppress the non-writable NumPy array warning
warnings.filterwarnings("ignore", category=UserWarning, message="The given NumPy array is not writable")

# Import student model
from student_model import HiFiGANStudent2D, StudentConfig
from student_model.utils import (
    channel_reduction_mapping, 
    transpose_to_standard_mapping,
    estimate_model_size,
    create_sample_input,
    save_model_info
)

def improved_weight_mapping(teacher_weights, student_model):
    """
    Improved weight mapping strategy that uses pattern matching and structural similarity
    to map weights from teacher to student model.
    
    Args:
        teacher_weights: Dictionary of teacher weights
        student_model: Instance of student model
        
    Returns:
        Dictionary of mapped weights keyed by student layer names
    """
    print("Starting improved weight mapping process...")
    
    # Print debug information about the teacher weights
    print("\nDEBUG: Teacher Model Weights Analysis")
    print("=" * 50)
    print(f"Total teacher weights: {len(teacher_weights)}")
    
    # Group weights by prefix to understand structure
    prefix_groups = {}
    for name in teacher_weights.keys():
        parts = name.split('.')
        if len(parts) >= 2:
            prefix = parts[0] + '.' + parts[1]
            if prefix not in prefix_groups:
                prefix_groups[prefix] = []
            prefix_groups[prefix].append(name)
    
    print("Found the following weight groups in teacher model:")
    for prefix, weights in prefix_groups.items():
        print(f"  - {prefix}: {len(weights)} weights")
    print("=" * 50)
    
    # Dictionary to store mapped weights
    mapped_weights = {}
    
    # Map initial convolution (conv_pre)
    print("\nMapping initial convolution layer...")
    initial_conv_candidates = []
    
    # Look for potential initial convolution weights
    for name, tensor in teacher_weights.items():
        # Initial convolutions are typically larger and at the beginning of the model
        if ('weight' in name and 
            len(tensor.shape) == 4 and  # 2D convolution
            tensor.shape[2] >= 5 and    # Kernel size at least 5x5
            'conv' in name.lower() and  # Contains 'conv' in name
            ('initial' in name.lower() or 'pre' in name.lower() or 'first' in name.lower() or
             any(s in name.lower() for s in ['in_', 'input', 'begin', 'start']))):
            initial_conv_candidates.append((name, tensor))
    
    # If no specific candidates found, look for any initial layers
    if not initial_conv_candidates:
        for name, tensor in teacher_weights.items():
            if ('weight' in name and 
                len(tensor.shape) == 4 and  # 2D convolution
                ('generator.conv' in name or  # Common pattern in HiFi-GAN
                 'generator.m_source' in name or
                 'generator.initial' in name)):
                initial_conv_candidates.append((name, tensor))
    
    # If still no candidates, look for any conv with appropriate shape
    if not initial_conv_candidates:
        for name, tensor in teacher_weights.items():
            if ('weight' in name and 
                len(tensor.shape) == 4 and  # 2D convolution
                tensor.shape[1] <= 2):      # Input channels typically small
                initial_conv_candidates.append((name, tensor))
    
    if initial_conv_candidates:
        # Sort candidates by shape similarity to student conv_pre
        student_shape = student_model.conv_pre.weight.shape
        initial_conv_candidates.sort(key=lambda x: shape_similarity_score(x[1].shape, student_shape))
        
        # Use the best candidate
        best_name, best_tensor = initial_conv_candidates[0]
        print(f"Selected '{best_name}' as initial convolution weight")
        
        # Reshape if needed
        if best_tensor.shape != student_shape:
            if len(best_tensor.shape) == len(student_shape):
                reshaped = adaptive_reshape(best_tensor, student_shape)
                mapped_weights['conv_pre.weight'] = reshaped
            else:
                print(f"Warning: Cannot reshape from {best_tensor.shape} to {student_shape}")
                mapped_weights['conv_pre.weight'] = student_model.conv_pre.weight.clone()
        else:
            mapped_weights['conv_pre.weight'] = best_tensor
    else:
        print("No suitable initial convolution weights found. Using random initialization.")
        mapped_weights['conv_pre.weight'] = student_model.conv_pre.weight.clone()
    
    # Map upsampling layers
    print("\nMapping upsampling layers...")
    
    # Look for potential upsampling convolution weights
    upsampling_candidates = []
    
    # HiFi-GAN typically uses a series of transposed convolutions for upsampling
    for name, tensor in teacher_weights.items():
        if ('weight' in name and 
            len(tensor.shape) == 4 and
            ('transpose' in name.lower() or 
             'up' in name.lower() or 
             'upsample' in name.lower() or
             # Common naming patterns in HiFi-GAN models
             'generator.ups' in name or
             'generator.resblocks' in name)):
            upsampling_candidates.append((name, tensor))
    
    # Group upsampling candidates by similarity
    upsampling_groups = []
    for name, tensor in upsampling_candidates:
        # Check if this tensor belongs to an existing group
        found_group = False
        for group in upsampling_groups:
            ref_shape = teacher_weights[group[0][0]].shape
            if tensor.shape[0] == ref_shape[0] or tensor.shape[1] == ref_shape[1]:
                group.append((name, tensor))
                found_group = True
                break
        
        if not found_group:
            # Create a new group
            upsampling_groups.append([(name, tensor)])
    
    # Sort groups by likely order in model
    upsampling_groups.sort(key=lambda group: (
        # Larger input channel dimension usually means earlier in the network
        -teacher_weights[group[0][0]].shape[1],
        group[0][0]  # Use name as secondary sort key
    ))
    
    # Map upsampling layers
    for i, up_layer in enumerate(student_model.upsamples):
        if i < len(upsampling_groups) and upsampling_groups[i]:
            # Find best matching weight in this group
            student_shape = up_layer[1].weight.shape
            candidates = upsampling_groups[i]
            candidates.sort(key=lambda x: shape_similarity_score(x[1].shape, student_shape))
            
            best_name, best_tensor = candidates[0]
            print(f"Selected '{best_name}' for upsampling layer {i}")
            
            # Reshape if needed
            if best_tensor.shape != student_shape:
                reshaped = adaptive_reshape(best_tensor, student_shape)
                mapped_weights[f"upsamples.{i}.1.weight"] = reshaped
            else:
                mapped_weights[f"upsamples.{i}.1.weight"] = best_tensor
        else:
            print(f"No suitable weights found for upsampling layer {i}. Using random initialization.")
            mapped_weights[f"upsamples.{i}.1.weight"] = up_layer[1].weight.clone()
    
    # Map MRF blocks
    print("\nMapping MRF blocks...")
    
    # Identify potential resblock weights (usually the largest group of weights)
    resblock_weights = []
    for name, tensor in teacher_weights.items():
        if ('weight' in name and 
            len(tensor.shape) == 4 and
            ('resblock' in name.lower() or 
             'res_block' in name.lower() or
             'conv_block' in name.lower() or
             'generator.resblocks' in name or
             'dilation' in name.lower())):
            resblock_weights.append((name, tensor))
    
    # Group by block pattern (usually blocks have same number of layers with different dilation)
    block_patterns = {}
    for name, tensor in resblock_weights:
        # Extract pattern from name (e.g., "generator.resblocks.0.1.2")
        parts = name.split('.')
        if len(parts) >= 3:
            pattern_key = '.'.join(parts[:-2])  # Group by module name without specific layer
            if pattern_key not in block_patterns:
                block_patterns[pattern_key] = []
            block_patterns[pattern_key].append((name, tensor))
    
    # Sort block patterns by size (larger blocks usually more important)
    sorted_patterns = sorted(block_patterns.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Map to student MRF blocks
    for i, mrf in enumerate(student_model.mrfs):
        print(f"Mapping MRF block {i}...")
        
        # Map each resblock in the MRF
        for k, resblock in enumerate(mrf.resblocks):
            # Try to find matching weights for this resblock
            matched_weights = False
            
            # Search for weights with matching shape
            for pattern_key, pattern_weights in sorted_patterns:
                if matched_weights:
                    break
                    
                # Check if this pattern has weights for all convolutions in the resblock
                if len(pattern_weights) >= len(resblock.convs):
                    # Group by likely layer position in the resblock
                    layer_groups = {}
                    for weight_name, weight_tensor in pattern_weights:
                        parts = weight_name.split('.')
                        if len(parts) >= 4:
                            # Use the last numeric part as layer identifier
                            layer_id = parts[-2]
                            if layer_id not in layer_groups:
                                layer_groups[layer_id] = []
                            layer_groups[layer_id].append((weight_name, weight_tensor))
                    
                    # Sort layer groups by name
                    sorted_layers = sorted(layer_groups.items())
                    
                    # Try to map each conv in the resblock
                    if len(sorted_layers) >= len(resblock.convs):
                        matched_weights = True
                        
                        for j, conv in enumerate(resblock.convs):
                            if j < len(sorted_layers):
                                _, layer_weights = sorted_layers[j]
                                
                                # Sort weights by name similarity to find weight and bias
                                layer_weights.sort(key=lambda x: x[0])
                                
                                if layer_weights:
                                    weight_name, weight_tensor = layer_weights[0]
                                    print(f"  Using '{weight_name}' for MRF {i}, resblock {k}, conv {j}")
                                    
                                    # Handle depthwise separable convolution
                                    if hasattr(conv, 'depthwise'):
                                        # Split weight for depthwise and pointwise
                                        depthwise_shape = conv.depthwise.weight.shape
                                        pointwise_shape = conv.pointwise.weight.shape
                                        
                                        # Create depthwise weight
                                        depthwise = adaptive_reshape(weight_tensor, depthwise_shape)
                                        mapped_weights[f"mrfs.{i}.resblocks.{k}.convs.{j}.depthwise.weight"] = depthwise
                                        
                                        # Create pointwise weight (identity mapping if possible)
                                        in_channels = depthwise_shape[0]
                                        pointwise = torch.zeros(pointwise_shape)
                                        for c in range(min(pointwise_shape[0], in_channels)):
                                            pointwise[c, c % in_channels] = 1.0
                                        
                                        mapped_weights[f"mrfs.{i}.resblocks.{k}.convs.{j}.pointwise.weight"] = pointwise
                                    else:
                                        # Standard convolution
                                        student_shape = conv.weight.shape
                                        reshaped = adaptive_reshape(weight_tensor, student_shape)
                                        mapped_weights[f"mrfs.{i}.resblocks.{k}.convs.{j}.weight"] = reshaped
                            else:
                                print(f"  Not enough layer weights for MRF {i}, resblock {k}, conv {j}")
                                matched_weights = False
            
            # If no weights were matched for this resblock
            if not matched_weights:
                print(f"  No matching weights found for MRF {i}, resblock {k}")
                for j, conv in enumerate(resblock.convs):
                    if hasattr(conv, 'depthwise'):
                        mapped_weights[f"mrfs.{i}.resblocks.{k}.convs.{j}.depthwise.weight"] = conv.depthwise.weight.clone()
                        mapped_weights[f"mrfs.{i}.resblocks.{k}.convs.{j}.pointwise.weight"] = conv.pointwise.weight.clone()
                    else:
                        mapped_weights[f"mrfs.{i}.resblocks.{k}.convs.{j}.weight"] = conv.weight.clone()
    
    # Map output convolution
    print("\nMapping output convolution layer...")
    output_candidates = []
    
    # Look for potential output convolutions
    for name, tensor in teacher_weights.items():
        if ('weight' in name and 
            len(tensor.shape) == 4 and
            tensor.shape[0] == 1 and  # Output typically has 1 channel
            ('output' in name.lower() or 
             'out_' in name.lower() or 
             'final' in name.lower() or
             'post' in name.lower() or
             'last' in name.lower())):
            output_candidates.append((name, tensor))
    
    # If no specific candidates found, look for layer that outputs 1 channel
    if not output_candidates:
        for name, tensor in teacher_weights.items():
            if ('weight' in name and 
                len(tensor.shape) == 4 and
                tensor.shape[0] == 1):  # Output channel is 1
                output_candidates.append((name, tensor))
    
    if output_candidates:
        # Sort candidates by shape similarity
        student_shape = student_model.conv_post.weight.shape
        output_candidates.sort(key=lambda x: shape_similarity_score(x[1].shape, student_shape))
        
        best_name, best_tensor = output_candidates[0]
        print(f"Selected '{best_name}' as output convolution weight")
        
        # Reshape if needed
        if best_tensor.shape != student_shape:
            reshaped = adaptive_reshape(best_tensor, student_shape)
            mapped_weights['conv_post.weight'] = reshaped
        else:
            mapped_weights['conv_post.weight'] = best_tensor
    else:
        print("No suitable output convolution weights found. Using random initialization.")
        mapped_weights['conv_post.weight'] = student_model.conv_post.weight.clone()
    
    print(f"\nCompleted mapping process with {len(mapped_weights)} mapped layers")
    return mapped_weights

def shape_similarity_score(shape1, shape2):
    """
    Calculate a similarity score between two tensor shapes.
    Lower score means more similar.
    
    Args:
        shape1: First tensor shape
        shape2: Second tensor shape
        
    Returns:
        Similarity score (lower is more similar)
    """
    if len(shape1) != len(shape2):
        return 1000  # Very different
    
    # Calculate normalized difference for each dimension
    score = 0
    for s1, s2 in zip(shape1, shape2):
        # Calculate relative difference
        if s1 == 0 and s2 == 0:
            diff = 0
        elif s1 == 0 or s2 == 0:
            diff = 1
        else:
            diff = abs(s1 - s2) / max(s1, s2)
        score += diff
    
    return score

def improved_initial_mapping(teacher_weights, student_model):
    """Improved mapping for the initial convolution layer specifically."""
    print("\nMapping initial convolution layer with specialized approach...")
    student_shape = student_model.conv_pre.weight.shape
    
    # First check for the most likely candidates by name
    candidates = []
    
    # Known common initial convolution names in HiFi-GAN
    likely_names = [
        'generator.conv_pre.weight',
        'generator.input_conv.weight',
        'generator.first_conv.weight'
    ]
    
    for name in likely_names:
        if name in teacher_weights:
            candidates.append((name, teacher_weights[name]))
            print(f"Found exact match: {name}")
    
    # If no exact matches, try pattern matching
    if not candidates:
        for name, tensor in teacher_weights.items():
            # Check common patterns in HiFi-GAN models
            if (('weight' in name) and 
                ('conv_pre' in name or 'input_conv' in name or 'first_conv' in name)):
                candidates.append((name, tensor))
                print(f"Found pattern match: {name}")
    
    # If still no candidates, look for any Conv2D with size info that makes sense for input
    if not candidates:
        for name, tensor in teacher_weights.items():
            if ('weight' in name and len(tensor.shape) == 4):
                # Initial convolution typically has small input channels (1-3)
                if tensor.shape[1] <= 3:
                    candidates.append((name, tensor))
                    print(f"Found potential initial conv by shape: {name} {tensor.shape}")
    
    # Check for 'generator.conv_pre.weight' specifically (common in HiFi-GAN)
    if 'generator.conv_pre.weight' in teacher_weights:
        print("Using 'generator.conv_pre.weight' as this is the standard initial conv in HiFi-GAN")
        conv_weight = teacher_weights['generator.conv_pre.weight']
        return robust_reshape(conv_weight, student_shape)
    
    # If we have candidates, choose the best one
    if candidates:
        # Prioritize candidates with kernel size >= 3
        valid_candidates = [c for c in candidates if (
            c[1].shape[2] >= 3 and c[1].shape[3] >= 3)]
        
        if valid_candidates:
            candidates = valid_candidates
        
        # Choose the one with closest shape to target
        name, tensor = candidates[0]
        print(f"Selected '{name}' for initial convolution with shape {tensor.shape}")
        return robust_reshape(tensor, student_shape)
    
    # Fallback - if no suitable convolutions found, initialize randomly
    print(f"No suitable initial convolution found, initializing randomly")
    import torch.nn as nn
    init_tensor = torch.zeros(student_shape)
    nn.init.kaiming_normal_(init_tensor)
    return init_tensor

def robust_reshape(tensor, target_shape):
    """
    Robustly reshape a tensor to the target shape, even for challenging cases.
    """
    print(f"Reshaping tensor from {tensor.shape} to {target_shape}")
    
    # Special case for the problematic initial tensor
    if tensor.shape == torch.Size([1, 1, 2, 1]) and target_shape == torch.Size([256, 1, 7, 7]):
        print("Using special handling for [1,1,2,1] -> [256,1,7,7] case")
        # Create new tensor with target shape
        result = torch.zeros(target_shape)
        
        # Fill with values derived from the source but properly scaled
        # Copy the values in a pattern that expands to fill the target shape
        for i in range(256):
            for h in range(7):
                for w in range(7):
                    # Use modulo to cycle through the source values
                    src_h = h % tensor.shape[2]
                    src_w = w % tensor.shape[3]
                    value = tensor[0, 0, src_h, src_w].item()
                    
                    # Scale the value based on position to create variation
                    scale = 0.5 + (i % 5) * 0.1  # Some variation by output channel
                    result[i, 0, h, w] = value * scale
        
        # Normalize to have similar statistics as the source
        src_mean = tensor.mean().item()
        src_std = tensor.std().item() or 0.01  # Fallback if std is 0
        
        result_mean = result.mean().item()
        result_std = result.std().item() or 0.01
        
        # Adjust to match statistics
        result = (result - result_mean) / result_std * src_std + src_mean
        
        return result
    
    # For all other cases, use the more general adaptive_reshape
    return adaptive_reshape(tensor, target_shape)

def adaptive_reshape(tensor, target_shape):
    """
    Adaptively reshape a tensor to match the target shape,
    handling different dimensionality and sizes.
    
    Args:
        tensor: Source tensor
        target_shape: Target shape
        
    Returns:
        Reshaped tensor
    """
    import torch
    import torch.nn.functional as F
    
    # Check if shapes already match
    if tensor.shape == target_shape:
        return tensor
    
    # Create new tensor with target shape
    reshaped = torch.zeros(target_shape, dtype=tensor.dtype)
    
    # Handle different dimensionality cases
    if len(tensor.shape) != len(target_shape):
        print(f"Warning: Dimension mismatch between tensor {tensor.shape} and target {target_shape}")
        
        # Case: Convert from 3D to 4D (or similar dimension changes)
        if len(tensor.shape) == 3 and len(target_shape) == 4:
            # Try to transform the 3D tensor to 4D
            t_expanded = tensor.unsqueeze(2)  # Convert [N,C,L] -> [N,C,1,L]
            
            # Check if we can reinterpret the dimensions
            if t_expanded.shape[0] == target_shape[0]:  # Output channels match
                # Initialize result with zeros
                out_c, in_c, kh, kw = target_shape
                
                # Copy available weights
                min_in_c = min(t_expanded.shape[1], in_c)
                min_kh = min(t_expanded.shape[2], kh)
                min_kw = min(t_expanded.shape[3], kw)
                
                # Copy what we can
                reshaped[:, :min_in_c, :min_kh, :min_kw] = t_expanded[:, :min_in_c, :min_kh, :min_kw]
                
                return reshaped
        
        # If we reach here, we'll create a random initialization with proper statistics
        print(f"Using randomized initialization for {target_shape} with statistics from source tensor")
        mean = tensor.mean().item()
        std = tensor.std().item() or 0.01  # Fallback to small std if 0
        
        import torch.nn.init as init
        init.normal_(reshaped, mean=mean, std=std)
        return reshaped
    
    # Handle spatial dimensions (last two for 4D tensors)
    if len(tensor.shape) == 4 and len(target_shape) == 4:
        # Get channel dimensions
        out_c, in_c, kh, kw = target_shape
        src_out_c, src_in_c, src_kh, src_kw = tensor.shape
        
        # Try spatial interpolation if the kernel shapes are valid for interpolation
        if src_kh > 1 and src_kw > 1 and kh > 1 and kw > 1:
            try:
                temp = F.interpolate(
                    tensor.unsqueeze(0),
                    size=(kh, kw),
                    mode='bilinear'
                ).squeeze(0)
                
                # Handle channel dimensions
                min_out_c = min(src_out_c, out_c)
                min_in_c = min(src_in_c, in_c)
                
                # Copy data with channel adaption
                reshaped[:min_out_c, :min_in_c] = temp[:min_out_c, :min_in_c]
            except (ValueError, RuntimeError) as e:
                print(f"Interpolation failed: {e}")
                # Fallback to simple copy of available values
                min_out_c = min(src_out_c, out_c)
                min_in_c = min(src_in_c, in_c)
                min_kh = min(src_kh, kh)
                min_kw = min(src_kw, kw)
                
                reshaped[:min_out_c, :min_in_c, :min_kh, :min_kw] = tensor[:min_out_c, :min_in_c, :min_kh, :min_kw]
        else:
            # Source or target has singleton dimensions, use direct copy
            min_out_c = min(src_out_c, out_c)
            min_in_c = min(src_in_c, in_c)
            min_kh = min(src_kh, kh)
            min_kw = min(src_kw, kw)
            
            reshaped[:min_out_c, :min_in_c, :min_kh, :min_kw] = tensor[:min_out_c, :min_in_c, :min_kh, :min_kw]
    
    return reshaped

def extract_weights_from_onnx(onnx_path):
    """
    Extract weights from ONNX model into a dictionary.
    
    Args:
        onnx_path: Path to the ONNX model file
        
    Returns:
        Dictionary mapping layer name to weight tensors
    """
    print(f"Loading ONNX model from {onnx_path}...")
    model = onnx.load(onnx_path)
    graph = model.graph
    
    # Create a dictionary to store weights
    weights = {}
    
    # Extract weights from initializers
    for initializer in graph.initializer:
        # Convert initializer to numpy array
        np_array = onnx.numpy_helper.to_array(initializer)
        # Convert to PyTorch tensor
        tensor = torch.from_numpy(np_array)
        # Store in dictionary with name
        weights[initializer.name] = tensor
        
    print(f"Extracted {len(weights)} weight tensors from ONNX model")
    return weights

def map_weights_to_student(teacher_weights, student_model, teacher_arch_json):
    """
    Map weights from teacher to student model based on mapping strategy.
    
    Args:
        teacher_weights: Dictionary of teacher weights
        student_model: Instance of student model
        teacher_arch_json: Path to teacher architecture JSON file
        
    Returns:
        Dictionary of mapped weights keyed by student layer names
    """
    print("Starting weight mapping process...")
    
    # Load teacher architecture information
    with open(teacher_arch_json, 'r') as f:
        teacher_info = json.load(f)
    
    # Print debug information about the teacher weights
    print("\nDEBUG: Weight Mapping Process")
    print("=" * 50)
    print(f"Total teacher weights: {len(teacher_weights)}")
    print("Sample of available teacher weight names:")
    sample_names = list(teacher_weights.keys())[:10]
    for name in sample_names:
        print(f"  - {name}: {teacher_weights[name].shape}")
    print("=" * 50)
    
    # Get student model mapping info
    mapping_info = student_model.get_mapping_info()
    
    # Dictionary to store mapped weights
    mapped_weights = {}
    
    # Map initial layers (conv_pre)
    print("Mapping initial convolution layer...")
    initial_mapping = teacher_info['student_design']['layer_mapping']['initial_layers']
    teacher_prefix = initial_mapping['teacher_prefix']
    reduction_factor = initial_mapping['reduction_factor']
    
    print(f"Looking for initial weights with prefix: '{teacher_prefix}'")
    matching_weights = [name for name in teacher_weights.keys() if teacher_prefix in name and "weight" in name]
    print(f"Found {len(matching_weights)} matching weights for initial layer")
    
    # Find the corresponding teacher weights for initial conv
    found_initial = False
    for name, tensor in teacher_weights.items():
        if teacher_prefix in name and "weight" in name:
            print(f"Using weight '{name}' for initial convolution")
            # Apply channel reduction
            reduced_weights = channel_reduction_mapping(tensor, reduction_factor)
            # Check if shapes match or need reshaping
            student_shape = student_model.conv_pre.weight.shape
            if reduced_weights.shape != student_shape:
                # Reshape or resize as needed
                if len(reduced_weights.shape) == len(student_shape):
                    # Simple reshaping needed
                    print(f"Reshaping initial weights from {reduced_weights.shape} to {student_shape}")
                    reduced_weights = F.interpolate(
                        reduced_weights.unsqueeze(0), 
                        size=student_shape[2:],
                        mode='bilinear'
                    ).squeeze(0)
                    # Make sure to match the output channels
                    reduced_weights = reduced_weights[:student_shape[0], :student_shape[1]]
            
            mapped_weights['conv_pre.weight'] = reduced_weights
            found_initial = True
            break
    
    if not found_initial:
        print(f"WARNING: No weights found for initial convolution with prefix '{teacher_prefix}'")
        print(f"Using random initialization for initial convolution")
        # Use random initialization for the initial convolution
        mapped_weights['conv_pre.weight'] = student_model.conv_pre.weight.clone()
    
    # Map upsampling layers
    print("Mapping upsampling layers...")
    for i, upsampling_map in enumerate(teacher_info['student_design']['layer_mapping']['upsampling_layers']):
        # Skip if i is out of range for student model's upsamples
        if i >= len(student_model.upsamples):
            print(f"Warning: Upsampling layer index {i} is out of range. Student model has {len(student_model.upsamples)} upsampling layers.")
            continue
            
        teacher_prefix = upsampling_map['teacher_prefix']
        reduction_factor = upsampling_map['reduction_factor']
        mapping_type = upsampling_map.get('mapping_type', 'direct')
        
        print(f"Looking for upsampling layer {i} weights with prefix: '{teacher_prefix}'")
        matching_weights = [name for name in teacher_weights.keys() if teacher_prefix in name and "weight" in name]
        print(f"Found {len(matching_weights)} matching weights for upsampling layer {i}")
        
        # Find the corresponding teacher weights
        found_upsampling = False
        for name, tensor in teacher_weights.items():
            if teacher_prefix in name and "weight" in name:
                print(f"Using weight '{name}' for upsampling layer {i}")
                # Convert transposed conv to standard
                if mapping_type == 'transpose_to_standard':
                    # First transform the weights
                    transformed = transpose_to_standard_mapping(tensor)
                    # Then apply channel reduction
                    reduced = channel_reduction_mapping(transformed, reduction_factor)
                else:
                    reduced = channel_reduction_mapping(tensor, reduction_factor)
                
                # Check if we need to resize/reshape
                student_layer_name = f"upsamples.{i}.1.weight"  # The conv after upsampling
                student_shape = student_model.upsamples[i][1].weight.shape
                
                if reduced.shape != student_shape:
                    print(f"Reshaping upsampling weights from {reduced.shape} to {student_shape}")
                    if len(reduced.shape) == len(student_shape):
                        reduced = F.interpolate(
                            reduced.unsqueeze(0),
                            size=student_shape[2:],
                            mode='bilinear'
                        ).squeeze(0)
                        # Make sure to match the channel dimensions
                        reduced = reduced[:student_shape[0], :student_shape[1]]
                
                mapped_weights[student_layer_name] = reduced
                found_upsampling = True
                break
        
        if not found_upsampling:
            print(f"WARNING: No weights found for upsampling layer {i} with prefix '{teacher_prefix}'")
            print(f"Using existing initialization for upsampling layer {i}")
            # Use existing initialization for this upsampling layer
            mapped_weights[f"upsamples.{i}.1.weight"] = student_model.upsamples[i][1].weight.clone()
    
    # Map MRF layers
    print("Mapping MRF layers...")
    mrf_mappings = teacher_info['student_design']['layer_mapping'].get('mrfs', [])
    student_mrf_count = len(student_model.mrfs)
    
    print(f"Found {len(mrf_mappings)} MRF mappings in teacher info, student model has {student_mrf_count} MRF blocks")
    
    for i, mrf_map in enumerate(mrf_mappings):
        # Skip if i is out of range for student model's MRFs
        if i >= student_mrf_count:
            print(f"Warning: MRF index {i} is out of range. Student model has {student_mrf_count} MRF blocks.")
            continue
            
        teacher_prefix = mrf_map['teacher_prefix']
        reduction_factor = mrf_map['reduction_factor']
        
        print(f"Processing MRF block {i} with prefix: '{teacher_prefix}'")
        matching_weights = [name for name in teacher_weights.keys() if teacher_prefix in name and "weight" in name]
        print(f"Found {len(matching_weights)} matching weights for MRF block {i}")
        
        # MRF blocks have multiple resblocks, each with multiple convolutions
        for k_idx, resblock in enumerate(student_model.mrfs[i].resblocks):
            for j, conv in enumerate(resblock.convs):
                # Handle depthwise separable convolution
                if hasattr(conv, 'depthwise'):
                    # Find teacher weights for this kernel size/dilation
                    found_mrf_conv = False
                    for name, tensor in teacher_weights.items():
                        # Try different pattern variations
                        patterns = [
                            f"{teacher_prefix}_kernel_{k_idx+1}_dilation_{j+1}",
                            f"{teacher_prefix}.kernel_{k_idx+1}.dilation_{j+1}",
                            f"{teacher_prefix}/kernel_{k_idx+1}/dilation_{j+1}"
                        ]
                        
                        if any(pattern in name for pattern in patterns) and "weight" in name:
                            print(f"Using weight '{name}' for MRF {i}, resblock {k_idx}, conv {j}")
                            # Apply reduction strategy
                            reduced = channel_reduction_mapping(tensor, reduction_factor)
                            
                            # For depthwise separable, we need to map to both depthwise and pointwise
                            # Reshape for depthwise conv
                            depthwise_shape = conv.depthwise.weight.shape
                            pointwise_shape = conv.pointwise.weight.shape
                            
                            if reduced.shape != depthwise_shape and len(reduced.shape) == len(depthwise_shape):
                                # Reshape to fit depthwise
                                depthwise = F.interpolate(
                                    reduced.unsqueeze(0),
                                    size=depthwise_shape[2:],
                                    mode='bilinear'
                                ).squeeze(0)
                                
                                # Take subset of channels for depthwise
                                depthwise = depthwise[:depthwise_shape[0], :depthwise_shape[1]]
                                
                                # Create pointwise from reduced channels
                                pointwise = torch.zeros(pointwise_shape)
                                in_channels = depthwise_shape[0]
                                for c in range(min(pointwise_shape[0], in_channels)):
                                    pointwise[c, c % in_channels] = 1.0
                                
                                # Store in mapped weights
                                mapped_weights[f"mrfs.{i}.resblocks.{k_idx}.convs.{j}.depthwise.weight"] = depthwise
                                mapped_weights[f"mrfs.{i}.resblocks.{k_idx}.convs.{j}.pointwise.weight"] = pointwise
                                found_mrf_conv = True
                                break
                    
                    if not found_mrf_conv:
                        print(f"WARNING: No weights found for MRF {i}, resblock {k_idx}, conv {j}")
                        print(f"Using existing initialization")
                        # Use existing initialization for this conv
                        mapped_weights[f"mrfs.{i}.resblocks.{k_idx}.convs.{j}.depthwise.weight"] = conv.depthwise.weight.clone()
                        mapped_weights[f"mrfs.{i}.resblocks.{k_idx}.convs.{j}.pointwise.weight"] = conv.pointwise.weight.clone()
                        
                else:
                    # Standard convolution
                    found_mrf_conv = False
                    for name, tensor in teacher_weights.items():
                        # Try different pattern variations
                        patterns = [
                            f"{teacher_prefix}_kernel_{k_idx+1}_dilation_{j+1}",
                            f"{teacher_prefix}.kernel_{k_idx+1}.dilation_{j+1}",
                            f"{teacher_prefix}/kernel_{k_idx+1}/dilation_{j+1}"
                        ]
                        
                        if any(pattern in name for pattern in patterns) and "weight" in name:
                            print(f"Using weight '{name}' for MRF {i}, resblock {k_idx}, conv {j}")
                            # Apply reduction
                            reduced = channel_reduction_mapping(tensor, reduction_factor)
                            
                            # Reshape if needed
                            student_shape = conv.weight.shape
                            if reduced.shape != student_shape and len(reduced.shape) == len(student_shape):
                                reduced = F.interpolate(
                                    reduced.unsqueeze(0),
                                    size=student_shape[2:],
                                    mode='bilinear'
                                ).squeeze(0)
                                reduced = reduced[:student_shape[0], :student_shape[1]]
                            
                            mapped_weights[f"mrfs.{i}.resblocks.{k_idx}.convs.{j}.weight"] = reduced
                            found_mrf_conv = True
                            break
                    
                    if not found_mrf_conv:
                        print(f"WARNING: No weights found for MRF {i}, resblock {k_idx}, conv {j}")
                        print(f"Using existing initialization")
                        # Use existing initialization for this conv
                        mapped_weights[f"mrfs.{i}.resblocks.{k_idx}.convs.{j}.weight"] = conv.weight.clone()
    
    # Map output layer (conv_post)
    print("Mapping output convolution layer...")
    output_mapping = teacher_info['student_design']['layer_mapping']['output_layer']
    teacher_prefix = output_mapping['teacher_prefix']
    reduction_factor = output_mapping['reduction_factor']
    
    print(f"Looking for output weights with prefix: '{teacher_prefix}'")
    matching_weights = [name for name in teacher_weights.keys() if teacher_prefix in name and "weight" in name]
    print(f"Found {len(matching_weights)} matching weights for output layer")
    
    found_output = False
    for name, tensor in teacher_weights.items():
        if teacher_prefix in name and "weight" in name:
            print(f"Using weight '{name}' for output convolution")
            # Direct mapping with reduction
            reduced = channel_reduction_mapping(tensor, reduction_factor)
            
            # Check shape match
            student_shape = student_model.conv_post.weight.shape
            if reduced.shape != student_shape and len(reduced.shape) == len(student_shape):
                reduced = F.interpolate(
                    reduced.unsqueeze(0),
                    size=student_shape[2:],
                    mode='bilinear'
                ).squeeze(0)
                reduced = reduced[:student_shape[0], :student_shape[1]]
            
            mapped_weights['conv_post.weight'] = reduced
            found_output = True
            break
    
    if not found_output:
        print(f"WARNING: No weights found for output convolution with prefix '{teacher_prefix}'")
        print(f"Using existing initialization for output convolution")
        # Use existing initialization for the output convolution
        mapped_weights['conv_post.weight'] = student_model.conv_post.weight.clone()
    
    print(f"Completed mapping process with {len(mapped_weights)} mapped layers")
    
    if len(mapped_weights) == 0:
        print("WARNING: No weights were mapped from teacher to student model!")
        print("Initializing weights with default initialization")
        # Initialize all weights with default parameters
        for name, param in student_model.named_parameters():
            if 'weight' in name:
                mapped_weights[name] = param.clone()
    
    return mapped_weights

def apply_mapped_weights(student_model, mapped_weights):
    """
    Apply the mapped weights to the student model with improved channel handling.
    
    Args:
        student_model: Instance of student model
        mapped_weights: Dictionary of mapped weights
        
    Returns:
        Updated student model with mapped weights
    """
    print("Applying mapped weights to student model with improved channel handling...")
    
    # Get state dict of the student model
    state_dict = student_model.state_dict()
    
    # Count how many weights were successfully mapped
    mapped_count = 0
    
    # Explicitly handle channel adaptation layers first
    if hasattr(student_model, 'channel_adaptations'):
        print("Initializing channel adaptation layers...")
        for i, adapt_layer in enumerate(student_model.channel_adaptations):
            if hasattr(adapt_layer, 'adaptation'):
                # Initialize with identity-like mapping for channel adaptation
                weight_name = f"channel_adaptations.{i}.adaptation.weight"
                if weight_name in state_dict:
                    # Create identity-like mapping for channel adaptation
                    weight = torch.zeros_like(state_dict[weight_name])
                    in_channels = adapt_layer.in_channels
                    out_channels = adapt_layer.out_channels
                    
                    # Identity mapping for shared channels, zero for others
                    for c in range(min(in_channels, out_channels)):
                        # Set a diagonal-like pattern
                        weight[c, c % in_channels] = 1.0
                    
                    state_dict[weight_name].copy_(weight)
                    print(f"  Initialized {weight_name} with identity-like mapping")
    
    # Handle final adaptation layer if it exists
    if hasattr(student_model, 'final_adaptation') and hasattr(student_model.final_adaptation, 'adaptation'):
        weight_name = "final_adaptation.adaptation.weight"
        if weight_name in state_dict:
            # Create identity-like mapping
            weight = torch.zeros_like(state_dict[weight_name])
            in_channels = student_model.final_adaptation.in_channels
            out_channels = student_model.final_adaptation.out_channels
            
            # Identity mapping for shared channels
            for c in range(min(in_channels, out_channels)):
                weight[c, c % in_channels] = 1.0
            
            state_dict[weight_name].copy_(weight)
            print(f"  Initialized {weight_name} with identity-like mapping")
    
    # Update weights that were mapped
    for name, tensor in mapped_weights.items():
        if name in state_dict:
            # Check if shapes match
            if state_dict[name].shape == tensor.shape:
                state_dict[name].copy_(tensor)
                mapped_count += 1
            else:
                print(f"Shape mismatch for {name}: expected {state_dict[name].shape}, got {tensor.shape}")
                # Try to reshape if possible
                if len(state_dict[name].shape) == len(tensor.shape):
                    try:
                        if len(tensor.shape) == 4:  # For 2D convolutions
                            reshaped = F.interpolate(
                                tensor.unsqueeze(0),
                                size=state_dict[name].shape[2:],
                                mode='bilinear'
                            ).squeeze(0)
                            
                            # Handle channel dimensions
                            if reshaped.shape[0] > state_dict[name].shape[0]:
                                reshaped = reshaped[:state_dict[name].shape[0]]
                            if reshaped.shape[1] > state_dict[name].shape[1]:
                                reshaped = reshaped[:, :state_dict[name].shape[1]]
                            
                            # Add additional padding if necessary
                            if reshaped.shape[0] < state_dict[name].shape[0]:
                                padding = torch.zeros(
                                    state_dict[name].shape[0] - reshaped.shape[0],
                                    reshaped.shape[1],
                                    reshaped.shape[2],
                                    reshaped.shape[3],
                                    device=reshaped.device
                                )
                                reshaped = torch.cat([reshaped, padding], dim=0)
                            
                            if reshaped.shape[1] < state_dict[name].shape[1]:
                                padding = torch.zeros(
                                    reshaped.shape[0],
                                    state_dict[name].shape[1] - reshaped.shape[1],
                                    reshaped.shape[2],
                                    reshaped.shape[3],
                                    device=reshaped.device
                                )
                                reshaped = torch.cat([reshaped, padding], dim=1)
                            
                            # Check if reshaping worked
                            if reshaped.shape == state_dict[name].shape:
                                state_dict[name].copy_(reshaped)
                                mapped_count += 1
                                print(f"Successfully reshaped weight for {name}")
                            else:
                                print(f"Reshaping failed for {name}, keeping original weights")
                        else:
                            # For non-4D tensors, just copy what we can
                            print(f"Cannot reshape non-4D tensor for {name}, using partial copy")
                    except Exception as e:
                        print(f"Error reshaping weight for {name}: {e}")
        else:
            print(f"Warning: {name} not found in student model")
    
    print(f"Successfully applied {mapped_count}/{len(mapped_weights)} mapped weights")
    
    # Initialize any remaining uninitialized parameters
    for name, param in state_dict.items():
        if 'weight' in name and torch.all(param == 0):
            print(f"Initializing zero parameter: {name}")
            if len(param.shape) == 4:  # Conv weight
                nn.init.kaiming_normal_(param)
            elif len(param.shape) == 1:  # Bias
                nn.init.zeros_(param)
            else:
                # Use xavier for other types of weights
                nn.init.xavier_normal_(param)
    
    # Load the updated state dict back into the model
    student_model.load_state_dict(state_dict)
    return student_model
def test_inference(model, dummy_input=None):
    """
    Test inference with the model to ensure it works, with improved debugging.
    
    Args:
        model: PyTorch model to test
        dummy_input: Optional dummy input tensor
        
    Returns:
        Output of model inference
    """
    print("Testing inference with dummy data...")
    
    # Create dummy input if not provided
    if dummy_input is None:
        dummy_input = create_sample_input('2D', batch_size=1)
    
    # Set model to evaluation mode
    model.eval()
    
    # Enable tensor shape tracking
    def hook_fn(module, input, output):
        print(f"Module: {module.__class__.__name__}")
        print(f"  Input shape: {[x.shape if isinstance(x, torch.Tensor) else type(x) for x in input]}")
        print(f"  Output shape: {output.shape}")
        return output
    
    # Register hooks for key modules
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Sequential)) or "adapt" in name.lower() or "mrf" in name.lower():
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    try:
        # Perform inference
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Inference successful! Output shape: {output.shape}")
        return output
    except Exception as e:
        print(f"Inference failed with error: {e}")
        raise
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
def main(onnx_path, student_arch_json, output_path):
    """
    Main function to map weights and test the student model.
    
    Args:
        onnx_path: Path to the teacher ONNX model
        student_arch_json: Path to the student architecture JSON file
        output_path: Path to save the student model
    """
    # Extract weights from teacher model
    teacher_weights = extract_weights_from_onnx(onnx_path)
    
    # Initialize student model
    print("Initializing student model...")
    student_model = HiFiGANStudent2D()
    
    # Use improved weight mapping strategy
    print("\n" + "="*50)
    print("USING IMPROVED WEIGHT MAPPING STRATEGY")
    print("="*50)
    
    # Create a dictionary to store mapped weights
    mapped_weights = {}
    
    # Print debug information about the teacher weights
    print("\nDEBUG: Teacher Model Weights Analysis")
    print("=" * 50)
    print(f"Total teacher weights: {len(teacher_weights)}")
    
    # Group weights by prefix to understand structure
    prefix_groups = {}
    for name in teacher_weights.keys():
        parts = name.split('.')
        if len(parts) >= 2:
            prefix = parts[0] + '.' + parts[1]
            if prefix not in prefix_groups:
                prefix_groups[prefix] = []
            prefix_groups[prefix].append(name)
    
    print("Found the following weight groups in teacher model:")
    for prefix, weights in prefix_groups.items():
        print(f"  - {prefix}: {len(weights)} weights")
    print("=" * 50)
    
    # Map initial convolution using specialized approach
    mapped_weights['conv_pre.weight'] = improved_initial_mapping(teacher_weights, student_model)
    
    # Map upsampling layers
    print("\nMapping upsampling layers...")
    # Get specific upsampling weights from the HiFi-GAN model
    ups_weights = [(name, tensor) for name, tensor in teacher_weights.items() 
                  if 'generator.ups' in name and 'weight' in name]
    
    # Sort by the numerical part of the name (e.g., generator.ups.0.weight)
    ups_weights.sort(key=lambda x: int(x[0].split('.')[2]) if x[0].split('.')[2].isdigit() else 999)
    
    # Map to student upsampling layers
    for i, up_layer in enumerate(student_model.upsamples):
        if i < len(ups_weights):
            name, tensor = ups_weights[i]
            print(f"Mapping upsampling layer {i} from {name} {tensor.shape}")
            
            # Get the convolution weight after upsampling
            student_shape = up_layer[1].weight.shape
            mapped_weights[f"upsamples.{i}.1.weight"] = adaptive_reshape(tensor, student_shape)
        else:
            print(f"No matching weight found for upsampling layer {i}, using random initialization")
            mapped_weights[f"upsamples.{i}.1.weight"] = up_layer[1].weight.clone()
    
    # Map MRF blocks
    print("\nMapping MRF blocks...")
    # Get resblock weights from the HiFi-GAN model
    resblock_weights = [(name, tensor) for name, tensor in teacher_weights.items() 
                       if 'generator.resblocks' in name and 'weight' in name]
    
    # Group by resblock number
    resblock_groups = {}
    for name, tensor in resblock_weights:
        parts = name.split('.')
        if len(parts) >= 3 and parts[2].isdigit():
            group_key = int(parts[2])
            if group_key not in resblock_groups:
                resblock_groups[group_key] = []
            resblock_groups[group_key].append((name, tensor))
    
    # Sort groups by resblock number
    sorted_groups = sorted(resblock_groups.items())
    
    # Map to student MRF blocks
    for i, mrf in enumerate(student_model.mrfs):
        print(f"Mapping MRF block {i}...")
        if i < len(sorted_groups):
            _, group_weights = sorted_groups[i]
            
            # Map each resblock in the MRF
            for k, resblock in enumerate(mrf.resblocks):
                # Calculate how many teacher weights to use per student resblock
                weights_per_resblock = len(group_weights) // len(mrf.resblocks)
                start_idx = k * weights_per_resblock
                end_idx = start_idx + weights_per_resblock
                
                # Get weights for this resblock
                rb_weights = group_weights[start_idx:end_idx]
                
                # Map convolutions
                for j, conv in enumerate(resblock.convs):
                    if j < len(rb_weights):
                        name, tensor = rb_weights[j]
                        print(f"  Mapping MRF {i}, resblock {k}, conv {j} from {name} {tensor.shape}")
                        
                        # Handle depthwise separable convolution
                        if hasattr(conv, 'depthwise'):
                            # Split the weight for depthwise and pointwise
                            depthwise_shape = conv.depthwise.weight.shape
                            pointwise_shape = conv.pointwise.weight.shape
                            
                            # Map depthwise
                            mapped_weights[f"mrfs.{i}.resblocks.{k}.convs.{j}.depthwise.weight"] = adaptive_reshape(tensor, depthwise_shape)
                            
                            # Create identity-like pointwise weight
                            in_channels = depthwise_shape[0]
                            pointwise = torch.zeros(pointwise_shape)
                            for c in range(min(pointwise_shape[0], in_channels)):
                                pointwise[c, c % in_channels] = 1.0
                            
                            mapped_weights[f"mrfs.{i}.resblocks.{k}.convs.{j}.pointwise.weight"] = pointwise
                        else:
                            # Standard convolution
                            student_shape = conv.weight.shape
                            mapped_weights[f"mrfs.{i}.resblocks.{k}.convs.{j}.weight"] = adaptive_reshape(tensor, student_shape)
                    else:
                        print(f"  No matching weight for MRF {i}, resblock {k}, conv {j}, using random initialization")
                        if hasattr(conv, 'depthwise'):
                            mapped_weights[f"mrfs.{i}.resblocks.{k}.convs.{j}.depthwise.weight"] = conv.depthwise.weight.clone()
                            mapped_weights[f"mrfs.{i}.resblocks.{k}.convs.{j}.pointwise.weight"] = conv.pointwise.weight.clone()
                        else:
                            mapped_weights[f"mrfs.{i}.resblocks.{k}.convs.{j}.weight"] = conv.weight.clone()
        else:
            print(f"No matching weights for MRF block {i}, using random initialization")
            for k, resblock in enumerate(mrf.resblocks):
                for j, conv in enumerate(resblock.convs):
                    if hasattr(conv, 'depthwise'):
                        mapped_weights[f"mrfs.{i}.resblocks.{k}.convs.{j}.depthwise.weight"] = conv.depthwise.weight.clone()
                        mapped_weights[f"mrfs.{i}.resblocks.{k}.convs.{j}.pointwise.weight"] = conv.pointwise.weight.clone()
                    else:
                        mapped_weights[f"mrfs.{i}.resblocks.{k}.convs.{j}.weight"] = conv.weight.clone()
    
    # Map output convolution
    print("\nMapping output convolution layer...")
    if 'generator.conv_post.weight' in teacher_weights:
        print("Using 'generator.conv_post.weight' for output convolution")
        output_shape = student_model.conv_post.weight.shape
        mapped_weights['conv_post.weight'] = adaptive_reshape(teacher_weights['generator.conv_post.weight'], output_shape)
    else:
        print("No suitable output convolution weight found, using random initialization")
        mapped_weights['conv_post.weight'] = student_model.conv_post.weight.clone()
    
    print(f"\nCompleted mapping process with {len(mapped_weights)} mapped layers")
    
    # Apply mapped weights to student model
    student_model = apply_mapped_weights(student_model, mapped_weights)
    
    # Test inference
    dummy_input = create_sample_input('2D', batch_size=1)
    test_inference(student_model, dummy_input)
    
    # Save the mapped student model
    print(f"Saving mapped student model to {output_path}...")
    torch.save(student_model.state_dict(), output_path)
    
    # Save model info
    info_path = os.path.splitext(output_path)[0] + "_info.json"
    save_model_info(student_model, info_path)
    
    # Print model size estimate
    size_info = estimate_model_size(student_model)
    print(f"Student model size: {size_info['total_parameters']:,} parameters")
    print(f"Estimated memory: {size_info['memory_megabytes']:.2f} MB")
    
    print("\nWeight mapping and testing completed successfully!")
    return student_model, size_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map weights from teacher to student HiFi-GAN model")
    parser.add_argument("--onnx_path", type=str, default="nsf_hifigan.onnx", help="Path to teacher ONNX model")
    parser.add_argument("--arch_json", type=str, default="student_architecture.json", help="Path to student architecture JSON")
    parser.add_argument("--output_path", type=str, default="student_model.pt", help="Path to save mapped student model")
    
    args = parser.parse_args()
    
    main(args.onnx_path, args.arch_json, args.output_path)