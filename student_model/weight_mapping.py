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
    
    # Get student model mapping info
    mapping_info = student_model.get_mapping_info()
    
    # Dictionary to store mapped weights
    mapped_weights = {}
    
    # Map initial layers (conv_pre)
    print("Mapping initial convolution layer...")
    initial_mapping = teacher_info['student_design']['layer_mapping']['initial_layers']
    teacher_prefix = initial_mapping['teacher_prefix']
    reduction_factor = initial_mapping['reduction_factor']
    
    # Find the corresponding teacher weights for initial conv
    for name, tensor in teacher_weights.items():
        if teacher_prefix in name and "weight" in name:
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
            
            mapped_weights['conv_pre.weight'] = reduced_weights
            break
    
    # Map upsampling layers
    print("Mapping upsampling layers...")
    for i, upsampling_map in enumerate(teacher_info['student_design']['layer_mapping']['upsampling_layers']):
        teacher_prefix = upsampling_map['teacher_prefix']
        reduction_factor = upsampling_map['reduction_factor']
        mapping_type = upsampling_map['mapping_type']
        
        # Find the corresponding teacher weights
        for name, tensor in teacher_weights.items():
            if teacher_prefix in name and "weight" in name:
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
                
                mapped_weights[student_layer_name] = reduced
                break
    
    # Map MRF layers
    print("Mapping MRF layers...")
    for i, mrf_map in enumerate(teacher_info['student_design']['layer_mapping']['mrfs']):
        teacher_prefix = mrf_map['teacher_prefix']
        reduction_factor = mrf_map['reduction_factor']
        
        # MRF blocks have multiple resblocks, each with multiple convolutions
        for k_idx, resblock in enumerate(student_model.mrfs[i].resblocks):
            for j, conv in enumerate(resblock.convs):
                # Handle depthwise separable convolution
                if hasattr(conv, 'depthwise'):
                    # Find teacher weights for this kernel size/dilation
                    for name, tensor in teacher_weights.items():
                        if teacher_prefix in name and f"kernel_{k_idx+1}" in name and f"dilation_{j+1}" in name and "weight" in name:
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
                                break
                else:
                    # Standard convolution
                    for name, tensor in teacher_weights.items():
                        if teacher_prefix in name and f"kernel_{k_idx+1}" in name and f"dilation_{j+1}" in name and "weight" in name:
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
                            break
    
    # Map output layer (conv_post)
    print("Mapping output convolution layer...")
    output_mapping = teacher_info['student_design']['layer_mapping']['output_layer']
    teacher_prefix = output_mapping['teacher_prefix']
    reduction_factor = output_mapping['reduction_factor']
    
    for name, tensor in teacher_weights.items():
        if teacher_prefix in name and "weight" in name:
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
            break
    
    print(f"Completed mapping process with {len(mapped_weights)} mapped layers")
    return mapped_weights

def apply_mapped_weights(student_model, mapped_weights):
    """
    Apply the mapped weights to the student model.
    
    Args:
        student_model: Instance of student model
        mapped_weights: Dictionary of mapped weights
        
    Returns:
        Updated student model with mapped weights
    """
    print("Applying mapped weights to student model...")
    
    # Get state dict of the student model
    state_dict = student_model.state_dict()
    
    # Count how many weights were successfully mapped
    mapped_count = 0
    
    # Update weights that were mapped
    for name, tensor in mapped_weights.items():
        if name in state_dict:
            # Check if shapes match
            if state_dict[name].shape == tensor.shape:
                state_dict[name].copy_(tensor)
                mapped_count += 1
            else:
                print(f"Shape mismatch for {name}: expected {state_dict[name].shape}, got {tensor.shape}")
        else:
            print(f"Warning: {name} not found in student model")
    
    print(f"Successfully applied {mapped_count}/{len(mapped_weights)} mapped weights")
    
    # Load the updated state dict back into the model
    student_model.load_state_dict(state_dict)
    return student_model

def test_inference(model, dummy_input=None):
    """
    Test inference with the model to ensure it works.
    
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
    
    # Perform inference
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Inference successful! Output shape: {output.shape}")
    return output

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
    
    # Map weights from teacher to student
    mapped_weights = map_weights_to_student(teacher_weights, student_model, student_arch_json)
    
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