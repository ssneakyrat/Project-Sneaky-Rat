#!/usr/bin/env python3
"""
test_student_model.py - Test the student model with dummy data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse

from student_model import HiFiGANStudent2D
from student_model.utils import create_sample_input, estimate_model_size

def load_student_model(model_path):
    """
    Load a student model from a saved checkpoint.
    
    Args:
        model_path: Path to the saved model checkpoint
        
    Returns:
        Loaded student model
    """
    print(f"Loading student model from {model_path}...")
    
    # Initialize model
    model = HiFiGANStudent2D()
    
    # Load state dict
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    # Set to eval mode
    model.eval()
    
    return model

def generate_dummy_spectrogram(height=80, width=80):
    """
    Generate a dummy mel-spectrogram for testing.
    
    Args:
        height: Height of the spectrogram
        width: Width of the spectrogram
        
    Returns:
        Dummy spectrogram tensor
    """
    # Create a dummy spectrogram with some patterns
    x = np.linspace(0, 4*np.pi, width)
    y = np.linspace(0, 2*np.pi, height)
    xx, yy = np.meshgrid(x, y)
    
    # Create some patterns
    spectrogram = np.sin(xx) * np.cos(yy)
    
    # Add some random noise
    spectrogram += np.random.normal(0, 0.1, (height, width))
    
    # Normalize to [0, 1]
    spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
    
    # Convert to PyTorch tensor [batch, channel, height, width]
    spectrogram_tensor = torch.from_numpy(spectrogram).float().unsqueeze(0).unsqueeze(0)
    
    # Ensure it's a contiguous tensor
    spectrogram_tensor = spectrogram_tensor.contiguous()
    
    return spectrogram_tensor

def visualize_output(input_tensor, output_tensor, save_path=None):
    """
    Visualize the input and output for comparison.
    
    Args:
        input_tensor: Input tensor [batch, channel, height, width]
        output_tensor: Output tensor [batch, channel, height, width]
        save_path: Optional path to save the visualization
    """
    # Convert tensors to numpy arrays
    input_np = input_tensor.squeeze().detach().cpu().numpy()
    output_np = output_tensor.squeeze().detach().cpu().numpy()
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot input spectrogram
    im1 = ax1.imshow(input_np, aspect='auto', cmap='viridis')
    ax1.set_title('Input Mel-Spectrogram')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frequency')
    fig.colorbar(im1, ax=ax1)
    
    # Plot output waveform as a 2D representation
    im2 = ax2.imshow(output_np, aspect='auto', cmap='viridis')
    ax2.set_title('Output Waveform (2D Representation)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Channel')
    fig.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.close()

def run_inference(model, input_tensor):
    """
    Run inference with the model.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor
        
    Returns:
        Output tensor from model
    """
    print("Running inference...")
    
    # Ensure model is in eval mode
    model.eval()
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Inference successful! Output shape: {output.shape}")
    return output

def main(model_path, output_dir):
    """
    Main function to test the student model.
    
    Args:
        model_path: Path to the saved student model
        output_dir: Directory to save outputs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the student model
    student_model = load_student_model(model_path)
    
    # Print model information
    model_size = estimate_model_size(student_model)
    print(f"Model parameters: {model_size['total_parameters']:,}")
    print(f"Estimated memory: {model_size['memory_megabytes']:.2f} MB")
    
    # Generate dummy input
    dummy_input = generate_dummy_spectrogram()
    print(f"Created dummy input with shape: {dummy_input.shape}")
    
    # Run inference
    output = run_inference(student_model, dummy_input)
    
    # Visualize results
    viz_path = os.path.join(output_dir, "inference_visualization.png")
    visualize_output(dummy_input, output, viz_path)
    
    # Save a sample of the output
    output_path = os.path.join(output_dir, "output_sample.npy")
    output_np = output.squeeze().detach().cpu().numpy()
    np.save(output_path, output_np)
    print(f"Output sample saved to {output_path}")
    
    print("\nTesting completed successfully!")
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test student HiFi-GAN model")
    parser.add_argument("--model_path", type=str, default="student_model.pt", help="Path to student model checkpoint")
    parser.add_argument("--output_dir", type=str, default="test_output", help="Directory to save outputs")
    
    args = parser.parse_args()
    
    main(args.model_path, args.output_dir)