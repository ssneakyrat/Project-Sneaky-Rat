#!/usr/bin/env python3
"""
run_weight_mapping_and_test.py - Run weight mapping and inference testing.
"""

import os
import argparse
import torch
import sys
from pathlib import Path
# Add current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our scripts
from student_model.weight_mapping import main as run_weight_mapping
from test_student_model import main as run_test

def main(onnx_path, arch_json, model_output_path, test_output_dir):
    """
    Run weight mapping and then test the resulting model.
    
    Args:
        onnx_path: Path to teacher ONNX model
        arch_json: Path to student architecture JSON
        model_output_path: Path to save the mapped student model
        test_output_dir: Directory to save test outputs
    """
    print("\n" + "="*50)
    print("STEP 1: WEIGHT MAPPING")
    print("="*50)
    
    # Run weight mapping
    student_model, size_info = run_weight_mapping(onnx_path, arch_json, model_output_path)
    
    print("\n" + "="*50)
    print("STEP 2: MODEL TESTING")
    print("="*50)
    
    # Run model testing
    output = run_test(model_output_path, test_output_dir)
    
    print("\n" + "="*50)
    print("PROCESS COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"Student model saved to: {model_output_path}")
    print(f"Test outputs saved to: {test_output_dir}")
    print(f"Student model parameters: {size_info['total_parameters']:,}")
    print(f"Parameter reduction: {(1 - size_info['total_parameters']/14168590)*100:.2f}%")
    
    return student_model, output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run weight mapping and test for HiFi-GAN student model")
    parser.add_argument("--onnx_path", type=str, default="distilation/nsf_hifigan.onnx", help="Path to teacher ONNX model")
    parser.add_argument("--arch_json", type=str, default="distilation/student_architecture.json", help="Path to student architecture JSON")
    parser.add_argument("--model_output", type=str, default="student_model.pt", help="Path to save student model")
    parser.add_argument("--test_output", type=str, default="test_output", help="Directory to save test outputs")
    
    args = parser.parse_args()
    
    main(args.onnx_path, args.arch_json, args.model_output, args.test_output)
