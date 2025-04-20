import onnx
import numpy as np
import onnxruntime as ort
from collections import defaultdict
import json
import os
import datetime

def analyze_hifigan_onnx(onnx_path):
    """
    Load and analyze a HiFi-GAN ONNX model to understand its architecture.
    """
    print(f"Loading ONNX model from {onnx_path}...")
    
    # Load the model
    model = onnx.load(onnx_path)
    
    # Basic model information
    print(f"Model IR version: {model.ir_version}")
    print(f"Producer: {model.producer_name}")
    
    # Initialize counters for different node types
    node_types = defaultdict(int)
    conv_layers = []
    transposed_conv_layers = []
    
    # Analyze the graph structure
    graph = model.graph
    print(f"\nModel Inputs: {[i.name for i in graph.input]}")
    print(f"Model Outputs: {[o.name for o in graph.output]}")
    
    # Analyze nodes
    print("\nAnalyzing model architecture...")
    for node in graph.node:
        node_types[node.op_type] += 1
        
        # Collect detailed info about convolution layers
        if node.op_type == 'Conv':
            attrs = {}
            for attr in node.attribute:
                if attr.name == 'kernel_shape':
                    # Handle both 1D and 2D kernel shapes
                    attrs[attr.name] = list(attr.ints)
                else:
                    # Handle other attributes appropriately
                    if hasattr(attr, 'i'):
                        attrs[attr.name] = attr.i
                    elif hasattr(attr, 'ints'):
                        attrs[attr.name] = list(attr.ints)
                    elif hasattr(attr, 'f'):
                        attrs[attr.name] = attr.f
                    elif hasattr(attr, 'fs'):
                        attrs[attr.name] = list(attr.fs)
                    elif hasattr(attr, 's'):
                        attrs[attr.name] = attr.s
                    elif hasattr(attr, 'ss'):
                        attrs[attr.name] = list(attr.ss)
            
            conv_layers.append({
                'name': node.name,
                'input': node.input,
                'output': node.output,
                'attributes': attrs
            })
        
        # Similar approach for transposed convolution layers
        elif node.op_type == 'ConvTranspose':
            attrs = {}
            for attr in node.attribute:
                if attr.name == 'kernel_shape':
                    attrs[attr.name] = list(attr.ints)
                else:
                    if hasattr(attr, 'i'):
                        attrs[attr.name] = attr.i
                    elif hasattr(attr, 'ints'):
                        attrs[attr.name] = list(attr.ints)
                    elif hasattr(attr, 'f'):
                        attrs[attr.name] = attr.f
                    elif hasattr(attr, 'fs'):
                        attrs[attr.name] = list(attr.fs)
                    elif hasattr(attr, 's'):
                        attrs[attr.name] = attr.s
                    elif hasattr(attr, 'ss'):
                        attrs[attr.name] = list(attr.ss)
            
            transposed_conv_layers.append({
                'name': node.name,
                'input': node.input,
                'output': node.output,
                'attributes': attrs
            })
    
    # Summarize node types
    print("\nNode Type Distribution:")
    for op_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {op_type}: {count}")
    
    # Analyze initializers (weights and biases)
    print("\nAnalyzing model parameters...")
    initializer_types = defaultdict(int)
    total_params = 0
    layer_params = defaultdict(int)
    
    # Additional detailed info for PyTorch conversion
    layer_details = []
    
    for initializer in graph.initializer:
        shape = [dim for dim in initializer.dims]
        num_params = np.prod(shape)
        total_params += num_params
        
        # Attempt to associate with layer type
        name_lower = initializer.name.lower()
        if "conv" in name_lower:
            if "weight" in name_lower:
                layer_type = "conv_weights"
            else:
                layer_type = "conv_biases"
        elif "upsample" in name_lower:
            layer_type = "upsample"
        elif "resblock" in name_lower:
            layer_type = "resblock"
        # Additional patterns for NSF-HiFiGAN 
        elif "f0_upsamp" in name_lower or "f0" in name_lower:
            layer_type = "f0_processing"
        elif "source" in name_lower:
            layer_type = "source_module"
        else:
            layer_type = "other"
            
        layer_params[layer_type] += num_params
        initializer_types[initializer.data_type] += 1
        
        # Collect detailed layer info for PyTorch implementation
        layer_details.append({
            'name': initializer.name,
            'shape': shape,
            'type': layer_type,
            'data_type': initializer.data_type
        })
    
    print(f"Total parameters: {total_params:,}")
    print("\nParameter Distribution:")
    for layer_type, count in sorted(layer_params.items(), key=lambda x: x[1], reverse=True):
        print(f"  {layer_type}: {count:,} ({100 * count / total_params:.2f}%)")
    
    # Try to identify upsampling factors
    upsampling_factors = []
    for conv in transposed_conv_layers:
        if 'strides' in conv['attributes']:
            upsampling_factors.append(conv['attributes']['strides'])
    
    print("\nDetected Upsampling Factors:", upsampling_factors)
    
    # Also collect the convolution dimensions to determine if 1D or 2D
    conv_dimensions = set()
    for conv in conv_layers:
        if 'kernel_shape' in conv['attributes']:
            conv_dimensions.add(len(conv['attributes']['kernel_shape']))
    
    print(f"\nDetected Convolution Dimensions: {conv_dimensions} (1 = 1D convs, 2 = 2D convs)")
    
    # Identify activation functions
    activation_types = set()
    for node in graph.node:
        if node.op_type in ['Relu', 'LeakyRelu', 'Sigmoid', 'Tanh', 'Elu', 'PRelu']:
            activation_types.add(node.op_type)
    
    print(f"\nDetected Activation Functions: {activation_types}")
    
    # Return analysis results
    return {
        'node_types': dict(node_types),
        'conv_layers': conv_layers,
        'transposed_conv_layers': transposed_conv_layers,
        'total_params': int(total_params),
        'layer_params': dict(layer_params),
        'upsampling_factors': upsampling_factors,
        'conv_dimensions': list(conv_dimensions),
        'model_inputs': [i.name for i in graph.input],
        'model_outputs': [o.name for o in graph.output],
        'layer_details': layer_details,
        'activation_functions': list(activation_types)
    }

def design_student_architecture(analysis):
    """
    Based on the analysis of the teacher model, suggest a student architecture
    with weight mapping strategy - enhanced for PyTorch 2.4.0 implementation.
    """
    # Extract key information from analysis
    total_params = analysis['total_params']
    upsampling_factors = analysis['upsampling_factors']
    conv_count = analysis['node_types'].get('Conv', 0)
    transposed_conv_count = analysis['node_types'].get('ConvTranspose', 0)
    conv_dimensions = analysis['conv_dimensions']
    activation_functions = analysis['activation_functions']
    
    # Design parameters
    channel_reduction_factor = 0.5  # Reduce channels by 50%
    target_params = int(total_params * 0.3)  # Target 30% of original params
    
    # Determine if 1D or 2D convolutions are used
    is_1d = 1 in conv_dimensions and 2 not in conv_dimensions
    conv_type = "1D" if is_1d else "2D"
    
    # Default activation function based on analysis
    default_activation = "LeakyRelu" if "LeakyRelu" in activation_functions else "Relu"
    
    print("\n" + "="*50)
    print("STUDENT ARCHITECTURE RECOMMENDATION")
    print("="*50)
    
    print(f"\nTeacher model has {total_params:,} parameters")
    print(f"Target: ~{target_params:,} parameters (70% reduction)")
    
    # Suggest architecture modifications
    print("\n1. Overall Architecture Modifications:")
    print("   - Replace transposed convolutions with upsampling + standard convolution")
    print("   - Reduce number of channels throughout the network")
    print("   - Simplify or reduce Multi-Receptive Field Fusion modules")
    print("   - Use depthwise separable convolutions for larger kernels")
    
    # Specific upsampling strategy
    print("\n2. Upsampling Strategy:")
    print(f"   - Teacher uses {len(upsampling_factors)} upsampling stages with factors: {upsampling_factors}")
    print("   - Student recommendation: Keep upsampling structure but replace with:")
    print("     * Nearest-neighbor upsampling followed by efficient convolutions")
    print("     * Reduce channels after each upsampling stage")
    
    # Channel reduction plan
    print("\n3. Channel Reduction Plan:")
    print("   - Initial layers: 50% channel reduction")
    print("   - Middle layers: 60% channel reduction")
    print("   - Final layers: 40% channel reduction (preserve quality in output stages)")
    
    # MRF simplification
    print("\n4. Multi-Receptive Field Fusion Simplification:")
    print("   - Reduce number of parallel paths")
    print("   - Use depthwise separable convolutions for larger kernels")
    print("   - Share weights across similar kernel paths where possible")
    
    # Generate PyTorch-specific architecture details
    # Define the student architecture in a PyTorch-friendly format
    
    # Define base channel counts (these would be derived from analysis in a full implementation)
    base_channels = 64 if conv_count > 20 else 32  # Estimated from model size
    
    # Simplified architecture definition for PyTorch
    if is_1d:  # 1D convolution based architecture
        student_arch = {
            'model_type': 'HiFiGANStudent',
            'conv_type': '1D',
            'input_channels': 1,  # Mel-spectrogram typically has 1 channel
            'base_channels': int(base_channels * channel_reduction_factor),
            'upsample_rates': upsampling_factors[0] if upsampling_factors else [4, 4, 4, 4],
            'upsample_kernel_sizes': [8, 8, 4, 4],  # Typical kernel sizes
            'upsample_initial_channel': int(512 * channel_reduction_factor),
            'resblock_kernel_sizes': [3, 5, 7],
            'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            'use_depthwise_separable': True,
            'simplified_mrf': True,
            'activation_function': default_activation,
            'leaky_relu_alpha': 0.1,
            'target_params': target_params
        }
    else:  # 2D convolution based architecture
        student_arch = {
            'model_type': 'HiFiGANStudent2D',
            'conv_type': '2D',
            'input_channels': 1,
            'base_channels': int(base_channels * channel_reduction_factor),
            'upsample_rates': upsampling_factors[0] if upsampling_factors else [4, 4, 4, 4],
            'upsample_kernel_sizes': [(8, 8), (8, 8), (4, 4), (4, 4)],  # 2D kernels
            'upsample_initial_channel': int(512 * channel_reduction_factor),
            'resblock_kernel_sizes': [(3, 3), (5, 5), (7, 7)],
            'resblock_dilation_sizes': [[(1, 1), (3, 3), (5, 5)], 
                                       [(1, 1), (3, 3), (5, 5)], 
                                       [(1, 1), (3, 3), (5, 5)]],
            'use_depthwise_separable': True,
            'simplified_mrf': True,
            'activation_function': default_activation,
            'leaky_relu_alpha': 0.1,
            'target_params': target_params
        }
    
    # Define the layer correspondence for weight mapping
    layer_mapping = {
        'initial_layers': {
            'teacher_prefix': 'initial_conv',
            'student_prefix': 'conv_pre',
            'reduction_factor': 0.5,
            'mapping_type': 'channel_reduction'
        },
        'upsampling_layers': [
            {
                'teacher_prefix': f'upsampling_{i+1}',
                'student_prefix': f'upsamples.{i}',
                'reduction_factor': 0.5,
                'mapping_type': 'transpose_to_standard'
            } for i in range(len(upsampling_factors) if upsampling_factors else 4)
        ],
        'mrfs': [
            {
                'teacher_prefix': f'mrf_block_{i+1}',
                'student_prefix': f'mrfs.{i}',
                'reduction_factor': 0.6,
                'mapping_type': 'mrf_simplification'
            } for i in range(len(upsampling_factors) if upsampling_factors else 4)
        ],
        'output_layer': {
            'teacher_prefix': 'output_conv',
            'student_prefix': 'conv_post',
            'reduction_factor': 0.4,
            'mapping_type': 'direct'
        }
    }
    
    # Complete student design with PyTorch implementation details
    student_design = {
        'recommended_architecture': student_arch,
        'layer_mapping': layer_mapping,
        'pytorch_version': '2.4.0',
        'framework': 'pytorch',
        'teacher_model_info': {
            'total_params': total_params,
            'conv_layers': conv_count,
            'transposed_conv_layers': transposed_conv_count,
            'upsampling_factors': upsampling_factors,
            'conv_dimensions': conv_dimensions,
            'activation_functions': activation_functions
        }
    }
    
    return student_design

def save_student_design_json(analysis, student_design, output_path="student_architecture.json"):
    """
    Save the student design and relevant teacher analysis to a JSON file.
    """
    # Combine analysis and student design into a complete specification
    design_spec = {
        'timestamp': datetime.datetime.now().isoformat(),
        'teacher_analysis': {
            'total_params': analysis['total_params'],
            'node_types': analysis['node_types'],
            'upsampling_factors': analysis['upsampling_factors'],
            'conv_dimensions': analysis['conv_dimensions'],
            'model_inputs': analysis['model_inputs'],
            'model_outputs': analysis['model_outputs'],
            'activation_functions': analysis['activation_functions']
        },
        'student_design': student_design
    }
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(design_spec, f, indent=2)
    
    print(f"\nStudent architecture design saved to {output_path}")
    return output_path

def main(onnx_path, output_json="student_architecture.json"):
    """
    Main function to analyze ONNX model, suggest student architecture,
    and save the design to a JSON file.
    """
    # Analyze the teacher model
    analysis = analyze_hifigan_onnx(onnx_path)
    
    # Design student architecture
    student_design = design_student_architecture(analysis)
    
    # Save the design to a JSON file
    json_path = save_student_design_json(analysis, student_design, output_json)
    
    # Output summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("\nTeacher Model:")
    print(f"- Total parameters: {analysis['total_params']:,}")
    print(f"- Conv layers: {analysis['node_types'].get('Conv', 0)}")
    print(f"- ConvTranspose layers: {analysis['node_types'].get('ConvTranspose', 0)}")
    
    print("\nRecommended Student Model:")
    print("- Expected parameters: ~30% of teacher")
    print("- Architecture: Simplified with efficient operations")
    print("- Weight initialization: Strategic mapping from teacher")
    
    print("\nNext steps:")
    print(f"1. Run build_student.py with the generated JSON file: {json_path}")
    print("2. Implement the weight mapping initialization using the generated model files")
    print("3. Set up knowledge distillation with feature matching")
    print("4. Fine-tune distillation hyperparameters")
    
    return analysis, student_design, json_path

# Example usage
if __name__ == "__main__":
    # Path to your ONNX model
    onnx_model_path = "nsf_hifigan.onnx"
    output_json_path = "student_architecture.json"
    
    # Run the analysis and architecture design
    main(onnx_model_path, output_json_path)