#!/usr/bin/env python3
"""
config.py - Configuration settings for the student HiFi-GAN model.
Auto-generated from teacher model analysis.
"""

class StudentConfig:
    """Configuration for the student HiFi-GAN model."""
    
    # Model type and dimensions
    model_type = 'HiFiGANStudent2D'
    conv_type = '2D'
    input_channels = 1
    base_channels = 32
    upsample_rates = 0
    upsample_kernel_sizes = [[8, 8], [8, 8], [4, 4], [4, 4]]
    upsample_initial_channel = 256
    resblock_kernel_sizes = [[3, 3], [5, 5], [7, 7]]
    resblock_dilation_sizes = [[[1, 1], [3, 3], [5, 5]], [[1, 1], [3, 3], [5, 5]], [[1, 1], [3, 3], [5, 5]]]
    use_depthwise_separable = True
    simplified_mrf = True
    activation_function = 'LeakyRelu'
    leaky_relu_alpha = 0.1
    target_params = 4250577

    # Teacher model information
    teacher_total_params = 14168590
    teacher_conv_layers = 98
    teacher_transposed_conv_layers = 5
    teacher_upsampling_factors = [0, 0, 0, 0, 0]
    teacher_conv_dimensions = [1, 2]
    teacher_activation_functions = ['LeakyRelu', 'Tanh']

    # Generation information
    generated_date = '2025-04-20T14:36:52.508399'
