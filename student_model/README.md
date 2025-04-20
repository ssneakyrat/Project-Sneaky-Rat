# Student HiFi-GAN Model

This is a PyTorch implementation of a student HiFi-GAN model for efficient audio synthesis through knowledge distillation.

## Model Architecture

- **Model Type**: HiFiGANStudent2D
- **Convolution Type**: 2D
- **Target Parameters**: ~4,250,577 (30% of teacher model)
- **Uses Depthwise Separable Convolutions**: True
- **Simplified MRF**: True
- **Activation Function**: LeakyRelu

## Project Status

### Completed:
- ✅ Teacher model analysis and architecture design
- ✅ Student model implementation with efficient layers
- ✅ Weight mapping from teacher to student model
- ✅ Inference testing with dummy data

### In Progress:
- 🔄 Knowledge distillation setup

### To Do:
- 📝 Set up training dataset and dataloader
- 📝 Implement distillation loss functions 
- 📝 Create training pipeline
- 📝 Fine-tune hyperparameters
- 📝 Evaluate audio quality metrics

## Usage

```python
from student_model import HiFiGANStudent2D
from student_model.config import StudentConfig

# Create model
model = HiFiGANStudent2D()

# OR create from config
config = StudentConfig()
model = HiFiGANStudent2D(config)

# Load pre-mapped weights
model.load_state_dict(torch.load('student_model.pt'))

# Forward pass
x = torch.randn(1, 1, 80, 80)  # [batch_size, channels, height, width]
y = model(x)
```

## Files

- `model.py`: Main model implementation
- `layers.py`: Custom layers like depthwise separable convolutions and MRF
- `config.py`: Configuration settings
- `utils.py`: Utility functions for the model
- `weight_mapping.py`: Script to map weights from teacher to student model
- `test_student_model.py`: Test script for inference with the student model
- `run_weight_mapping_and_test.py`: Runner script for weight mapping and testing

## Weight Mapping

The weight mapping process transfers knowledge from the teacher model to the student model by:

1. Extracting weights from the ONNX teacher model
2. Applying channel reduction strategies based on importance
3. Converting transposed convolutions to standard convolutions
4. Simplifying Multi-Receptive Field fusion modules
5. Applying the mapped weights to the student model

Run the weight mapping and testing with:

```bash
python run_weight_mapping_and_test.py --onnx_path nsf_hifigan.onnx --arch_json student_architecture.json
```

## Next Steps

The immediate next step is to implement the knowledge distillation training pipeline:

1. Create a dataset loader for mel-spectrograms and corresponding audio
2. Implement feature matching loss between teacher and student intermediate layers
3. Set up the training loop with appropriate optimization strategies
4. Add validation metrics to track audio quality during training

This will enable the student model to learn from the teacher while maintaining high audio quality with significantly fewer parameters.