# Student HiFi-GAN Model

This is a PyTorch implementation of a student HiFi-GAN model for efficient audio synthesis.

## Model Architecture

- **Model Type**: HiFiGANStudent2D
- **Convolution Type**: 2D
- **Target Parameters**: ~4,250,577 (30% of teacher model)
- **Uses Depthwise Separable Convolutions**: True
- **Simplified MRF**: True
- **Activation Function**: LeakyRelu

## Usage

```python
from student_model import HiFiGANStudent2D
from student_model.config import StudentConfig

# Create model
model = HiFiGANStudent2D()

# OR create from config
config = StudentConfig()
model = HiFiGANStudent2D(config)

# Forward pass
x = torch.randn(1, 1, 80)  # [batch_size, channels, time]
y = model(x)
```

## Files

- `model.py`: Main model implementation
- `layers.py`: Custom layers like depthwise separable convolutions and MRF
- `config.py`: Configuration settings
- `utils.py`: Utility functions for the model

## Next Steps

1. Initialize weights from teacher model
2. Set up knowledge distillation
3. Fine-tune the student model
