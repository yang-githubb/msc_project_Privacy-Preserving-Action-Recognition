# Privacy-Preserving Video Anonymization with Temporal Consistency

This project implements a full-body video anonymization system that replaces people in videos with consistent, synthetic figures while preserving action recognition capabilities.

## Project Overview

### Research Context
Recent advances in video analysis have made privacy preservation critical. Traditional anonymization methods like blurring or pixelation often distort important visual features needed for downstream tasks like action recognition. This project addresses the challenge of maintaining temporal consistency in synthetic identities across video frames while preserving privacy.

### Key Features
- **Pose-conditioned GAN**: Generates synthetic figures based on human pose keypoints
- **Temporal Consistency**: Maintains identity consistency across video frames
- **Action Recognition**: Evaluates utility preservation through action classification
- **Privacy Evaluation**: Comprehensive assessment of anonymization effectiveness

## Project Structure

```
├── config.py                          # Configuration settings
├── requirements.txt                   # Python dependencies
├── README.md                         # This file
├── src/
│   ├── data/                         # Dataset loading and preprocessing
│   ├── models/                       # Neural network models
│   ├── training/                     # Training pipeline
│   └── evaluation/                   # Evaluation metrics
├── scripts/
│   ├── test_job.slurm               # GPU test script
│   └── test_pytorch_job.py          # PyTorch test script
├── data/                             # Dataset directory
├── models/                           # Saved model checkpoints
└── outputs/                          # Training outputs and logs
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.2.0+
- CUDA 11.8+ (for GPU training)

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create necessary directories
python -c "from config import Config; Config().create_directories()"
```

## Usage

### Testing Environment
```bash
# Test GPU availability
sbatch scripts/test_job.slurm

# Test PyTorch installation
python scripts/test_pytorch_job.py
```

## Configuration

Key parameters can be modified in `config.py`:

```python
class Config:
    # Video settings
    VIDEO_HEIGHT = 160
    VIDEO_WIDTH = 288
    
    # Model settings
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0002
    
    # Training settings
    NUM_EPOCHS = 100
    TEMPORAL_WINDOW_SIZE = 5
```

## Development Status

- [x] Project structure setup
- [x] Configuration system
- [x] Basic documentation
- [ ] Dataset preprocessing
- [ ] Model implementation
- [ ] Training pipeline
- [ ] Evaluation framework

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please contact the project maintainers. 