#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=tc067-s2737744
#SBATCH --output=../outputs/anonymization_training_%j.out
#SBATCH --error=../outputs/anonymization_training_%j.err
#SBATCH --job-name=video_anon

# Load required modules
module load pytorch/2.2.0-gpu
module load cuda/11.8

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create output directory
mkdir -p ../outputs

# Print job information
echo "=== Video Anonymization Training Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $(pwd)"
echo "Start Time: $(date)"

# Check GPU availability
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Run training (placeholder for now)
echo "Training script will be implemented in Week 2"
echo "Current status: Environment setup complete"

# Print completion information
echo "Environment setup completed at: $(date)"
echo "Job finished successfully!" 