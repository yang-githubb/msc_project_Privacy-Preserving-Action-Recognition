#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --account=tc067-s2737744
#SBATCH --output=../outputs/test_gpu_output.txt

# Load GPU-compatible PyTorch module
module load pytorch/2.2.0-gpu

# Run your Python script
srun python ./test_pytorch_job.py
