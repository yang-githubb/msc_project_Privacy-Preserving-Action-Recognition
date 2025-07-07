#!/bin/bash
#SBATCH --job-name=kinetics_vid2vid
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=tc067-s2737744
#SBATCH --output=/work/tc067/tc067/s2737744/logs/prepare_kinetic_for_vid2vid/prepare_kinetics_vid2vid_%j.out
#SBATCH --error=/work/tc067/tc067/s2737744/logs/prepare_kinetic_for_vid2vid/prepare_kinetics_vid2vid_%j.err

# Set working directory
cd /work/tc067/tc067/s2737744

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/work/tc067/tc067/s2737744/vid2vid:/work/tc067/tc067/s2737744/detectron2:/work/tc067/tc067/s2737744/detectron2/projects/DensePose:$PYTHONPATH"

# Use Python directly from the conda environment
PYTHON_PATH="/work/tc067/tc067/s2737744/miniconda3/envs/project/bin/python"

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "CUDA device: $CUDA_VISIBLE_DEVICES"
echo "Python path: $PYTHON_PATH"
echo "Python version: $($PYTHON_PATH --version)"

# Run the Kinetics vid2vid processing
echo "Starting Kinetics vid2vid processing..."
$PYTHON_PATH scripts/process_kinetics_vid2vid.py \
    --kinetics_path Dataset/kinetics-dataset/k400 \
    --output_dir output/kinetics_vid2vid_output \
    --split test \
    --max_videos 10 \
    --max_frames_per_video 30

echo "Processing completed!"
