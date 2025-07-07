#!/bin/bash
#SBATCH --job-name=ucf101_basketball_vid2vid
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=tc067-s2737744
#SBATCH --output=/work/tc067/tc067/s2737744/logs/ucf101_vid2vid/ucf101_basketball_vid2vid_%j.out
#SBATCH --error=/work/tc067/tc067/s2737744/logs/ucf101_vid2vid/ucf101_basketball_vid2vid_%j.err

cd /work/tc067/tc067/s2737744

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/work/tc067/tc067/s2737744/vid2vid:/work/tc067/tc067/s2737744/detectron2:/work/tc067/tc067/s2737744/detectron2/projects/DensePose:$PYTHONPATH"

PYTHON_PATH="/work/tc067/tc067/s2737744/miniconda3/envs/project/bin/python"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "CUDA device: $CUDA_VISIBLE_DEVICES"
echo "Python path: $PYTHON_PATH"
echo "Python version: $($PYTHON_PATH --version)"

$PYTHON_PATH scripts/process_ucf101_vid2vid.py \
    --ucf101_path Dataset/ucf101/UCF-101/Basketball \
    --output_dir output/ucf101_basketball_vid2vid_output \
    --max_videos 10 \
    --max_frames_per_video 30

echo "Processing completed!"