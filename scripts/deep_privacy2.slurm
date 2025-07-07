#!/bin/bash
#SBATCH --job-name=dp2_anonymize
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=tc067-s2737744
#SBATCH --output=/work/tc067/tc067/s2737744/logs/dp/dp2_%j.out
#SBATCH --error=/work/tc067/tc067/s2737744/logs/dp/dp2_%j.err

# Fix 1: Set writable cache directories
export IOPATH_CACHE=/work/tc067/tc067/s2737744/shared_cache/iopath
export FVCORE_CACHE=/work/tc067/tc067/s2737744/shared_cache/fvcore
export XDG_CACHE_HOME=/work/tc067/tc067/s2737744/shared_cache/xdg
export MPLCONFIGDIR=/work/tc067/tc067/s2737744/shared_cache/mpl

# Fix 2: Add detectron2 to Python path for local config files
export PYTHONPATH=/work/tc067/tc067/s2737744/detectron2:$PYTHONPATH

# Fix 3: Activate conda properly
source /work/tc067/tc067/s2737744/miniconda3/etc/profile.d/conda.sh
conda activate dp2

# Go to the DeepPrivacy2 directory
cd /work/tc067/tc067/s2737744/deep_privacy2

# Run the anonymizer
python3 anonymize.py configs/anonymizers/FB_cse.py \
-i /work/tc067/tc067/s2737744/Dataset/ucf101/UCF-101/Basketball/v_Basketball_g01_c01.avi \
--output_path /work/tc067/tc067/s2737744/output/dp_output.mp4
