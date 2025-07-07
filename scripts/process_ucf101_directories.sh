#!/bin/bash
#SBATCH --job-name=dp2_all_ucf101
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=tc067-s2737744
#SBATCH --output=/work/tc067/tc067/s2737744/logs/dp/all_ucf101_%j.out
#SBATCH --error=/work/tc067/tc067/s2737744/logs/dp/all_ucf101_%j.err

# Fix Conda
source /work/tc067/tc067/s2737744/miniconda3/etc/profile.d/conda.sh
conda activate dp2

# Fix cache dirs
export IOPATH_CACHE=/work/tc067/tc067/s2737744/shared_cache/iopath
export FVCORE_CACHE=/work/tc067/tc067/s2737744/shared_cache/fvcore
export XDG_CACHE_HOME=/work/tc067/tc067/s2737744/shared_cache/xdg
export MPLCONFIGDIR=/work/tc067/tc067/s2737744/shared_cache/mpl
export PYTHONPATH=/work/tc067/tc067/s2737744/detectron2:$PYTHONPATH

# Set working directory
cd /work/tc067/tc067/s2737744/deep_privacy2

UCF_DIR="/work/tc067/tc067/s2737744/Dataset/ucf101/UCF-101"
OUTPUT_BASE="/work/tc067/tc067/s2737744/output/ucf101_anonymized"
CONFIG_PATH="configs/anonymizers/FB_cse.py"

mkdir -p "$OUTPUT_BASE"

echo "Starting batch processing of all UCF-101 videos..."

find "$UCF_DIR" -type f -name "*.avi" | while read -r video; do
    class_dir=$(basename "$(dirname "$video")")
    video_name=$(basename "$video" .avi)
    output_dir="$OUTPUT_BASE/$class_dir"
    output_file="$output_dir/${video_name}_anonymized.mp4"

    mkdir -p "$output_dir"

    if [[ -f "$output_file" ]]; then
        echo "Skipping already processed: $output_file"
        continue
    fi

    echo "Processing $video → $output_file"
    python3 anonymize.py "$CONFIG_PATH" \
        -i "$video" \
        --output_path "$output_file"

    echo "Done with $video"
    echo "------------------------------"
done

echo "✅ All videos processed!"
