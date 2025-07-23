#!/bin/bash
#SBATCH --job-name=dp2_array
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --account=tc067-s2737744
#SBATCH --array=0-3
#SBATCH --output=/work/tc067/tc067/s2737744/logs/dp/array_%A_%a.out
#SBATCH --error=/work/tc067/tc067/s2737744/logs/dp/array_%A_%a.err

source /work/tc067/tc067/s2737744/miniconda3/etc/profile.d/conda.sh
conda activate dp2

export IOPATH_CACHE=/work/tc067/tc067/s2737744/shared_cache/iopath
export FVCORE_CACHE=/work/tc067/tc067/s2737744/shared_cache/fvcore
export XDG_CACHE_HOME=/work/tc067/tc067/s2737744/shared_cache/xdg
export MPLCONFIGDIR=/work/tc067/tc067/s2737744/shared_cache/mpl
export PYTHONPATH=/work/tc067/tc067/s2737744/detectron2:$PYTHONPATH

cd /work/tc067/tc067/s2737744/deep_privacy2

LIST_FILE="/work/tc067/tc067/s2737744/ucf101_part_$(printf '%02d' $SLURM_ARRAY_TASK_ID)"
OUTPUT_BASE="/work/tc067/tc067/s2737744/output/ucf101_anonymized"
CONFIG_PATH="configs/anonymizers/FB_cse.py"

mkdir -p "$OUTPUT_BASE"

while read -r video; do
    class_dir=$(basename "$(dirname "$video")")
    video_name=$(basename "$video" .avi)
    output_dir="$OUTPUT_BASE/$class_dir"
    output_file="$output_dir/${video_name}_anonymized.mp4"

    mkdir -p "$output_dir"

    if [[ -f "$output_file" ]]; then
        echo "Skipping $output_file"
        continue
    fi

    python3 anonymize.py "$CONFIG_PATH" -i "$video" --output_path "$output_file"
done < "$LIST_FILE"

echo "✅ Task $SLURM_ARRAY_TASK_ID done."