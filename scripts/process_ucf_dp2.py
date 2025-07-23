import os
import subprocess
from pathlib import Path

# Paths
UCF_ROOT = Path("/work/tc067/tc067/s2737744/Dataset/ucf101/UCF-101")
OUTPUT_ROOT = Path("/work/tc067/tc067/s2737744/output/ucf101_anonymized")
CONFIG_PATH = "configs/anonymizers/FB_cse.py"
ANON_SCRIPT = "/work/tc067/tc067/s2737744/deep_privacy2/anonymize.py"
WORKDIR = "/work/tc067/tc067/s2737744/deep_privacy2"

# Environment variables already set in current shell, so no need to re-set or re-activate
def get_all_videos(ucf_root):
    return sorted(ucf_root.rglob("*.avi"))

def get_output_path(video_path):
    class_dir = video_path.parent.name
    video_name = video_path.stem
    output_dir = OUTPUT_ROOT / class_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{video_name}_anonymized.mp4"

def process_video(video_path):
    output_path = get_output_path(video_path)
    
    if output_path.exists():
        print(f"✅ Already processed: {output_path.name}")
        return

    print(f"🚀 Processing: {video_path.name}")
    cmd = [
        "python3", ANON_SCRIPT,
        CONFIG_PATH,
        "-i", str(video_path),
        "--output_path", str(output_path)
    ]
    subprocess.run(cmd, cwd=WORKDIR)

def main():
    videos = get_all_videos(UCF_ROOT)
    print(f"🎬 Total videos found: {len(videos)}")
    for video_path in videos:
        try:
            process_video(video_path)
        except Exception as e:
            print(f"❌ Failed to process {video_path.name}: {e}")

if __name__ == "__main__":
    main()
