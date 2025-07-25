import os
from pathlib import Path
import shutil

def organize_reid_dataset(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    query_dir = output_dir / 'query'
    gallery_dir = output_dir / 'gallery'
    query_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)

    pid = 1
    for person_folder in sorted(input_dir.iterdir()):
        if not person_folder.is_dir():
            continue
        crops = sorted([f for f in person_folder.iterdir() if f.suffix.lower() in ['.jpg', '.png']])
        if not crops:
            continue
        # First crop to query
        query_name = f"{pid:04d}_c1.jpg"
        shutil.copy(str(crops[0]), str(query_dir / query_name))
        # Rest to gallery
        for idx, crop in enumerate(crops[1:], start=1):
            gallery_name = f"{pid:04d}_c2_{idx:03d}.jpg"
            shutil.copy(str(crop), str(gallery_dir / gallery_name))
        pid += 1
    print(f"Organized dataset into {query_dir} and {gallery_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Organize person crops into Torchreid query/gallery folders.")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with person crop subfolders')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for Torchreid dataset')
    args = parser.parse_args()
    organize_reid_dataset(args.input_dir, args.output_dir) 