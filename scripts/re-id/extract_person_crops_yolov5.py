import os
import cv2
import torch
from pathlib import Path

# Load YOLOv5s model from torch.hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0]

def extract_person_crops(frames_dir, output_dir, max_frames=None):
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_files = sorted([f for f in frames_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png']])
    if max_frames:
        frame_files = frame_files[:max_frames]
    crop_idx = 1
    for frame_path in frame_files:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        results = model(img)
        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_fname = f"person_{crop_idx:03d}.jpg"
            cv2.imwrite(str(output_dir / crop_fname), crop)
            crop_idx += 1
    print(f"Saved {crop_idx-1} person crops to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract person crops from frames using YOLOv5.")
    parser.add_argument('--frames_dir', type=str, required=True, help='Directory with input frames')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save person crops')
    parser.add_argument('--max_frames', type=int, default=None, help='Max number of frames to process')
    args = parser.parse_args()
    extract_person_crops(args.frames_dir, args.output_dir, args.max_frames) 