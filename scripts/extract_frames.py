import cv2
import os
import argparse

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_dir}/frame_{idx:05d}.jpg", frame)
        idx += 1
    cap.release()
    print(f"Extracted {idx-1} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from two videos (original and anonymized)")
    parser.add_argument('--original_video', type=str, required=True, help='Path to the original video')
    parser.add_argument('--original_output', type=str, required=True, help='Output directory for original frames')
    parser.add_argument('--anon_video', type=str, required=True, help='Path to the anonymized video')
    parser.add_argument('--anon_output', type=str, required=True, help='Output directory for anonymized frames')
    args = parser.parse_args()

    extract_frames(args.original_video, args.original_output)
    extract_frames(args.anon_video, args.anon_output)