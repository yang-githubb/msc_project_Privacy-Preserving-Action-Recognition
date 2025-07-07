#!/usr/bin/env python3
"""
Process Kinetics videos with Vid2Vid framework
Extracts frames, detects poses, and generates enhanced videos
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import subprocess
import sys

class KineticsVid2VidProcessor:
    def __init__(self, kinetics_path="../kinetics-dataset/k400", output_dir="kinetics_vid2vid_output"):
        self.kinetics_path = Path(kinetics_path)
        self.output_dir = Path(output_dir)
        self.frame_rate = 30
        self.frame_size = (256, 256)  # Standard size for vid2vid
        
    def create_output_dirs(self, video_name):
        """Create output directory structure for a video"""
        video_output = self.output_dir / video_name
        
        # Create directories
        (video_output / "frames").mkdir(parents=True, exist_ok=True)
        (video_output / "poses").mkdir(parents=True, exist_ok=True)
        (video_output / "enhanced").mkdir(parents=True, exist_ok=True)
        (video_output / "metadata").mkdir(parents=True, exist_ok=True)
        
        return video_output
    
    def extract_frames(self, video_path, output_dir, max_frames=None):
        """Extract frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        frame_count = 0
        extracted_frames = []
        
        with tqdm(total=total_frames, desc=f"Extracting frames from {video_path.name}") as pbar:
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save frame
                frame_filename = f"frame_{frame_count:05d}.jpg"
                frame_path = output_dir / "frames" / frame_filename
                cv2.imwrite(str(frame_path), frame)
                
                extracted_frames.append({
                    'frame_number': frame_count,
                    'filename': frame_filename,
                    'path': str(frame_path)
                })
                
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        
        # Save frame metadata
        metadata = {
            'video_name': video_path.name,
            'total_frames': len(extracted_frames),
            'fps': fps,
            'frame_size': self.frame_size,
            'frames': extracted_frames
        }
        
        with open(output_dir / "metadata" / "frames_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Extracted {len(extracted_frames)} frames from {video_path.name}")
        return True
    
    def detect_poses(self, frames_dir, poses_dir):
        """Detect poses using DensePose and save visualizations"""
        import torch
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        from densepose import add_densepose_config
        from densepose.vis.extractor import DensePoseResultExtractor
        from densepose.vis.densepose_results import DensePoseResultsCoarseSegmentationVisualizer

        cfg = get_cfg()
        add_densepose_config(cfg)
        config_path = "/work/tc067/tc067/s2737744/detectron2/projects/DensePose/configs/cse/densepose_rcnn_R_50_FPN_s1x.yaml"
        cfg.merge_from_file(config_path)
        
        # Try to use local model first, otherwise download from model zoo
        local_model_path = "/work/tc067/tc067/s2737744/models/DensePose_ResNet50_FPN_s1x-e2e.pkl"
        if os.path.exists(local_model_path):
            cfg.MODEL.WEIGHTS = local_model_path
            print(f"Using local CSE model: {local_model_path}")
        else:
            # Use model zoo URL for CSE model
            cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
            print(f"Using CSE model from model zoo: {cfg.MODEL.WEIGHTS}")
        
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        predictor = DefaultPredictor(cfg)

        frame_files = sorted(list(frames_dir.glob("*.jpg")))
        print(f"Running DensePose on {len(frame_files)} frames...")

        processed_frames = 0
        for frame_file in tqdm(frame_files, desc="DensePose"):
            try:
                img = cv2.imread(str(frame_file))
                if img is None:
                    print(f"Could not read image: {frame_file}")
                    continue
                    
                outputs = predictor(img)
                instances = outputs["instances"]
                
                # Check if instances has DensePose predictions and is not empty
                if hasattr(instances, "pred_densepose") and len(instances) > 0:
                    try:
                        # Extract DensePose results using the proper extractor
                        extractor = DensePoseResultExtractor()
                        densepose_results, boxes_xywh = extractor(instances)
                        
                        if densepose_results is not None and len(densepose_results) > 0:
                            # Create visualization using the coarse segmentation visualizer
                            visualizer = DensePoseResultsCoarseSegmentationVisualizer()
                            vis_img = visualizer.visualize(img, (densepose_results, boxes_xywh))
                            
                            # Create a more visible overlay by blending with original image
                            alpha = 0.7  # Transparency for overlay
                            blended_img = cv2.addWeighted(img, 1-alpha, vis_img, alpha, 0)
                            
                            out_path = poses_dir / (frame_file.stem + "_pose.jpg")
                            cv2.imwrite(str(out_path), blended_img)
                            processed_frames += 1
                        else:
                            print(f"No valid DensePose results for frame {frame_file.name}, skipping.")
                            continue
                            
                    except Exception as e:
                        print(f"Error visualizing DensePose for frame {frame_file.name}: {e}")
                        continue
                else:
                    print(f"No DensePose results for frame {frame_file.name}, skipping.")
                    continue
                    
            except Exception as e:
                print(f"Error processing frame {frame_file.name}: {e}")
                continue

        print(f"✅ DensePose pose extraction completed. Processed {processed_frames}/{len(frame_files)} frames.")
        return processed_frames > 0
    
    def prepare_vid2vid_data(self, video_output):
        """Prepare data in the format expected by vid2vid"""
        # Create vid2vid dataset structure
        vid2vid_data = video_output / "vid2vid_data"
        (vid2vid_data / "source").mkdir(parents=True, exist_ok=True)
        (vid2vid_data / "target").mkdir(parents=True, exist_ok=True)
        (vid2vid_data / "pose").mkdir(parents=True, exist_ok=True)
        
        # Copy frames to source and target (for now, they're the same)
        frames_dir = video_output / "frames"
        poses_dir = video_output / "poses"
        
        frame_files = sorted(list(frames_dir.glob("*.jpg")))
        pose_files = sorted(list(poses_dir.glob("*_pose.jpg")))
        
        if len(pose_files) == 0:
            print("No pose files found. Cannot prepare vid2vid data without poses.")
            return None
        
        if len(frame_files) != len(pose_files):
            print(f"Warning: Number of frame files ({len(frame_files)}) does not match number of pose files ({len(pose_files)})")
            # Use the minimum number to avoid errors
            min_count = min(len(frame_files), len(pose_files))
            frame_files = frame_files[:min_count]
            pose_files = pose_files[:min_count]
        
        for i, (frame_file, pose_file) in enumerate(zip(frame_files, pose_files)):
            source_path = vid2vid_data / "source" / f"{i:05d}.jpg"
            target_path = vid2vid_data / "target" / f"{i:05d}.jpg"
            pose_path = vid2vid_data / "pose" / f"{i:05d}.jpg"
            
            frame_img = cv2.imread(str(frame_file))
            pose_img = cv2.imread(str(pose_file))

            if frame_img is None or pose_img is None:
                print(f"Warning: Could not read frame or pose image for index {i}")
                continue

            frame_resized = cv2.resize(frame_img, self.frame_size)
            pose_resized = cv2.resize(pose_img, self.frame_size)

            cv2.imwrite(str(source_path), frame_resized)
            cv2.imwrite(str(target_path), frame_resized)
            cv2.imwrite(str(pose_path), pose_resized)
        
        print(f"Prepared vid2vid data for {len(frame_files)} frames")
        return vid2vid_data
    
    def run_vid2vid_test(self, vid2vid_data, video_name):
        """Run vid2vid test using the original vid2vid test.py script"""
        print(f"Running vid2vid test on {video_name}...")
        
        # Change to vid2vid directory
        original_dir = os.getcwd()
        vid2vid_dir = "/work/tc067/tc067/s2737744/vid2vid"
        os.chdir(vid2vid_dir)
        
        try:
            # Run vid2vid test with our prepared data
            cmd = f"""python test.py \
                --dataroot {vid2vid_data} \
                --name pose2vid_test \
                --model vid2vid \
                --dataset_mode pose \
                --input_nc 6 \
                --output_nc 3 \
                --loadSize 256 \
                --fineSize 256 \
                --resize_or_crop scaleWidth_and_crop \
                --no_flip \
                --phase test \
                --how_many 30 \
                --results_dir {vid2vid_data}/results"""
            
            print(f"Running command: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Vid2vid test completed successfully!")
                print(f"Results saved to: {vid2vid_data}/results")
            else:
                print(f"❌ Vid2vid test failed with error: {result.stderr}")
                
        except Exception as e:
            print(f"Error running vid2vid test: {e}")
        finally:
            # Return to original directory
            os.chdir(original_dir)
        
        return True
    
    def process_video(self, video_path, max_frames=None):
        video_name = video_path.stem
        video_output = self.create_output_dirs(video_name)
        
        print(f"\nProcessing video: {video_name}")
        print("=" * 50)
        
        print("Step 1: Extracting frames...")
        if not self.extract_frames(video_path, video_output, max_frames):
            return False
        
        print("\nStep 2: Detecting poses...")
        frames_dir = video_output / "frames"
        poses_dir = video_output / "poses"
        if not self.detect_poses(frames_dir, poses_dir):
            print("No poses detected. Skipping vid2vid preparation.")
            return False
        
        print("\nStep 3: Preparing vid2vid data...")
        vid2vid_data = self.prepare_vid2vid_data(video_output)
        if vid2vid_data is None:
            print("Could not prepare vid2vid data. Skipping vid2vid test.")
            return False
        
        print("\nStep 4: Running vid2vid test...")
        self.run_vid2vid_test(vid2vid_data, video_name)
        
        print(f"\nProcessing completed for {video_name}")
        print(f"Output saved to: {video_output}")
        return True
    
    def process_split(self, split='test', max_videos=None, max_frames_per_video=None):
        split_dir = self.kinetics_path / split
        
        if not split_dir.exists():
            print(f"Split directory {split_dir} does not exist!")
            return
        
        video_files = list(split_dir.glob("*.mp4"))
        if max_videos:
            video_files = video_files[:max_videos]
        
        print(f"Found {len(video_files)} videos in {split} split")
        
        processed_count = 0
        for video_file in video_files:
            try:
                success = self.process_video(video_file, max_frames_per_video)
                if success:
                    processed_count += 1
            except Exception as e:
                print(f"Error processing {video_file.name}: {e}")
                continue
        
        print(f"\nProcessing completed!")
        print(f"Successfully processed {processed_count}/{len(video_files)} videos")
        print(f"Output directory: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Process Kinetics videos with Vid2Vid')
    parser.add_argument('--kinetics_path', type=str, default='../kinetics-dataset/k400')
    parser.add_argument('--output_dir', type=str, default='kinetics_vid2vid_output')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--max_videos', type=int, default=None)
    parser.add_argument('--max_frames_per_video', type=int, default=None)
    
    args = parser.parse_args()
    
    processor = KineticsVid2VidProcessor(
        kinetics_path=args.kinetics_path,
        output_dir=args.output_dir
    )
    
    processor.process_split(
        split=args.split,
        max_videos=args.max_videos,
        max_frames_per_video=args.max_frames_per_video
    )

if __name__ == "__main__":
    main()
