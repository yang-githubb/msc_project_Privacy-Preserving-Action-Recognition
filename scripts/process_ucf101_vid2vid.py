from pathlib import Path
import argparse
import cv2
import numpy as np
import json
from tqdm import tqdm
import os
import subprocess

class Vid2VidProcessor:
    def __init__(self, ucf101_path, output_dir):
        self.ucf101_path = Path(ucf101_path)
        self.output_dir = Path(output_dir)
        self.frame_size = (256, 256)  # Standard size for vid2vid

    def create_output_dirs(self, video_name):
        """Create output directory structure for a video"""
        video_output = self.output_dir / video_name
        
        # Create directories
        (video_output / "frames").mkdir(parents=True, exist_ok=True)
        (video_output / "poses").mkdir(parents=True, exist_ok=True)
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
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
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
                            # Create visualization using the coarse segmentation visualizer for CSE model
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

    def process_video(self, video_path, max_frames=None):
        """Process a single video"""
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
        
        print(f"\nProcessing completed for {video_name}")
        print(f"Output saved to: {video_output}")
        return True

    def process_videos(self, max_videos=None, max_frames_per_video=None):
        video_files = list(self.ucf101_path.glob("*.avi"))
        if max_videos:
            video_files = video_files[:max_videos]
        print(f"Found {len(video_files)} videos in {self.ucf101_path}")
        processed_count = 0
        processed_videos = []
        for video_file in video_files:
            try:
                success = self.process_video(video_file, max_frames_per_video)
                if success:
                    processed_count += 1
                    video_name = video_file.stem
                    video_output = self.output_dir / video_name
                    processed_videos.append((video_name, video_output))
            except Exception as e:
                print(f"Error processing {video_file.name}: {e}")
                continue
        print(f"\nProcessing completed!")
        print(f"Successfully processed {processed_count}/{len(video_files)} videos")
        print(f"Output directory: {self.output_dir}")
        # Write summary file
        summary_path = Path(self.output_dir) / "all_outputs.txt"
        with open(summary_path, "w") as f:
            for video_name, video_output in processed_videos:
                f.write(f"{video_name}: {video_output}\n")
        print(f"Summary of outputs written to {summary_path}")

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

def main():
    parser = argparse.ArgumentParser(description='Process UCF101 videos with Vid2Vid')
    parser.add_argument('--ucf101_path', type=str, required=True, help='Path to UCF101 video directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed videos')
    parser.add_argument('--max_videos', type=int, default=None, help='Maximum number of videos to process')
    parser.add_argument('--max_frames_per_video', type=int, default=None, help='Maximum frames per video')
    
    args = parser.parse_args()
    
    processor = Vid2VidProcessor(
        ucf101_path=args.ucf101_path,
        output_dir=args.output_dir
    )
    
    processor.process_videos(
        max_videos=args.max_videos,
        max_frames_per_video=args.max_frames_per_video
    )

if __name__ == "__main__":
    main() 